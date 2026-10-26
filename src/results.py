"""
results.py — the ONE channel through which notebooks talk to each other and to the report.

WHY THIS FILE EXISTS: without it, the number in the abstract is whatever you last
typed, and it drifts away from the number the model actually produced the moment
you rerun anything. Here every notebook writes its numbers to `results.json`, and
the report reads only from there. `make clean && make run` therefore regenerates
the report exactly, and there is no path by which a stale MAE survives into the
submission.

The second job is ORDERING. `require()` raises with the name of what is missing,
so notebook 04 cannot silently run before notebook 01 and quietly use the default
lag set instead of the PACF-justified one.

Contract:
    R.save('nb04', dict(...))   merge a notebook's numbers (shallow merge)
    R.stale_keys()                keys left over from an EARLIER run of that notebook
    R.load()                      the whole dict
    R.get('nb04', 'worst_day')  one value, None if absent
    R.require('nb01', 'chosen_lags')  one value, raises if absent

Everything written must be JSON-serialisable. numpy scalars are NOT — cast with
float()/int() at the call site, deliberately, so you notice when a "number" is
actually an array.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from src import config as cfg

# ONE ID PER PROCESS, and a notebook runs in a fresh kernel, so one id per execution
# of one notebook. This is the thing `_written_at` could never be: a way to tell a key
# written by THIS run from a key left behind by an earlier one.
#
# A timestamp cannot do it. Different cells of the same notebook legitimately write
# minutes apart, so "does this key's timestamp match the newest one" reports most of
# a perfectly healthy notebook as stale. The question is not *when* a key was written,
# it is *which execution* wrote it, and only an identity answers that.
#
# Deliberately NOT reproducible. Everything else in this project that is hashed uses
# zlib.crc32 precisely because `hash()` is salted per process; here the per-process
# variation is the entire point.
RUN_ID = uuid.uuid4().hex[:12]


class MissingResultError(KeyError):
    """Raised when a notebook asks for something an earlier notebook never wrote."""


def _atomic_write(path, payload: dict) -> None:
    """Write JSON via a temp file + rename.

    WHY: a crash halfway through a plain `open(..., 'w')` leaves a truncated
    results.json, and the next notebook fails with a JSONDecodeError that tells you
    nothing about what actually went wrong. Rename is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load() -> dict:
    """Return the whole results dict ({} if nothing has been written yet)."""
    if not cfg.RESULTS.exists():
        return {}
    try:
        return json.loads(cfg.RESULTS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{cfg.RESULTS} is corrupt ({exc}). Delete it and rerun the pipeline "
            f"from notebook 00 — every notebook rewrites its own block."
        ) from exc


READONLY_ENV = "NETLOAD_RESULTS_READONLY"
# Ledger keys that hold a wall-clock duration. The only ones in results.json today
# are `seconds` (notebook 04 ladder and sensitivity axes, notebook 05 algorithm timings).
TIMING_KEYS = frozenset({"seconds"})


def _readonly_verify(nb: str, values: dict[str, Any]) -> dict:
    """Redraw mode: write nothing, and PROVE the run reproduces the ledger.

    WHY THIS EXISTS. Fixing a label on a notebook 00 figure means re-executing
    Notebook 00, and every `R.save` in it restamps `nb00._written_at`; the
    `figure_staleness` guard then convicts every figure of every later notebook,
    and the honest remedy is a full multi-hour re-run of a pipeline whose numbers
    did not change. With `NETLOAD_RESULTS_READONLY=1` in the kernel's environment
    the ledger is left untouched and, instead, every value the cell would have
    written is compared to the value already stored. A difference raises: a
    figure redrawn from numbers the ledger does not hold is a figure the report
    cannot cite, and the run must stop there rather than ship it.

    Floats are compared at 1e-6 relative; NaN equals NaN; everything else exact.
    Keys named in `TIMING_KEYS` are skipped: a wall-clock duration is a fact about
    the machine on the day, not a result, and the first redraw of notebook 04 was
    stopped by four of them (`ladder/*/seconds`) while every MAE matched.
    """
    import math

    stored = load().get(nb, {})
    incoming = json.loads(json.dumps(values, default=str))
    missing = object()
    diffs: list[tuple[str, Any, Any]] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in a.items():
                if k in TIMING_KEYS:
                    continue
                walk(v, b.get(k, missing), f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, bool) or isinstance(b, bool):
            # True == 1 in Python, so a flag stored as a number would otherwise pass
            if type(a) is not type(b) or a != b:
                diffs.append((path, a, b if b is not missing else "<absent>"))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # relative 1e-6; the absolute floor is tiny so that p-values such as 1.2e-20
            # are compared too, not waved through as "both below 1e-6"
            if not ((math.isnan(a) and math.isnan(b))
                    or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)):
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b if b is not missing else "<absent>"))

    for k, v in incoming.items():
        walk(v, stored.get(k, missing), f"{nb}/{k}")
    if diffs:
        shown = "\n".join(f"  {p}: run={a!r} ledger={b!r}" for p, a, b in diffs[:25])
        raise RuntimeError(
            f"{READONLY_ENV} is set and this run does NOT reproduce the ledger for "
            f"'{nb}' ({len(diffs)} value(s) differ). Nothing was written. Either "
            f"the notebook is not deterministic or its inputs changed; re-run it for "
            f"real, from this notebook onward.\n{shown}")
    print(f"[results read-only] {nb}: {len(incoming)} key(s) verified against "
          f"the ledger, nothing written")
    return stored


def save(nb: str, values: dict[str, Any]) -> dict:
    """Merge `values` into `results.json` under key `notebook`, and stamp the time.

    Shallow merge, not replace: rerunning one cell of notebook 04 does not wipe the
    keys written by the other cells of notebook 04.

    With `NETLOAD_RESULTS_READONLY` set, nothing is merged or stamped — see
    `_readonly_verify`, which is what a figure-only redraw runs under.
    """
    if not isinstance(values, dict):
        raise TypeError(f"save() takes a dict, got {type(values).__name__}")
    if os.environ.get(READONLY_ENV):
        return _readonly_verify(nb, values)
    payload = load()
    block = payload.get(nb, {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    block.update(values)

    # A TIMESTAMP PER KEY, not just per notebook. `_written_at` advances on the
    # notebook's FIRST save, which in notebook 04 is a cell holding setup constants
    # that never change — so a run that dies later leaves the whole block stamped
    # fresh with most of its values from the previous run. That happened on
    # 2026-08-22 (36 stale values under a fresh stamp) and again on 2026-08-23,
    # when the search cell timed out after the earlier cells had already saved.
    #
    # The notebook stamp certifies only "some cell of this notebook ran then". These
    # certify each value, so a half-written notebook is visible instead of having to
    # be reconstructed by diffing against a backup someone remembered to take.
    stamps = block.get("_keys_written_at", {})
    runs = block.get("_keys_run_id", {})
    if not isinstance(stamps, dict):
        stamps = {}
    if not isinstance(runs, dict):
        runs = {}
    for k in values:
        stamps[k] = now
        runs[k] = RUN_ID
    block["_keys_written_at"] = stamps
    block["_keys_run_id"] = runs
    block["_run_id"] = RUN_ID
    block["_written_at"] = now
    payload[nb] = block
    _atomic_write(cfg.RESULTS, payload)
    return block


def key_ages(nb: str | None = None) -> dict:
    """{notebook: {key: timestamp}} — which values came from which run.

    Use this after any interrupted run. A notebook whose keys carry two or more
    distinct timestamps has been written by two or more runs, and the older ones
    are stale by definition unless the newer run deliberately left them alone.
    """
    payload = load()
    nbs = [nb] if nb else sorted(k for k in payload if k.startswith("nb"))
    return {s: payload.get(s, {}).get("_keys_written_at", {}) for s in nbs}


def stale_keys(nb: str | None = None) -> dict:
    """{notebook: [keys left behind by an EARLIER execution of that notebook]}.

    Empty is the healthy answer. Anything else is a partial run: some cells of the
    notebook wrote in the latest execution and some did not, so the block mixes two
    vintages while presenting one `_written_at` — which is precisely what happened
    on 2026-08-22 (36 stale values under a fresh stamp) and again on 2026-08-23,
    when the search cell timed out after the earlier cells had already saved.

    Compared by RUN, not by clock. Cells of one notebook write minutes apart, so a
    timestamp comparison calls most of a healthy notebook stale and the check would be
    ignored within a week.

    A KEY WITH NO RUN ID IS THE OLDEST VINTAGE THERE IS, and reading this function's
    first version taught that the hard way on 2026-08-24. It iterated `_keys_run_id`
    and compared only the keys it found there — so the twenty-eight notebook 04 keys
    written before stamping existed were never compared to anything, and the block
    reported clean while `holdout_mae`, `skill` and `g1_met` all came from a
    superseded run. The set to walk is the block's DATA keys, not the stamp dict:
    absence from the stamp dict is the strongest possible evidence of age, and it
    was being read as no evidence at all.
    """
    payload = load()
    nbs = ([nb] if nb
              else sorted(k for k in payload if k.startswith("nb")))
    out = {}
    for s in nbs:
        block = payload.get(s, {})
        current = block.get("_run_id")
        if current is None:
            continue
        runs = block.get("_keys_run_id", {})
        if not isinstance(runs, dict):
            runs = {}
        older = sorted(k for k in block
                       if not k.startswith("_") and runs.get(k) != current)
        if older:
            out[s] = older
    return out


# The notebook ORDER is the medium the two previous guards do not cover. `stale_keys`
# asks whether one block mixes vintages; it cannot see a block that is internally
# consistent and entirely older than the panel it claims to describe — which is what
# notebooks 05 through 07 were on 2026-08-24, every one of them written before notebook 00 was
# last rewritten.
#
# THIS IS THE ONE PLACE THE CLOCK IS THE RIGHT INSTRUMENT. Within a block it fails,
# because cells of one notebook legitimately write minutes apart. Across blocks it
# cannot fail, because notebooks run strictly in order and hours apart: a downstream
# notebook stamped EARLIER than an upstream one has no innocent explanation.
# An audit notebook assembles the report and AUDITS the ledger: it reads every other block and
# nothing reads it. Numeric order is therefore NOT dependency order at the end of the
# chain. `run_pipeline` re-runs 09 last precisely so its counts see the last notebook, which
# makes the audit notebook the NEWEST block -- and a guard that equates "lower number" with
# "upstream" then convicts a healthy ledger of exactly the rot it was built to find.
#
# The exclusion is one-directional and that asymmetry is the point: the audit notebook never
# counts as an INPUT to anything, and is still required to postdate everything it
# audits, any later notebook included.
AUDIT_BLOCKS = frozenset({9})


def upstream_block_numbers(nb: int, present: list[int]) -> list[int]:
    """The notebooks `notebook` depends on, by dependency order rather than by number."""
    if nb in AUDIT_BLOCKS:
        return [n for n in present if n != nb]
    return [n for n in present if n < nb and n not in AUDIT_BLOCKS]


def downstream_staleness() -> dict:
    """{notebook: [upstream notebooks written after it]}. Empty is the healthy answer."""
    payload = load()

    # ORDER BY NOTEBOOK NUMBER, NOT BY NAME. String order puts "nb10" before
    # "nb02", so the day a later notebook existed, every notebook 02-07 would have been
    # convicted of predating a notebook that runs AFTER it. Caught 2026-09-02 while
    # adding a later notebook, before it fired.
    def _num(name: str) -> int:
        return int(name[2:]) if name[2:].isdigit() else -1

    stamped = {}
    for s in sorted((k for k in payload if k.startswith("nb")), key=_num):
        at = payload[s].get("_written_at")
        if isinstance(at, str):
            stamped[s] = at
    out = {}
    present = [_num(u) for u in stamped]
    for s, at in stamped.items():
        deps = set(upstream_block_numbers(_num(s), present))
        newer = sorted((u for u, u_at in stamped.items()
                        if _num(u) in deps and u_at > at), key=_num)
        if newer:
            out[s] = newer
    return out


def get(nb: str, key: str, default: Any = None) -> Any:
    """One value, or `default`. Use when absence is acceptable."""
    return load().get(nb, {}).get(key, default)


def require(nb: str, key: str) -> Any:
    """One value, or a loud failure naming the notebook that should have written it.

    This is the notebook-ordering guard. Read the error message: it tells you which
    notebook to run, not just that a key is missing.
    """
    payload = load()
    if nb not in payload:
        raise MissingResultError(
            f"'{nb}' has not been written yet. Run notebook "
            f"{nb.replace('nb', '').zfill(2)} first "
            f"(notebooks present: {sorted(k for k in payload if k.startswith('nb')) or 'none'})."
        )
    if key not in payload[nb]:
        raise MissingResultError(
            f"'{key}' missing from '{nb}'. Keys present: {sorted(payload[nb])}. "
            f"The cell that writes it is at the end of the {nb} notebook."
        )
    return payload[nb][key]


def summary() -> str:
    """One line per notebook — what has run and when. A status summary for the reader."""
    payload = load()
    if not payload:
        return "results.json is empty — no notebook has run."
    lines = []
    for nb in sorted(payload):
        block = payload[nb]
        when = block.get("_written_at", "?")
        keys = ", ".join(sorted(k for k in block if not k.startswith("_")))
        lines.append(f"{nb:8s} {when}  [{keys}]")
    return "\n".join(lines)


if __name__ == "__main__":  # `python -m src.results`
    print(summary())
