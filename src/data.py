"""The panel: raw pull, the cleaning contract, the net-load target (demand - wind - solar), and a
synthetic panel with the same schema for the tests.
"""
from __future__ import annotations


# ---- Raw panel, cleaning and target ------------------------------------------------------------

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config as cfg

MAX_INTERP_HOURS = 3
MIN_TEMP_COVERAGE = 0.90     # below this the weather join is treated as failed

# --- plausibility bounds ----------------------------------------------------
# The first real pull (2019-2024, six BAs) contained values that are physically
# impossible, and the original contract caught none of them because it only
# clipped NEGATIVES. Two real examples from that vintage:
#
#   SWPP 2023-06-13 02:00Z   demand 3,621,097 MW and wind 3,589,445 MW, roughly
#                            100x SWPP's true peak. It is INVISIBLE downstream
#                            because the error cancels in the identity:
#                            3,621,097 - 3,589,445 - 32 = 31,620 MW, an entirely
#                            ordinary net load. The target looks fine; the
#                            components are garbage.
#
#   CISO 2019-12-20 00-02Z   demand 14, 91 and 96 MW against a normal of ~24,000,
#                            while generation kept reporting normally. That is a
#                            demand-telemetry failure, and it produced the most
#                            negative net load in the whole panel (-4,314 MW) --
#                            which would otherwise have been read as a headline
#                            renewable over-generation event.
#
# The bounds are deliberately WIDE and are computed per zone from that zone's own
# distribution, so they are scale-free and no legitimate hour is at risk. They are
# applied BEFORE interpolation, so a short corrupt episode is repaired by the same
# <=3h rule as any other dropout, and a long one is dropped.
#
# They must NOT be tightened into an outlier filter. Genuine extremes are the
# subject of this project: CISO 2019-04-10 23:00Z has demand 12,193 MW against
# 3,398 MW wind + 9,422 MW solar -- renewable share 1.05, a real >100% renewable
# hour -- and it survives these bounds untouched.
DEMAND_LOW_FRAC = 0.10      # below 10% of the zone's median demand
DEMAND_HIGH_MULT = 10.0     # above 10x the zone's median demand
GEN_HIGH_MULT = 5.0         # above 5x the zone's 99.9th percentile for that fuel

# Columns the plausibility rule applies to. Weather is excluded (a temperature of
# 0 is not a defect), and storage columns are excluded from the LOWER bound
# because they are legitimately negative while charging.
_POWER_COLS = (
    "demand", "demand_forecast", "net_generation", "solar", "wind", "hydro",
    "nuclear", "gas", "coal", "oil", "other", "geothermal", "unknown_fuel",
    "battery", "pumped_storage", "other_storage", "unknown_storage",
    "solar_battery", "wind_battery",
)


def flag_implausible(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set physically impossible values to NaN, per zone, and count what was flagged.

    Returns the frame and a {column: count} report. NaN rather than clip, because
    an impossible value carries no information about what the true value was --
    clipping it to a bound would invent a number, whereas NaN hands it to the
    interpolation contract, which is the part of this module that is allowed to
    estimate.
    """
    df = df.copy()
    counts: dict[str, int] = {}
    cols = [c for c in _POWER_COLS if c in df.columns]

    for z, idx in df.groupby(cfg.GROUP_COL).groups.items():
        block = df.loc[idx, cols]

        # demand gets a two-sided bound: a balancing authority always serves load,
        # so a near-zero demand is a reporting failure, not a quiet night.
        if "demand" in block:
            med = block["demand"].median()
            if pd.notna(med) and med > 0:
                bad = ((block["demand"] < DEMAND_LOW_FRAC * med)
                       | (block["demand"] > DEMAND_HIGH_MULT * med)).to_numpy()
                if bad.any():
                    counts["demand"] = counts.get("demand", 0) + int(bad.sum())
                    df.loc[block.index[bad], "demand"] = np.nan

        # generation gets an UPPER bound only -- any fuel may legitimately be zero.
        # Referenced to p99.9 rather than the median because a fuel that is off
        # most of the time (solar at night) has a median of zero.
        #
        # CAVEAT, stated because it is the obvious attack on this rule: the bound
        # tightens as a column's dispersion falls, because p99.9 collapses towards
        # the constant. On a truly constant column any deviation would be flagged.
        # That is safe here only because every real fuel column has genuine spread
        # over six years -- 5x the 99.9th percentile of a zone's solar output is
        # about five times its installed capacity. It would NOT be safe on a column
        # that is constant by construction, and nothing like that is in this panel.
        for c in cols:
            if c == "demand":
                continue
            hi = block[c].quantile(0.999)
            if pd.isna(hi) or hi <= 0:
                continue          # all-NaN or all-zero for this zone: nothing to test
            bad = (block[c].abs() > GEN_HIGH_MULT * abs(hi)).to_numpy()
            if bad.any():
                counts[c] = counts.get(c, 0) + int(bad.sum())
                df.loc[block.index[bad], c] = np.nan

    return df, counts


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(cfg.ROOT),
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _write_manifest(df: pd.DataFrame, source: str, notes: dict, path=None) -> dict:
    """Merge a provenance record into manifest.json.

    `path` is the artefact being described, so notebook 00 hashes panel_raw.parquet
    and notebook 01 hashes panel.parquet. Merged rather than replaced, so notebook 01
    writing its cleaning report does not erase notebook 00's gathering record.
    """
    path = cfg.PANEL if path is None else path
    manifest = {}
    if cfg.MANIFEST.exists():
        try:
            manifest = json.loads(cfg.MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest |= {
        "data_source": source,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "rows": int(len(df)),
        "zones": sorted(df[cfg.GROUP_COL].unique().tolist()),
        "start": str(df[cfg.TIME_COL].min()),
        "end": str(df[cfg.TIME_COL].max()),
        "columns": sorted(df.columns.tolist()),
        "seed": cfg.SEED,
        "target": cfg.TARGET,
        "artefact": path.name,
        "panel_sha256_16": _sha256(path) if path.exists() else None,
        "citations": {
            "eia930": "U.S. Energy Information Administration, Hourly Electric Grid "
                      "Monitor (EIA-930). Public domain.",
            "open_meteo": "Open-Meteo historical weather API (ERA5 reanalysis, ECMWF). "
                          "CC-BY-4.0.",
        },
        **notes,
    }
    cfg.MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# the cleaning contract
# ---------------------------------------------------------------------------

def interpolate_short_runs(s: pd.Series, limit: int = MAX_INTERP_HOURS) -> pd.Series:
    """Linearly fill runs of NaN no longer than `limit`; leave longer runs alone.

    THE BUG THIS REPLACES, because it is subtle and it shipped. The contract says
    "interpolate gaps of <= 3 hours only", and the code said
    `interpolate(limit=3, limit_area="inside")`. `limit` is a pandas FILL LIMIT,
    not a gate on the run: it fills the FIRST THREE NaNs OF EVERY GAP, however
    long. So each of the four 24-hour outages had its first three hours filled and
    its remaining 21 dropped, and the twelve values that entered the panel were
    three points read off a straight line drawn across a whole day of an outage —
    the exact quantity the contract's own rationale calls "normal operation that
    never happened". The rule was written to forbid that and the implementation
    performed it.

    Run length is measured per column, not per row: `demand` may be out for a day
    while `wind` is out for an hour, and repairing them by the same rule would be
    wrong in one direction or the other.

    Leading and trailing runs are never filled at all (`limit_area="inside"`),
    because extrapolating off the end of a series is not interpolation.
    """
    isna = s.isna()
    if not isna.any():
        return s
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(isna & (run_len <= limit)), filled)


def missing_run_census(s: pd.Series, limit: int = MAX_INTERP_HOURS) -> dict:
    """How many NaN runs this column has, split at the cap. Counted, never assumed."""
    isna = s.isna()
    if not isna.any():
        return {"runs": 0, "runs_repaired": 0, "runs_left_whole": 0,
                "hours_repaired": 0, "hours_left_whole": 0}
    run_id = (isna != isna.shift()).cumsum()
    lengths = isna.groupby(run_id).sum()
    lengths = lengths[lengths > 0]
    short, long_ = lengths[lengths <= limit], lengths[lengths > limit]
    return {"runs": int(len(lengths)),
            "runs_repaired": int(len(short)),
            "runs_left_whole": int(len(long_)),
            "hours_repaired": int(short.sum()),
            "hours_left_whole": int(long_.sum())}


def clean_panel(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, regularise to a complete hourly grid, interpolate short gaps only.

    Every defect is COUNTED and returned, not silently repaired — the counts go
    into the manifest and into docs/data_dictionary.md, and a spike in any of
    them between vintages is the signal that something upstream changed.
    """
    report: dict = {}
    df = df.copy()
    df[cfg.TIME_COL] = pd.to_datetime(df[cfg.TIME_COL])

    before = len(df)
    df = df.drop_duplicates(subset=[cfg.GROUP_COL, cfg.TIME_COL], keep="last")
    report["duplicate_rows_dropped"] = before - len(df)

    # A complete hourly grid per zone. A missing HOUR is invisible until a lag
    # feature quietly reaches back 167 hours instead of 168.
    full = []
    for z, g in df.groupby(cfg.GROUP_COL):
        idx = pd.date_range(g[cfg.TIME_COL].min(), g[cfg.TIME_COL].max(), freq="h")
        g = g.set_index(cfg.TIME_COL).reindex(idx)
        g[cfg.GROUP_COL] = z
        g.index.name = cfg.TIME_COL
        full.append(g.reset_index())
    df = pd.concat(full, ignore_index=True)
    report["rows_after_regular_grid"] = int(len(df))

    # Physically impossible values -> NaN, BEFORE the missing-value census, so the
    # counts below include them and the interpolation rule gets a chance to repair
    # a short corrupt episode. See flag_implausible for the two real examples.
    df, implausible = flag_implausible(df)
    report["implausible_values_nulled"] = implausible

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    report["missing_before_interp"] = {
        c: int(df[c].isna().sum()) for c in numeric if df[c].isna().any()
    }

    df = df.sort_values([cfg.GROUP_COL, cfg.TIME_COL])

    # The cap is a WHOLE-RUN GATE: a run of <= 3 missing hours is repaired
    # entirely, a longer one is not touched at all. See interpolate_short_runs
    # for the pandas fill-limit bug this replaced, and why the twelve values it
    # admitted were the worst possible ones to admit.
    census: dict[str, dict] = {}
    for c in numeric:
        if df[c].isna().any():
            census[c] = {"zones": {}}
    for z, idx in df.groupby(cfg.GROUP_COL).groups.items():
        for c in numeric:
            col = df.loc[idx, c]
            if not col.isna().any():
                continue
            census[c]["zones"][z] = missing_run_census(col)
            df.loc[idx, c] = interpolate_short_runs(col)

    report["interp_limit_hours"] = MAX_INTERP_HOURS
    report["interp_rule"] = "whole-run gate: a run of <= 3 NaN is filled, a longer run is not"
    report["interp_runs"] = {
        c: {"runs": sum(v["runs"] for v in blk["zones"].values()),
            "runs_repaired": sum(v["runs_repaired"] for v in blk["zones"].values()),
            "runs_left_whole": sum(v["runs_left_whole"] for v in blk["zones"].values()),
            "hours_repaired": sum(v["hours_repaired"] for v in blk["zones"].values()),
            "hours_left_whole": sum(v["hours_left_whole"] for v in blk["zones"].values())}
        for c, blk in census.items()
    }
    report["missing_after_interp"] = {
        c: int(df[c].isna().sum()) for c in numeric if df[c].isna().any()
    }

    # Physically impossible values: generation is never negative. Clipped, counted,
    # and the count is a data-quality finding, not housekeeping.
    for c in ("solar", "wind", "hydro", "nuclear"):
        if c in df.columns:
            neg = int((df[c] < 0).sum())
            if neg:
                report.setdefault("negative_generation_clipped", {})[c] = neg
                df[c] = df[c].clip(lower=0)

    return df.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True), report


def derive_net_load(df: pd.DataFrame) -> pd.DataFrame:
    """net_load = demand - wind - solar.

    This is the quantity the system must actually serve from dispatchable
    resources, and it is the reason the project is not a demand-forecasting
    exercise. Because the identity is exact, `demand`, `wind` and `solar` can
    never be features — see MutualInfoSelector.DROP_ALWAYS.

    KNOWN INCOMPLETENESS, stated rather than silently corrected. EIA publishes two
    hybrid codes — `SNB` (solar with integrated battery storage) and `WNB` (wind
    with integrated battery) — whose output is NOT included in `SUN` or `WND`. So
    where a respondent reports them, this identity slightly OVERSTATES net load.

    The identity is nevertheless left as demand − wind − solar, for three reasons:
    the magnitude is negligible in this vintage (only BPAT reports `SNB`, from
    2024-07-15, at most 50 MW against a BPAT peak near 11,700 MW — under 0.5%);
    hybrid plants are partly dispatchable, so folding them into a *variable*
    renewable term is not obviously more correct; and the identity is the documented
    contract that `tests/test_data_and_clusters.py::test_net_load_identity` and
    `MutualInfoSelector.DROP_ALWAYS` are both written against.

    Revisit if a respondent's hybrid capacity becomes material — the check is
    `solar_battery.max() / demand.max()` per zone.

    MISSING COMPONENTS. A zone that never reports a non-zero value of a component
    (NYIS solar) has none of it, and 0 is its value. A component the zone does report
    that is still missing here is an outage longer than the interpolation cap, and
    the cleaning contract applies: net load cannot be recovered, the target is NaN
    and build_clean_panel drops the hour. Filling it with 0 instead scored 24 hours of
    CISO on 2 November 2024 (wind and solar both missing) as if net load were demand.
    """
    df = df.copy()
    for c in ("wind", "solar"):
        if c not in df.columns:
            df[c] = 0.0
    comp = {}
    for c in ("wind", "solar"):
        has_it = df.groupby(cfg.GROUP_COL)[c].transform(lambda s: bool((s.fillna(0) != 0).any()))
        comp[c] = df[c].where(has_it, df[c].fillna(0))
    df[cfg.TARGET] = df["demand"] - comp["wind"] - comp["solar"]
    df["renewable_share"] = (
        (comp["wind"] + comp["solar"]) / df["demand"]
    ).replace([np.inf, -np.inf], np.nan)

    # `ramp` — the hour-over-hour change in net load. This is the quantity
    # flexibility is actually procured against, and notebook 01 profiles it directly.
    #
    # It is computed by TIMESTAMP, not by row position, for the same reason the
    # lag features are: the cleaning contract drops unrecoverable hours, so
    # `shift(1)` across an outage would report a 40-hour swing as an hourly ramp
    # and put a spurious outlier in the tail of every distribution in notebook 01.
    #
    # It is a DESCRIPTIVE column, never a feature: ramp(t) contains net_load(t),
    # so a model given it would be reading its own answer. It is listed in
    # MutualInfoSelector.DROP_ALWAYS, and tests/test_leakage.py asserts that.
    df = df.sort_values([cfg.GROUP_COL, cfg.TIME_COL])
    prev = pd.MultiIndex.from_arrays([
        df[cfg.GROUP_COL].to_numpy(),
        (pd.to_datetime(df[cfg.TIME_COL]) - pd.Timedelta(hours=1)).to_numpy(),
    ])
    keyed = pd.Series(
        df[cfg.TARGET].to_numpy(),
        index=pd.MultiIndex.from_arrays([
            df[cfg.GROUP_COL].to_numpy(), pd.to_datetime(df[cfg.TIME_COL]).to_numpy()
        ]),
    )
    keyed = keyed[~keyed.index.duplicated(keep="last")]
    df["ramp"] = df[cfg.TARGET].to_numpy() - keyed.reindex(prev).to_numpy()
    return df


# ---------------------------------------------------------------------------
# Notebook 00 — gather. Nothing is repaired here.
# ---------------------------------------------------------------------------
def load_raw_panel(force: bool = False, verbose: bool = True,
                   allow_synthetic: bool = True) -> pd.DataFrame:
    """Pull both APIs, join on (zone, timestamp), and freeze the result UNCHANGED.

        load_raw_panel()             read panel_raw.parquet if it exists
        load_raw_panel(force=True)   re-pull and rebuild
        load_raw_panel(allow_synthetic=False)   fail rather than fall back

    WHAT THIS DELIBERATELY DOES NOT DO: no deduplication, no regular hourly grid,
    no interpolation, no clipping, no plausibility bounds, no derived columns, no
    dropping of anything. The output is what the two APIs actually served.

    WHY THE SPLIT MATTERS: the defects ARE the subject of notebook 01. If notebook 00
    repaired them, notebook 01 would be analysing the repair rather than the data, and
    every missingness figure would show the tidied version. The raw table stays on
    disk so any cleaning decision can be re-argued against the original evidence.

    The one thing that IS enforced here is the weather join, because a failed join
    is a gathering failure rather than a data defect — see the coverage check below.
    """
    if cfg.PANEL_RAW.exists() and not force:
        df = pd.read_parquet(cfg.PANEL_RAW)
        if verbose:
            src = json.loads(cfg.MANIFEST.read_text(encoding="utf-8")).get("data_source", "?") \
                if cfg.MANIFEST.exists() else "?"
            print(f"panel_raw.parquet: {len(df):,} rows, source={src} "
                  f"(pass force=True to re-pull)")
        return df

    key = os.environ.get("EIA_API_KEY")
    # The per-zone API pulls are committed under data/raw/. When every zone is there,
    # the panel is rebuilt from those files and no key is needed; the synthetic
    # generator is reached only when there is neither a key nor a cached pull.
    cached_pulls = all(
        (cfg.DATA_RAW / f"eia930_{z}_{cfg.START[:4]}_{cfg.END[:4]}.parquet").exists()
        for z in cfg.ZONES
    )
    if key or cached_pulls:
        source = "eia930+open-meteo"
        if verbose:
            if key:
                print("Building the panel from EIA-930 + Open-Meteo ERA5.")
                print("First run pulls several hundred requests and takes a few minutes; "
                      "everything is cached to data/raw/, so reruns are free.\n")
            else:
                print("Building the panel from the cached EIA-930 and Open-Meteo pulls in "
                      "data/raw/ (no EIA key needed for a rebuild).\n")
        from src import sources

        power = sources.load_all(cfg.ZONES, cfg.START, cfg.END, key, verbose=verbose)
        weather = sources.load_all_weather(cfg.ZONES, cfg.START, cfg.END,
                                                  verbose=verbose)
        power[cfg.TIME_COL] = pd.to_datetime(power[cfg.TIME_COL])
        weather[cfg.TIME_COL] = pd.to_datetime(weather[cfg.TIME_COL])
        df = power.merge(weather, on=[cfg.GROUP_COL, cfg.TIME_COL], how="left")

        # Step 3: the assertion that keeps a silent weather failure from surviving.
        if "temp" not in df.columns:
            raise RuntimeError(
                "The weather join produced no `temp` column. Both sides must be UTC "
                "and both must use the same zone codes — see src/sources.py."
            )
        coverage = float(df["temp"].notna().mean())
        if coverage < MIN_TEMP_COVERAGE:
            raise RuntimeError(
                f"`temp` is only {coverage:.1%} populated (need >= {MIN_TEMP_COVERAGE:.0%}). "
                f"This is almost always a timezone mismatch on the join, not a gap in "
                f"ERA5. Do NOT proceed: the model would train with no weather and you "
                f"would conclude that weather does not matter."
            )
    elif allow_synthetic:
        source = "synthetic"
        if verbose:
            print("=" * 72)
            print("NO EIA_API_KEY FOUND — falling back to the synthetic generator.")
            print("The architecture will run end to end, but NO NUMBER FROM THIS RUN")
            print("MAY ENTER THE SUBMITTED REPORT. Get a free key in about a minute:")
            print("  https://www.eia.gov/opendata/   ->  echo 'EIA_API_KEY=...' > .env")
            print("=" * 72)

        df = make_panel()
        coverage = 1.0
    else:
        raise RuntimeError(
            "No EIA_API_KEY and allow_synthetic=False. Register at "
            "https://www.eia.gov/opendata/ and put the key in .env."
        )

    # Sorted by (timestamp, zone) for readability only. Notebook 01 re-sorts after
    # cleaning, and PanelTimeSeriesSplit checks the ordering itself.
    df = df.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True)
    df.to_parquet(cfg.PANEL_RAW, index=False)
    manifest = _write_manifest(df, source, {"temp_coverage": round(coverage, 4)},
                               path=cfg.PANEL_RAW)

    if verbose:
        print(f"\npanel_raw.parquet written: {len(df):,} rows x {df.shape[1]} columns")
        print(f"  zones   : {manifest['zones']}")
        print(f"  span    : {manifest['start']} -> {manifest['end']}")
        print(f"  source  : {source}")
        print(f"  sha256  : {manifest['panel_sha256_16']}")
        print("  NOT cleaned. Defects are found and repaired in notebook 01 (EDA).")
    return df


# ---------------------------------------------------------------------------
# Notebook 01 — clean. Called from the EDA notebook, not from notebook 00.
# ---------------------------------------------------------------------------
def build_clean_panel(raw: pd.DataFrame | None = None, verbose: bool = True
                      ) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning contract to the raw panel and write panel.parquet.

    This is NOTEBOOK 01's function, not notebook 00's. It is here rather than in the
    notebook because it is reusable, tested logic; the notebook narrates the
    decisions and shows the evidence for each one.

    Order matters: plausibility bounds -> dedupe -> regular grid -> interpolate
    -> clip -> derive -> drop unrecoverable targets. See clean_panel.
    """
    raw = load_raw_panel(verbose=verbose) if raw is None else raw

    df, report = clean_panel(raw, verbose=verbose)
    df = derive_net_load(df)

    before = len(df)
    df = df.dropna(subset=[cfg.TARGET])
    report["rows_dropped_missing_target"] = before - len(df)

    df = df.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True)
    df.to_parquet(cfg.PANEL, index=False)

    source = (json.loads(cfg.MANIFEST.read_text(encoding="utf-8")).get("data_source", "?")
              if cfg.MANIFEST.exists() else "?")
    _write_manifest(df, source, {"cleaning": report}, path=cfg.PANEL)

    if verbose:
        print(f"\npanel.parquet written: {len(df):,} rows x {df.shape[1]} columns")
        print(f"  dropped from raw: {len(raw) - len(df):,} rows")
        if report.get("missing_after_interp"):
            print(f"  gaps left after {MAX_INTERP_HOURS}h interpolation: "
                  f"{report['missing_after_interp']}")
    return df, report


def load_panel(verbose: bool = True) -> pd.DataFrame:
    """Read the CLEANED panel. Every notebook from 2 onward uses this.

    Raises rather than silently building it, because a notebook that quietly cleaned
    its own input would be using a different vintage from the one notebook 01 argued
    about in writing.
    """
    if not cfg.PANEL.exists():
        raise FileNotFoundError(
            "data/processed/panel.parquet does not exist. It is written by "
            "NOTEBOOK 01 (notebooks/01_eda_statistics.ipynb), which is where the "
            "cleaning contract is applied and justified. Notebook 00 writes "
            "panel_raw.parquet only."
        )
    df = pd.read_parquet(cfg.PANEL)
    if verbose:
        print(f"panel.parquet: {len(df):,} rows x {df.shape[1]} columns")
    return df


def data_dictionary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Column, unit, source, range, defect — the table docs/data_dictionary.md needs.

    Generated rather than typed, so it cannot drift from the panel it describes.
    """
    # Defaults to the cleaned panel, falling back to the raw one, so this works in
    # Notebook 00 (before panel.parquet exists) as well as afterwards.
    if df is None:
        df = pd.read_parquet(cfg.PANEL if cfg.PANEL.exists() else cfg.PANEL_RAW)
    units = {
        "demand": "MW", "demand_forecast": "MW", "solar": "MW", "wind": "MW",
        "hydro": "MW", "nuclear": "MW", "gas": "MW", "coal": "MW",
        "oil": "MW", "other": "MW", "net_generation": "MW", "total_interchange": "MW",
        "geothermal": "MW", "unknown_fuel": "MW", "battery": "MW",
        "pumped_storage": "MW", "other_storage": "MW", "unknown_storage": "MW",
        "solar_battery": "MW", "wind_battery": "MW",
        "net_load": "MW", "ramp": "MW/h", "temp": "deg C", "humidity": "%",
        "wind_speed": "km/h", "cloud_cover": "%", "renewable_share": "fraction",
        # The eleven ERA5 columns added on 2026-08-23 (cfg.WEATHER_FEATURES). Without
        # these entries they printed unit "-" in the generated dictionary.
        "ghi": "W/m2", "dni_horiz": "W/m2", "dhi": "W/m2",
        "wind_speed_100m": "km/h", "wind_gust": "km/h",
        "apparent_temp": "deg C", "dew_point": "deg C",
        "snowfall": "cm/h", "snow_depth": "m", "precipitation": "mm/h",
        "pressure": "hPa",
        "timestamp": "UTC hour", "zone": "EIA-930 respondent code",
    }
    # Every column the API actually serves is EIA-930. `oil`, `other`,
    # `net_generation` and `total_interchange` were previously falling through to
    # "derived", which mislabels the provenance of four columns in a graded artefact.
    # Only net_load, renewable_share and ramp are genuinely computed here.
    eia_cols = ("demand", "demand_forecast", "solar", "wind", "hydro", "nuclear",
                "gas", "coal", "oil", "other", "net_generation", "total_interchange",
                "geothermal", "unknown_fuel", "battery", "pumped_storage",
                "other_storage", "unknown_storage", "solar_battery", "wind_battery")
    weather_cols = tuple(cfg.WEATHER_FEATURES)   # all fifteen ERA5 columns
    src = {c: ("EIA-930" if c in eia_cols
               else "Open-Meteo ERA5" if c in weather_cols
               else "key" if c in (cfg.TIME_COL, cfg.GROUP_COL)
               else "derived") for c in df.columns}
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "unit": units.get(c, "-"),
            "source": src.get(c, "-"),
            "min": round(float(s.min()), 2) if pd.api.types.is_numeric_dtype(s) else "-",
            "max": round(float(s.max()), 2) if pd.api.types.is_numeric_dtype(s) else "-",
            "missing_%": round(100 * float(s.isna().mean()), 3),
        })
    return pd.DataFrame(rows)


# ---- Synthetic panel (tests only) --------------------------------------------------------------

import zlib

import numpy as np
import pandas as pd

from src import config as cfg

# Scale (MW) and renewable character per zone. Roughly the real ordering, which
# is the point: a pipeline that only works when all zones are the same size is
# not a pipeline that will work on the real panel.
ZONE_PROFILE = {
    "CISO": dict(base=28_000, solar=0.34, wind=0.08, winter_peaking=False),
    "ERCO": dict(base=48_000, solar=0.16, wind=0.30, winter_peaking=False),
    "MISO": dict(base=72_000, solar=0.04, wind=0.16, winter_peaking=False),
    "SWPP": dict(base=30_000, solar=0.05, wind=0.38, winter_peaking=False),
    "BPAT": dict(base=7_000, solar=0.01, wind=0.14, winter_peaking=True),
    "NYIS": dict(base=18_000, solar=0.05, wind=0.04, winter_peaking=False),
}


def _ou_noise(n: int, rng: np.random.Generator, theta: float = 0.02,
              sigma: float = 1.0) -> np.ndarray:
    """Ornstein-Uhlenbeck noise — autocorrelated, so the ACF in notebook 01 is not a spike.

    White noise would make the PACF exercise meaningless and would let a model
    with no lag features score as well as one with them.
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0 - x[i - 1]) + sigma * rng.standard_normal()
    return x


def make_panel(zones=cfg.ZONES, start: str = cfg.START, end: str = cfg.END,
               seed: int = cfg.SEED) -> pd.DataFrame:
    """A panel with the same columns the real loader produces."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start.replace("T", " ") + ":00"),
                        pd.Timestamp(end.replace("T", " ") + ":00"), freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    dow = idx.dayofweek.to_numpy()
    yearfrac = (idx.year.to_numpy() - idx.year.min()) + doy / 365.25

    frames = []
    for z in zones:
        p = ZONE_PROFILE.get(z, dict(base=20_000, solar=0.1, wind=0.1, winter_peaking=False))
        # zlib.crc32, NOT hash(). Python salts string hashing per process
        # (PYTHONHASHSEED), so `hash(z)` gives a different panel on every run —
        # in a project whose central claim is reproducibility. crc32 is stable
        # across processes, machines and Python versions.
        rz = np.random.default_rng(seed + zlib.crc32(z.encode()) % 10_000)

        # Temperature: annual sine + diurnal + AR noise, phase-shifted by latitude proxy.
        temp = (12 + 12 * np.sin(2 * np.pi * (doy - 105) / 365.25)
                + 3 * np.sin(2 * np.pi * (hour - 15) / 24)
                + 0.6 * _ou_noise(n, rz, theta=0.05, sigma=0.9))

        # Demand: the hockey stick — cooling above 18C, heating below 12C.
        cdd = np.clip(temp - 18, 0, None)
        hdd = np.clip(12 - temp, 0, None)
        diurnal = 0.16 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.06 * np.sin(2 * np.pi * (hour - 19) / 12)
        weekend = np.where(dow >= 5, -0.07, 0.0)
        growth = 1 + 0.012 * yearfrac
        heat_w = 0.020 if p["winter_peaking"] else 0.008
        cool_w = 0.010 if not p["winter_peaking"] else 0.004
        demand = p["base"] * growth * (1 + diurnal + weekend + cool_w * cdd + heat_w * hdd)
        demand += p["base"] * 0.012 * _ou_noise(n, rz, theta=0.03, sigma=0.8)

        # Solar: a bell centred on local noon, EXACTLY zero at night, seasonally scaled.
        # Renewable capacity grows over the window — the project is about rising
        # penetration, so the fallback has to contain it.
        cap_growth = 1 + 0.18 * yearfrac
        clear = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
        season = 0.75 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365.25)
        clouds = np.clip(0.75 + 0.25 * _ou_noise(n, rz, theta=0.08, sigma=0.5), 0.15, 1.0)
        solar = p["solar"] * p["base"] * cap_growth * clear * season * clouds
        solar[hour < 6] = 0.0
        solar[hour > 19] = 0.0

        # Wind: persistent, mildly anti-correlated with summer, never negative.
        w = 0.5 + 0.35 * _ou_noise(n, rz, theta=0.01, sigma=0.5)
        w += 0.12 * np.sin(2 * np.pi * (doy - 300) / 365.25)
        wind = p["wind"] * p["base"] * cap_growth * np.clip(w, 0.02, 1.6)

        hydro = 0.10 * p["base"] * (1 + 0.3 * np.sin(2 * np.pi * (doy - 150) / 365.25))
        if z == "BPAT":
            hydro *= 6.0

        df = pd.DataFrame({
            cfg.TIME_COL: idx,
            cfg.GROUP_COL: z,
            "demand": demand,
            # The day-ahead forecast benchmark: the truth plus a plausible error,
            # so notebook 04 has an external baseline to beat even offline.
            "demand_forecast": demand * (1 + 0.018 * rz.standard_normal(n)),
            "solar": solar,
            "wind": wind,
            "hydro": hydro,
            "nuclear": 0.12 * p["base"] * np.ones(n),
            "gas": np.nan,
            "coal": np.nan,
            "temp": temp,
            "humidity": np.clip(60 + 20 * rz.standard_normal(n), 5, 100),
            "wind_speed": np.clip(4 + 3 * np.abs(rz.standard_normal(n)), 0, 30),
            "cloud_cover": (1 - clouds) * 100,
        })
        # Realistic defects, so the cleaning contract in notebook 00 has something to
        # find: scattered dropouts and one multi-day outage per zone.
        miss = rz.random(n) < 0.002
        df.loc[miss, ["demand", "solar", "wind"]] = np.nan
        outage_start = rz.integers(0, n - 96)
        df.loc[outage_start:outage_start + 60, ["demand", "solar", "wind"]] = np.nan
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True)

