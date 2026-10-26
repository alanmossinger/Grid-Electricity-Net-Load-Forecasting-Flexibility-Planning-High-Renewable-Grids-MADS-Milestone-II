"""Execute the notebooks in order, each in a fresh kernel, writing outputs in place.

    python run_notebooks.py            # 00 .. 08
    python run_notebooks.py --from 04  # 04 .. 08
    python run_notebooks.py --only 08  # one notebook
    python run_notebooks.py --only 04 --n-jobs 4   # fewer parallel workers

On Windows a joblib worker pool can deadlock late in a long kernel: the notebook stops
with no error and no CPU use (notebook 04 did once, in its last cell, which now runs
sequentially). Each cell times out after 4 hours rather than hanging forever; if that
happens, re-run the notebook with --n-jobs 4 (or 1). The results do not depend on it.

The notebooks depend on each other through data/processed/results.json: notebook 04
raises if notebook 01 has not written its lag sets, and so on. Re-running a notebook
therefore means re-running everything after it. Runtimes on a laptop: 00-03 about
15 minutes in total, 04 one to three hours (the hyperparameter search), 05 ten
minutes, 06 about an hour, 07 about twenty minutes, 08 one minute.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
KERNEL = "electricity-grid"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="00")
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-jobs", default=None,
                    help="worker processes for scikit-learn (sets NETLOAD_N_JOBS; default all cores)")
    args = ap.parse_args()
    # the kernels inherit this environment
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")   # avoids BrokenProcessPool / WinError 6 on Windows
    if args.n_jobs:
        os.environ["NETLOAD_N_JOBS"] = str(int(args.n_jobs))
    nbs = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.only:
        nbs = [p for p in nbs if p.name.startswith(args.only.zfill(2))]
    else:
        nbs = [p for p in nbs if p.name[:2] >= args.start.zfill(2)]
    if not nbs:
        print(f"no notebook matches --from {args.start!r} / --only {args.only!r}", file=sys.stderr)
        return 1
    for p in nbs:
        t0 = time.time()
        print(f"running {p.name} ...", flush=True)
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, timeout=14_400, kernel_name=KERNEL,
                                resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        finally:
            # the kernel `make setup` registers, so that Jupyter opens the notebook on the project venv
            nb.metadata["kernelspec"] = {"display_name": "Python 3 (electricity-grid)",
                                         "language": "python", "name": KERNEL}
            nbformat.write(nb, p)
        print(f"  ok  {p.name}  {time.time() - t0:,.0f}s", flush=True)
    print("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
