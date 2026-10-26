"""Shared fixtures. A small synthetic panel keeps the suite under a few seconds —
a test suite that takes ten minutes is a test suite you stop running."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg          # noqa: E402
from src import data as data_loader
synthetic = data_loader  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    """~90 days x 3 zones, cleaned and with net_load derived. Never touches the network."""
    raw = synthetic.make_panel(zones=["CISO", "ERCO", "BPAT"],
                               start="2023-01-01T00", end="2023-03-31T23")
    df, _ = data_loader.clean_panel(raw, verbose=False)
    df = data_loader.derive_net_load(df).dropna(subset=[cfg.TARGET])
    return df.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True)
