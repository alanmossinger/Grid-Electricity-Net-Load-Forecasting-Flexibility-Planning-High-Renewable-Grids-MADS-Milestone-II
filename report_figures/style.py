"""Shared style for the compact report figures: drawn at the size they print in the report, so text is readable.

Every figure is 6.7 in wide (the report's text width) and at most about 3.6 in tall, text at 8 pt (ticks 7 pt),
saved at 250 dpi under reports/figures/report/. All values come from data/processed/results.json (the ledger)
or the processed data files; nothing is re-fitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "figures" / "report"
WIDTH = 6.7          # inches, the report's text width
DPI = 250

ZONE_COLOR = {"CISO": "#E8692E", "ERCO": "#2C9A3C", "MISO": "#D62728", "SWPP": "#8C564B",
              "BPAT": "#1F77B4", "NYIS": "#9467BD"}
ZONE_NAME = {"CISO": "California", "ERCO": "Texas", "MISO": "MISO", "SWPP": "Plains",
             "BPAT": "Bonneville", "NYIS": "New York"}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.4,
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def ledger() -> dict:
    return json.loads((ROOT / "data" / "processed" / "results.json").read_text(encoding="utf-8"))


def figure(height: float, **kw):
    setup()
    return plt.subplots(figsize=(WIDTH, height), **kw)


def panel_label(ax, letter: str):
    """'(a)' in the top-left corner, outside the axes, as the captions name the panels."""
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, weight="bold")


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
