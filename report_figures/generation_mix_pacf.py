"""The changing generation mix (left) and the PACF that sets the lag sets (right).

Redraws two notebook-01 figures at print size, from data/processed/panel.parquet, with the notebook's own
definitions (notebook 01 Pass 7 cell for 01_fuel_mix_over_time.png, Pass 5 cell for 01_pacf.png):

(a) the fleet's generation mix as shares (the notebook's panel b): storage codes excluded, a fuel a zone
    reports in under 1% of its hours set to a structural zero, zones summed per hour, monthly mean, shares.
    The two labelled wind + solar values are the notebook's headline shares (first and last 12 months, zone-hours
    with a missing reported fuel left out), checked against results.json nb01.fuel_mix.
(b) wind + solar share of each region's own generation, monthly (faint) and 12-month centred mean (bold)
    (the notebook's panel c).
(c) PACF of hourly net load per zone, lags 1-200, on each zone's longest contiguous segment, method 'ldb',
    six zones overlaid; the notebook drew one bar panel per zone. Mean |PACF| at lags 1, 24 and 168 is printed
    from results.json nb01.pacf_mean_abs (and checked against the recomputation).
Dropped: the notebook's panel (a) (the same mix in GW; the report quotes only shares) and the boxed note on
California's reporting gap (the gap is still marked by the dashed lines).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import pacf

try:
    from .style import ZONE_COLOR, ZONE_NAME, figure, ledger, panel_label, save
except ImportError:
    from style import ZONE_COLOR, ZONE_NAME, figure, ledger, panel_label, save
from src import config as cfg  # noqa: E402  (style.py puts the repository root on sys.path)

STACK = ["coal", "oil", "gas", "nuclear", "hydro", "other", "wind", "solar"]
FUEL_COLOR = {"coal": "#3b3b3b", "oil": "#7b5e3b", "gas": "#e8833a", "nuclear": "#b5417f",
              "hydro": "#2b7bba", "other": "#9aa0a6", "wind": "#3fa34d", "solar": "#f2c14e"}
NLAGS = 200


def _mix(df: pd.DataFrame, zones: list[str]):
    d = df.copy()
    for f in STACK:
        reports = d.groupby(cfg.GROUP_COL)[f].transform(lambda s: s.notna().mean() >= 0.01)
        d[f] = d[f].where(reports, 0.0)
    mon = d.groupby(cfg.TIME_COL)[STACK].sum(min_count=1).resample("MS").mean() / 1000.0
    share = mon.div(mon.sum(axis=1), axis=0) * 100.0
    ciso_missing = (d[d[cfg.GROUP_COL] == "CISO"].set_index(cfg.TIME_COL)["hydro"]
                    .isna().resample("MS").mean())
    blk = ciso_missing[ciso_missing > 0.5]
    gap = (blk.index.min(), blk.index.max() + pd.offsets.MonthEnd(1))
    ws = {}
    for z in zones:
        dz = d[d[cfg.GROUP_COL] == z].set_index(cfg.TIME_COL)[STACK].resample("MS").mean()
        ws[z] = dz[["wind", "solar"]].sum(axis=1) / dz.sum(axis=1) * 100
    full = d.copy()
    full.loc[full[STACK].isna().any(axis=1), STACK] = np.nan
    mon_c = full.groupby(cfg.TIME_COL)[STACK].sum(min_count=1).resample("MS").mean() / 1000.0
    share_c = mon_c.div(mon_c.sum(axis=1), axis=0) * 100.0
    v0 = share_c.iloc[:12][["wind", "solar"]].sum(axis=1).mean()
    v1 = share_c.iloc[-12:][["wind", "solar"]].sum(axis=1).mean()
    return share, ws, gap, v0, v1


def _pacf(df: pd.DataFrame, zones: list[str]):
    segments = {}
    for z in zones:
        s = df[df[cfg.GROUP_COL] == z].set_index(cfg.TIME_COL)[cfg.TARGET].sort_index()
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        missing = full.isna().to_numpy()
        best_start = best_len = cur = start = 0
        for i, gap in enumerate(missing):
            if gap:
                cur = 0
            else:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best_len:
                    best_start, best_len = start, cur
        segments[z] = full.iloc[best_start:best_start + best_len]
    P = {z: pacf(s.values, nlags=NLAGS, method="ldb") for z, s in segments.items()}
    ci = {z: 1.96 / np.sqrt(len(s)) for z, s in segments.items()}
    return P, ci, {z: len(s) for z, s in segments.items()}


def draw():
    df = pd.read_parquet(cfg.PANEL)
    df[cfg.TIME_COL] = pd.to_datetime(df[cfg.TIME_COL])
    zones = sorted(df[cfg.GROUP_COL].unique())
    L = ledger()["nb01"]
    fm = L["fuel_mix"]

    share, ws, gap, v0, v1 = _mix(df, zones)
    assert round(v0, 1) == round(fm["vre_share_first12_pct"], 1) and round(v1, 1) == round(fm["vre_share_last12_pct"], 1)
    P, ci, nseg = _pacf(df, zones)
    assert nseg == L["pacf_segment_hours"]
    for lag in (1, 24, 168):
        m = np.mean([abs(P[z][lag]) for z in zones])
        assert round(float(m), 4) == L["pacf_mean_abs"][str(lag)], (lag, m)

    fig, _ax = figure(3.3)
    _ax.remove()
    # left column: (a) and (b) on the same years; the spacer column holds (a)'s legend and (b)'s labels
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.42, 0.92], height_ratios=[1, 1], wspace=0.05, hspace=0.32,
                          left=0.08, right=0.99, top=0.97, bottom=0.11)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 2])

    # (a) shares
    ax = ax_a
    ax.stackplot(share.index, *[share[f] for f in STACK], labels=STACK,
                 colors=[FUEL_COLOR[f] for f in STACK], lw=0)
    # same x range as (b), so the years line up
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of generation, %")
    ax.grid(False)
    for g in gap:
        ax.axvline(g, color="red", ls="--", lw=0.8)
    ax.text(gap[0] + (gap[1] - gap[0]) / 2, 3, "CA gap", fontsize=7, color="white", ha="center", va="bottom")
    ax.annotate(f"wind + solar {v0:.1f}%", xy=(share.index[1], 100 - v0 / 2), ha="left", va="center",
                fontsize=7, fontweight="bold", color="white")
    ax.annotate(f"wind + solar {v1:.1f}%", xy=(share.index[-2], 100 - v1 / 2), ha="right", va="center",
                fontsize=7, fontweight="bold", color="white")
    ax.set_xticks(pd.date_range(share.index.min(), share.index.max(), freq="YS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=7,
              handlelength=0.9, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.15)
    panel_label(ax, "a")

    # (b) wind + solar share by region
    ax = ax_b
    ends = {}
    for z in zones:
        s = ws[z]
        col = ZONE_COLOR[z]
        ax.plot(s.index, s.values, lw=0.5, alpha=0.22, color=col)
        roll = s.rolling(12, center=True).mean()
        ax.plot(s.index, roll, lw=1.6, color=col)
        last = roll.dropna()
        ends[z] = (last.index[-1], last.iloc[-1])
    placed = []
    for z in sorted(ends, key=lambda z: ends[z][1]):
        y = ends[z][1] if not placed else max(ends[z][1], placed[-1] + 7.0)
        placed.append(y)
        ax.text(1.02, y, ZONE_NAME[z], color=ZONE_COLOR[z], transform=ax.get_yaxis_transform(), clip_on=False,
                fontsize=7, fontweight="bold", va="center")
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_xticks(pd.date_range(share.index.min(), share.index.max(), freq="YS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, 56)
    ax.set_ylabel("wind + solar share, %")
    panel_label(ax, "b")

    # (c) PACF, six zones overlaid
    ax = ax_c
    lags = np.arange(1, NLAGS + 1)
    band = max(ci.values())
    ax.axhspan(-band, band, color="crimson", alpha=0.18, lw=0)
    for lag, col in ((24, "green"), (48, "purple"), (72, "darkorange"), (168, "black")):
        ax.axvline(lag, color=col, ls=":", lw=0.9)
    for z in zones:
        ax.plot(lags, P[z][1:NLAGS + 1], lw=0.6, color=ZONE_COLOR[z], alpha=0.85, label=ZONE_NAME[z])
    ma = L["pacf_mean_abs"]
    ax.annotate(f"lag 1: {ma['1']:.3f}", xy=(1, 0.98), xytext=(14, 0.9), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    ax.annotate(f"lag 24: {ma['24']:.3f}", xy=(24, 0.36), xytext=(36, 0.45), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    ax.annotate(f"lag 168: {ma['168']:.3f}", xy=(168, 0.1), xytext=(118, 0.3), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    ax.text(0.98, 0.985, "mean |PACF| over\nthe six zones", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="0.3")
    ax.set_xlim(0, NLAGS + 2)
    ax.set_xticks([0, 24, 48, 72, 100, 168, 200])
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("partial autocorrelation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=1, fontsize=7, handlelength=1.2, labelspacing=0.2)
    panel_label(ax, "c")

    return save(fig, "generation_mix_pacf")


if __name__ == "__main__":
    print(draw())
