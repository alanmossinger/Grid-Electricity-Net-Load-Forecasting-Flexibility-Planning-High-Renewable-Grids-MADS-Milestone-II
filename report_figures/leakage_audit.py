"""Two leaks, measured before removal.

(a) The nine same-hour fuel columns summed against net load, one small panel per zone, with the 45-degree line:
    redrawn from data/processed/panel.parquet exactly as notebook 02 (cell drawing 02_leakage_fuel_columns.png):
    panel sorted by (timestamp, zone), fuel_sum = the nine FUELS columns with NaN as 0, r = Pearson on every row of
    the zone, scatter = g.sample(min(4000, n), random_state=cfg.SEED), red where r > 0.97.
(b) Cross-validated MAE by horizon, admissible against leaked rolling windows, from results.json
    nb04.rolling_leak.by_horizon (notebook 04 section 4.2, 04_leak_cost.png panel a). The optimism figures that
    were the notebook's panel b are printed beside each horizon instead of drawn as bars.
Dropped: the 'gas alone' scatter and the identity text box (02_leakage_fuel_columns.png, last two cells), each
zone's 'mean gap' line, and 04_leak_cost.png's bar panel (its values are the percentages printed in (b)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .style import ZONE_NAME, figure, ledger, panel_label, save
except ImportError:
    from style import ZONE_NAME, figure, ledger, panel_label, save
from src import config as cfg  # noqa: E402  (style.py puts the repository root on sys.path)

FUELS = ["coal", "gas", "nuclear", "oil", "other", "hydro", "battery", "unknown_storage", "solar_battery"]
BAD, ACC = "#c1121f", "#3a7ca5"
ADM, LEAK = "#3fa34d", "#c1121f"


def draw():
    panel = pd.read_parquet(cfg.PANEL).sort_values([cfg.TIME_COL, cfg.GROUP_COL])
    d = panel.copy()
    d["fuel_sum"] = d[FUELS].fillna(0).sum(axis=1)
    leak = ledger()["nb04"]["rolling_leak"]["by_horizon"]
    hs = sorted(int(h) for h in leak)

    fig, _ax = figure(3.2)
    _ax.remove()
    outer = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.2, left=0.085, right=0.985, top=0.93, bottom=0.13)
    inner = outer[0, 0].subgridspec(2, 3, wspace=0.45, hspace=0.62)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(3)]

    for k, (ax, (z, g)) in enumerate(zip(axes, d.groupby(cfg.GROUP_COL))):
        r = float(np.corrcoef(g["fuel_sum"], g[cfg.TARGET])[0, 1])
        s = g.sample(min(4000, len(g)), random_state=cfg.SEED)
        col = BAD if r > .97 else ACC
        ax.scatter(s[cfg.TARGET] / 1000, s["fuel_sum"] / 1000, s=0.6, alpha=.25, color=col, lw=0, rasterized=True)
        lim = [0, max(s[cfg.TARGET].max(), s["fuel_sum"].max()) / 1000 * 1.05]
        ax.plot(lim, lim, "k--", lw=.7)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(False)
        ax.set_title(f"{ZONE_NAME[z]}  r = {r:.4f}", fontsize=7, weight="normal", color=col, pad=2)
        ax.locator_params(nbins=3)
        if k % 3 == 0:
            ax.set_ylabel("fuel sum (GW)", labelpad=1)
        if k >= 3:
            ax.set_xlabel("net load (GW)", labelpad=1)
    axes[0].text(-0.42, 1.04, "(a)", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.5,
                 weight="bold")

    ax = fig.add_subplot(outer[0, 1])
    adm = [leak[str(h)]["admissible"] for h in hs]
    lk = [leak[str(h)]["leaked"] for h in hs]
    ax.plot(hs, adm, "o-", lw=1.6, ms=4.5, color=ADM, label="admissible: window ends\nat the origin")
    ax.plot(hs, lk, "s--", lw=1.6, ms=4.5, color=LEAK, label="leaked: window ends at t-1")
    # the optimism (admissible minus leaked), the notebook's panel b, as a small table in the empty corner
    ax.fill_between(hs, lk, adm, color=LEAK, alpha=0.08, lw=0)
    rows = ["optimism of the leak:"] + [
        f"h = {h}: {leak[str(h)]['optimism_mw']:,.0f} MW ({leak[str(h)]['optimism_pct']:.1f}%)"
        for h in hs if leak[str(h)]["optimism_mw"] > 1]
    ax.text(0.97, 0.06, "\n".join(rows), transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=LEAK, linespacing=1.3)
    ax.annotate(f"both\n{leak['1']['admissible']:,.1f}", (1, leak["1"]["admissible"]), xytext=(-3, -7),
                textcoords="offset points", ha="left", va="top", fontsize=7)
    ax.set_xticks(hs)
    ax.set_xlim(-5, 79)
    ax.set_ylim(0, 3500)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("CV MAE (MW)")
    ax.legend(loc="upper left", fontsize=7, handlelength=2.2, borderaxespad=0.1)
    panel_label(ax, "b")
    return save(fig, "leakage_audit")


if __name__ == "__main__":
    print(draw())
