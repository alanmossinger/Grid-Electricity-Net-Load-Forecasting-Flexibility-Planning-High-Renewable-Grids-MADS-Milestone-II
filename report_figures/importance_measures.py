"""Two importance measures (ablation vs permutation) and permutation importance by horizon.

Replaces 08_ablation_vs_permutation.png (notebook 08, cell drawing it) and 07_importance_by_horizon.png
(notebook 07 §7.2b). Every value is read from the ledger, defined exactly as in the notebooks:
  (a) nb04.ablation.families[f].delta_mw            -- CV-MAE increase after refitting without the family
      nb04.hyperparam_sensitivity.fold_noise_floor_mw -- dashed "fold noise floor"
  (b) nb07.permutation_by_family[f]                 -- held-out MAE increase, family shuffled (h=24 champion)
  (c) nb07.importance_by_horizon.share[h][f]        -- share of the total permutation importance
  (d) nb07.importance_by_horizon.mae_increase_mw[h][f]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

# the family colours of notebook 07
COLS = {"lag": "#e8833a", "rolling": "#7e57c2", "weather": "#2b7bba",
        "calendar": "#9aa0a6", "exogenous": "#2a9d8f", "zone": "#b5417f"}
STACK = ["lag", "rolling", "exogenous", "weather", "calendar", "zone"]   # notebook 07 stacking order
HORIZONS = ["1", "24", "48", "72"]


def draw() -> Path:
    L = S.ledger()
    s4, s7 = L["nb04"], L["nb07"]
    fams = ["weather", "calendar", "lag", "zone", "exogenous", "rolling"]
    abl = {f: s4["ablation"]["families"][f]["delta_mw"] for f in fams}
    perm = {f: s7["permutation_by_family"][f] for f in fams}
    order = sorted(fams, key=lambda f: perm[f], reverse=True)      # as notebook 08: by permutation MW
    floor = s4["hyperparam_sensitivity"]["fold_noise_floor_mw"]
    share = s7["importance_by_horizon"]["share"]
    mw = s7["importance_by_horizon"]["mae_increase_mw"]

    fig, axs = S.figure(3.45, nrows=2, ncols=2, gridspec_kw=dict(height_ratios=[1, 1.08]))
    y = np.arange(len(order))

    # (a) ablation, log axis, negative shown red at 1 (as the notebook)
    ax = axs[0, 0]
    a = np.array([abl[f] for f in order])
    ax.barh(y, np.clip(a, 1, None), color=["#6aa84f" if v > 0 else "#cc4125" for v in a],
            edgecolor="#333", lw=0.5, height=0.7)
    for yy, v in zip(y, a):
        ax.text(max(v, 1) * 1.2, yy, f"{v:+,.0f} MW", va="center", fontsize=6.5, zorder=4,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.axvline(floor, color="#7f7f7f", ls="--", lw=0.9)
    ax.text(floor * 1.1, -0.95, "fold noise floor", fontsize=6.5, color="#555", va="center")
    ax.set_title("ablation: refit without the family", loc="left", fontweight="normal", fontsize=8)
    S.panel_label(ax, "a")

    # (b) permutation, same log axis
    ax = axs[0, 1]
    p = np.array([perm[f] for f in order])
    ax.barh(y, p, color="#3d85c6", edgecolor="#333", lw=0.5, height=0.7)
    for yy, v in zip(y, p):
        ax.text(v * 1.2, yy, f"{v:,.0f} MW", va="center", fontsize=6.5)
    ax.set_title("permutation: shuffle in the fitted model", loc="left", fontweight="normal", fontsize=8)
    S.panel_label(ax, "b")

    for ax in axs[0]:
        ax.set_xscale("log"); ax.set_xlim(0.8, 3e4)
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_ylim(len(order) - 0.5, -1.4)
        ax.set_xlabel("MAE increase (MW, log scale)", labelpad=1)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    axs[0, 1].set_yticklabels([])

    # (c) share of total permutation importance by horizon, stacked
    ax = axs[1, 0]
    xs = np.arange(len(HORIZONS))
    bottom = np.zeros(len(HORIZONS))
    for f in STACK:
        vals = np.array([100 * share[h][f] for h in HORIZONS])
        ax.bar(xs, vals, 0.62, bottom=bottom, color=COLS[f], label=f)
        for j, v in enumerate(vals):
            if v > 4:
                ax.text(xs[j], bottom[j] + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100); ax.set_ylabel("share of total (%)")
    ax.set_xlabel("forecast horizon (hours)", labelpad=1)
    ax.grid(axis="x", visible=False)
    S.panel_label(ax, "c")

    # (d) the same in MW
    ax = axs[1, 1]
    hx = [int(h) for h in HORIZONS]
    for f in STACK:
        ax.plot(hx, [mw[h][f] for h in HORIZONS], "o-", lw=1.4, ms=3.5, color=COLS[f], label=f)
    ax.set_xticks(hx); ax.set_xlabel("forecast horizon (hours)", labelpad=1)
    ax.set_ylabel("MAE increase (MW)")
    ax.set_ylim(-500, 20500)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    S.panel_label(ax, "d")

    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels, ncol=2, loc="upper right", fontsize=6.5, handlelength=1.4,
                     columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.6, w_pad=1.2)
    return S.save(fig, "importance_measures")


if __name__ == "__main__":
    print(draw())
