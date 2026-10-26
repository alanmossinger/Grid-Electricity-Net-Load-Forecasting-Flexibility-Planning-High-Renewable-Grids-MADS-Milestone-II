"""Why the selector keeps every feature.

Every value comes from results.json nb02, written by notebook 02 sections 2.20-2.22:
(a) forward-selection curve, MAE on the last CV fold against the number of MI-ranked features kept
    (nb02.forward_curve); the 39 lowest-ranked features are worth forward_curve[30] - forward_curve[69].
    Dashed line: within 1% of the best MAE; dash-dot: the knee k (nb02.forward_knee_k), as in 02_forward_selection.png.
(b) mean |SHAP| per feature, top 20, log axis (nb02.shap_mean_abs), coloured by feature family as notebook 02 does
    (fam(): '_rel' -> relative, otherwise src.features.feature_family).
(c) MI rank against SHAP rank (ranks of nb02.mi_mean and nb02.shap_mean_abs, descending, ties averaged, as
    pandas .rank(ascending=False) in the notebook), Spearman rho recomputed and checked against
    nb02.shap_mi_rank_spearman; the six largest disagreements are labelled as in the notebook.
Nothing is refitted: the SHAP values were saved by notebook 02 as means, which is all both panels need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .style import figure, ledger, panel_label, save
except ImportError:
    from style import figure, ledger, panel_label, save
from src import features as P  # noqa: E402  (style.py puts the repository root on sys.path)

FAMC = {"relative": "#3fa34d", "lag": "#e8833a", "rolling": "#7e57c2", "weather": "#2b7bba",
        "exogenous": "#2a9d8f", "calendar": "#9aa0a6", "zone": "#b5417f", "other": "#888"}


def fam(c: str) -> str:
    return "relative" if c.endswith("_rel") else P.feature_family(c)


def draw():
    L = ledger()["nb02"]
    fs = pd.Series({int(k): v for k, v in L["forward_curve"].items()}).sort_index()
    knee = int(L["forward_knee_k"])
    mi_mean = pd.Series(L["mi_mean"]).sort_values(ascending=False)
    shap_imp = pd.Series(L["shap_mean_abs"]).sort_values(ascending=False)
    assert set(mi_mean.index) == set(shap_imp.index) and len(shap_imp) == L["n_features"]

    mi_r = mi_mean.rank(ascending=False)
    sh_r = shap_imp.rank(ascending=False)
    common = list(shap_imp.index)
    rho = float(pd.Series([mi_r[c] for c in common]).corr(pd.Series([sh_r[c] for c in common]), method="spearman"))
    assert abs(rho - L["shap_mi_rank_spearman"]) < 1e-9, rho
    top10 = shap_imp.head(10).sum() / shap_imp.sum() * 100
    tail = fs[30] - fs[fs.index.max()]

    fig, _ax = figure(3.35)
    _ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.62, left=0.075, right=0.94, top=0.95,
                          bottom=0.13)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) forward-selection curve
    ax = ax_a
    ax.plot(fs.index, fs.values, "o-", lw=1.5, ms=3.5, color="#e8833a")
    ax.axhline(fs.min(), color="0.6", ls=":", lw=0.8)
    ax.axhline(fs.min() * 1.01, color="#c1121f", ls="--", lw=0.9)
    ax.text(2, fs.min() - 4, "within 1% of best MAE", fontsize=7, color="#c1121f", va="top")
    ax.axvline(knee, color="#3fa34d", lw=1.1, ls="-.")
    ax.annotate("", xy=(fs.index.max(), fs.iloc[-1]), xytext=(30, fs[30]),
                arrowprops=dict(arrowstyle="->", color="#3fa34d", lw=1.1, connectionstyle="arc3,rad=-0.25"))
    ax.text(63, 905, f"no knee: the 39\nlowest-ranked\nfeatures are worth\n{tail:.2f} MW", fontsize=7,
            color="#3fa34d", ha="right", va="top", fontweight="bold")
    ax.set_xlim(-2, 72)
    ax.set_ylim(550, 1250)
    ax.set_xlabel("features kept (MI order)")
    ax.set_ylabel("MAE, last CV fold (MW)")
    panel_label(ax, "a")

    # (b) mean |SHAP|, top 20, log axis
    ax = ax_b
    sh = shap_imp.head(20)[::-1]
    ax.barh(range(len(sh)), sh.values, color=[FAMC[fam(c)] for c in sh.index], height=0.75)
    ax.set_xscale("log")
    ax.set_xlim(10, shap_imp.max() * 3)
    ax.set_yticks(range(len(sh)))
    ax.set_yticklabels(sh.index, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_ylim(-0.6, len(sh) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.text(shap_imp.iloc[0] * 1.15, len(sh) - 1, f"{shap_imp.iloc[0]:,.0f}", fontsize=7, va="center", ha="left")
    ax.text(0.97, 0.47, f"top 10 hold\n{top10:.1f}% of\n|SHAP|", transform=ax.transAxes, ha="right", va="center",
            fontsize=7)
    ax.set_xlabel("mean |SHAP| (MW, log)")
    present = [f for f in FAMC if any(fam(c) == f for c in shap_imp.index)]
    handles = [plt.Line2D([], [], color=FAMC[f], lw=5, label=f) for f in present]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, handlelength=1.0, labelspacing=0.2,
              borderaxespad=0.1)
    panel_label(ax, "b")

    # (c) MI rank against SHAP rank
    ax = ax_c
    ax.scatter([mi_r[c] for c in common], [sh_r[c] for c in common], c=[FAMC[fam(c)] for c in common],
               s=14, edgecolor="0.3", linewidths=0.4, zorder=3)
    lim = len(common) + 1
    ax.plot([0, lim], [0, lim], "--", color="0.55", lw=0.9, zorder=1)
    disagree = sorted(common, key=lambda c: -abs(mi_r[c] - sh_r[c]))[:6]
    # points within two ranks of each other share one label (oil_lag24 and zone_is_BPAT sit side by side on the
    # tied bottom row), written under that row so it covers no point
    groups: list[list[str]] = []
    for c in disagree:
        for g in groups:
            if abs(mi_r[g[0]] - mi_r[c]) <= 2 and sh_r[g[0]] == sh_r[c]:
                g.append(c)
                break
        else:
            groups.append([c])
    for g in groups:
        x = float(np.mean([mi_r[c] for c in g])); y = float(sh_r[g[0]])
        if len(g) > 1:
            ax.annotate(", ".join(sorted(g)), (x, y), xytext=(-1, y + 5.5), textcoords="data", fontsize=6.5,
                        ha="right", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
        else:
            ax.annotate(g[0], (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points", ha="left")
    ax.set_title(f"Spearman ρ = {rho:+.2f}", fontsize=7.5, pad=3)
    ax.set_xlim(lim + 2, -2)
    ax.set_ylim(lim + 7, -2)
    ax.set_xlabel("MI rank (1 = highest MI)")
    ax.set_ylabel("SHAP rank (1 = most used)")
    panel_label(ax, "c")
    return save(fig, "feature_selection")


if __name__ == "__main__":
    print(draw())
