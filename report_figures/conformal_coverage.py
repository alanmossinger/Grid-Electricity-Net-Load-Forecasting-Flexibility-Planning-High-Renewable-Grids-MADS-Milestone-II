"""Coverage against width for every interval construction, and per-zone coverage and median.

Replaces 08_coverage_vs_width.png (notebook 08 §8.6) and 06_mondrian_by_zone.png (notebook 06 §6.2c).
Every value is read from the ledger, defined exactly as in the notebooks:
  (a) nb06.interval_comparison[method].{coverage, mean_width, winkler}, sorted by Winkler (the notebook's
      numbering 1-9); nb06.primary_method is the shipped one (red)
  (b) nb06.coverage_by_zone_pooled / nb06.coverage_by_zone (coverage), nb06.mondrian_margins_mw (margin per side)
  (c) nb06.median_by_zone["before shift" | "pooled shift" | "Mondrian shift"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

ZONES = ["CISO", "ERCO", "MISO", "SWPP", "BPAT", "NYIS"]       # cfg.ZONES order, as the notebook
POOLED_C, MOND_C, NOM_C = "#e9a03b", "#3fa34d", "#1d3557"         # notebook 06 colours
# label offsets (points) for the numbers of the nine methods; 1, 2 and 4 sit within 60 MW of each other
# and 5/6 share a width, so each gets its own side
SHORT = {"per-level shifts, Mondrian by zone": "per-level shifts, Mondrian",
         "CQR additive, Mondrian by zone": "CQR additive, Mondrian",
         "CQR scaled, Mondrian by zone": "CQR scaled, Mondrian"}
OFFSETS = {1: (0, 6), 2: (8, -9), 3: (6, 5), 4: (-8, 5), 5: (-7, 4), 6: (-7, -6), 7: (0, 6), 8: (0, 6), 9: (0, 6)}


def draw() -> Path:
    s6 = S.ledger()["nb06"]
    icd = sorted(s6["interval_comparison"].items(), key=lambda kv: kv[1]["winkler"])
    shipped = s6["primary_method"]

    fig = S.figure(3.4)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1], hspace=0.62, wspace=0.27,
                          left=0.085, right=0.995, top=0.95, bottom=0.12)
    ax = fig.add_subplot(gs[:, 0])

    # (a) coverage against width -------------------------------------------------------------
    ax.axhspan(0.85, 0.95, color="#d9ead3", alpha=0.45, zorder=0, lw=0)
    ax.axhline(0.90, color="#7f7f7f", ls="--", lw=0.9)
    ax.text(6880, 0.9025, "nominal 90%", fontsize=6.5, color="#555")
    ax.text(10750, 0.944, "tolerance ±5 pts", ha="right", va="top", fontsize=6.5, color="#38761d")
    names, wink = [], []
    for k, (name, r) in enumerate(icd, 1):
        is_s = name == shipped
        col = "#c00000" if is_s else "#3d85c6"
        ax.scatter(r["mean_width"], r["coverage"], s=34 if is_s else 20, color=col, edgecolor="#222",
                   lw=0.5, zorder=3)
        ax.annotate(str(k), (r["mean_width"], r["coverage"]), textcoords="offset points",
                    xytext=OFFSETS[k], ha="center", va="center", fontsize=6.5, weight="bold",
                    color="#c00000" if is_s else "#1f1f1f")
        names.append(f"{k}  {SHORT.get(name, name)}" + (" (shipped)" if is_s else ""))
        wink.append(f"{r['winkler']:,.0f}")
    # the key, in the empty lower-right corner (no data there: every point right of 7,200 MW is above 0.88)
    x0, x1, ytop, dy = 0.235, 0.985, 0.40, 0.040
    ax.text(x0, ytop + 0.012, "method", transform=ax.transAxes, fontsize=6.2, weight="bold", va="bottom")
    ax.text(x1, ytop + 0.012, "Winkler", transform=ax.transAxes, fontsize=6.2, weight="bold", va="bottom",
            ha="right")
    for i, (n, w) in enumerate(zip(names, wink)):
        col = "#c00000" if "shipped" in n else "#1f1f1f"
        ax.text(x0, ytop - i * dy, n, transform=ax.transAxes, fontsize=6.2, va="top", color=col)
        ax.text(x1, ytop - i * dy, w, transform=ax.transAxes, fontsize=6.2, va="top", ha="right", color=col)
    ax.set_xlim(6800, 10800); ax.set_ylim(0.77, 0.955); ax.set_xticks([7000, 8000, 9000, 10000])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("mean interval width (MW)")
    ax.set_ylabel("coverage of the nominal 90% band")
    S.panel_label(ax, "a")

    # (b) coverage by zone, pooled against Mondrian ------------------------------------------------
    bp = {z["group"]: z["coverage"] for z in s6["coverage_by_zone_pooled"]}
    bm = {z["group"]: z["coverage"] for z in s6["coverage_by_zone"]}
    mm = s6["mondrian_margins_mw"]
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(ZONES)); w = 0.45
    ax.axhspan(85, 95, color=MOND_C, alpha=0.10, zorder=0, lw=0)
    # the nominal line runs behind the bars, so the values printed inside the bar tops never cross it
    ax.axhline(90, ls="--", color=NOM_C, lw=1.0, zorder=1.5)
    ax.bar(xs - w / 2, [100 * bp[z] for z in ZONES], w, color=POOLED_C, label="pooled CQR", zorder=2)
    ax.bar(xs + w / 2, [100 * bm[z] for z in ZONES], w, color=MOND_C, label="Mondrian CQR (shipped)", zorder=2)
    for j, z in enumerate(ZONES):
        for dx, v, bold in ((-w / 2, 100 * bp[z], False), (w / 2, 100 * bm[z], True)):
            ax.text(xs[j] + dx, v - 0.5, f"{v:.1f}", ha="center", va="top", fontsize=5.4,
                    weight="bold" if bold else "normal", color="white", zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([f"{z}\n{mm[z]:+,.0f}" for z in ZONES], fontsize=6.3)
    ax.set_xlabel("zone, with the Mondrian margin per side (MW)", fontsize=6.8, labelpad=1)
    ax.set_ylim(74, 106); ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.set_ylabel("coverage (%)")
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, fontsize=6.3, borderaxespad=0.1, handlelength=1.2, columnspacing=0.8)
    S.panel_label(ax, "b")

    # (c) fraction of outcomes below the median forecast, by zone ------------------------------------
    med = s6["median_by_zone"]
    ax = fig.add_subplot(gs[1, 1])
    ax.axhspan(0.49, 0.51, color=MOND_C, alpha=0.12, zorder=0, lw=0)
    ax.axhline(0.5, ls="--", color=NOM_C, lw=1.0)
    ax.plot(xs, [med["before shift"][z] for z in ZONES], "o", ms=3.8, color="#c1121f", label="before any shift")
    ax.plot(xs, [med["pooled shift"][z] for z in ZONES], "^", ms=3.8, color=POOLED_C, label="pooled shift")
    ax.plot(xs, [med["Mondrian shift"][z] for z in ZONES], "s", ms=4.2, color=MOND_C, label="Mondrian shift")
    for j, z in enumerate(ZONES):
        ax.annotate(f"{med['Mondrian shift'][z]:.3f}", (xs[j], med["Mondrian shift"][z]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=5.8, weight="bold",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_xticks(xs); ax.set_xticklabels(ZONES, fontsize=6.3)
    ax.set_xlim(-0.5, len(ZONES) - 0.2)
    ax.set_ylim(0.42, 0.68); ax.set_yticks([0.45, 0.50, 0.55, 0.60])
    ax.set_ylabel("fraction below median")
    ax.legend(loc="upper center", ncol=3, fontsize=6.0, borderaxespad=0.1, handlelength=1.0,
              columnspacing=0.7, handletextpad=0.3)
    S.panel_label(ax, "c")
    return S.save(fig, "conformal_coverage")


if __name__ == "__main__":
    print(draw())
