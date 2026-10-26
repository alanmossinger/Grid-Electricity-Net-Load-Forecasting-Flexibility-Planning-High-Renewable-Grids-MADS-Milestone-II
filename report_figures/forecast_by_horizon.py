"""Error and skill by horizon, and the rolling-origin refits.

(a) held-out MAE of seasonal naive t-168, DF composite v2 (rebuilt per horizon) and the per-horizon champion, and
(b) skill against each benchmark (ledger nb04.per_horizon_skill, as notebook 04 section 4.9b draws
04_skill_by_horizon.png); (c) champion refit at four rolling origins and scored on the following quarter, against
seasonal naive, with the primary holdout MAE as the reference line (ledger nb04.rolling_origins, as notebook 04
section 4.15b draws 04_rolling_origins.png).
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

NAIVE_C, V2_C, MODEL_C, OK_C = "#c1121f", "#e8833a", "#1d3557", "#3fa34d"


def draw() -> Path:
    led = S.ledger()["nb04"]
    ph = led["per_horizon_skill"]
    hs = sorted(int(h) for h in ph)
    fig, _ax = S.figure(3.3)
    _ax.remove()
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.62, wspace=0.24, left=0.085, right=0.995, top=0.97, bottom=0.12)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_axisbelow(True)

    # (a) MAE by horizon
    w = 0.27
    x = np.arange(len(hs))
    ax_a.bar(x - w, [ph[str(h)]["naive_mae"] for h in hs], w, color=NAIVE_C, label="seasonal naive t-168")
    ax_a.bar(x, [ph[str(h)]["composite_v2_mae"] for h in hs], w, color=V2_C, label="DF composite v2")
    ax_a.bar(x + w, [ph[str(h)]["model_mae"] for h in hs], w, color=MODEL_C, label="champion (tuned per horizon)")
    for j, h in enumerate(hs):
        v = ph[str(h)]["model_mae"]
        ax_a.annotate(f"{v:,.0f}", (j + w, v), xytext=(0, 2), textcoords="offset points", ha="center",
                      fontsize=6.3, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"h={h}" for h in hs])
    ax_a.set_xlabel("forecast horizon (hours)")
    ax_a.set_ylabel("holdout MAE (MW)")
    ax_a.set_ylim(0, 6300)
    ax_a.grid(axis="x", visible=False)
    ax_a.legend(loc="upper center", ncol=2, fontsize=6.2, handlelength=1.0, columnspacing=0.7,
                borderaxespad=0.1, labelspacing=0.2)
    S.panel_label(ax_a, "a")

    # (b) skill by horizon
    sn = [100 * ph[str(h)]["skill_vs_naive"] for h in hs]
    sv = [100 * ph[str(h)]["skill_vs_v2"] for h in hs]
    ax_b.axvspan(36, 76, color="0.85", alpha=0.5, zorder=0, lw=0)
    ax_b.plot(hs, sn, "o-", lw=1.5, ms=3.5, color=NAIVE_C, label="vs naive")
    ax_b.plot(hs, sv, "s-", lw=1.5, ms=3.5, color=V2_C, label="vs composite v2")
    for h, a, b in zip(hs, sn, sv):
        ax_b.annotate(f"{a:.1f}%", (h, a), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.2,
                      color=NAIVE_C)
        first = h == hs[0]      # at h=1 the two labels would print on each other: this one goes right, below
        ax_b.annotate(f"{b:.1f}%", (h, b), xytext=(6, -2) if first else (0, 4), textcoords="offset points",
                      ha="left" if first else "center", va="top" if first else "baseline", fontsize=6.2,
                      color="#b35a1f")
    ax_b.axhline(0, color="0.3", lw=0.8)
    ax_b.set_ylim(-8, 150)
    ax_b.set_xlim(-4, 78)
    ax_b.set_xticks(hs)
    ax_b.set_xlabel("forecast horizon (hours)")
    ax_b.set_ylabel("skill (%)")
    ax_b.text(56, 128, "benchmark keeps a day-ahead\ndemand forecast: lower bound", ha="center", va="top",
              fontsize=6, color="0.35")
    ax_b.legend(loc="upper left", fontsize=6.2, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
    S.panel_label(ax_b, "b")

    # (c) rolling origins
    ro = led["rolling_origins"]
    rows = ro["rows"]
    xs = np.arange(len(rows))
    ax_c.bar(xs - 0.2, [r["mae"] for r in rows], 0.4, color=MODEL_C, label="champion")
    ax_c.bar(xs + 0.2, [r["naive_mae"] for r in rows], 0.4, color=NAIVE_C, label="seasonal naive t-168")
    for j, r in enumerate(rows):
        ax_c.annotate(f"{r['mae']:,.0f}", (j - 0.2, r["mae"]), xytext=(0, 2), textcoords="offset points",
                      ha="center", fontsize=6.3, fontweight="bold")
        ax_c.annotate(f"{r['naive_mae']:,.0f}", (j + 0.2, r["naive_mae"]), xytext=(0, 2), textcoords="offset points",
                      ha="center", fontsize=6.3, color="#7a0c15")
    prim = ro["primary_holdout_mae"]
    ax_c.axhline(prim, color=OK_C, ls="--", lw=1.2, label=f"primary holdout {prim:,.0f} MW (no January)")
    labels = []
    for r in rows:
        m = [int(v) for v in r["months"]]
        span = f"{calendar.month_abbr[m[0]]}–{calendar.month_abbr[m[-1]]}"
        labels.append(f"{r['origin']} ({span})\nskill vs naive {100 * r['skill_vs_naive']:.1f}%")
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(labels, fontsize=6.5)
    ax_c.set_ylabel("MAE (MW)")
    ax_c.set_xlabel("rolling origin (quarter scored after it)", labelpad=1)
    ax_c.set_ylim(0, 6600)
    ax_c.set_xlim(-0.5, len(rows) - 0.5)
    ax_c.grid(axis="x", visible=False)
    ax_c.legend(loc="upper center", ncol=3, fontsize=6.3, handlelength=1.6, borderaxespad=0.1)
    S.panel_label(ax_c, "c")
    return S.save(fig, "forecast_by_horizon")


if __name__ == "__main__":
    print(draw())
