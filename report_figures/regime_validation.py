"""Are the regimes zones in disguise, and does the label help the forecast?

(a) zone x regime crosstab (ledger nb05.regime_by_zone, as notebook 05 section 5.4c draws 05_regime_vs_zone.png);
(b) each regime's mean GHI minus its own zone's mean (ledger nb05.regime_weather_within_zone[*].ghi_vs_zone_mean);
(c) change in cross-validated MAE when a regime label is added to the champion (ledger nb05.downstream, rounded
exactly as notebook 05 section 5.5 rounds them for 05_downstream_utility.png), paired p-values from
ledger nb05.downstream_paired, grey band = the largest fold-to-fold SD of the four arms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

REGIME_COLOR = {"winter-shaped": "#1f77b4", "conventional": "#ff7f0e", "DUCK": "#2ca02c"}   # as Figure 9
ARMS = [  # ledger key, label on the figure, colour (notebook 05's)
    ("0  baseline (no cluster feature)", "baseline (no regime label)", "0.45"),
    ("1  cluster_prev (admissible, batch day-ahead)", "yesterday's label\n(admissible, batch day-ahead)", "#3fa34d"),
    ("1b cluster_prev2 (admissible at every origin)", "label two days back\n(admissible at every origin)", "#2a9d8f"),
    ("2  cluster_same (LEAK)", "today's own label\n(LEAK, control)", "#c1121f"),
]


def _setup(led):
    zones = sorted(led["regime_by_zone"])
    regimes = sorted(next(iter(led["regime_by_zone"].values())))          # DUCK, conventional, winter-shaped
    ticks = [S.ZONE_NAME[z] for z in zones]
    xs = np.arange(len(zones))
    return zones, regimes, ticks, xs


def _crosstab(ax_a, led, letter, short=False):
    zones, regimes, ticks, xs = _setup(led)
    bottom = np.zeros(len(zones))
    for rg in regimes:
        v = np.array([led["regime_by_zone"][z][rg] for z in zones], dtype=float)
        ax_a.bar(xs, v, 0.75, bottom=bottom, color=REGIME_COLOR[rg], label=rg)
        bottom += v
    conc = led["regime_zone_concentration"]
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(ticks, rotation=45 if short else 25, ha="right", rotation_mode="anchor")
    ax_a.set_ylabel("zone-days")
    ax_a.set_ylim(0, 3250)
    ax_a.grid(axis="x", visible=False)
    if short:
        ax_a.set_ylim(0, 3900)
        ax_a.legend(loc="upper right", ncol=1, fontsize=6.0, handlelength=1.0, borderaxespad=0.1)
    else:
        ax_a.legend(loc="upper center", ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=0.8,
                    borderaxespad=0.1)
    ax_a.text(1, bottom[1] + 25, f"{conc['share']:.0%} of all DUCK days", ha="center", va="bottom",
              fontsize=6.3, color="0.25")
    S.panel_label(ax_a, letter)


def _ghi(ax_b, led, letter):
    zones, regimes, ticks, xs = _setup(led)
    within = led["regime_weather_within_zone"]
    W = 0.8 / len(regimes)
    for j, rg in enumerate(regimes):
        v = [within[f"{z}/{rg}"]["ghi_vs_zone_mean"] for z in zones]
        ax_b.bar(xs + j * W - 0.4 + W / 2, v, W, color=REGIME_COLOR[rg], label=rg)
    ax_b.axhline(0, color="0.3", lw=0.8)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(ticks, rotation=25, ha="right", rotation_mode="anchor")
    ax_b.set_ylabel("GHI minus zone's\nown mean (W/m²)")
    ax_b.grid(axis="x", visible=False)
    S.panel_label(ax_b, letter)


def _downstream(ax_c, led, letter, short=False):
    ds, paired = led["downstream"], led["downstream_paired"]
    base = ds[ARMS[0][0]]["cv_mae_mean"]
    rows = []
    for key, lab, col in ARMS:
        a = ds[key]
        rows.append({"label": lab, "col": col, "cv_mae": round(a["cv_mae_mean"], 1), "sd": round(a["cv_mae_sd"], 1),
                     "delta": round(a["cv_mae_mean"] - base, 1),
                     "p": paired[key]["paired_p"] if key in paired else None})
    sd_max = max(r["sd"] for r in rows)
    assert abs(sd_max - led["downstream_fold_sd_max_mw"]) < 0.05
    ax_c.axvspan(-sd_max, sd_max, color="0.85", alpha=0.6, zorder=0, lw=0)
    ys = np.arange(len(rows))
    ax_c.barh(ys, [r["delta"] for r in rows], color=[r["col"] for r in rows], height=0.6)
    ax_c.axvline(0, ls="--", lw=1.0, color="0.3")
    for y, r in zip(ys, rows):
        d = r["delta"]
        txt = f"{d:+,.0f} MW (abs {r['cv_mae']:,.0f}, sd {r['sd']:,.0f})"
        if r["p"] is not None:
            txt += f", paired p = {r['p']:.3f}"
        ax_c.annotate(txt, (min(d, 0) if d < 0 else d, y), xytext=(-4 if d < 0 else 4, 0), textcoords="offset points",
                      va="center", ha="right" if d < 0 else "left", fontsize=6.5)
    ax_c.text(sd_max - 4, 2.0, f"grey band: largest\nfold-to-fold SD,\n{sd_max:,.0f} MW", ha="right", va="center",
              fontsize=6.3, color="0.35")
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels([r["label"] for r in rows], fontsize=6.5, linespacing=1.0)
    ax_c.set_xlim(-240, 160)
    ax_c.set_ylim(-0.5, len(rows) - 0.3)
    ax_c.grid(axis="y", visible=False)
    ax_c.set_xlabel(f"change in CV MAE vs the {base:,.0f} MW baseline (MW); negative is better" if short else
                    f"change in CV MAE vs baseline (MW), negative is better; baseline = {base:,.0f} MW")
    S.panel_label(ax_c, letter)


SHORT_LABELS = ["baseline\n(no regime label)", "yesterday's label", "label two days back",
                "today's own label\n(leak, control)"]


def _downstream_short(ax, led, letter):
    """Panel (c) of the full figure, compact: the same rounded deltas and paired p-values, labels to the right."""
    ds, paired = led["downstream"], led["downstream_paired"]
    base = ds[ARMS[0][0]]["cv_mae_mean"]
    rows = [{"col": col, "delta": round(ds[key]["cv_mae_mean"] - base, 1), "sd": round(ds[key]["cv_mae_sd"], 1),
             "p": paired[key]["paired_p"] if key in paired else None} for key, _, col in ARMS]
    sd_max = max(r["sd"] for r in rows)
    assert abs(sd_max - led["downstream_fold_sd_max_mw"]) < 0.05
    ax.axvspan(-sd_max, sd_max, color="0.85", alpha=0.6, zorder=0, lw=0)
    ys = np.arange(len(rows))
    ax.barh(ys, [r["delta"] for r in rows], color=[r["col"] for r in rows], height=0.6)
    ax.axvline(0, ls="--", lw=1.0, color="0.3")
    for y, r in zip(ys, rows):
        txt = f"{r['delta']:+,.0f} MW" + (f", paired p = {r['p']:.3f}" if r["p"] is not None else "")
        ax.annotate(txt, (0, y), xytext=(5, 0), textcoords="offset points", va="center", ha="left", fontsize=6.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(SHORT_LABELS, fontsize=6.5, linespacing=1.0)
    ax.set_xlim(-sd_max * 1.05, sd_max * 1.05)
    ax.set_ylim(-0.5, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"change in CV MAE vs the {base:,.0f} MW baseline (MW);\n"
                  f"grey band = largest fold SD ({sd_max:,.0f} MW)",
                  linespacing=1.1)
    S.panel_label(ax, letter)


def draw() -> Path:
    """(a) crosstab, (b) GHI signature, (c) downstream utility: the full technical report's figure."""
    led = S.ledger()["nb05"]
    fig, _ax = S.figure(3.45)
    _ax.remove()
    top = fig.add_gridspec(1, 2, left=0.09, right=0.99, top=0.97, bottom=0.62, wspace=0.3)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(fig.add_gridspec(1, 1, left=0.25, right=0.99, top=0.42, bottom=0.12)[0, 0])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_axisbelow(True)
    _crosstab(ax_a, led, "a")
    _ghi(ax_b, led, "b")
    _downstream(ax_c, led, "c")
    return S.save(fig, "regime_validation")


def draw_short() -> Path:
    """The 20-page report's version: (a) crosstab and (b) downstream utility, one row."""
    led = S.ledger()["nb05"]
    fig, _ax = S.figure(2.5)
    _ax.remove()
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.99, top=0.93, bottom=0.25, wspace=0.55, width_ratios=[1.0, 1.35])
    ax_a, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (ax_a, ax_c):
        ax.set_axisbelow(True)
    _crosstab(ax_a, led, "a", short=True)
    _downstream_short(ax_c, led, "b")
    return S.save(fig, "regime_validation_short")


if __name__ == "__main__":
    print(draw())
    print(draw_short())
