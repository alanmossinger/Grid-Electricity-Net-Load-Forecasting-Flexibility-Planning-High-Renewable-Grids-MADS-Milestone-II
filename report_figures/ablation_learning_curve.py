"""Inputs matter more than settings.

(a) cross-validated MAE increase when each feature family is removed and the model refit, with the fold-noise floor
(ledger nb04.ablation and nb04.hyperparam_sensitivity.fold_noise_floor_mw, as notebook 04 section 4.10 draws
04_ablation.png); (b) training and validation MAE against training-set size, each size the most recent slice
(ledger nb04.learning_curve.rows, panel a of 04_learning_curve.png, notebook 04 section 4.11).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S


def draw() -> Path:
    led = S.ledger()["nb04"]
    fig, (ax_a, ax_b) = S.figure(2.7, ncols=2, gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.3})
    fig.subplots_adjust(left=0.115, right=0.985, top=0.95, bottom=0.17)

    # (a) ablation, families in descending order of the MAE increase (as E.ablation_by_family sorts them)
    fam = pd.DataFrame(led["ablation"]["families"]).T.sort_values("delta_mw", ascending=False)
    noise = led["hyperparam_sensitivity"]["fold_noise_floor_mw"]
    ys = np.arange(len(fam))[::-1]
    cols = ["#c1121f" if d > noise else "#9aa0a6" for d in fam["delta_mw"]]
    ax_a.axvspan(-noise, noise, color="#9aa0a6", alpha=0.2, lw=0)
    ax_a.barh(ys, fam["delta_mw"], color=cols, height=0.6)
    for y, (name, r) in zip(ys, fam.iterrows()):
        d = r["delta_mw"]
        txt = f"{d:+,.0f} MW ({r['pct_of_full']:+.1f}%)"
        if d > 400:      # the weather bar: label inside its end, so the axis does not have to stretch for it
            ax_a.annotate(txt, (d, y), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                          fontsize=6.5, color="white", weight="bold")
        else:
            # every other label sits right of its bar end, or right of zero for the one negative bar
            ax_a.annotate(txt, (max(d, 0), y), xytext=(4, 0), textcoords="offset points", ha="left",
                          va="center", fontsize=6.5)
    ax_a.axvline(0, color="0.3", lw=0.8)
    ax_a.annotate(f"fold-noise floor ±{noise:,.0f} MW", (noise, len(fam) - 0.55), xytext=(3, 0),
                  textcoords="offset points", fontsize=6.2, color="0.35", va="center")
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(fam.index)
    ax_a.set_xlim(-70, 760)
    ax_a.set_ylim(-0.6, len(fam) - 0.3)
    ax_a.grid(axis="y", visible=False)
    ax_a.set_axisbelow(True)
    ax_a.set_xlabel("CV MAE increase when the family\nis removed and the model refit (MW)")
    S.panel_label(ax_a, "a")

    # (b) learning curve
    lc = pd.DataFrame(led["learning_curve"]["rows"])
    lc["train_start"] = pd.to_datetime(lc["train_start"])
    ax_b.fill_between(lc["n_rows"], lc["train_mae"], lc["valid_mae"], color="#9aa0a6", alpha=0.2, lw=0)
    ax_b.plot(lc["n_rows"], lc["valid_mae"], "o-", lw=1.6, ms=3.5, color="#1d3557", label="validation (dev's last 15%)")
    ax_b.plot(lc["n_rows"], lc["train_mae"], "s--", lw=1.4, ms=3.2, color="#3fa34d", label="training (in-sample)")
    for i, r in lc.iterrows():      # alternately above and below the line, so neighbouring labels do not collide
        above = i % 2 == 0
        ax_b.annotate(f"{r['train_start']:%b %Y}", (r["n_rows"], r["valid_mae"]), xytext=(0, 5 if above else -6),
                      textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=6,
                      color="0.35")
    ax_b.set_xlim(0, 250_000)
    ax_b.set_ylim(500, 4300)
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax_b.set_xlabel("training rows (most recent slice;\nlabel = where the slice starts)")
    ax_b.set_ylabel("MAE (MW)")
    ax_b.legend(loc="upper right", fontsize=6.5, handlelength=2.0, borderaxespad=0.2)
    S.panel_label(ax_b, "b")
    return S.save(fig, "ablation_learning_curve")


if __name__ == "__main__":
    print(draw())
