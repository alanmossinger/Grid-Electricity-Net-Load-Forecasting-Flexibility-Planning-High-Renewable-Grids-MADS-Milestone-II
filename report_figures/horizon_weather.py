"""90% coverage per horizon, and the holdout re-scored on operational weather forecasts.

Replaces 06_coverage_by_horizon.png (notebook 06 §6.2b, panel a only) and 06_operational_weather.png
(notebook 06 §6.3b, both panels). Every value is read from the ledger, defined exactly as in notebook 06:
  (a) nb06.coverage_by_horizon[h].{coverage_raw, coverage_cqr_pooled, coverage_cqr}
      and .{worst_zone, worst_zone_coverage} (the zone whose Mondrian coverage is furthest from 90%)
  (b) nb06.operational_weather_by_horizon[h].{mae_reanalysis, mae_operational, penalty_mw}
  (c) nb06.operational_weather_by_horizon[h].penalty_pct for the horizons with an archived lead (24, 48, 72),
      nb06.operational_weather_modelled_floor_pct (the dashed "modelled floor at h=72")
"""
from __future__ import annotations

from pathlib import Path

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

H = [1, 24, 48, 72]
RAW_C, POOLED_C, MOND_C, NOM_C = "#c1121f", "#e9a03b", "#3fa34d", "#1d3557"   # notebook 06 colours


def draw() -> Path:
    s6 = S.ledger()["nb06"]
    cb = s6["coverage_by_horizon"]
    op = s6["operational_weather_by_horizon"]
    floor = s6["operational_weather_modelled_floor_pct"]

    fig = S.figure(3.0)[0]
    fig.clf()
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 0.85], wspace=0.42,
                          left=0.075, right=0.995, top=0.80, bottom=0.15)

    # (a) coverage per horizon ----------------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.axhspan(88, 92, color=MOND_C, alpha=0.10, zorder=0, lw=0)
    ax.axhline(90, ls="--", color=NOM_C, lw=1.0, label="nominal 90%")
    ax.plot(H, [100 * cb[str(h)]["coverage_raw"] for h in H], "o--", ms=3.5, color=RAW_C, label="raw quantile GBM")
    ax.plot(H, [100 * cb[str(h)]["coverage_cqr_pooled"] for h in H], "^-", ms=3.5, lw=1.1, color=POOLED_C,
            label="pooled CQR")
    ax.plot(H, [100 * cb[str(h)]["coverage_cqr"] for h in H], "s-", ms=4, lw=1.6, color=MOND_C,
            label="Mondrian CQR (shipped)")
    ax.plot(H, [100 * cb[str(h)]["worst_zone_coverage"] for h in H], "D", ms=3.5, mfc="white", mec="#555",
            mew=0.8, ls="none", label="worst zone, Mondrian")
    for h in H:
        c = 100 * cb[str(h)]["coverage_cqr"]
        ax.annotate(f"{c:.1f}", (h, c), xytext=(0, -7) if h == 1 else (0, 5), textcoords="offset points",
                    ha="center", va="center", fontsize=6.2, weight="bold", color="#1f5f28",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        wz, wc = cb[str(h)]["worst_zone"], 100 * cb[str(h)]["worst_zone_coverage"]
        # h=1: to the right; h=24 (below the Mondrian line): underneath; h=48, 72: above, on two lines
        txt, off, ha, va = {1: (f"{S.ZONE_NAME[wz]} {wc:.1f}", (5, 0), "left", "center"),
                            24: (f"{S.ZONE_NAME[wz]} {wc:.1f}", (0, -5), "center", "top")}.get(
            h, (f"{S.ZONE_NAME[wz]}|{wc:.1f}", (0, 5), "center", "bottom"))
        ax.annotate(txt.replace("|", chr(10)), (h, wc), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=5.8, color="#444", linespacing=1.0)
    ax.set_xticks(H); ax.set_xlim(-3, 76)
    # the shipped band's mean width at each horizon (the text quotes it beside this panel)
    ax.set_xticklabels([f"{h}\n{cb[str(h)]['width_cqr']:,.0f}" for h in H], linespacing=1.1)
    ax.set_ylim(74.5, 97.5)
    ax.set_xlabel("forecast horizon (hours);\nMondrian band mean width below (MW)", labelpad=1, linespacing=1.1)
    ax.set_ylabel("coverage of the 90% band (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2, fontsize=6.0, handlelength=1.6,
              columnspacing=0.8, borderaxespad=0)
    S.panel_label(ax, "a")
    ax.texts[-1].set_position((-0.02, 1.27))          # the label sits above the legend

    # (b) holdout MAE, reanalysis against the forecast as issued -------------------------------------
    ax = fig.add_subplot(gs[1])
    re = [op[str(h)]["mae_reanalysis"] for h in H]
    fc = [op[str(h)]["mae_operational"] for h in H]
    ax.plot(H, re, "o-", lw=1.6, ms=3.8, color=MOND_C, label="reanalysis weather\n(perfect foresight)")
    ax.plot(H, fc, "s--", lw=1.4, ms=3.4, color=RAW_C, label="forecast as issued\nat that lead")
    for h in H:
        pm = op[str(h)]["penalty_mw"]
        y = (op[str(h)]["mae_reanalysis"] + op[str(h)]["mae_operational"]) / 2
        ax.annotate(f"{pm:+,.0f} MW", (h, y), xytext=(4, -8) if h != 72 else (-3, 9),
                    textcoords="offset points", ha="left" if h != 72 else "right", va="center", fontsize=6.2)
    ax.set_xticks(H); ax.set_xlim(-3, 76)
    ax.set_ylim(0, 2900)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("forecast horizon (hours)", labelpad=1)
    ax.set_ylabel("holdout MAE (MW)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=1, fontsize=6.0, handlelength=1.8,
              borderaxespad=0)
    S.panel_label(ax, "b")
    ax.texts[-1].set_position((-0.02, 1.27))

    # (c) the measured penalty in % of the reanalysis score -------------------------------------------
    ax = fig.add_subplot(gs[2])
    hs = [h for h in H if op[str(h)]["lead_day"] is not None]
    pct = [op[str(h)]["penalty_pct"] for h in hs]
    ax.bar([str(h) for h in hs], pct, 0.6, color=RAW_C, label="measured", zorder=2)
    ax.axhline(floor, ls="--", color="#e8833a", lw=1.2, label=f"modelled floor\nat h=72 ({floor:.2f}%)", zorder=3)
    for j, v in enumerate(pct):
        ax.annotate(f"{v:+.2f}%", (j, v), xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2, weight="bold")
    ax.axhline(0, color="#333", lw=0.6, zorder=2)
    ax.set_ylim(-0.75, 1.85)
    ax.set_xlabel("forecast horizon (hours)", labelpad=1)
    ax.set_ylabel("MAE penalty (% of reanalysis)")
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=1, fontsize=6.0, handlelength=1.6,
              borderaxespad=0)
    S.panel_label(ax, "c")
    ax.texts[-1].set_position((-0.02, 1.27))
    return S.save(fig, "horizon_weather")


if __name__ == "__main__":
    print(draw())
