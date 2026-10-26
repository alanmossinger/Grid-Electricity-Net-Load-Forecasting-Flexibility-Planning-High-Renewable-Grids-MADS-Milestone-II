"""Two causal designs, no claim.

(a) Difference-in-differences event study (03_did.png panel a): quarterly means of the deseasonalised evening ramp,
    treated (CISO, ERCO, MISO) and control, relative to 2022-01-01, with the pre-period linear trends dotted.
    Recomputed from data/processed/panel.parquet with notebook 03's own definitions (cell 3: zone-local days,
    evening = local hours 16-21, complete days only; cell 12: src.causal.deseasonalise fitted on the pre-period,
    src.causal.event_study_frame with quarterly bins clipped to [-12, 9]). These are group means, nothing is
    fitted; the pre-trend slopes and the 2x2 cells are checked against results.json nb03.
    The pooled 2x2 estimate and its interval are printed from nb03.did_coef / nb03.did_ci.
(b) Covariate balance before and after propensity matching (03_matching.png panel a, the Love plot).
    The standardised mean differences were not saved to results.json; they are taken from the balance table that
    notebook 03 printed in the same run (cell 15 output, 3 decimals), and the worst |SMD| after matching is checked
    against nb03.matched_worst_smd_after. Pair counts from nb03.matched_n_pairs / matched_share; the cross-zone
    share (98.5%) is from the same cell's printed output.
Not redrawn: 03_matching.png panel b (propensity-score overlap) needs the fitted propensity scores, which were
never saved; drawing it would mean refitting the propensity model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .style import figure, ledger, panel_label, save
except ImportError:
    from style import figure, ledger, panel_label, save
from src import causal as C  # noqa: E402  (style.py puts the repository root on sys.path)
from src import config as cfg  # noqa: E402

TREAT_C, CTRL_C = "#c1121f", "#1d3557"

# notebook 03, cell 15 output (same run as results.json nb03): standardised mean differences
SMD = pd.DataFrame({"smd_before": {"demand": 0.559, "humidity": -0.588, "temp": 0.989, "wind_speed": -0.462},
                    "smd_after": {"demand": 0.151, "humidity": -0.111, "temp": 0.135, "wind_speed": -0.021}})
CROSS_ZONE_PCT = 98.5        # same output: "98.5% of the matched pairs compare days from different zones"
N_TREATED = 3941             # same output: "matched 2,108 of 3,941 treated (53.5%)"


def _daily() -> pd.DataFrame:
    df = pd.read_parquet(cfg.PANEL)
    _local = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for _zone, _tz in cfg.ZONE_TZ.items():
        _m = (df[cfg.GROUP_COL] == _zone).to_numpy()
        _local[_m] = (df.loc[_m, cfg.TIME_COL].dt.tz_localize("UTC").dt.tz_convert(_tz).dt.tz_localize(None))
    df["date"] = _local.dt.date
    df["hour"] = _local.dt.hour
    evening = df[df["hour"].between(16, 21)]
    daily = (df.groupby([cfg.GROUP_COL, "date"])
               .agg(solar_share=("solar", "sum"), demand=("demand", "sum"),
                    temp=("temp", "mean"), humidity=("humidity", "mean"),
                    wind_speed=("wind_speed", "mean"), cloud_cover=("cloud_cover", "mean"),
                    n_hours=("solar", "size"))
               .assign(solar_share=lambda d: d["solar_share"] / d["demand"]))
    daily = daily[daily["n_hours"] >= 23].drop(columns="n_hours")
    daily["evening_ramp"] = (evening.groupby([cfg.GROUP_COL, "date"])[cfg.TARGET]
                             .apply(lambda s: s.max() - s.min()))
    daily = daily.reset_index().dropna()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def draw():
    L = ledger()["nb03"]
    daily = _daily()
    treated = L["did_treated_zones"]
    t0 = pd.Timestamp(L["did_t0"])
    f = daily.assign(treated=daily[cfg.GROUP_COL].isin(treated).astype(int),
                     post=(daily["date"] >= t0).astype(int))
    f["ramp_adj"] = C.deseasonalise(f.assign(month=f["date"].dt.month), outcome="evening_ramp",
                                    unit=cfg.GROUP_COL, season="month", fit_mask=(f["post"] == 0))
    did = C.did_2x2(f, outcome="ramp_adj")
    assert abs(did["did"] - L["did_hand_computed"]) < 1e-6, did["did"]
    f["rel"] = C.event_study_frame(f, time="date", t0=t0, freq="Q")
    ev = f.groupby(["rel", "treated"])["ramp_adj"].mean().unstack()
    pre = ev[ev.index < 0]
    fit_t, fit_c = np.polyfit(pre.index, pre[1], 1), np.polyfit(pre.index, pre[0], 1)
    assert abs(fit_t[0] - L["parallel_pre_slope_treated"]) < 1e-6
    assert abs(fit_c[0] - L["parallel_pre_slope_control"]) < 1e-6
    assert abs(SMD["smd_after"].abs().max() - round(L["matched_worst_smd_after"], 3)) < 1e-9

    fig, _ax = figure(3.0)
    _ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3, left=0.09, right=0.99, top=0.95, bottom=0.15)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) event study
    ax = ax_a
    names = ", ".join({"CISO": "California", "ERCO": "Texas", "MISO": "MISO"}.get(z, z) for z in treated)
    ax.plot(ev.index, ev[1], marker="o", ms=3.5, lw=1.5, color=TREAT_C, label=f"treated:\n{names}")
    ax.plot(ev.index, ev[0], marker="s", ms=3.2, lw=1.5, color=CTRL_C, label="control")
    ax.plot(pre.index, np.poly1d(fit_t)(pre.index), color=TREAT_C, ls=":", lw=1.0)
    ax.plot(pre.index, np.poly1d(fit_c)(pre.index), color=CTRL_C, ls=":", lw=1.0)
    ax.axvline(-0.5, color="0.4", ls="--", lw=1.0)
    ax.text(-0.2, 4300, f"t0 = {t0:%Y-%m-%d}", fontsize=7, color="0.3", ha="left", va="top")
    lo, hi = L["did_ci"]
    ax.text(0.02, 0.97, f"pooled 2×2 DiD {L['did_coef']:+,.1f} MW\n95% CI [{lo:,.1f}, {hi:+,.1f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("quarters relative to t0 (−1 = last pre-quarter; +9 pools 9–11)")
    ax.set_ylabel("deseasonalised evening ramp (MW)")
    ax.set_xticks([-12, -8, -4, 0, 4, 8])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=7)
    panel_label(ax, "a")

    # (b) Love plot
    ax = ax_b
    order = SMD["smd_before"].abs().sort_values().index
    y = np.arange(len(order))
    for i, c in enumerate(order):
        ax.plot([abs(SMD.loc[c, "smd_after"]), abs(SMD.loc[c, "smd_before"])], [i, i], color="0.8", lw=1, zorder=1)
    ax.scatter(SMD.loc[order, "smd_before"].abs(), y, color=TREAT_C, s=22, label="before", zorder=3)
    ax.scatter(SMD.loc[order, "smd_after"].abs(), y, color="#3fa34d", marker="s", s=20, label="after", zorder=3)
    ax.axvline(0.1, color="0.4", ls="--", lw=1.0, label="0.1 rule")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4 + 2.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("|standardised mean difference|")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=7, ncol=3, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1,
              handlelength=1.4)
    ax.text(0.98, 0.86, f"{L['matched_n_pairs']:,} of {N_TREATED:,} treated days\nmatched "
            f"({L['matched_share'] * 100:.1f}%)\n{CROSS_ZONE_PCT:.1f}% of pairs cross zones",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    panel_label(ax, "b")
    return save(fig, "causal_designs")


if __name__ == "__main__":
    print(draw())
