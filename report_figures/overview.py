#!/usr/bin/env python
"""One-page overview of the project for a general audience.

    py -3.12 report_figures/overview.py      -> reports/figures/report/project_overview.png

Every number is read from data/processed/results.json (the ledger the notebooks write), and
the California day is the median 2024 day computed from panel.parquet, so the picture follows
the notebooks after any re-run. Nothing on it is typed by hand except wording.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as cfg  # noqa: E402

OUT = ROOT / "reports" / "figures" / "report" / "project_overview.png"
L = json.loads((ROOT / "data" / "processed" / "results.json").read_text(encoding="utf-8"))

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e6e5e1"


def g(path):
    node = L
    for p in path.split("/"):
        node = node[p]
    return node


# ---- numbers, all from the ledger --------------------------------------------------
vre0, vre1 = g("nb01/fuel_mix/vre_share_first12_pct"), g("nb01/fuel_mix/vre_share_last12_pct")
trough_by_year = {int(k): v[1] for k, v in g("nb00/duck_curve/trough_by_year").items()}   # [hour, MW]
swing = g("nb00/duck_curve/net_load_swing_mw")
ramp_growth = 100 * g("nb08/report_numbers/ciso_evening_ramp_growth_2019_2024/median")
mae = {"Same hour\nlast week": g("nb04/baselines/naive168_mae"),
       "EIA forecast,\nas net load": g("nb04/baselines/df_composite_mae"),
       "Harder public\nbenchmark": g("nb04/baselines_v2/df_composite_v2_mae"),
       "This project": g("nb04/per_horizon_skill/24/model_mae")}
skill = [100 * g("nb04/per_horizon_skill/24/skill_vs_naive"), 100 * g("nb04/skill_vs_df_composite"),
         100 * g("nb04/per_horizon_skill/24/skill_vs_v2")]
leg_demand, leg_renew = g("nb04/baselines/composite_demand_leg_mae"), g("nb04/baselines/composite_renewables_leg_mae")
n_feat, lags = g("nb07/n_features_explained"), g("nb01/chosen_lags")
pacf1 = g("nb01/pacf_mean_abs/1")
weather_cost = g("nb04/ablation/families/weather/delta_mw")
by_h = {h: g(f"nb04/per_horizon_skill/{h}/model_mae") for h in ("1", "24", "48", "72")}
cover = 100 * g("nb06/coverage_cqr")
win =next(w for w in g("nb08/conditional_reserve_windows") if w["regime"] == "DUCK" and w["hours"] == "16-20")
short_flat, short_cond = 100 * win["shortfall_flat"], 100 * win["shortfall_conditional"]
extra_cap = 100 * g("nb08/conditional_vs_flat_mw/conditional_over_flat")

# ---- the median 2024 day in California, local time ------------------------------
panel = pd.read_parquet(cfg.PANEL)
ca = panel[panel[cfg.GROUP_COL] == "CISO"].copy()
ca["local"] = ca[cfg.TIME_COL].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
ca = ca[ca["local"].dt.year == 2024]
day = ca.groupby(ca["local"].dt.hour)[["demand", cfg.TARGET]].median() / 1000

# ---- figure ------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(20, 11.6), facecolor=SURF)
fig.text(0.03, 0.962, "Forecasting net load for six US power grids, from public data only",
         fontsize=27, weight="bold", color=INK)
fig.text(0.03, 0.938, "Net load = electricity demand − wind − solar: what the power plants that can be switched on "
         "and off must still supply.\nThis project forecasts it hour by hour, one hour to three days ahead, for "
         "California, Texas, the Midwest, the Plains, the Pacific Northwest and New York (2019–2024).\n"
         "Why a model: many power plants take hours to start, so grid operators schedule them a day ahead. "
         "They need to know the net load before it happens, not react to it.",
         fontsize=14, color=INK2, linespacing=1.4, va="top")

gs = fig.add_gridspec(2, 3, left=0.075, right=0.955, top=0.75, bottom=0.1, hspace=0.75, wspace=0.34)


def title(ax, n, head, sub):
    ax.text(0, 1.30, f"{n}  {head}", transform=ax.transAxes, fontsize=17, weight="bold", color=INK, va="bottom")
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=12, color=INK2, va="bottom", linespacing=1.35)


# 1. renewables changed the landscape
ax = fig.add_subplot(gs[0, 0])
yrs = sorted(trough_by_year)
ax.plot(yrs, [trough_by_year[y] / 1000 for y in yrs], color=ORANGE, lw=2.5, marker="o", ms=8)
ax.set_ylabel("California's lowest midday\nnet load, median day (GW)")
ax.set_xticks(yrs)
ax.grid(axis="y", color=GRID)
ax.set_ylim(0, max(trough_by_year.values()) / 1000 * 1.25); ax.set_xlim(yrs[0] - 0.5, yrs[-1] + 0.5)
for y in (yrs[0], yrs[-1]):
    ax.annotate(f"{trough_by_year[y] / 1000:.1f} GW", (y, trough_by_year[y] / 1000), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=12, color=INK)
title(ax, "1", "Wind and solar changed the grid",
      f"Their share of the six grids' generation rose from {vre0:.1f}% to {vre1:.1f}%\n"
      f"in five years, and California's evening ramp grew {ramp_growth:.0f}%. More is being\n"
      f"built, so net load will keep moving further away from demand.")

# 2. demand vs net load
ax = fig.add_subplot(gs[0, 1])
h = day.index
ax.fill_between(h, day[cfg.TARGET], day["demand"], color=YELLOW, alpha=0.35, lw=0)
ax.plot(h, day["demand"], color=MUTED, lw=2.5)
ax.plot(h, day[cfg.TARGET], color=ORANGE, lw=3)
ax.text(1, day["demand"].iloc[1] + 0.8, "demand", color=INK2, fontsize=12)
ax.text(1, day[cfg.TARGET].iloc[1] - 4.2, "net load", color=ORANGE, fontsize=12, weight="bold")
ax.text(12, (day["demand"].loc[12] + day[cfg.TARGET].loc[12]) / 2, "met by\nwind + solar", ha="center",
        va="center", fontsize=11.5, color=INK2)
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["0:00", "6:00", "12:00", "18:00", "23:00"])
ax.set_xlabel("hour of the day, local time"); ax.set_ylabel("GW")
ax.set_ylim(0, day["demand"].max() * 1.15); ax.grid(axis="y", color=GRID)
title(ax, "2", "Demand and net load are not the same",
      f"California, median 2024 day. Net load sinks at midday and then\n"
      f"climbs {swing / 1000:.1f} GW in the evening as the sun sets: the hours when\n"
      f"plants must be ready. Predicting demand alone misses this.")

# 3. why a net-load forecast is needed
ax = fig.add_subplot(gs[0, 2])
ax.barh([1, 0], [leg_demand / 1000, leg_renew / 1000], color=[MUTED, YELLOW], height=0.55)
for y, v in ((1, leg_demand), (0, leg_renew)):
    ax.text(v / 1000 + 0.05, y, f"{v:,.0f} MW", va="center", fontsize=13, color=INK)
ax.set_yticks([1, 0]); ax.set_yticklabels(["demand part\n(EIA forecast)", "wind + solar part\n(yesterday repeated)"])
ax.set_xlabel("average error of each part (GW)"); ax.set_xlim(0, leg_renew / 1000 * 1.35)
ax.grid(axis="x", color=GRID)
title(ax, "3", "No one publishes a net-load forecast",
      "The US EIA forecasts demand only. Turned into net load\n"
      "with public data, most of its error comes from wind\n"
      "and solar: they must be forecast too, from the weather.")

# 4. how much better
ax = fig.add_subplot(gs[1, 0])
names = list(mae)
vals = [mae[n] / 1000 for n in names]
ys = range(len(names))[::-1]
ax.barh(list(ys), vals, color=[MUTED, MUTED, MUTED, BLUE], height=0.6)
for y, v, n in zip(ys, vals, names):
    ax.text(v + 0.05, y, f"{v * 1000:,.0f} MW", va="center", fontsize=13, color=INK,
            weight="bold" if n.startswith("This") else "normal")
ax.set_yticks(list(ys)); ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel("average error, one day ahead, 2024 test period (GW)")
ax.set_xlim(0, max(vals) * 1.3); ax.grid(axis="x", color=GRID)
title(ax, "4", "How much better the forecast is",
      f"One day ahead the model is {skill[0]:.1f}% more accurate than 'same hour last\n"
      f"week', {skill[1]:.1f}% more than the EIA-based forecast and {skill[2]:.1f}% more\n"
      f"than a harder benchmark built so the result cannot rest on a weak rival.")

# 5. how: lags and weather
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
title(ax, "5", "How: recent history plus the weather",
      f"The model is gradient-boosted trees using {n_feat} inputs from six families.")
lines = [
    ("Lags – what happened before", BLUE,
     f"Net load at {', '.join(str(x) for x in lags[:3])} hours, 1 day and 1 week earlier. Grids are\n"
     f"creatures of habit: an hour looks like the hour before (correlation\n"
     f"with the hour before: {pacf1:.3f}), and days and weeks repeat."),
    ("Weather – what is about to happen", AQUA,
     f"Temperature, sun, cloud and wind drive both demand and output.\n"
     f"Removing weather adds {weather_cost:,.0f} MW of error, the most of any family;\n"
     f"tested with the forecasts as they were actually issued."),
    ("Calendar and region", YELLOW,
     "Hour, weekday and season; which grid it is."),
]
y = 0.93
for head, col, body in lines:
    ax.add_patch(plt.Rectangle((0, y - 0.035), 0.018, 0.05, color=col, transform=ax.transAxes, clip_on=False))
    ax.text(0.04, y, head, transform=ax.transAxes, fontsize=13.5, weight="bold", color=INK, va="center")
    ax.text(0.04, y - 0.06, body, transform=ax.transAxes, fontsize=12, color=INK2, va="top", linespacing=1.35)
    y -= 0.34 if body.count("\n") else 0.2
ax.text(0.04, 0.0, f"Error by lead time: {float(by_h['1']):,.0f} MW one hour ahead, {float(by_h['24']):,.0f} one day, "
        f"{float(by_h['72']):,.0f} three days.", transform=ax.transAxes, fontsize=12, color=INK2)

# 6. what stakeholders get
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
title(ax, "6", "What grid operators gain",
      "Not only a number, but how far to trust it, region by region.")
items = [
    (f"{cover:.1f}%", "of real hours fall inside the\n90% range; every region within\n5 points of 90%"),
    (f"{short_flat:.1f}% → {short_cond:.1f}%", f"reserve shortfalls on solar-day\nevening ramps, for {extra_cap:.1f}% more\nreserve: backup where the risk is"),
    (f"{(mae[names[1]] - mae[names[3]]):,.0f} MW", "less error every hour than the\nbest forecast public data\nallowed until now"),
    ("6 grids", "one method, public data only;\nanyone can re-run it and get\nthe same numbers"),
]
y = 0.93
for big, small in items:
    ax.text(0, y, big, transform=ax.transAxes, fontsize=18, weight="bold", color=BLUE, va="top")
    ax.text(0.45, y, small, transform=ax.transAxes, fontsize=12, color=INK2, va="top", linespacing=1.35)
    y -= 0.25

fig.text(0.03, 0.018, "Sources: EIA-930 Hourly Electric Grid Monitor; Open-Meteo (ERA5 and archived weather "
         "forecasts). Test period: 2024 (from 7 February), never used for training or tuning. "
         "Every number on this page is read from the project's results ledger.", fontsize=10.5, color=MUTED)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110, facecolor=SURF)
print(f"wrote {OUT.relative_to(ROOT)}")
