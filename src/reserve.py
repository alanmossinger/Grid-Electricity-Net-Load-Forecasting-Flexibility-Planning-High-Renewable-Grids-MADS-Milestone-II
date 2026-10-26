"""Reserve sizing and its valuation: the library behind notebooks 05 and 08.

WHY THIS MODULE EXISTS. The per-horizon comparison and the reserve valuation were
first produced by an uncommitted ad-hoc session. They drifted from the notebooks --
the seasonal-naive MAE at h=48 disagreed between two files in this repository -- and
they were quoted in the report with no code behind them. Everything they did now lives
here, so that one implementation serves the notebook that narrates it, the scripts that
regenerate the report tables, and the tests that pin it.

THE FOUR CORRECTIONS THIS MODULE ENCODES, each of which lowers the valuation:

1. UPWARD reserve is sized from the UPPER TAIL OF THE SIGNED ERROR, not from a quantile
   of |error|. Capacity held to cover an under-forecast is not used when the forecast was
   too high. This is not cosmetic: the EIA composite over-forecasts net load badly -- in
   MISO its p95 down-error is nearly twice its p95 up-error -- so a quantile of |error|
   credited the model for curing a bias that upward reserve never pays for. Both
   quantities are computed by `tails`, so the size of the correction stays auditable.

2. Each balancing authority procures its own reserve, so the six-zone quantity is the
   SUM OF PER-ZONE QUANTILES. The pooled quantile is a different number -- at h=24 they
   differ by a factor of four -- and mixing them was the defect a reviewer found in the
   first version of the table.

3. A day-ahead benchmark is issued ONCE, mid-morning on D-1, for all 24 hours of D, so
   its lead times run about 14-38 h. A model re-run every hour at h=24 knows more than
   that for late-day hours. `build_scored_frames` therefore also builds a FIXED-ORIGIN
   frame, giving each target hour the NEXT LARGER available horizon (h=24 out to a 24-h
   lead, h=48 beyond). Every hour is then forecast from an origin at least as early as
   the real one, which makes it a LOWER BOUND on fixed-origin skill.

4. At h=1 the reserve saving is measured against t-1 PERSISTENCE, not against a
   day-ahead composite -- see `bench_col`. Pricing a one-hour-ahead model against a
   benchmark issued the previous morning produced the largest dollar figure on the page
   out of an information-set mismatch.

AND THE PRICE IS A PROXY. Spinning reserve is a contingency product, sized from the
largest single credible outage, not from forecast error. Forecast error is covered by
regulation, CAISO's flexible ramping product and MISO's ramp capability product, which
frequently clear near zero, and day-ahead error is largely resolved by intraday
recommitment rather than by capacity held all year. So `price_scenarios` returns three
scenarios rather than one, and the honest output of this module is a surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import models as BL
from src import config as cfg
from src import features as P

TIME, GROUP, TARGET = cfg.TIME_COL, cfg.GROUP_COL, cfg.TARGET

ZONES = ["BPAT", "CISO", "ERCO", "MISO", "NYIS", "SWPP"]
HORIZONS = (1, 24, 48, 72)
COVERAGES = (0.90, 0.95, 0.975, 0.99)
HOURS_PER_YEAR = 8760
LEG_VALID_DAYS = 90        # the dev tail notebook 04 selects the renewable legs on
ISSUE_HOUR_LOCAL = 10      # a day-ahead forecast is submitted mid-morning on D-1
TARGET_SHORTFALL = 0.05    # the 5% a p95 requirement leaves uncovered by definition
FIXED_ORIGIN = "fixed_origin"

# The two row rules, named because the difference between them matters. `LEDGER_COLS`
# is notebook 04's own -- model, v2, naive, target -- and is the only one that can be
# checked against the ledger. `ALL_COLS` additionally requires v1, because any table
# printing a v1 column must score every forecast in it on identical rows. They differ
# by eighteen rows at h=48.
LEDGER_COLS = ("y_pred", "v2", "naive168", TARGET)
ALL_COLS = ("y_pred", "v1", "v2", "naive168", TARGET)


# ===========================================================================
# 1. local civil time, for presentation only
# ===========================================================================
def local_hour(frame: pd.DataFrame) -> pd.Series:
    """Local civil hour of every row, as a float.

    The panel is stored in UTC and the model stays in UTC. This is for the
    fixed-origin lead-time arithmetic and for the by-hour reserve tables, both of
    which are facts about the operator's clock. Never build a feature from it.
    """
    t = pd.to_datetime(frame[TIME])
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for zone, tz in cfg.ZONE_TZ.items():
        m = (frame[GROUP] == zone).to_numpy()
        if m.any():
            out[m] = (t[m].dt.tz_localize("UTC").dt.tz_convert(tz)
                      .dt.tz_localize(None).dt.hour.to_numpy(dtype=float))
    if out.isna().any():
        raise ValueError(
            f"{int(out.isna().sum())} rows have a zone missing from cfg.ZONE_TZ: "
            f"{sorted(set(frame.loc[out.isna(), GROUP]))}")
    return out


def lead_hours(frame: pd.DataFrame, issue_hour: int = ISSUE_HOUR_LOCAL) -> pd.Series:
    """Hours from a D-1 issuance at `issue_hour` local to each target hour of D.

    Runs from 24 - issue_hour (the first hour of D) to 47 - issue_hour (the last), so
    14 to 37 hours for a 10:00 issuance. This is the quantity the benchmark actually
    faces and the rolling h=24 model does not.
    """
    return (24 - issue_hour) + local_hour(frame)


# ===========================================================================
# 2. the scored frames
# ===========================================================================
def build_scored_frames(panel: pd.DataFrame, cut, predictions: pd.DataFrame,
                        issue_hour: int = ISSUE_HOUR_LOCAL) -> tuple[dict, dict]:
    """One scored frame per horizon plus the fixed-origin emulation.

    Returns `(frames, guard_frames)`: the second uses notebook 04's row rule and is
    what `check_against_ledger` compares, the first adds v1 and is what the tables use.

    The benchmark is REBUILT here from `src.baselines` rather than read from a file, so
    the leg selection, the fit window and the horizon-matched persistence lag are the
    same objects notebook 04 used. Reading a saved benchmark would make the guard below
    check the report against itself.
    """
    cut = pd.Timestamp(cut)
    dev = panel[panel[TIME] <= cut]
    hold = panel[panel[TIME] > cut].sort_values([GROUP, TIME]).copy()

    chosen = BL.select_leg_methods(
        dev, BL.fit_renewable_legs(dev, valid_days=LEG_VALID_DAYS),
        valid_days=LEG_VALID_DAYS)
    coefs = BL.fit_renewable_legs(dev)      # legs refit on all of dev once the method is fixed

    hold["naive168"] = BL.lag_by_time(hold, TARGET, 168, GROUP, TIME)
    hold["persist1"] = BL.lag_by_time(hold, TARGET, 1, GROUP, TIME)
    hold["local_hour"] = local_hour(hold)
    hold["wind_p24"] = BL.lag_by_time(hold, "wind", 24, GROUP, TIME)
    hold["solar_p24"] = BL.lag_by_time(hold, "solar", 24, GROUP, TIME)
    hold["v1"] = hold["demand_forecast"] - hold["wind_p24"] - hold["solar_p24"]

    keyed = hold.set_index([GROUP, TIME]).index
    pred_by_h = {}
    for h in HORIZONS:
        src = predictions[predictions["horizon"] == h].set_index([GROUP, TIME])["y_pred"]
        pred_by_h[h] = pd.Series(keyed.map(src).to_numpy(), index=hold.index)

    cols = [GROUP, TIME, TARGET, "naive168", "persist1", "local_hour", "v1"]
    frames, guard = {}, {}

    def add(key, y_pred, lag_hours, lead=None):
        f = hold[cols].copy()
        f["y_pred"] = y_pred
        f["v2"] = BL.composite_v2_selected(
            hold, coefs, chosen, lag_hours=lag_hours)["df_composite_v2"].to_numpy()
        if lead is not None:
            f["lead_hours"] = lead
        guard[key] = f[f[list(LEDGER_COLS)].notna().all(axis=1)].copy()
        frames[key] = f[f[list(ALL_COLS)].notna().all(axis=1)].copy()

    for h in HORIZONS:
        # The renewable legs reach back a whole number of days >= h: at h=48 an operator
        # cannot use yesterday's output, so the benchmark has to degrade with the horizon
        # too. Defaulting the lag to 24 everywhere would hand the baseline a day of
        # information it does not have.
        add(str(h), pred_by_h[h], int(np.ceil(h / 24) * 24))

    lead = lead_hours(hold, issue_hour)
    add(FIXED_ORIGIN, np.where(lead > 24, pred_by_h[48], pred_by_h[24]), 24, lead=lead)
    return frames, guard


def frame_keys() -> list[str]:
    """The frame keys in report order."""
    return [str(h) for h in HORIZONS] + [FIXED_ORIGIN]


def mae(frame: pd.DataFrame, col: str) -> float:
    return float((frame[col] - frame[TARGET]).abs().mean())


def check_against_ledger(guard_frames: dict, per_horizon_skill: dict,
                         tol_mw: float = 0.01) -> None:
    """Reproduce notebook 04's per-horizon table, or raise and write nothing.

    This is the guard the first version of the report lacked. It fired once during
    development, on an eighteen-row difference in the row filter, which is exactly the
    class of drift it exists to catch.
    """
    for h in HORIZONS:
        g, want = guard_frames[str(h)], per_horizon_skill[str(h)]
        for col, key in (("y_pred", "model_mae"), ("v2", "composite_v2_mae"),
                         ("naive168", "naive_mae")):
            got, exp = mae(g, col), float(want[key])
            if abs(got - exp) > tol_mw:
                raise AssertionError(
                    f"h={h} {key}: rebuilt {got:,.4f} MW but the ledger says "
                    f"{exp:,.4f} MW. The notebooks and this module disagree; fix that "
                    "before trusting any number downstream.")
        if len(g) != int(want["n"]):
            raise AssertionError(
                f"h={h}: {len(g):,} scored rows here against {int(want['n']):,} in the "
                "ledger. The row filters have diverged.")


# ===========================================================================
# 3. prices
# ===========================================================================
def price_scenarios(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Three capacity-price scenarios, all observed, none of them the right product.

    `spin_mean` is what the first version of the table used. `spin_median` exists
    because CISO's spin mean is four times its median -- a handful of scarcity hours
    carry it -- and a valuation whose sign changes between the mean and the median of
    the same year of the same product has to say so. `ramp_proxy` uses MISO's
    supplemental-reserve price as a stand-in for the flexibility products that actually
    cover forecast error, which clear near zero.

    Only CISO and MISO have an accessible price. ERCOT's API sits behind a credentialed
    login and Bonneville is a federal self-supplier with no market at all, so the other
    four zones take the mean of the two and every output labels them as proxies.
    """
    spin = prices[((prices["zone"] == "CISO") & (prices["product"] == "SR"))
                  | ((prices["zone"] == "MISO") & (prices["product"] == "SPIN"))]
    supp = float(prices[(prices["zone"] == "MISO")
                        & (prices["product"] == "SUPP")]["price_usd_per_mw_h"].mean())
    observed = {
        "spin_mean": spin.groupby("zone")["price_usd_per_mw_h"].mean().to_dict(),
        "spin_median": spin.groupby("zone")["price_usd_per_mw_h"].median().to_dict(),
        "ramp_proxy": {"CISO": supp, "MISO": supp},
    }
    out = {}
    for name, base in observed.items():
        proxy = float(np.mean(list(base.values())))
        out[name] = {z: float(base.get(z, proxy)) for z in ZONES}
        out[name]["__proxy__"] = proxy
    return out


def priced_zones() -> tuple[str, ...]:
    """The zones whose reserve price is observed rather than a proxy."""
    return ("CISO", "MISO")


# ===========================================================================
# 4. the tails, signed and absolute
# ===========================================================================
def bench_col(key: str) -> str:
    """Which benchmark a horizon's reserve saving is measured against.

    At h=1 it is t-1 persistence, NOT the day-ahead composite: a one-hour-ahead model
    reads the last hour of history and a forecast issued the previous morning does not,
    so comparing them measures the information gap rather than the model.
    """
    return "persist1" if key == "1" else "v2"


def tails(frame: pd.DataFrame, coverage: float = 0.95,
          bench: str = "v2") -> pd.DataFrame:
    """Per-zone reserve requirements at `coverage`, signed both ways and absolute.

    `up` is the quantile of (actual - forecast): the shortfall upward reserve covers.
    `down` is its mirror, which a different product pays for. `abs` is the old, wrong
    quantity, kept so the size of the correction is auditable rather than asserted.
    """
    def one(d: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "up_model": float(np.quantile(d[TARGET] - d["y_pred"], coverage)),
            "up_bench": float(np.quantile(d[TARGET] - d[bench], coverage)),
            "down_model": float(np.quantile(d["y_pred"] - d[TARGET], coverage)),
            "down_bench": float(np.quantile(d[bench] - d[TARGET], coverage)),
            "abs_model": float(np.quantile((d["y_pred"] - d[TARGET]).abs(), coverage)),
            "abs_bench": float(np.quantile((d[bench] - d[TARGET]).abs(), coverage)),
        })
    # rows where the benchmark has no value (persistence on the hour after a dropped hour) are
    # left out of both quantiles, so model and benchmark are sized on the same hours
    frame = frame.dropna(subset=[TARGET, "y_pred", bench])
    per = frame.groupby(GROUP)[[TARGET, "y_pred", bench]].apply(one)
    per["up_saved"] = per["up_bench"] - per["up_model"]
    per["down_saved"] = per["down_bench"] - per["down_model"]
    per["abs_saved"] = per["abs_bench"] - per["abs_model"]
    return per


def value_per_year(per: pd.DataFrame, prices: dict[str, float],
                   col: str = "up_saved") -> float:
    """Annual cost of the headroom in `col`, at one price scenario."""
    return float(sum(per.loc[z, col] * prices[z] * HOURS_PER_YEAR for z in per.index))


def horizon_table(frames: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per horizon: every MAE, every skill, and the reserve quantities."""
    spin = price_scenarios(prices)["spin_mean"]
    rows = []
    for key in frame_keys():
        g = frames[key]
        bench = bench_col(key)
        per = tails(g, 0.95, bench)
        rows.append({
            "horizon": key, "rows": len(g), "reserve_benchmark": bench,
            "mae_model": mae(g, "y_pred"), "mae_v1": mae(g, "v1"),
            "mae_v2": mae(g, "v2"), "mae_naive168": mae(g, "naive168"),
            "mae_persist1": mae(g, "persist1"),
            "skill_vs_v1": 1 - mae(g, "y_pred") / mae(g, "v1"),
            "skill_vs_v2": 1 - mae(g, "y_pred") / mae(g, "v2"),
            "skill_vs_naive": 1 - mae(g, "y_pred") / mae(g, "naive168"),
            "skill_vs_persist1": 1 - mae(g, "y_pred") / mae(g, "persist1"),
            "pooled_p95_up_model": float(np.quantile(g[TARGET] - g["y_pred"], 0.95)),
            "pooled_p95_up_bench": float(np.quantile(g[TARGET] - g[bench], 0.95)),
            "pooled_p95_abs_model": float(
                np.quantile((g["y_pred"] - g[TARGET]).abs(), 0.95)),
            "sum_zone_p95_up_model": float(per["up_model"].sum()),
            "sum_zone_p95_up_bench": float(per["up_bench"].sum()),
            "up_saved_MW": float(per["up_saved"].sum()),
            "down_saved_MW": float(per["down_saved"].sum()),
            "abs_saved_MW": float(per["abs_saved"].sum()),
            "usd_per_year_spin_mean": value_per_year(per, spin),
        })
    return pd.DataFrame(rows)


def zone_table(frames: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon, zone), with the price and whether it is observed."""
    spin = price_scenarios(prices)["spin_mean"]
    rows = []
    for key in frame_keys():
        bench = bench_col(key)
        per = tails(frames[key], 0.95, bench)
        for z in per.index:
            rows.append({
                "horizon": key, "zone": z, "reserve_benchmark": bench,
                **{c: float(per.loc[z, c]) for c in per.columns},
                "usd_per_mw_h": spin[z], "priced": z in priced_zones(),
                "usd_per_year": float(
                    per.loc[z, "up_saved"] * spin[z] * HOURS_PER_YEAR),
            })
    return pd.DataFrame(rows)


def sensitivity_surface(frames: dict, prices: pd.DataFrame,
                        coverages=COVERAGES) -> pd.DataFrame:
    """The price x coverage surface goal G5 requires beside any dollar figure.

    A point estimate from this table would be a claim the data does not support: at
    p95 the day-ahead figure changes sign between the mean and the median of one
    year of one product.
    """
    scen = price_scenarios(prices)
    rows = []
    for key in frame_keys():
        bench = bench_col(key)
        for cov in coverages:
            per = tails(frames[key], cov, bench)
            for name, pm in scen.items():
                rows.append({
                    "horizon": key, "coverage": cov, "price_scenario": name,
                    "up_saved_MW": float(per["up_saved"].sum()),
                    "usd_per_year": value_per_year(per, pm),
                })
    return pd.DataFrame(rows)


# ===========================================================================
# 5. the conditional band against a flat quantile, at EQUAL RISK
# ===========================================================================
def hourly_with_band(predictions: pd.DataFrame, quantiles: pd.DataFrame,
                     cluster_labels: pd.DataFrame, regime_names: dict,
                     horizon: int = 24) -> pd.DataFrame:
    """The day-ahead holdout with its point forecast, calibrated band and regime.

    The regime join keys on LOCAL calendar date via `P.local_calendar_date`, the one
    helper that exists because three cells once merged local-date labels onto UTC dates
    and the join succeeded silently every time.
    """
    q = quantiles.merge(
        predictions[predictions["horizon"] == horizon][[TIME, GROUP, "y_pred"]],
        on=[TIME, GROUP], how="left")
    if q["y_pred"].isna().any():
        raise AssertionError(
            f"{int(q['y_pred'].isna().sum())} quantile rows have no h={horizon} "
            "point forecast: the quantile file and the prediction file do not align")
    q["err_up"] = q["actual"] - q["y_pred"]          # positive = short = needs up-reserve
    q["band_up"] = (q["cqr_hi"] - q["y_pred"]).clip(lower=0)
    q["local_hour"] = local_hour(q)
    q["local_date"] = P.local_calendar_date(q)

    lab = normalise_labels(cluster_labels, regime_names)
    # A LEFT join, not inner: a few hundred holdout hours fall on days the clustering
    # never labelled, and they belong in the reserve tables as an explicit `unlabelled`
    # row rather than being quietly dropped out of the denominator. The fill matters:
    # `groupby` drops NaN keys silently, so without it those rows would vanish from every
    # by-regime table while still counting in the zone totals -- two tables that no longer
    # add up, for a reason nothing on the page would explain. Notebook 06's
    # `coverage_by_regime` uses the same label for the same rows.
    out = q.merge(
        lab.rename(columns={"date": "local_date"})[[GROUP, "local_date", "regime"]],
        on=[GROUP, "local_date"], how="left")
    out["regime"] = out["regime"].fillna("unlabelled")
    return out


def scale_to_shortfall(err_up: pd.Series, band: pd.Series,
                       target: float = TARGET_SHORTFALL,
                       lo: float = 0.05, hi: float = 5.0,
                       iters: int = 80) -> float:
    """The multiplier making this band's realised shortfall rate exactly `target`.

    WHY SCALE AT ALL. Comparing a flat quantile against a conditional band on
    megawatts alone is rigged: a wider band always covers more hours. Equalising the
    realised risk first is the only comparison that answers the question.

    Bisection rather than a closed form because the band varies row by row, so there is
    no single quantile to invert -- which is the entire point of a conditional
    requirement.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if float((err_up > mid * band).mean()) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def equal_risk_comparison(hourly: pd.DataFrame,
                          target: float = TARGET_SHORTFALL) -> pd.DataFrame:
    """Flat per-zone quantile against the conditional band at the same realised risk."""
    rows = []
    for zone, d in hourly.groupby(GROUP):
        flat = float(np.quantile(d["err_up"], 1 - target))
        s = scale_to_shortfall(d["err_up"], d["band_up"], target)
        dyn = s * d["band_up"]
        rows.append({
            "zone": zone, "flat_p95_MW": flat, "conditional_mean_MW": float(dyn.mean()),
            "conditional_p95_MW": float(np.quantile(dyn, 0.95)),
            "conditional_min_MW": float(dyn.min()),
            "conditional_max_MW": float(dyn.max()), "scale": s,
            "shortfall_flat": float((d["err_up"] > flat).mean()),
            "shortfall_conditional": float((d["err_up"] > dyn).mean()),
            "saving_MW": flat - float(dyn.mean()),
        })
    return pd.DataFrame(rows).set_index("zone")


def risk_concentration(hourly: pd.DataFrame, comparison: pd.DataFrame):
    """Where a flat requirement is short once it is right on average.

    Returns `(by_regime, by_regime_hour)`. A flat p95 misses 5% of hours by
    construction; these tables say which 5%, and the answer is the evening ramp.
    """
    q = hourly.copy()
    q["flat_req"] = q[GROUP].map(comparison["flat_p95_MW"])
    q["cond_req"] = q["band_up"] * q[GROUP].map(comparison["scale"])
    q["short_flat"] = q["err_up"] > q["flat_req"]
    q["short_cond"] = q["err_up"] > q["cond_req"]
    agg = dict(rows=("short_flat", "size"), shortfall_flat=("short_flat", "mean"),
               shortfall_conditional=("short_cond", "mean"),
               mean_req_flat=("flat_req", "mean"),
               mean_req_conditional=("cond_req", "mean"))
    return (q.groupby("regime").agg(**agg),
            q.groupby(["regime", "local_hour"]).agg(**agg).reset_index())


def window_shortfall(by_regime_hour: pd.DataFrame, regime: str,
                     lo: int, hi: int) -> dict:
    """Row-weighted shortfall rates for one regime over a local-hour window."""
    d = by_regime_hour[(by_regime_hour["regime"] == regime)
                       & by_regime_hour["local_hour"].between(lo, hi)]
    if d.empty:
        raise ValueError(f"no rows for regime {regime!r} in hours {lo}-{hi}")
    return {
        "regime": regime, "hours": f"{lo:02d}-{hi:02d}", "rows": int(d["rows"].sum()),
        "shortfall_flat": float(np.average(d["shortfall_flat"], weights=d["rows"])),
        "shortfall_conditional": float(
            np.average(d["shortfall_conditional"], weights=d["rows"])),
    }


# ===========================================================================
# 6. are the regimes zones in disguise?
# ===========================================================================
def normalise_labels(cluster_labels, regime_names: dict) -> pd.DataFrame:
    """A `zone / date / cluster / regime` frame with `date` as tz-naive midnight.

    Accepts either the saved `cluster_labels.parquet` or notebook 05's in-memory
    `labels` Series indexed by (zone, date), so the notebook and the scripts cannot
    drift apart on the shape of this object.

    THE DATE NORMALISATION IS THE POINT. Every join in this module keys on a local
    calendar date, and a label frame whose dates are `datetime.date` merges to nothing
    against one whose dates are `Timestamp` -- silently, returning an empty frame or a
    column of NaN rather than an error. That failure mode has already cost this project
    one wrong published number.
    """
    lab = (cluster_labels.reset_index() if isinstance(cluster_labels, pd.Series)
           else cluster_labels.copy())
    missing = {GROUP, "date", "cluster"} - set(lab.columns)
    if missing:
        raise ValueError(f"label frame is missing {sorted(missing)}; got {list(lab.columns)}")
    lab["date"] = pd.to_datetime(lab["date"]).dt.normalize()
    lab["regime"] = lab["cluster"].astype(str).map(regime_names)
    if lab["regime"].isna().any():
        unknown = sorted(set(lab.loc[lab["regime"].isna(), "cluster"].astype(str)))
        raise ValueError(f"clusters {unknown} have no name in regime_names")
    return lab


def regime_by_zone(cluster_labels, regime_names: dict) -> pd.DataFrame:
    """Zone-days by zone and regime.

    The question this answers is whether a pooled weather contrast between regimes is
    really a contrast between grids. If one zone supplies most of a regime's days, then
    "duck days are sunnier" may be "California is sunnier than the Pacific Northwest".
    """
    lab = normalise_labels(cluster_labels, regime_names)
    return pd.crosstab(lab[GROUP], lab["regime"])


def regime_weather_within_zone(panel: pd.DataFrame, cluster_labels,
                               regime_names: dict,
                               weather=("temp", "cloud_cover", "ghi"),
                               min_coverage: float = 0.95) -> pd.DataFrame:
    """Each regime's weather as a departure from ITS OWN ZONE's mean.

    The pooled version of this check cannot separate regime from geography. Measuring
    within zone can: whatever survives here is a property of the daily shape, and
    whatever does not was a between-zone contrast wearing a regime's name.

    `min_coverage` guards the join. An inner merge on a mis-keyed date silently drops
    rows instead of failing, so the share of labelled zone-days that found weather is
    checked rather than assumed.
    """
    weather = [c for c in weather if c in panel.columns]
    if not weather:
        raise ValueError("none of the requested weather columns are in the panel")
    lab = normalise_labels(cluster_labels, regime_names)
    p = panel.copy()
    p["local_date"] = P.local_calendar_date(p)
    daily = (p.groupby([GROUP, "local_date"])[weather].mean().reset_index()
             .merge(lab.rename(columns={"date": "local_date"})[[GROUP, "local_date", "regime"]],
                    on=[GROUP, "local_date"]))
    coverage = len(daily) / max(len(lab), 1)
    if coverage < min_coverage:
        raise AssertionError(
            f"only {coverage:.1%} of {len(lab):,} labelled zone-days matched a weather "
            f"day ({len(daily):,} rows survived the join). A local-date key that does "
            "not line up drops rows silently, so this is refused rather than reported.")
    rows = []
    for zone, dz in daily.groupby(GROUP):
        base = dz[weather].mean()
        for regime, dr in dz.groupby("regime"):
            rows.append({"zone": zone, "regime": regime, "zone_days": len(dr),
                         **{f"{c}_vs_zone_mean": float(dr[c].mean() - base[c])
                            for c in weather}})
    return pd.DataFrame(rows)
