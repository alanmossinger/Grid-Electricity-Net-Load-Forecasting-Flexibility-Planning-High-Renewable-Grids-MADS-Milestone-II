"""Models: the benchmarks (seasonal naive, the EIA composite and its strengthened version), the
linear rung with its diagnostics, and the shared training frames and scorer used by notebooks 05-06.
"""
from __future__ import annotations


# ---- Benchmarks --------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from src import config as cfg


def lag_by_time(frame: pd.DataFrame, col: str, hours: int,
                group_col: str = cfg.GROUP_COL, time_col: str = cfg.TIME_COL) -> pd.Series:
    """`col` at t - hours in the same zone, looked up by TIMESTAMP within `frame`; NaN where
    that hour is not in the frame. Aligned to `frame`'s index.

    Every benchmark lag goes through here. A positional `groupby(zone).shift(168)` counts
    rows, and the panel has holes (the cleaning contract drops unrecoverable hours): on the
    rows after a 25-hour step (24 missing hours) a row shift reaches 192 hours back and hands the seasonal-naive
    benchmark the wrong week. The model's own lags are looked up by timestamp
    (GroupLagFeatures); the benchmarks must be built the same way.
    """
    keys = pd.MultiIndex.from_arrays([frame[group_col].to_numpy(), frame[time_col].to_numpy()])
    src = pd.Series(frame[col].to_numpy(), index=keys)
    src = src[~src.index.duplicated(keep="last")]
    want = pd.MultiIndex.from_arrays([frame[group_col].to_numpy(),
                                      (frame[time_col] - pd.Timedelta(hours=int(hours))).to_numpy()])
    return pd.Series(src.reindex(want).to_numpy(), index=frame.index)


def capped_cube(speed_kmh, rated_kmh: float = cfg.WIND_RATED_KMH):
    """min(v, v_rated)^3 / 1000 — the turbine power curve, in one expression.

    Output goes as the cube of hub-height wind speed below rated speed and is flat
    above it. Dividing by 1000 keeps the column on a scale a linear coefficient can
    be read on; it changes nothing about the fit.
    """
    v = np.clip(np.asarray(speed_kmh, dtype=float), 0.0, rated_kmh)
    return v ** 3 / 1000.0


def _design(block: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The regressors for one leg. Physical, small, and the same in fit and predict."""
    if kind == "wind":
        return pd.DataFrame({
            "cube": capped_cube(block["wind_speed_100m"]),
            "speed": block["wind_speed_100m"].astype(float),
            # Above rated speed output is flat, and above cut-out it collapses. One
            # indicator is enough to let the line bend where the physics does.
            "above_rated": (block["wind_speed_100m"] > cfg.WIND_RATED_KMH).astype(float),
        })
    if kind == "solar":
        return pd.DataFrame({
            "ghi": block["ghi"].astype(float),
            "dni": block["dni_horiz"].astype(float),
            # Panels lose efficiency as they heat. A small term, and it is the one
            # correction a solar engineer would insist on.
            "ghi_x_temp": block["ghi"].astype(float) * block["temp"].astype(float) / 100.0,
        })
    raise ValueError(f"kind must be 'wind' or 'solar', got {kind!r}")


def fit_renewable_legs(dev: pd.DataFrame, fit_window_days: int = 365,
                       valid_days: int = 0,
                       time_col: str = cfg.TIME_COL,
                       group_col: str = cfg.GROUP_COL) -> dict:
    """Per-zone least-squares coefficients for the wind and solar legs.

    Fitted on the last `fit_window_days` of `dev` and on nothing else. The holdout
    is never touched here, which is the whole point of the signature: a benchmark
    fitted on the block it is scored against would be a better model than the
    champion for reasons that have nothing to do with forecasting.

    `valid_days` holds the tail of dev out of the fit as well, so that
    `select_leg_methods` can compare the weather arm against persistence on rows the
    weather arm has not seen. Leaving it at 0 fits right up to the dev boundary,
    which is what the SHIPPED benchmark does once the method per leg is settled.
    """
    end = dev[time_col].max() - pd.Timedelta(days=valid_days)
    cut = end - pd.Timedelta(days=fit_window_days)
    recent = dev[(dev[time_col] > cut) & (dev[time_col] <= end)]
    coefs: dict = {"fit_from": str(recent[time_col].min()),
                   "fit_to": str(recent[time_col].max()),
                   "fit_rows": int(len(recent)), "zones": {}}
    for zone, block in recent.groupby(group_col):
        entry = {}
        for kind in ("wind", "solar"):
            if kind not in block.columns:
                continue
            X = _design(block, kind)
            y = block[kind].astype(float)
            ok = X.notna().all(axis=1) & y.notna()
            if ok.sum() < 100:
                entry[kind] = None            # a zone that does not report this fuel
                continue
            A = np.column_stack([np.ones(int(ok.sum())), X[ok].to_numpy()])
            beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
            entry[kind] = {"names": ["intercept"] + list(X.columns),
                           "beta": [float(b) for b in beta],
                           "n": int(ok.sum())}
        coefs["zones"][zone] = entry
    return coefs


def predict_renewable_legs(frame: pd.DataFrame, coefs: dict,
                           group_col: str = cfg.GROUP_COL) -> pd.DataFrame:
    """`wind_hat` and `solar_hat` for every row, clipped at zero.

    Clipped because a linear model can predict negative generation and no turbine
    or panel produces less than nothing. Clipping at the physical floor is not
    tuning — it is the same rule the cleaning contract applies to the observed
    columns.
    """
    out = pd.DataFrame(index=frame.index, dtype=float)
    for kind in ("wind", "solar"):
        pred = pd.Series(np.nan, index=frame.index, dtype=float)
        for zone, block in frame.groupby(group_col):
            entry = coefs["zones"].get(zone, {}).get(kind)
            if entry is None:
                pred.loc[block.index] = 0.0    # a zone with no such fleet forecasts 0
                continue
            X = _design(block, kind)
            A = np.column_stack([np.ones(len(X)), X.to_numpy()])
            pred.loc[block.index] = A @ np.asarray(entry["beta"])
        out[f"{kind}_hat"] = pred.clip(lower=0.0)
    return out


def composite_v2(frame: pd.DataFrame, coefs: dict,
                 demand_forecast_col: str = "demand_forecast") -> pd.Series:
    """DF_composite_v2 = EIA's demand forecast minus the weather-driven legs."""
    legs = predict_renewable_legs(frame, coefs)
    return frame[demand_forecast_col] - legs["wind_hat"] - legs["solar_hat"]


def leg_errors(frame: pd.DataFrame, coefs: dict,
               group_col: str = cfg.GROUP_COL, time_col: str = cfg.TIME_COL) -> dict:
    """MAE of each leg, weather-driven against 24-hour persistence, on the same rows.

    Reported because the whole argument for v2 rests on this comparison. If the
    weather legs are not materially better than persistence, then v2 is not a harder
    benchmark and saying so is the finding.
    """
    f = frame.sort_values([group_col, time_col]).copy()
    legs = predict_renewable_legs(f, coefs)
    f["wind_hat"], f["solar_hat"] = legs["wind_hat"], legs["solar_hat"]
    f["wind_p24"] = lag_by_time(f, "wind", 24, group_col, time_col)
    f["solar_p24"] = lag_by_time(f, "solar", 24, group_col, time_col)
    ok = f[["wind_hat", "solar_hat", "wind_p24", "solar_p24",
            "wind", "solar"]].notna().all(axis=1)
    g = f[ok]
    out = {"n_scored": int(len(g))}
    for kind in ("wind", "solar"):
        out[f"{kind}_persistence_mae"] = float((g[f"{kind}_p24"] - g[kind]).abs().mean())
        out[f"{kind}_weather_mae"] = float((g[f"{kind}_hat"] - g[kind]).abs().mean())
    out["combined_persistence_mae"] = float(
        ((g["wind_p24"] + g["solar_p24"]) - (g["wind"] + g["solar"])).abs().mean())
    out["combined_weather_mae"] = float(
        ((g["wind_hat"] + g["solar_hat"]) - (g["wind"] + g["solar"])).abs().mean())
    return out


# ---------------------------------------------------------------------------
# choosing a method PER LEG, on validation data, which is the actual diligence
# ---------------------------------------------------------------------------
# THE MEASUREMENT THAT FORCED THIS, and it is the most interesting thing in the
# module. Fitted on the last year of dev and scored on the holdout:
#
#     wind      persistence 2,798.2 MW   weather regression 2,184.6 MW   (-22%)
#     solar     persistence   343.7 MW   weather regression   946.0 MW  (+175%)
#
# The wind result is what everyone expects. The solar result is the opposite, and
# the cause is not a bad regression — it is CAPACITY GROWTH. ERCO's mean solar
# output goes from 1,919 MW across dev to 5,703 MW across the holdout; MISO's from
# 323 to 1,773. The fleet roughly tripled. A coefficient fitted on history predicts
# yesterday's fleet from today's sunshine, while t-24 persistence carries the
# CURRENT installed capacity for free and only has to get the weather delta right.
# On a fast-growing fleet that trade favours persistence, and no amount of care with
# the irradiance model changes it.
#
# A weather-SCALED persistence leg -- solar(t-24) x ghi(t)/ghi(t-24) -- was tried as
# the obvious way to have both, and it lost to plain persistence on both legs
# (solar 577 vs 344, wind 3,055 vs 2,796). The hour-on-hour irradiance ratio is too
# noisy to carry the correction. Recorded because it is the first thing a reader
# will propose.
#
# So the benchmark picks per leg. The SELECTION IS MADE ON A DEV VALIDATION SLICE,
# never on the holdout -- choosing the better leg by looking at the block the claim
# is scored on would make the benchmark a function of the test set, which is the
# same offence this project spends ten notebooks avoiding.

LEG_METHODS = ("persistence", "weather")


def _leg_predictions(frame: pd.DataFrame, coefs: dict, kind: str,
                     group_col: str = cfg.GROUP_COL,
                     time_col: str = cfg.TIME_COL,
                     lag_hours: int = 24) -> pd.DataFrame:
    """Both candidate forecasts for one leg, aligned to `frame`'s own index.

    `lag_hours` follows the HORIZON the benchmark is being built for. At h=48 an
    operator cannot use yesterday's output as persistence — it is not observed at
    the origin — so the leg must reach back 48 hours, and the benchmark gets
    correspondingly worse. Defaulting it to 24 and reusing that at h=72 would hand
    the baseline a day of information it does not have, which is the same offence
    the whole project is built against, committed on the benchmark's behalf.
    """
    f = frame.sort_values([group_col, time_col])
    out = pd.DataFrame(index=f.index)
    out["persistence"] = lag_by_time(f, kind, lag_hours, group_col, time_col)
    out["weather"] = predict_renewable_legs(f, coefs)[f"{kind}_hat"]
    out["actual"] = f[kind]
    return out.reindex(frame.index)


def select_leg_methods(dev: pd.DataFrame, coefs: dict, valid_days: int = 90,
                       time_col: str = cfg.TIME_COL,
                       group_col: str = cfg.GROUP_COL) -> dict:
    """Pick 'persistence' or 'weather' for each leg on the tail of dev.

    `coefs` must have been fitted on dev EXCLUDING this validation slice, or the
    weather arm is being judged on its own training rows and will win by memory.
    `fit_renewable_legs(..., valid_days=...)` is the way to arrange that.
    """
    cut = dev[time_col].max() - pd.Timedelta(days=valid_days)
    valid = dev[dev[time_col] > cut]
    chosen = {"valid_from": str(valid[time_col].min()),
              "valid_to": str(valid[time_col].max()),
              "valid_rows": int(len(valid)), "legs": {}}
    for kind in ("wind", "solar"):
        preds = _leg_predictions(valid, coefs, kind, group_col, time_col)
        ok = preds.notna().all(axis=1)
        scores = {m: float((preds.loc[ok, m] - preds.loc[ok, "actual"]).abs().mean())
                  for m in LEG_METHODS}
        winner = min(scores, key=scores.get)
        chosen["legs"][kind] = {"method": winner, "validation_mae": scores,
                                "rows": int(ok.sum())}
    return chosen


def composite_v2_selected(frame: pd.DataFrame, coefs: dict, chosen: dict,
                          demand_forecast_col: str = "demand_forecast",
                          group_col: str = cfg.GROUP_COL,
                          time_col: str = cfg.TIME_COL,
                          lag_hours: int = 24) -> pd.DataFrame:
    """DF_composite_v2 using each leg's validation-selected method.

    Returns a frame with `wind_leg`, `solar_leg` and `df_composite_v2`, indexed like
    `frame`, so the caller can score the legs and the composite on the same rows.
    """
    out = pd.DataFrame(index=frame.index)
    for kind in ("wind", "solar"):
        method = chosen["legs"][kind]["method"]
        out[f"{kind}_leg"] = _leg_predictions(frame, coefs, kind, group_col,
                                              time_col, lag_hours)[method]
    out["df_composite_v2"] = (frame[demand_forecast_col]
                              - out["wind_leg"] - out["solar_leg"])
    return out


# ---- Linear regression and its assumptions -----------------------------------------------------

import numpy as np
import pandas as pd

from src import config as cfg


# ===========================================================================
# 1. The design matrix the ML models actually see
# ===========================================================================
def design_matrix(pipe, frame, drop_constant: bool = True,
                  dummy_prefixes=("zone_is_",), drop_reference: bool = True) -> pd.DataFrame:
    """Run a fitted supervised pipeline up to (not including) the estimator.

    WHY NOT JUST REGRESS ON THE RAW COLUMNS. Because then the linear model and the
    boosted model would be fed different information and the comparison in the ladder
    would be meaningless. OLS here sees exactly the engineered feature set the
    tree models see — same lags, same calendar encodings, same MI selection — so the
    difference in score is attributable to the estimator and nothing else.

    THE DUMMY VARIABLE TRAP, and it is the reason `drop_reference` exists. The
    pipeline one-hot encodes six zones into six columns, because a tree does not care
    and the extra column costs it nothing. Add an intercept to that and the design is
    EXACTLY singular: the six dummies sum to the constant column. statsmodels will not
    crash — it falls back to a pseudo-inverse — it will simply return coefficients
    that are not uniquely identified, standard errors computed from a rank-deficient
    matrix, and a condition number around 1e15. Measured here before the fix: VIF
    62,563 on `zone_is_ERCO`.

    Dropping one dummy makes the remaining five read as contrasts AGAINST the omitted
    zone, which is what a zone coefficient means in a fixed-effects specification.
    The omitted level is returned in the frame's `.attrs` so the notebook can name it
    rather than leaving the reader to guess what the baseline is.

    Constant columns are dropped for the same reason: they carry no information and
    they make X'X singular.
    """
    from sklearn.pipeline import Pipeline

    pre = Pipeline(pipe.steps[:-1])
    Xt = pre.transform(frame)
    names = list(pipe.named_steps["select"].get_feature_names_out())
    Xt = np.asarray(Xt)
    if Xt.shape[1] != len(names):
        raise ValueError(
            f"design matrix has {Xt.shape[1]} columns but `select` names "
            f"{len(names)}. A step after `select` is adding or dropping columns; "
            f"the names and the matrix must correspond or every coefficient is "
            f"attributed to the wrong variable."
        )
    X = pd.DataFrame(Xt, columns=names, index=frame.index)
    if drop_constant:
        X = X.loc[:, X.std(axis=0) > 0]

    reference = {}
    if drop_reference:
        for pref in dummy_prefixes:
            group = sorted(c for c in X.columns if c.startswith(pref))
            if len(group) > 1:
                reference[pref] = group[0]
                X = X.drop(columns=[group[0]])
    X.attrs["reference_levels"] = reference
    return X


# ===========================================================================
# 2. The fit, and the table an examiner expects to see
# ===========================================================================
def ols_fit(X, y, cov: str = "nonrobust", hac_lags: int = 24, times=None):
    """Fit OLS with an intercept under one of three covariance estimators.

        nonrobust  the textbook one. Valid only if the errors are independent and
                   homoscedastic — which §4.4 shows they are not.
        HC3        heteroscedasticity-consistent. Fixes non-constant variance;
                   does nothing about autocorrelation.
        HAC        Newey-West. Fixes BOTH, by summing autocovariances out to
                   `hac_lags`. This is the one to quote on hourly data.

    The coefficients are IDENTICAL under all three — only the standard errors move.
    That is the single most useful thing to understand about robust inference: it
    does not fix the estimate, it fixes your confidence in it.

    A PANEL. On rows of six zones sorted by time, `hac_lags=24` ROWS reach four hours
    back, into other zones. Pass `times` (each row's timestamp, rows sorted by time):
    the scores are summed across zones within each hour and the Newey-West sum runs over
    hours (Driscoll-Kraay, statsmodels 'hac-groupsum'), so 24 lags are 24 hours, and
    zones that err together in the same hour are not counted as independent.
    """
    import pandas as pd
    import statsmodels.api as sm

    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    model = sm.OLS(np.asarray(y, dtype=float), Xc)
    if cov == "HAC" and times is not None:
        t = pd.to_datetime(pd.Series(np.asarray(times)))
        if not t.is_monotonic_increasing:
            raise ValueError("rows must be sorted by time for the panel HAC")
        codes = pd.factorize(t, sort=True)[0]
        return model.fit(cov_type="hac-groupsum", cov_kwds={"time": codes, "maxlags": int(hac_lags)})
    if cov == "HAC":
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    if cov == "HC3":
        return model.fit(cov_type="HC3")
    return model.fit()


def coefficient_table(fit, names) -> pd.DataFrame:
    """coef · std err · t · P>|t| · 95% CI, one row per term, intercept first."""
    idx = ["const"] + list(names)
    ci = fit.conf_int()
    return pd.DataFrame({
        "coef": fit.params, "std_err": fit.bse, "t": fit.tvalues, "p": fit.pvalues,
        "ci_lo": ci[:, 0] if isinstance(ci, np.ndarray) else ci[0].to_numpy(),
        "ci_hi": ci[:, 1] if isinstance(ci, np.ndarray) else ci[1].to_numpy(),
    }, index=idx)


def fit_summary(fit) -> dict:
    """The global statistics: R², adjusted R², F and its p-value, AIC, BIC.

    WHY ADJUSTED R² IS THE ONE TO QUOTE with 40-odd regressors: plain R² cannot fall
    when a variable is added, so on a wide design it rewards clutter. The adjustment
    charges for each parameter.

    WHY THE F-STATISTIC IS NEARLY USELESS HERE, and it is worth saying so rather than
    printing it without comment: it tests whether ALL slopes are zero jointly. With
    268,000 rows and a lag-1-style predictor in the design, that null was never
    remotely plausible. A vast F with p = 0 is not evidence of a good model; it is
    evidence the null was a straw man.
    """
    return {
        "r2": float(fit.rsquared), "adj_r2": float(fit.rsquared_adj),
        "f_stat": float(fit.fvalue), "f_pvalue": float(fit.f_pvalue),
        "aic": float(fit.aic), "bic": float(fit.bic),
        "n": int(fit.nobs), "df_model": int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "sigma": float(np.sqrt(fit.mse_resid)),
    }


def within_group_r2(y, resid, groups) -> dict:
    """R² recomputed against each unit's OWN mean instead of the grand mean.

    WHY THE HEADLINE R² IS INFLATED ON THIS PANEL, and by a lot. Total R² measures
    variance explained around the GRAND mean of all six zones pooled. Those zones
    differ enormously in size — mean net load runs from a few GW to tens of GW — so a
    large share of the total variance is simply "which zone is this row?". A model
    that knew nothing but the zone dummies would already score a high R².

    Recomputing against each zone's own mean removes that free variance and asks the
    question actually of interest: given that we are in ERCO, how much of ERCO's own
    hour-to-hour variation is explained? The gap between the two numbers is the share
    of the headline R² that was cross-sectional rather than predictive.
    """
    y = np.asarray(y, dtype=float)
    r = np.asarray(resid, dtype=float)
    g = pd.Series(np.asarray(groups))
    ss_res = float((r ** 2).sum())
    ss_tot_grand = float(((y - y.mean()) ** 2).sum())
    demeaned = y - g.map(pd.Series(y).groupby(g).mean()).to_numpy()
    ss_tot_within = float((demeaned ** 2).sum())
    return {
        "r2_total": 1 - ss_res / ss_tot_grand,
        "r2_within": 1 - ss_res / ss_tot_within,
        "between_share_of_variance": 1 - ss_tot_within / ss_tot_grand,
    }


# ===========================================================================
# 3. The assumption battery
# ===========================================================================
def variance_inflation(X, sample: int = 20_000, seed: int = cfg.SEED) -> pd.DataFrame:
    """VIF per column: how much a coefficient's variance is inflated by collinearity.

        VIF_j = 1 / (1 - R²_j),  R²_j from regressing column j on all the others.

    VIF = 1 means orthogonal. 5 is a common eyebrow-raise, 10 the conventional alarm.
    VIF = 100 means the standard error is TEN times what it would be under
    orthogonality — the coefficient is estimated ten times less precisely than the
    sample size suggests.

    Computed on a subsample because it costs one regression per column; the estimate
    is stable well below the full 268k rows and the alternative is minutes of compute
    for a diagnostic read to one decimal place.
    """
    from sklearn.linear_model import LinearRegression

    Xs = X.sample(n=min(sample, len(X)), random_state=seed) if len(X) > sample else X
    A = np.asarray(Xs, dtype=float)
    rows = []
    for j, col in enumerate(Xs.columns):
        others = np.delete(A, j, axis=1)
        r2 = LinearRegression().fit(others, A[:, j]).score(others, A[:, j])
        vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "vif": float(vif), "r2_on_others": float(r2)})
    return pd.DataFrame(rows).set_index("feature").sort_values("vif", ascending=False)


def condition_number(X) -> float:
    """Condition number of the standardised design — collinearity as one number.

    Standardised first, or the number would mostly reflect that net load is in the
    thousands and `hour_sin` is in [-1, 1], which is a units artefact rather than
    collinearity. Above 30 is the textbook concern threshold.
    """
    A = np.asarray(X, dtype=float)
    A = (A - A.mean(0)) / np.where(A.std(0) == 0, 1, A.std(0))
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def _within_group_serial(resid, groups, order, max_lag: int):
    """Durbin-Watson and residual ACF computed WITHIN each unit, ordered by time.

    WHY THIS IS NOT OPTIONAL ON A PANEL, and getting it wrong is silent. This project
    sorts the frame `(timestamp, zone)` so that a positional CV split is a
    chronological one. The consequence for diagnostics is that two ADJACENT ROWS are
    six different zones at the same hour, not one zone at consecutive hours. Running
    Durbin-Watson straight down that column measures the CROSS-SECTIONAL correlation
    between zones at a fixed instant and calls it serial correlation.

    Measured on this project's residuals: the naive row-order DW is 1.70, which reads
    as "no strong signal", while the within-zone DW is far lower and the within-zone
    residual ACF at lag 24 is large. The naive number is not a weaker version of the
    right answer; it is an answer to a different question.
    """
    from statsmodels.stats.stattools import durbin_watson

    df = pd.DataFrame({"r": np.asarray(resid, dtype=float),
                       "g": np.asarray(groups), "t": np.asarray(order)})
    dws, weights, acfs = [], [], []
    for _, sub in df.groupby("g", sort=False):
        r = sub.sort_values("t")["r"].to_numpy()
        if len(r) < max_lag + 10:
            continue
        dws.append(float(durbin_watson(r)))
        weights.append(len(r))
        acfs.append([float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in range(1, max_lag + 1)])
    if not dws:
        raise ValueError("no group had enough rows for a serial-correlation diagnostic")
    w = np.asarray(weights, dtype=float)
    return (float(np.average(dws, weights=w)),
            list(np.average(np.asarray(acfs), axis=0, weights=w)),
            int(len(dws)))


def assumption_battery(fit, X, resid_acf_lags: int = 48, groups=None, order=None) -> dict:
    """Test all five classical assumptions and return the verdicts.

    Every test here is reported with its NULL stated, because a p-value whose null
    the reader has to guess is not a result. Note the direction of each null: for
    Breusch-Pagan and Jarque-Bera the null is the GOOD case (homoscedastic, normal),
    so a small p-value is bad news.

    Pass `groups` and `order` on panel data — see `_within_group_serial`. Both the
    naive and the within-unit serial statistics are returned so the notebook can show
    the difference rather than assert it.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    resid = np.asarray(fit.resid, dtype=float)

    dw_naive = float(durbin_watson(resid))
    jb_stat, jb_p, skew, kurt = [float(v) for v in jarque_bera(resid)]

    if groups is not None:
        dw, acf_w, n_groups = _within_group_serial(resid, groups, order, resid_acf_lags)
    else:
        dw, acf_w, n_groups = dw_naive, None, 0

    # Breusch-Pagan on a subsample: the statistic is an n-scaled quantity and on
    # 268k rows it is astronomically large whatever the effect size, so the p-value
    # carries no information. The auxiliary R² is the effect size and is reported.
    n_bp = min(30_000, len(resid))
    idx = np.linspace(0, len(resid) - 1, n_bp).astype(int)
    import statsmodels.api as sm
    exog = sm.add_constant(np.asarray(X, dtype=float)[idx], has_constant="add")
    bp_stat, bp_p, _, _ = het_breuschpagan(resid[idx], exog)

    acf_rows = [float(np.corrcoef(resid[:-k], resid[k:])[0, 1])
                for k in range(1, resid_acf_lags + 1)]
    acf = acf_w if acf_w is not None else acf_rows

    return {
        "durbin_watson": dw,
        "durbin_watson_row_order": dw_naive,
        "serial_stats_within_group": groups is not None,
        "n_groups_used": n_groups,
        "dw_null": "errors are serially uncorrelated (DW = 2)",
        "dw_verdict": ("positive autocorrelation" if dw < 1.5
                       else "negative autocorrelation" if dw > 2.5 else "no strong signal"),
        "resid_acf_lag1": acf[0], "resid_acf_lag24": acf[23] if len(acf) >= 24 else None,
        "resid_acf": acf,
        "resid_acf_row_order": acf_rows,
        "breusch_pagan_stat": float(bp_stat), "breusch_pagan_p": float(bp_p),
        "bp_null": "error variance is constant (homoscedastic)",
        "bp_n_used": int(n_bp),
        "jarque_bera_stat": jb_stat, "jarque_bera_p": jb_p,
        "jb_null": "errors are normally distributed",
        "resid_skew": skew, "resid_kurtosis": kurt,
        "condition_number": condition_number(X),
    }


def coefficient_stability(X, y, n_blocks: int = 4, seed: int = cfg.SEED) -> pd.DataFrame:
    """Refit on consecutive time blocks and report how much each coefficient moves.

    WHY THIS IS THE DECISIVE COLLINEARITY DIAGNOSTIC. A VIF is a statement about the
    design matrix; this is a statement about what actually happens to the answer. If
    a coefficient is +400 on one block and −250 on the next, it has no stable
    interpretation, and no amount of significance stars rescues it. `sign_flips`
    counts exactly that.

    Blocks are CONSECUTIVE, not random: on a time series a random split would leave
    each block with the same seasonal composition and understate the instability.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    edges = np.linspace(0, len(Xa), n_blocks + 1).astype(int)
    fits = []
    for i in range(n_blocks):
        sl = slice(edges[i], edges[i + 1])
        f = ols_fit(Xa[sl], ya[sl])
        fits.append(f.params[1:])                    # drop the intercept
    M = np.vstack(fits)
    mean = M.mean(0)
    return (pd.DataFrame({
        "feature": list(X.columns),
        "mean": mean,
        "sd_across_blocks": M.std(0, ddof=1),
        "min": M.min(0), "max": M.max(0),
        "sign_flips": ((M > 0).any(0) & (M < 0).any(0)),
        "cv_abs": np.where(np.abs(mean) > 1e-9, M.std(0, ddof=1) / np.abs(mean), np.inf),
    }).set_index("feature").sort_values("sd_across_blocks", ascending=False))


def significance_shift(fit_plain, fit_robust, names, alpha: float = 0.05) -> dict:
    """How many coefficients lose significance once the standard errors are honest.

    This is the number that makes robust inference concrete. The coefficients are
    identical between the two fits; only the standard errors differ. Anything that
    was 'significant' under the naive estimator and is not under the robust one was
    never significant — it was autocorrelation being counted as independent evidence.
    """
    idx = ["const"] + list(names)
    p0 = pd.Series(fit_plain.pvalues, index=idx)
    p1 = pd.Series(fit_robust.pvalues, index=idx)
    lost = [n for n in idx if n != "const" and p0[n] < alpha <= p1[n]]
    gained = [n for n in idx if n != "const" and p1[n] < alpha <= p0[n]]
    return {
        "n_significant_naive": int((p0.drop("const") < alpha).sum()),
        "n_significant_robust": int((p1.drop("const") < alpha).sum()),
        "lost_significance": lost, "gained_significance": gained,
        "n_terms": len(idx) - 1,
        "median_se_inflation": float(np.median(np.asarray(fit_robust.bse)
                                               / np.asarray(fit_plain.bse))),
    }


# ---- Training frames and the downstream scorer -------------------------------------------------

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import features as P

# Columns that may never appear in an exported dataset AS FEATURES.
#
# `ramp` contains net_load(t) and is never needed for anything, so it stays out
# of the file entirely. `demand`, `wind` and `solar` are a different case since
# EXOG_LAG_FEATURES widened: they are LAG SOURCES, so the file must contain them
# for `wind_lag24` to be reconstructible, and DROP_ALWAYS removes the raw column
# from the feature set at the `select` step.
#
# BE CLEAR ABOUT WHAT THIS COSTS. The export used to be structurally incapable of
# leaking -- the target's components simply were not in the file. That is no
# longer true. dataset.parquet now holds demand, wind and solar, and
# net_load = demand - wind - solar exactly, so anyone who fits on these columns
# directly instead of through the pipeline gets a perfect score and a worthless
# model. The protection is now procedural (replicate.py goes through
# make_supervised_pipeline, which drops them) rather than structural. That is a
# real reduction in safety and it is the price of lagging the renewable history.
NEVER_EXPORT = ("ramp",)
FORBIDDEN = NEVER_EXPORT


def _estimator(kind):
    """Fresh estimator per call -- never share a fitted object between folds."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=120, min_samples_leaf=5,
                                     random_state=cfg.SEED, n_jobs=cfg.N_JOBS)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=cfg.SEED)
    if kind == "quantile_gbm_median":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.5,
                                             random_state=cfg.SEED)
    if kind == "curtailment_classifier":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=cfg.SEED)
    raise ValueError(f"unknown estimator kind {kind!r}")


# The curtailment label, lifted from notebook 04 section 4.15 so the exported
# dataset carries the same definition the notebook scored.
#
# WHY A LABEL BUILT FROM THE SAME-HOUR TARGET IS NOT A LEAK. A label is supposed
# to be the answer; the question is whether any FEATURE restates it. `net_load`,
# `renewable_share`, `wind` and `solar` are all in DROP_ALWAYS, so none of them
# reaches the model as a raw feature. Their LAGGED copies do, and that is the
# ordinary history an operator has. If this arm ever scored near-perfectly, that
# guard is the first thing to check.
#
# WHY THE THRESHOLDS ARE FITTED ON DEV ONLY. They are quantiles -- parameters
# learned from data. Computing them on the full panel would let the holdout's own
# distribution define its labels, which is the same error as fitting a scaler
# outside the fold.


# The four fitted rungs of notebook 04's ladder. The two baselines (seasonal
# naive, EIA DF composite) are deliberately absent: they fit no parameters and
# have no dataset to replicate -- their holdout reference is in results.json.
MODELS = {
    "ridge": {
        "ladder_rung": "1  ridge",
        "estimator": lambda: _estimator("ridge"),
        "what_it_is": "linear baseline; scored, never read as importances (VIF)",
    },
    "random_forest": {
        "ladder_rung": "2  random forest",
        "estimator": lambda: _estimator("random_forest"),
        "what_it_is": "bagged trees; the non-linear control for the booster",
    },
    "hist_gradient_boosting": {
        "ladder_rung": "3  hist gradient boosting",
        "estimator": lambda: _estimator("hist_gradient_boosting"),
        "what_it_is": "the champion; carried into notebooks 06 and 07",
    },
    "quantile_gbm_median": {
        "ladder_rung": "4  quantile GBM (median)",
        "estimator": lambda: _estimator("quantile_gbm_median"),
        "what_it_is": "median loss; licenses notebook 06 to reuse this feature set",
    },
    # Not a ladder rung -- a second arm with a different estimand. A regression
    # MAE cannot answer "will this hour be a problem?", and that question is a
    # decision, so it is a classification.
    "curtailment_classifier": {
        "ladder_rung": None,
        "estimator": lambda: _estimator("curtailment_classifier"),
        "what_it_is": "curtailment-risk arm; label is dev-fitted zone quantiles",
        "task": "classification",
        "results_key": "classification",
    },
}


def analysis_frames(horizon: int = 24):
    """The panel, dev and holdout blocks, split exactly as notebook 04 §4.1 does.

    Time-major sort first: on a panel sorted (zone, timestamp) a positional cut
    lands inside one zone's history. Sorted (timestamp, zone) every positional
    block is a block of time containing all six zones.
    """
    time, group = cfg.TIME_COL, cfg.GROUP_COL
    df = pd.read_parquet(cfg.PANEL).sort_values([time, group]).reset_index(drop=True)
    assert df[time].is_monotonic_increasing, "panel must be time-major before splitting"
    cut = cfg.HOLDOUT_CUT
    dev = df[df[time] <= cut].reset_index(drop=True)
    hold = df[df[time] > cut].reset_index(drop=True)
    return df, dev, hold, cut


def panel_hash() -> str:
    """crc32 of the panel bytes. `hash()` is salted per process; this is not."""
    return format(zlib.crc32(Path(cfg.PANEL).read_bytes()) & 0xFFFFFFFF, "08x")


# ==========================================================================
# The bridge notebook 05 scores against
# ==========================================================================

def score_with_extra_features(extra: pd.DataFrame | None = None,
                              model: str = "hist_gradient_boosting",
                              horizon: int = 24) -> dict:
    """CV MAE of a notebook 04 model, optionally with extra columns joined on.

    WHY THIS EXISTS. Notebook 05 has to answer "does the cluster label help?", and
    that is only a real answer if the with- and without- numbers come from the
    SAME estimator, the SAME folds and the SAME gap. Two MAEs computed in two
    notebooks under two setups are not a comparison. Call this with `extra=None`
    for the baseline and with the label frame for the treatment, and the
    difference means something.

    `extra` must be indexed so it aligns row-for-row with the dev block after
    the time-major sort -- pass a frame with the same length as `dev`, or one
    carrying `timestamp` and `zone` to merge on.
    """
    from sklearn.model_selection import cross_val_score

    time, group, target = cfg.TIME_COL, cfg.GROUP_COL, cfg.TARGET
    lags, gap = cfg.lags_for(horizon), cfg.cv_gap_for(horizon)
    _df, dev, _hold, _cut = analysis_frames(horizon)
    numeric = list(cfg.NUMERIC_FEATURES)

    if extra is not None:
        extra = extra.copy()
        if {time, group}.issubset(extra.columns):
            dev = dev.merge(extra, on=[time, group], how="left")
        elif len(extra) == len(dev):
            for c in extra.columns:
                dev[c] = extra[c].values
        else:
            raise ValueError(
                f"extra has {len(extra)} rows against dev's {len(dev)} and carries "
                f"neither {time} nor {group} to merge on")
        added = [c for c in extra.columns if c not in (time, group)]
        bad = [c for c in added if c in FORBIDDEN]
        assert not bad, f"refusing to add forbidden columns as features: {bad}"
        numeric += added
    else:
        added = []

    est = MODELS[model]["estimator"]()
    pipe = P.make_supervised_pipeline(numeric, lags=lags, model=est)
    cv = P.make_ts_cv(n_splits=cfg.N_SPLITS, gap=gap)
    scores = -cross_val_score(pipe, dev, dev[target], cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=cfg.N_JOBS)
    return {"model": model, "added_features": added,
            "cv_mae_mean": float(scores.mean()), "cv_mae_sd": float(scores.std()),
            "folds": [float(s) for s in scores], "n_dev": int(len(dev))}

