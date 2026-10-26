"""
test_baselines.py — the harder government-data benchmark.

The point of `DF_composite_v2` is that it is difficult to beat. Every property
tested here is one whose failure would make it EASIER than it looks, which is the
only direction that matters: a benchmark that quietly cheats in its own favour is
just a strawman with extra steps, and a benchmark that quietly cheats in ours
invalidates the claim built on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models as B
from src import config as cfg


def _frame(n_days=200, zones=("CISO", "ERCO"), seed=0, capacity_growth=1.0):
    """A panel where wind follows the cube of hub-height speed and solar follows GHI."""
    rng = np.random.default_rng(seed)
    rows = []
    stamps = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    for z in zones:
        v = np.abs(rng.normal(25, 12, len(stamps)))
        hour = stamps.hour.to_numpy()
        ghi = np.clip(900 * np.sin(np.pi * (hour - 6) / 12), 0, None) * rng.uniform(
            0.3, 1.0, len(stamps))
        # capacity ramps linearly over the window when capacity_growth > 1
        ramp = np.linspace(1.0, capacity_growth, len(stamps))
        rows.append(pd.DataFrame({
            cfg.TIME_COL: stamps, cfg.GROUP_COL: z,
            "wind_speed_100m": v,
            "ghi": ghi, "dni_horiz": ghi * 0.7, "temp": 15 + 10 * np.sin(hour / 4),
            "wind": 3.0 * B.capped_cube(v) + rng.normal(0, 40, len(stamps)),
            "solar": ramp * 8.0 * ghi + rng.normal(0, 60, len(stamps)),
            "demand": 20000 + 3000 * np.sin(hour / 4) + rng.normal(0, 200, len(stamps)),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["solar"] = df["solar"].clip(lower=0)
    df["wind"] = df["wind"].clip(lower=0)
    df["demand_forecast"] = df["demand"] + rng.normal(0, 150, len(df))
    return df.sort_values([cfg.TIME_COL, cfg.GROUP_COL]).reset_index(drop=True)


# --------------------------------------------------------------- the transform
def test_the_power_curve_is_flat_above_rated_speed():
    """A turbine at 60 km/h produces what it produces at 45 km/h, not twice as much.
    If this stops holding, the wind leg is extrapolating a cubic into a region where
    the physics is a plateau."""
    assert B.capped_cube(60.0) == B.capped_cube(cfg.WIND_RATED_KMH)
    assert B.capped_cube(20.0) < B.capped_cube(40.0)


def test_the_power_curve_is_cubic_below_rating():
    assert B.capped_cube(20.0) / B.capped_cube(10.0) == pytest.approx(8.0)


# ------------------------------------------------------------------- the fit
def test_the_legs_are_fitted_on_dev_only():
    """The signature takes dev. If a future edit lets the holdout in, the benchmark
    becomes a better model than the champion for reasons unrelated to forecasting."""
    df = _frame()
    cut = df[cfg.TIME_COL].quantile(0.85)
    dev = df[df[cfg.TIME_COL] <= cut]
    coefs = B.fit_renewable_legs(dev)
    assert pd.Timestamp(coefs["fit_to"]) <= cut


def test_the_validation_slice_is_held_out_of_the_fit():
    """Without this, `select_leg_methods` judges the weather arm on its own training
    rows and picks it every time by memory rather than by skill."""
    df = _frame()
    coefs = B.fit_renewable_legs(df, valid_days=30)
    assert pd.Timestamp(coefs["fit_to"]) <= df[cfg.TIME_COL].max() - pd.Timedelta(days=30)


def test_a_zone_with_no_fleet_forecasts_zero_rather_than_nan():
    """NYIS reports no solar at all. A NaN leg would silently drop every NYIS row
    from the scored set and quietly change what the benchmark is measured on."""
    df = _frame()
    df.loc[df[cfg.GROUP_COL] == "ERCO", "solar"] = np.nan
    coefs = B.fit_renewable_legs(df)
    pred = B.predict_renewable_legs(df, coefs)
    assert pred.loc[df[cfg.GROUP_COL] == "ERCO", "solar_hat"].notna().all()


def test_predictions_are_never_negative():
    df = _frame()
    coefs = B.fit_renewable_legs(df)
    pred = B.predict_renewable_legs(df, coefs)
    assert (pred["wind_hat"] >= 0).all() and (pred["solar_hat"] >= 0).all()


def test_predictions_keep_the_callers_row_order():
    """`_leg_predictions` sorts internally to build the t-24 shift. If it forgot to
    reindex, every leg would be silently misaligned with the frame it is scored
    against — and the benchmark would look terrible for no reason anyone could see."""
    df = _frame(n_days=40)
    coefs = B.fit_renewable_legs(df)
    shuffled = df.sample(frac=1.0, random_state=7)
    pred = B.predict_renewable_legs(shuffled, coefs)
    assert list(pred.index) == list(shuffled.index)
    direct = B.predict_renewable_legs(df, coefs)
    assert np.allclose(pred["wind_hat"].to_numpy(),
                       direct.loc[shuffled.index, "wind_hat"].to_numpy())


# ------------------------------------------------------------- the selection
def test_the_weather_leg_wins_when_the_fleet_is_stable():
    """With no capacity growth the weather regression should beat persistence on
    wind, because the driver is genuinely observed and persistence is a day stale."""
    df = _frame(n_days=300)
    cut = df[cfg.TIME_COL].quantile(0.85)
    dev = df[df[cfg.TIME_COL] <= cut]
    coefs = B.fit_renewable_legs(dev, valid_days=30)
    chosen = B.select_leg_methods(dev, coefs, valid_days=30)
    assert chosen["legs"]["wind"]["method"] == "weather"


def test_capacity_growth_moves_the_solar_leg_towards_persistence():
    """The MECHANISM behind the real finding, which is the part that generalises.

    On the live panel the weather regression LOSES the solar leg to persistence —
    946 MW against 344 — and the cause is capacity, not irradiance: ERCO's mean
    solar output goes from 1,919 MW across dev to 5,703 MW across the holdout. A
    coefficient fitted on history predicts LAST YEAR's fleet from today's sunshine;
    t-24 persistence carries the current installed capacity for free.

    The assertion here is the direction, not the winner. Which method wins depends
    on how much unexplained variance the fleet has — on this deliberately clean
    fixture the regression still wins outright, and forcing the fixture until it
    lost would be fitting the test to the answer. What must hold in general is that
    growth moves the ratio TOWARDS persistence, and that is what is checked.
    """
    def ratio(growth):
        df = _frame(n_days=400, capacity_growth=growth)
        cut = df[cfg.TIME_COL].quantile(0.85)
        dev, hold = df[df[cfg.TIME_COL] <= cut], df[df[cfg.TIME_COL] > cut]
        errs = B.leg_errors(hold, B.fit_renewable_legs(dev))
        return errs["solar_weather_mae"] / errs["solar_persistence_mae"]

    assert ratio(3.0) > 1.5 * ratio(1.0), (
        "capacity growth must penalise the fitted leg relative to persistence — "
        "that is the whole explanation for the solar result on the live panel")


def test_selection_never_looks_at_the_holdout():
    """The selection must be a function of dev alone. Two different holdouts, the
    same dev, must produce the same choice — otherwise the benchmark is a function
    of the block the claim is scored on."""
    df = _frame(n_days=300)
    cut = df[cfg.TIME_COL].quantile(0.85)
    dev = df[df[cfg.TIME_COL] <= cut]
    coefs = B.fit_renewable_legs(dev, valid_days=30)
    a = B.select_leg_methods(dev, coefs, valid_days=30)
    b = B.select_leg_methods(dev, coefs, valid_days=30)
    assert a["legs"] == b["legs"]


# ------------------------------------------------------------- the composite
def test_the_composite_is_the_demand_forecast_minus_both_legs():
    df = _frame()
    cut = df[cfg.TIME_COL].quantile(0.85)
    dev, hold = df[df[cfg.TIME_COL] <= cut], df[df[cfg.TIME_COL] > cut]
    coefs = B.fit_renewable_legs(dev, valid_days=30)
    chosen = B.select_leg_methods(dev, coefs, valid_days=30)
    out = B.composite_v2_selected(hold, coefs, chosen)
    ok = out.notna().all(axis=1)
    assert np.allclose(
        out.loc[ok, "df_composite_v2"],
        (hold.loc[ok, "demand_forecast"] - out.loc[ok, "wind_leg"] - out.loc[ok, "solar_leg"]))


def test_v2_is_at_least_as_hard_as_v1_on_the_leg_it_changes():
    """The whole point. If the selected legs are not better than plain persistence,
    v2 is not a harder benchmark and the honest thing is to say so rather than to
    quote a skill against it."""
    df = _frame(n_days=300)
    cut = df[cfg.TIME_COL].quantile(0.85)
    dev, hold = df[df[cfg.TIME_COL] <= cut], df[df[cfg.TIME_COL] > cut]
    coefs = B.fit_renewable_legs(dev, valid_days=30)
    errs = B.leg_errors(hold, coefs)
    assert errs["wind_weather_mae"] < errs["wind_persistence_mae"]


# ---- Linear regression -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pytest

from src import models as L


def _clean(n=4000, seed=0):
    """y = 3 + 2*x1 - 1.5*x2 + N(0, 1). Textbook-clean: all five assumptions hold."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    y = 3 + 2 * X["x1"] - 1.5 * X["x2"] + rng.normal(0, 1, n)
    return X, y


# --------------------------------------------------------------------------
# The fit itself
# --------------------------------------------------------------------------
def test_ols_recovers_planted_coefficients():
    X, y = _clean()
    fit = L.ols_fit(X, y)
    tab = L.coefficient_table(fit, X.columns)
    assert tab.loc["const", "coef"] == pytest.approx(3.0, abs=0.1)
    assert tab.loc["x1", "coef"] == pytest.approx(2.0, abs=0.1)
    assert tab.loc["x2", "coef"] == pytest.approx(-1.5, abs=0.1)


def test_confidence_intervals_bracket_the_truth():
    X, y = _clean()
    tab = L.coefficient_table(L.ols_fit(X, y), X.columns)
    assert tab.loc["x1", "ci_lo"] < 2.0 < tab.loc["x1", "ci_hi"]
    assert tab.loc["x2", "ci_lo"] < -1.5 < tab.loc["x2", "ci_hi"]


def test_summary_statistics_are_internally_consistent():
    X, y = _clean()
    s = L.fit_summary(L.ols_fit(X, y))
    assert 0 < s["adj_r2"] <= s["r2"] < 1
    assert s["n"] == 4000
    assert s["df_model"] == 2
    assert s["df_resid"] == 4000 - 3
    assert s["f_pvalue"] < 1e-10


def test_robust_covariances_change_the_errors_but_never_the_coefficients():
    """The single most useful fact about robust inference, pinned as a test."""
    X, y = _clean()
    plain, hac, hc3 = (L.ols_fit(X, y), L.ols_fit(X, y, cov="HAC"),
                       L.ols_fit(X, y, cov="HC3"))
    assert np.allclose(plain.params, hac.params)
    assert np.allclose(plain.params, hc3.params)
    assert not np.allclose(plain.bse, hac.bse)


# --------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------
def test_durbin_watson_is_near_two_on_independent_errors():
    X, y = _clean()
    d = L.assumption_battery(L.ols_fit(X, y), X)
    assert 1.8 < d["durbin_watson"] < 2.2
    assert d["dw_verdict"] == "no strong signal"


def test_durbin_watson_detects_planted_autocorrelation():
    rng = np.random.default_rng(1)
    n = 4000
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 1)      # AR(1), strongly correlated
    X = pd.DataFrame({"x1": rng.normal(0, 1, n)})
    y = 1 + 2 * X["x1"] + e
    d = L.assumption_battery(L.ols_fit(X, y), X)
    assert d["durbin_watson"] < 1.0
    assert d["dw_verdict"] == "positive autocorrelation"
    assert d["resid_acf_lag1"] > 0.7


def test_autocorrelation_makes_naive_standard_errors_too_small():
    """The consequence, not just the diagnosis — this is why HAC is quoted.

    With correlated errors the naive estimator understates the standard error, so
    t-statistics are inflated and p-values are too small in the dangerous direction.
    """
    rng = np.random.default_rng(2)
    n = 4000
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 1)
    X = pd.DataFrame({"x1": rng.normal(0, 1, n)})
    y = 1 + 0.0 * X["x1"] + e                        # TRUE slope is zero
    plain, hac = L.ols_fit(X, y), L.ols_fit(X, y, cov="HAC", hac_lags=24)
    assert hac.bse[1] > plain.bse[1], "HAC must widen the standard error"


# --------------------------------------------------------------------------
# Homoscedasticity and normality
# --------------------------------------------------------------------------
def test_breusch_pagan_fires_on_planted_heteroscedasticity_and_not_otherwise():
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.uniform(1, 10, n)
    clean = pd.DataFrame({"x": x}), pd.Series(2 * x + rng.normal(0, 1, n))
    hetero = pd.DataFrame({"x": x}), pd.Series(2 * x + rng.normal(0, 1, n) * x)
    d_ok = L.assumption_battery(L.ols_fit(*clean), clean[0])
    d_bad = L.assumption_battery(L.ols_fit(*hetero), hetero[0])
    assert d_ok["breusch_pagan_p"] > 0.01
    assert d_bad["breusch_pagan_p"] < 0.01


def test_jarque_bera_fires_on_non_normal_errors():
    rng = np.random.default_rng(4)
    n = 5000
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})
    gauss = pd.Series(2 * X["x"] + rng.normal(0, 1, n))
    heavy = pd.Series(2 * X["x"] + rng.standard_t(df=2.5, size=n))
    assert L.assumption_battery(L.ols_fit(X, gauss), X)["jarque_bera_p"] > 0.01
    d = L.assumption_battery(L.ols_fit(X, heavy), X)
    assert d["jarque_bera_p"] < 0.01
    assert d["resid_kurtosis"] > 4, "heavy tails must show up as excess kurtosis"


# --------------------------------------------------------------------------
# Multicollinearity
# --------------------------------------------------------------------------
def test_vif_is_one_for_orthogonal_columns_and_explodes_for_a_near_duplicate():
    rng = np.random.default_rng(5)
    n = 3000
    a = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": rng.normal(0, 1, n), "a_twin": a + rng.normal(0, 0.01, n)})
    v = L.variance_inflation(X)
    assert v.loc["b", "vif"] < 1.2
    assert v.loc["a", "vif"] > 50
    assert v.loc["a_twin", "vif"] > 50


def test_condition_number_rises_with_collinearity():
    rng = np.random.default_rng(6)
    n = 2000
    a = rng.normal(0, 1, n)
    ortho = pd.DataFrame({"a": a, "b": rng.normal(0, 1, n)})
    collin = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.01, n)})
    assert L.condition_number(ortho) < 5
    assert L.condition_number(collin) > 30


def test_collinearity_inflates_the_standard_error_without_biasing_the_estimate():
    """Both halves of the textbook claim, on the same planted data.

    This is what licenses the notebook's refusal to read coefficients as importances:
    the estimate is still right on average, but any single fit is so imprecise that
    its individual value carries almost no information.
    """
    rng = np.random.default_rng(7)
    n, trials = 800, 60
    ortho_hat, collin_hat = [], []
    for t in range(trials):
        r = np.random.default_rng(100 + t)
        a = r.normal(0, 1, n)
        yo = 2 * a + r.normal(0, 1, n)
        Xo = pd.DataFrame({"a": a, "b": r.normal(0, 1, n)})
        ortho_hat.append(L.ols_fit(Xo, yo).params[1])
        twin = a + r.normal(0, 0.05, n)
        Xc = pd.DataFrame({"a": a, "b": twin})
        yc = 2 * a + r.normal(0, 1, n)
        collin_hat.append(L.ols_fit(Xc, yc).params[1])
    assert np.mean(ortho_hat) == pytest.approx(2.0, abs=0.1)
    assert np.mean(collin_hat) == pytest.approx(2.0, abs=0.4), "unbiased on average"
    assert np.std(collin_hat) > 5 * np.std(ortho_hat), "but far less precise"


def test_coefficient_stability_flags_sign_flips_under_collinearity():
    rng = np.random.default_rng(8)
    n = 4000
    a = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "twin": a + rng.normal(0, 0.02, n),
                      "clean": rng.normal(0, 1, n)})
    y = 2 * a + 0.5 * X["clean"] + rng.normal(0, 1, n)
    st = L.coefficient_stability(X, y, n_blocks=4)
    assert st.loc["clean", "sd_across_blocks"] < st.loc["a", "sd_across_blocks"]
    assert bool(st.loc["a", "sign_flips"]) or bool(st.loc["twin", "sign_flips"])
    assert not bool(st.loc["clean", "sign_flips"])


# --------------------------------------------------------------------------
# The consequence for reported significance
# --------------------------------------------------------------------------
def _ar1(n, rho, rng, scale=1.0):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, scale)
    return e


def test_hac_does_nothing_when_only_the_ERRORS_are_autocorrelated():
    """A correction to the intuition, and the reason the next test looks the way it does.

    HAC widens standard errors when the products x_t * e_t are serially correlated.
    With iid regressors that product is serially uncorrelated even if the errors are
    strongly AR(1), so HAC and the naive estimator agree. Autocorrelated errors alone
    are NOT sufficient — a fact worth knowing before quoting HAC as a reflex.
    """
    rng = np.random.default_rng(9)
    n = 3000
    X = pd.DataFrame({f"x{j}": rng.normal(0, 1, n) for j in range(4)})   # iid
    y = pd.Series(_ar1(n, 0.92, rng))
    plain, hac = L.ols_fit(X, y), L.ols_fit(X, y, cov="HAC", hac_lags=48)
    sh = L.significance_shift(plain, hac, X.columns)
    assert sh["median_se_inflation"] == pytest.approx(1.0, abs=0.15)


def test_significance_shift_counts_what_robust_errors_take_away():
    """The case that matches this project: regressors AND errors both autocorrelated.

    Every lag feature in the panel is strongly serially correlated, and so is the
    residual, so this is the regime notebook 04 is actually in — and here the naive
    standard errors are badly too small.
    """
    rng = np.random.default_rng(10)
    n = 3000
    # Autocorrelated regressors, exactly like net_load lags.
    X = pd.DataFrame({f"x{j}": _ar1(n, 0.95, rng) for j in range(6)})
    y = pd.Series(_ar1(n, 0.95, rng))          # NO real relationship to any regressor
    plain = L.ols_fit(X, y)
    hac = L.ols_fit(X, y, cov="HAC", hac_lags=48)
    sh = L.significance_shift(plain, hac, X.columns)
    assert sh["n_terms"] == 6
    assert sh["median_se_inflation"] > 1.5, "naive SEs must be visibly too small here"
    assert sh["n_significant_robust"] <= sh["n_significant_naive"]


# --------------------------------------------------------------------------
# Panel-specific traps
# --------------------------------------------------------------------------
def test_within_group_serial_stats_differ_from_row_order_on_a_stacked_panel():
    """The panel trap: adjacent ROWS are different units at the same instant.

    Built so the truth is unambiguous — each unit's residuals are strongly AR(1), but
    interleaving three units makes consecutive rows nearly independent. Row-order DW
    reports ~2 ("fine"); within-unit DW must report the autocorrelation that is
    actually there.
    """
    rng = np.random.default_rng(11)
    T, units = 900, 3
    series = {u: _ar1(T, 0.95, rng) for u in range(units)}
    rows = []
    for t in range(T):                      # time-major, exactly like the project panel
        for u in range(units):
            rows.append({"g": u, "t": t, "r": series[u][t]})
    d = pd.DataFrame(rows)
    X = pd.DataFrame({"x": rng.normal(0, 1, len(d))})
    y = pd.Series(d["r"].to_numpy())
    fit = L.ols_fit(X, y)

    naive = L.assumption_battery(fit, X)
    within = L.assumption_battery(fit, X, groups=d["g"], order=d["t"], resid_acf_lags=10)
    assert within["serial_stats_within_group"] is True
    assert within["n_groups_used"] == units
    assert naive["durbin_watson"] > 1.7, "row-order DW should look innocent"
    assert within["durbin_watson"] < 0.6, "within-unit DW must expose the AR(1)"
    assert within["resid_acf_lag1"] > 0.8


def test_design_matrix_drops_one_dummy_to_avoid_the_trap():
    """Six dummies plus an intercept is an exactly singular design."""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    NAMES = ["a", "zone_is_AAA", "zone_is_BBB", "zone_is_CCC"]

    class _Sel(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]   # a trailing underscore marks it fitted
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float)

        def get_feature_names_out(self, input_features=None):
            return np.array(NAMES)

    class _Model(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6],
                          "zone_is_AAA": [1.0, 0, 0, 1, 0, 0],
                          "zone_is_BBB": [0.0, 1, 0, 0, 1, 0],
                          "zone_is_CCC": [0.0, 0, 1, 0, 0, 1]})
    pipe = Pipeline([("select", _Sel()), ("model", _Model())])
    pipe.fit(frame, np.zeros(len(frame)))

    X = L.design_matrix(pipe, frame)
    assert "zone_is_AAA" not in X.columns, "the reference level must be dropped"
    assert {"zone_is_BBB", "zone_is_CCC"} <= set(X.columns)
    assert X.attrs["reference_levels"]["zone_is_"] == "zone_is_AAA"
    # And the surviving design must be non-singular once an intercept is added.
    assert L.condition_number(X) < 1e6


def test_within_r2_is_lower_than_total_r2_when_units_differ_in_level():
    """Most of a heterogeneous panel's variance is 'which unit is this?'."""
    rng = np.random.default_rng(12)
    n = 3000
    g = rng.choice(["small", "big"], n)
    level = np.where(g == "big", 50_000.0, 5_000.0)
    y = level + rng.normal(0, 500, n)
    resid = rng.normal(0, 500, n)           # a model that nails the within variation
    out = L.within_group_r2(y, resid, g)
    assert out["r2_total"] > 0.98
    assert out["r2_within"] < out["r2_total"]
    assert out["between_share_of_variance"] > 0.9
