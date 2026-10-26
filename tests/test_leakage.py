"""
test_leakage.py — the tests that answer "how do you know there is no leakage?"
with an assertion instead of a promise.

Every test here corresponds to a claim you will make out loud in the oral exam.
If one of them fails, the claim is false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import features as P


def test_history_buffer_never_contains_future(panel):
    """The lag buffer holds training rows only — the core structural claim."""
    cv = P.make_ts_cv(n_splits=3, gap=24)
    for tr, te in cv.split(panel):
        train, test = panel.iloc[tr], panel.iloc[te]
        lagger = P.GroupLagFeatures(lags=(1, 24), roll_windows=(24,)).fit(train)
        assert lagger.history_[cfg.TIME_COL].max() <= train[cfg.TIME_COL].max()
        assert lagger.history_[cfg.TIME_COL].max() < test[cfg.TIME_COL].min()


def _seam_nan_rate(out, source_frame):
    """Mean NaN rate of the lag columns, net of each source column's own gaps.

    A lag column is NaN for two unrelated reasons: the seam (no history reaches
    back far enough) and the source column simply being missing at that hour.
    Only the first is what the buffer controls, so the second is subtracted.

    `oil` is 50% missing across the panel and `coal` 16.7%, so without this the
    mean rate reports data coverage rather than buffer behaviour.
    """
    import numpy as np

    lag_cols = [c for c in out.columns if "_lag" in c]
    assert lag_cols, "no lag columns were produced at all"
    excess = []
    for c in lag_cols:
        source = c.rsplit("_lag", 1)[0]
        base = (source_frame[source].isna().mean()
                if source in source_frame.columns else 0.0)
        excess.append(max(0.0, out[c].isna().mean() - base))
    return float(np.mean(excess))


def test_buffer_fills_the_dev_holdout_seam(panel):
    """The seam the buffer exists for is the CONTIGUOUS one: notebook 04 fits on `dev`
    and predicts `hold`, which starts the very next hour. Without the buffer the
    first 168 hours of every zone in the holdout would be NaN — the rows you most
    want to score. With it, the NaN rate is ~0.

    (Inside cross-validation there is a deliberate `gap` between train and
    validation, so a lag shorter than the gap cannot be recovered there and those
    seam rows are imputed. That is the gap doing its job, not the buffer failing —
    `test_seam_nans_are_bounded_by_the_gap` pins the size of it.)"""
    cut = panel[cfg.TIME_COL].quantile(0.85)
    dev, hold = panel[panel[cfg.TIME_COL] <= cut], panel[panel[cfg.TIME_COL] > cut]
    lags = (1, 24)

    fitted = P.GroupLagFeatures(lags=lags, roll_windows=(24,)).fit(dev)
    with_buffer = _seam_nan_rate(fitted.transform(hold), hold)

    cold = P.GroupLagFeatures(lags=lags, roll_windows=(24,)).fit(hold)
    without = _seam_nan_rate(cold.transform(hold), hold)

    assert with_buffer < without, "the history buffer is not doing its job"
    assert with_buffer < 0.01, f"seam-attributable NaN rate is {with_buffer:.3f}"


def test_seam_nans_are_bounded_by_the_gap(panel):
    """Inside CV, unavoidable seam NaNs must be bounded — a large rate means the
    buffer is not being prepended at all."""
    gap, lags = 24, (1, 24)
    cv = P.make_ts_cv(n_splits=3, gap=gap)
    tr, te = list(cv.split(panel))[-1]
    fitted = P.GroupLagFeatures(lags=lags, roll_windows=(24,)).fit(panel.iloc[tr])
    rate = _seam_nan_rate(fitted.transform(panel.iloc[te]), panel.iloc[te])
    fold_hours = panel.iloc[te][cfg.TIME_COL].nunique()
    assert rate <= (gap + max(lags)) / fold_hours + 0.02, f"seam NaN rate {rate:.3f} is too high"


def test_with_history_fills_a_non_contiguous_seam_without_future(panel):
    """Notebook 06's case: fit on an early block, forecast a block that starts much
    later. The fitted buffer cannot reach the new block's lags; `with_history` rebuilds
    it from the observed frame before the block, and never from the block itself."""
    ts = panel[cfg.TIME_COL]
    c1, c2 = ts.quantile(0.55), ts.quantile(0.80)
    early, observed, later = panel[ts <= c1], panel[ts <= c2], panel[ts > c2]
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,))
    pipe.fit(early, early[cfg.TARGET])
    lag = pipe.named_steps["lag"]
    stale = _seam_nan_rate(lag.transform(later), later)
    P.with_history(pipe, observed)
    fresh = _seam_nan_rate(lag.transform(later), later)
    assert lag.history_[cfg.TIME_COL].max() <= observed[cfg.TIME_COL].max()
    assert lag.history_[cfg.TIME_COL].max() < later[cfg.TIME_COL].min()
    assert fresh < stale, "with_history did not supply the missing lags"
    assert fresh < 0.01, f"seam-attributable NaN rate is {fresh:.3f}"


def test_lag_values_are_actually_the_past(panel):
    """A lag24 column must equal the target 24 hours earlier for the same zone."""
    lagger = P.GroupLagFeatures(lags=(24,), roll_windows=()).fit(panel)
    out = lagger.transform(panel)
    z = panel[cfg.GROUP_COL].iloc[0]
    sub = out[out[cfg.GROUP_COL] == z].sort_values(cfg.TIME_COL)
    expected = sub[cfg.TARGET].shift(24)
    got = sub[f"{cfg.TARGET}_lag24"]
    mask = expected.notna() & got.notna()
    assert mask.sum() > 100
    assert np.allclose(expected[mask], got[mask])


def test_lags_survive_a_hole_in_the_timeline(panel):
    """Puncture the panel and check the lag is still the value 24 hours earlier.

    This is the regression test for the row-shift bug: `shift(24)` reaches back 24
    ROWS, which after an outage is not 24 hours, and nothing about the output
    looks wrong."""
    z = panel[cfg.GROUP_COL].iloc[0]
    sub = panel[panel[cfg.GROUP_COL] == z].sort_values(cfg.TIME_COL).reset_index(drop=True)
    punctured = sub.drop(index=range(200, 210)).reset_index(drop=True)  # a 10-hour hole

    out = P.GroupLagFeatures(lags=(24,), roll_windows=()).fit(punctured).transform(punctured)
    truth = punctured.set_index(cfg.TIME_COL)[cfg.TARGET]
    want = out[cfg.TIME_COL] - pd.Timedelta(hours=24)
    expected = truth.reindex(want).to_numpy()
    got = out[f"{cfg.TARGET}_lag24"].to_numpy()
    mask = ~np.isnan(expected) & ~np.isnan(got)
    assert mask.sum() > 100
    assert np.allclose(expected[mask], got[mask])
    # The hole must surface as NaN, not as a wrong number quietly filled in.
    assert np.isnan(got[np.isnan(expected)]).all()


def test_rolling_window_excludes_the_current_hour(panel):
    """rolling(24) on rows would include the value being predicted."""
    lagger = P.GroupLagFeatures(lags=(1,), roll_windows=(24,)).fit(panel)
    out = lagger.transform(panel)
    z = panel[cfg.GROUP_COL].iloc[0]
    sub = out[out[cfg.GROUP_COL] == z].sort_values(cfg.TIME_COL)
    # A rolling mean that included the current hour would correlate ~1.0 with it.
    r = sub[[cfg.TARGET, f"{cfg.TARGET}_roll24_mean"]].dropna()
    assert r.corr().iloc[0, 1] < 0.995


@pytest.mark.parametrize("horizon", [1, 24, 48, 72])
def test_rolling_window_ends_at_the_forecast_origin(panel, horizon):
    """The rolling window must stop at t-h, not at t-1.

    THE LEAK THIS PINS (found in notebook 04, 2026-08-14). `closed='left'` excludes the
    current hour and nothing else, so at h=24 the 24-hour rolling mean covered
    [t-24, t-1] — twenty-three hours that do not exist at a forecast origin of t-24.
    The test above passed throughout, because the window did exclude hour t; it was
    the twenty-three hours after the origin that nobody was checking.

    Asserted as an exact identity against the admissible window rather than as a
    correlation bound, because a correlation bound is what let the original slip
    through. At h=1 the two definitions coincide, which is why notebook 02's numbers
    survived the fix unchanged.
    """
    lags = cfg.lags_for(horizon)
    W = 24
    lagger = P.GroupLagFeatures(lags=lags, roll_windows=(W,)).fit(panel)
    out = lagger.transform(panel)
    z = panel[cfg.GROUP_COL].iloc[0]
    sub = out[out[cfg.GROUP_COL] == z].sort_values(cfg.TIME_COL).set_index(cfg.TIME_COL)

    checked = 0
    for t in sub.index[-40:]:
        got = sub[f"{cfg.TARGET}_roll{W}_mean"].loc[t]
        window = sub[cfg.TARGET].loc[
            t - pd.Timedelta(hours=W + horizon - 1): t - pd.Timedelta(hours=horizon)
        ]
        if np.isnan(got) or len(window) < W:
            continue
        assert got == pytest.approx(window.mean(), rel=1e-9), (
            f"h={horizon}: rolling mean at {t} is {got:,.3f}, but the admissible "
            f"window [t-{W + horizon - 1}, t-{horizon}] gives {window.mean():,.3f}"
        )
        checked += 1
    assert checked >= 10, f"only {checked} rows were checkable; the test proved little"


def test_the_rolling_fix_is_a_no_op_at_horizon_one(panel):
    """notebook 02 chose its feature set at h=1. If the fix moved h=1, that choice moved.

    This is the test that licenses saying "notebook 02's numbers are unchanged" out loud.
    """
    lagger = P.GroupLagFeatures(lags=(1, 24, 168), roll_windows=(24, 168))
    out = lagger.fit(panel).transform(panel)
    ind = out.sort_values([cfg.GROUP_COL, cfg.TIME_COL]).set_index(cfg.TIME_COL)
    for W in (24, 168):
        naive = (ind.groupby(cfg.GROUP_COL)[cfg.TARGET]
                 .rolling(f"{W}h", closed="left", min_periods=max(2, W // 4)).mean()
                 .reset_index(level=0, drop=True))
        got = out.sort_values([cfg.GROUP_COL, cfg.TIME_COL])[f"{cfg.TARGET}_roll{W}_mean"]
        both = pd.DataFrame({"a": naive.to_numpy(), "b": got.to_numpy()}).dropna()
        assert len(both) > 100
        assert np.allclose(both["a"], both["b"], rtol=1e-9)


def test_zero_lag_is_rejected():
    """A lag of 0 is the target itself. It must fail loudly, not quietly score 0.0 MAE."""
    import pytest
    with pytest.raises(ValueError):
        P.GroupLagFeatures(lags=(0, 24)).fit(pd.DataFrame({
            cfg.TIME_COL: pd.date_range("2023-01-01", periods=10, freq="h"),
            cfg.GROUP_COL: "CISO",
            cfg.TARGET: range(10),
        }))


def test_target_components_can_never_be_features(panel):
    """demand, wind and solar define net_load exactly. The selector must drop them."""
    sel = P.MutualInfoSelector(k="all").fit(panel, panel[cfg.TARGET])
    for banned in ("demand", "wind", "solar", cfg.TARGET):
        assert banned not in sel.selected_, f"{banned} survived into the feature set"


@pytest.mark.parametrize("banned", cfg.REALISED_GENERATION_COLS + ["renewable_share"])
def test_realised_generation_can_never_be_a_feature(panel, banned):
    """The nine fuel columns sum to the target; renewable_share IS the target.

    Notebook 02 audit, 2026-08-13. Measured on the real panel, the nine fuel columns
    reconstruct net load at r = 0.9997 (ERCO), and
    net_load == demand * (1 - renewable_share) to a residual of 0.0 MW. They are
    also same-hour realised values, unknowable at any forecast origin.

    Parameterised deliberately: a failure names the offending column rather than
    reporting that one of ten assertions in a loop went wrong.
    """
    sel = P.MutualInfoSelector(k="all").fit(panel, panel[cfg.TARGET])
    assert banned not in sel.selected_, (
        f"{banned!r} survived into the feature set. It is same-hour realised "
        f"generation and is part of the target's accounting identity."
    )


def test_no_unvetted_column_becomes_a_feature(panel):
    """The closed-world guard: nothing reaches the model without being vetted.

    The parameterised test above catches the columns we already know about. This
    one catches the NEXT one. EIA added `battery`, `unknown_storage` and
    `solar_battery` partway through this project's own window, and each arrived
    as a silently admissible feature. A new fuel code must fail this test rather
    than quietly improve the MAE.

    The rule: every selected feature is either engineered here (a lag, a calendar
    column, an encoding) or an explicitly vetted exogenous column. There is no
    third category.
    """
    sel = P.MutualInfoSelector(k="all").fit(panel, panel[cfg.TARGET])
    engineered = ("_lag", "_roll", "hour_", "dow_", "doy_", "month_", "is_", "zone_")
    unvetted = [
        c for c in sel.selected_
        if c not in cfg.WEATHER_FEATURES
        and c not in cfg.NUMERIC_FEATURES
        and not any(marker in c for marker in engineered)
    ]
    assert not unvetted, (
        f"unvetted panel columns reached the feature set: {unvetted}. Either add "
        f"them to cfg.WEATHER_FEATURES with a written justification, or to "
        f"cfg.REALISED_GENERATION_COLS if they are same-hour realised values."
    )


@pytest.mark.parametrize("h", sorted(cfg.LAGS_BY_HORIZON))
def test_every_generated_lag_is_observable_at_its_origin(panel, h):
    """EVERY lag column, exogenous included — not just the target's.

    `test_every_horizon_lag_set_is_observable_at_its_origin` checks the numbers in
    `cfg.LAGS_BY_HORIZON`. It cannot catch a lag the transformer invents on its own,
    and that is exactly what happened: `GroupLagFeatures` hardcoded the exogenous
    lag at 24 hours for every horizon, so the h=48 and h=72 pipelines carried
    `temp_lag24` — a value from 24 or 48 hours AFTER the forecast origin.

    This test reads the lag out of the produced column NAMES, so it constrains
    whatever the transformer actually built rather than what it was asked for.
    """
    lags = cfg.LAGS_BY_HORIZON[h]
    out = P.GroupLagFeatures(lags=lags).fit(panel).transform(panel)
    found = {c: int(c.rsplit("_lag", 1)[1]) for c in out.columns if "_lag" in c}
    assert found, "no lag columns were generated at all"
    offenders = {c: L for c, L in found.items() if L < h}
    assert not offenders, (
        f"h={h}: these lag columns are not observable at an origin {h} h before the "
        f"target: {offenders}. A lag L is admissible only if L >= h."
    )


def test_exog_lag_stays_aligned_to_the_daily_cycle(panel):
    """The exogenous lag must be a whole number of days, not merely >= h.

    Rounding the horizon up to the next multiple of 24 is what preserves the
    same-hour-of-day comparison that makes a temperature lag meaningful. Rounding to
    h exactly would give `temp_lag50` at h=50 — admissible, but comparing 09:00 with
    07:00 two days earlier, which is a different thing to know.
    """
    for h, expected in [(1, 24), (24, 24), (48, 48), (72, 72)]:
        got = P.GroupLagFeatures(lags=cfg.LAGS_BY_HORIZON[h])._exog_lag
        assert got == expected, f"h={h}: exog lag {got}, expected {expected}"
        assert got % 24 == 0


def test_per_zone_relative_learns_nothing(panel):
    """The whole safety argument for this family is that `fit` estimates nothing.

    If a future edit gives it a fitted parameter, it silently becomes a leak surface
    that has to be refit per fold. Fitting on two disjoint halves must give
    byte-identical output on the same rows.
    """
    step = P.PerZoneRelative()
    lagged = P.GroupLagFeatures(lags=cfg.LAGS_BY_HORIZON[1]).fit(panel).transform(panel)
    half = len(lagged) // 2
    a = step.fit(lagged.iloc[:half]).transform(lagged)
    b = step.fit(lagged.iloc[half:]).transform(lagged)
    pd.testing.assert_frame_equal(a, b)
    assert not [k for k in vars(step) if k.endswith("_")], (
        f"PerZoneRelative grew fitted state: {[k for k in vars(step) if k.endswith('_')]}"
    )


def test_per_zone_relative_is_the_stated_ratio(panel):
    """Each _rel column must be exactly its source over the zone's weekly mean."""
    lagged = P.GroupLagFeatures(lags=cfg.LAGS_BY_HORIZON[1]).fit(panel).transform(panel)
    step = P.PerZoneRelative()
    out = step.fit(lagged).transform(lagged)
    den = out[step.denom_col_]
    made = [c for c in out.columns if c.endswith("_rel")]
    assert made, "no relative features were produced"
    for c in made:
        src = c[: -len("_rel")]
        expected = out[src].to_numpy() / den.where(den > step.min_denom).to_numpy()
        np.testing.assert_allclose(out[c].to_numpy(), expected, equal_nan=True)
    # The denominator over itself is the constant 1 and carries nothing.
    assert f"{step.denom_col_}_rel" not in out.columns
    assert np.isfinite(out[made].to_numpy()[~np.isnan(out[made].to_numpy())]).all(), \
        "a relative feature is infinite - the denominator guard failed"


def test_per_zone_relative_respects_the_horizon(panel):
    """It divides existing lag columns, so it can never reintroduce a short lag."""
    for h in sorted(cfg.LAGS_BY_HORIZON):
        lagged = P.GroupLagFeatures(lags=cfg.LAGS_BY_HORIZON[h]).fit(panel).transform(panel)
        out = P.PerZoneRelative().fit(lagged).transform(lagged)
        found = {c: int(c.rsplit("_lag", 1)[1].split("_")[0])
                 for c in out.columns if "_lag" in c}
        bad = {c: L for c, L in found.items() if L < h}
        assert not bad, f"h={h}: relative features reintroduced short lags {bad}"


def test_weather_noise_is_off_by_default(panel):
    """The default pipeline must be byte-identical to one with no noise step.

    An assumption that silently moves a headline score is worse than no assumption.
    `WEATHER_NOISE_SCALE` is 0.0 and this test is what keeps it honest: if someone
    flips the default, every notebook result changes and this fails first.
    """
    assert cfg.WEATHER_NOISE_SCALE == 0.0, "the shipped default must be 0 (no noise)"
    step = P.WeatherForecastNoise(horizon=72)
    assert step.sigma_ == 0.0
    out = step.fit(panel).transform(panel)
    pd.testing.assert_frame_equal(out, panel)


@pytest.mark.parametrize("h,lead", sorted(cfg.NWP_LEAD_DAY_BY_HORIZON.items()))
def test_weather_noise_grows_with_the_horizon(panel, h, lead):
    """Injected error must follow the published curve, and h=1 must stay clean."""
    step = P.WeatherForecastNoise(horizon=h, scale=1.0)
    if lead == 0:                       # h=1: no published sub-daily value
        assert step.sigma_ == 0.0
        return
    expected = cfg.NWP_TEMP_MAE_C_BY_LEAD_DAY[lead] * cfg.MAE_TO_SIGMA
    assert step.sigma_ == pytest.approx(expected)
    out = step.fit(panel).transform(panel)
    # The realised spread should match the requested sigma, and nothing else moves.
    assert (out["temp"] - panel["temp"]).std() == pytest.approx(step.sigma_, rel=0.05)
    for c in ("humidity", "wind_speed", "cloud_cover", cfg.TARGET):
        pd.testing.assert_series_equal(out[c], panel[c])


def test_weather_noise_is_monotone_in_the_horizon():
    """h=24 < h=48 < h=72. If this inverts, the curve was entered wrongly."""
    s = [P.WeatherForecastNoise(horizon=h, scale=1.0).sigma_ for h in (24, 48, 72)]
    assert s == sorted(s) and s[0] > 0, s


def test_weather_noise_never_touches_the_lagged_column(panel):
    """The lag is observed at the origin. Degrading it would be a modelling error.

    This is the test that pins the step ORDER down: build the full pipeline with
    noise on, and confirm the lagged temperature is identical to the one produced
    with noise off, while the same-hour column is not.
    """
    lags = cfg.LAGS_BY_HORIZON[72]

    def through_weather(scale):
        """Run calendar -> lag -> weather in pipeline order, as the pipeline does."""
        pipe = P.make_supervised_pipeline(lags=lags, weather_noise_scale=scale)
        out = panel
        for name in ("calendar", "lag", "weather"):
            out = pipe.named_steps[name].fit(out, panel[cfg.TARGET]).transform(out)
        return out

    a, b = through_weather(0.0), through_weather(1.0)
    lag_col = f"temp_lag{P.GroupLagFeatures(lags=lags)._exog_lag}"
    assert lag_col in a.columns
    pd.testing.assert_series_equal(a[lag_col], b[lag_col])
    assert not a["temp"].equals(b["temp"]), "the same-hour column should be degraded"


def test_cv_folds_are_chronological_and_gapped(panel):
    """Validation always follows training, with a real gap measured in HOURS."""
    gap = 48
    cv = P.make_ts_cv(n_splits=3, gap=gap)
    for tr, te in cv.split(panel):
        train_max = panel.iloc[tr][cfg.TIME_COL].max()
        test_min = panel.iloc[te][cfg.TIME_COL].min()
        assert test_min > train_max
        assert (test_min - train_max) >= pd.Timedelta(hours=gap)


def test_no_timestamp_straddles_the_split(panel):
    """The panel-aware splitter must never put the same hour on both sides."""
    cv = P.make_ts_cv(n_splits=3, gap=24)
    for tr, te in cv.split(panel):
        assert not (set(panel.iloc[tr][cfg.TIME_COL]) & set(panel.iloc[te][cfg.TIME_COL]))


def test_target_encoder_is_refit_per_fold(panel):
    """Fitted inside the pipeline, the encoding must differ between folds."""
    cv = P.make_ts_cv(n_splits=3, gap=24)
    seen = []
    for tr, _ in cv.split(panel):
        enc = P.SmoothedTargetEncoder().fit(panel.iloc[tr], panel.iloc[tr][cfg.TARGET])
        seen.append(tuple(round(v, 4) for _, v in sorted(enc.mapping_.items())))
    assert len(set(seen)) > 1, "the encoder produced identical maps — it is not being refit"


def test_ramp_is_never_a_feature(panel):
    """ramp(t) = net_load(t) - net_load(t-1) contains the current target. It is a
    Notebook 01 descriptive column and a flexibility framing device, never an input."""
    from src import data as data_loader
    df = data_loader.derive_net_load(panel.drop(columns=["ramp"], errors="ignore"))
    sel = P.MutualInfoSelector(k="all").fit(df, df[cfg.TARGET])
    assert "ramp" not in sel.selected_


# ---------------------------------------------------------------- horizons
def test_every_horizon_lag_set_is_observable_at_its_origin():
    """notebook 01's contract, in assertion form.

    Forecasting net_load(t) from an origin at t-h means a lag of L hours is
    knowable only if L >= h. A lag below the horizon is information the
    operator does not have at the moment the forecast is issued; including it
    is leakage that looks exactly like skill. Lag 1 is the specific trap — it
    is the strongest predictor in the panel and it is legal ONLY at h=1.
    """
    assert set(cfg.LAGS_BY_HORIZON) == set(cfg.HORIZONS), (
        "HORIZONS and LAGS_BY_HORIZON disagree; every horizon needs a lag set"
    )
    for h, lags in cfg.LAGS_BY_HORIZON.items():
        assert lags, f"horizon {h}h has an empty lag set"
        assert min(lags) >= h, (
            f"horizon {h}h includes lag {min(lags)}, which is not observable "
            f"at the t-{h}h forecast origin"
        )
        assert list(lags) == sorted(set(lags)), (
            f"horizon {h}h lag set is not sorted/unique: {lags}"
        )


def test_lags_for_refuses_an_unknown_horizon():
    """A silent fallback to the h=1 lags at h=72 would hand the model 71 hours
    it cannot have, and the only symptom would be a suspiciously good score."""
    import pytest
    assert cfg.lags_for(24) == cfg.LAGS_BY_HORIZON[24]
    with pytest.raises(KeyError):
        cfg.lags_for(12)


@pytest.mark.parametrize("horizon", [1, 24, 48, 72])
def test_cv_gap_covers_the_rolling_reach_back(panel, horizon):
    """The gap must exceed how far a validation row's features actually reach.

    With the rolling window ending at the origin, the reach-back is
    max(ROLL_WINDOWS) + h - 1, so the flat CV_GAP = 168 is right only at h=1.
    """
    gap = cfg.cv_gap_for(horizon)
    lags = cfg.lags_for(horizon)
    reach = max(max(lags), max(cfg.ROLL_WINDOWS) + horizon - 1)
    assert gap >= reach, f"h={horizon}: gap {gap} < reach-back {reach}"
    assert cfg.cv_gap_for(1) == cfg.CV_GAP, "h=1 must keep the historical constant"


def test_split_gap_is_honoured_in_hours_not_rows(panel):
    """A gap set in hours must appear as that many hours on the timeline."""
    gap = cfg.cv_gap_for(72)
    cv = P.make_ts_cv(n_splits=3, gap=gap)
    ts = pd.to_datetime(panel[cfg.TIME_COL])
    for tr, va in cv.split(panel):
        delta = (ts.iloc[va].min() - ts.iloc[tr].max()) / pd.Timedelta(hours=1)
        assert delta >= gap, f"only {delta:.0f}h between train end and validation start"
