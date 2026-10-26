"""Pins on `src.reserve`: the four corrections, and the guard that caught the drift.

WHY THESE TESTS. The reserve valuation was wrong in four independent ways and every one
of them was invisible in the output: a quantile of |error| looks exactly like a quantile
of the signed error, a pooled quantile looks exactly like a sum of per-zone quantiles,
and a rolling h=24 score looks exactly like a day-ahead one. Nothing in the numbers
announces which you are looking at. So each correction gets a test that fails if the
code reverts to the wrong quantity, built on synthetic data where the right answer is
known by construction rather than by reading it off the holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import reserve as RS

TIME, GROUP, TARGET = cfg.TIME_COL, cfg.GROUP_COL, cfg.TARGET
_PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


# ---------------------------------------------------------------------------
# a synthetic holdout whose error distributions are known by construction
# ---------------------------------------------------------------------------
@pytest.fixture
def biased_frame() -> pd.DataFrame:
    """Model error symmetric; benchmark error shifted to OVER-forecast by 1,000 MW.

    This is the real situation in miniature. The EIA composite over-forecasts net load,
    so its |error| tail is dominated by errors on the down side -- the side upward
    reserve is never held against. A valuation built on |error| therefore credits the
    model for curing a bias the product does not pay for, and this fixture makes that
    difference arithmetic rather than rhetorical.
    """
    rng = np.random.default_rng(0)
    n = 4000
    rows = []
    for zone in ("CISO", "MISO"):
        actual = rng.normal(20_000, 3_000, n)
        rows.append(pd.DataFrame({
            GROUP: zone,
            TIME: pd.date_range("2024-02-07", periods=n, freq="h"),
            TARGET: actual,
            "y_pred": actual + rng.normal(0, 1_000, n),        # symmetric
            "v2": actual + rng.normal(1_000, 1_000, n),        # over-forecasts
            "persist1": actual + rng.normal(0, 2_000, n),
        }))
    return pd.concat(rows, ignore_index=True)


def test_upward_reserve_uses_the_signed_tail_not_the_absolute_one(biased_frame):
    """Correction 1, and the one that halved the valuation.

    Against a benchmark that over-forecasts, the |error| saving must come out LARGER
    than the signed upward saving. If they are equal, the signed path is not being
    taken.
    """
    per = RS.tails(biased_frame, 0.95, "v2")
    signed, absolute = per["up_saved"].sum(), per["abs_saved"].sum()
    assert absolute > signed + 500, (
        f"the |error| saving ({absolute:,.0f} MW) should exceed the signed upward saving "
        f"({signed:,.0f} MW) against an over-forecasting benchmark; if they agree, "
        "`tails` is not distinguishing the two tails")


def test_the_two_signed_tails_are_not_interchangeable(biased_frame):
    """A benchmark that over-forecasts has a SMALL up-tail and a LARGE down-tail."""
    per = RS.tails(biased_frame, 0.95, "v2")
    for zone in per.index:
        assert per.loc[zone, "down_bench"] > per.loc[zone, "up_bench"], (
            f"{zone}: this benchmark over-forecasts by construction, so its down-tail "
            "must be the larger one")
        assert per.loc[zone, "abs_bench"] >= per.loc[zone, "up_bench"], (
            f"{zone}: a quantile of |error| cannot be below the same quantile of the "
            "upward error")


def test_the_six_zone_figure_is_a_sum_of_per_zone_quantiles(biased_frame):
    """Correction 2. A pooled quantile is a different number and must not be summed."""
    per = RS.tails(biased_frame, 0.95, "v2")
    pooled = float(np.quantile(biased_frame[TARGET] - biased_frame["y_pred"], 0.95))
    assert per["up_model"].sum() > pooled, (
        "the sum of per-zone p95s should exceed the pooled p95 on a multi-zone panel; "
        "the two were mixed in the first version of this table")


def test_the_one_hour_benchmark_is_persistence_not_the_day_ahead_composite():
    """Correction 4: an information-set match, not a horizon label."""
    assert RS.bench_col("1") == "persist1"
    for key in ("24", "48", "72", RS.FIXED_ORIGIN):
        assert RS.bench_col(key) == "v2", f"{key} must be scored against the composite"


def test_the_coverage_axis_moves_the_requirement_the_right_way(biased_frame):
    """A higher coverage target can only ask for more capacity, never less."""
    reqs = [RS.tails(biased_frame, c, "v2")["up_model"].sum() for c in RS.COVERAGES]
    assert reqs == sorted(reqs), f"requirement must rise with coverage, got {reqs}"


def test_value_per_year_is_megawatts_times_price_times_hours():
    per = pd.DataFrame({"up_saved": [100.0, 200.0]}, index=["CISO", "MISO"])
    prices = {"CISO": 2.0, "MISO": 3.0}
    assert RS.value_per_year(per, prices) == pytest.approx(
        (100 * 2 + 200 * 3) * RS.HOURS_PER_YEAR)


# ---------------------------------------------------------------------------
# the fixed-origin lead-time arithmetic
# ---------------------------------------------------------------------------
def test_a_day_ahead_issuance_spans_the_expected_lead_times():
    """Correction 3. One issue time on D-1 means leads of 14 to 37 h, not a flat 24."""
    frame = pd.DataFrame({
        GROUP: ["CISO"] * 24,
        TIME: pd.date_range("2024-06-01 08:00", periods=24, freq="h"),
    })
    lead = RS.lead_hours(frame, issue_hour=10)
    assert lead.min() == 14 and lead.max() == 37, (
        f"leads should run 14-37 h for a 10:00 D-1 issuance, got "
        f"{lead.min()}-{lead.max()}")
    assert (lead > 24).any() and (lead <= 24).any(), (
        "a real issuance straddles the 24-hour mark, which is why the fixed-origin "
        "frame needs two horizons")


def test_local_hour_refuses_a_zone_it_has_no_timezone_for():
    frame = pd.DataFrame({GROUP: ["ATLANTIS"], TIME: [pd.Timestamp("2024-06-01")]})
    with pytest.raises(ValueError, match="ZONE_TZ"):
        RS.local_hour(frame)


# ---------------------------------------------------------------------------
# the ledger guard
# ---------------------------------------------------------------------------
def _guard_frame(mae_target: float) -> dict:
    n = 1000
    actual = np.full(n, 20_000.0)
    return {str(h): pd.DataFrame({
        GROUP: "CISO", TIME: pd.date_range("2024-02-07", periods=n, freq="h"),
        TARGET: actual, "y_pred": actual + mae_target,
        "v2": actual + 100.0, "naive168": actual + 200.0,
    }) for h in RS.HORIZONS}


def test_the_guard_passes_when_the_rebuild_matches():
    want = {str(h): {"model_mae": 50.0, "composite_v2_mae": 100.0,
                     "naive_mae": 200.0, "n": 1000} for h in RS.HORIZONS}
    RS.check_against_ledger(_guard_frame(50.0), want)          # must not raise


def test_the_guard_catches_a_drifted_number():
    """The defect that started all of this: a report a megawatt away from its notebook."""
    want = {str(h): {"model_mae": 50.0, "composite_v2_mae": 100.0,
                     "naive_mae": 200.0, "n": 1000} for h in RS.HORIZONS}
    with pytest.raises(AssertionError, match="ledger says"):
        RS.check_against_ledger(_guard_frame(51.5), want)


def test_the_guard_catches_a_changed_row_filter():
    """An eighteen-row difference in a row rule is exactly what it caught in practice."""
    frames = _guard_frame(50.0)
    frames["24"] = frames["24"].iloc[:-18]
    want = {str(h): {"model_mae": 50.0, "composite_v2_mae": 100.0,
                     "naive_mae": 200.0, "n": 1000} for h in RS.HORIZONS}
    with pytest.raises(AssertionError, match="row filters have diverged"):
        RS.check_against_ledger(frames, want)


# ---------------------------------------------------------------------------
# the equal-risk comparison
# ---------------------------------------------------------------------------
def test_scaling_a_band_hits_the_target_shortfall_rate():
    """Without this the flat-versus-conditional comparison is rigged by width."""
    rng = np.random.default_rng(1)
    err = pd.Series(rng.normal(0, 1_000, 20_000))
    band = pd.Series(np.full(20_000, 1_000.0))
    s = RS.scale_to_shortfall(err, band, target=0.05)
    assert float((err > s * band).mean()) == pytest.approx(0.05, abs=0.002)


def test_a_wider_band_is_short_less_often():
    """The monotonicity the bisection relies on."""
    rng = np.random.default_rng(2)
    err = pd.Series(rng.normal(0, 1_000, 5_000))
    band = pd.Series(np.full(5_000, 1_000.0))
    rates = [float((err > k * band).mean()) for k in (0.5, 1.0, 2.0, 3.0)]
    assert rates == sorted(rates, reverse=True), rates


def test_equal_risk_comparison_equalises_the_risk():
    rng = np.random.default_rng(3)
    n = 6_000
    hourly = pd.concat([pd.DataFrame({
        GROUP: z, "err_up": rng.normal(0, sd, n),
        "band_up": np.full(n, sd * 1.5), "local_hour": np.tile(np.arange(24), n // 24),
        "regime": "conventional",
    }) for z, sd in (("CISO", 1_000.0), ("ERCO", 4_000.0))], ignore_index=True)
    cmp = RS.equal_risk_comparison(hourly)
    assert np.allclose(cmp["shortfall_flat"], 0.05, atol=0.003)
    assert np.allclose(cmp["shortfall_conditional"], 0.05, atol=0.003)


def test_hourly_with_band_refuses_a_misaligned_prediction_file():
    q = pd.DataFrame({TIME: pd.to_datetime(["2024-02-07"]), GROUP: ["CISO"],
                      "actual": [20_000.0], "cqr_hi": [23_000.0]})
    preds = pd.DataFrame({TIME: pd.to_datetime(["2024-02-08"]), GROUP: ["CISO"],
                          "horizon": [24], "y_pred": [19_000.0]})
    with pytest.raises(AssertionError, match="do not align"):
        RS.hourly_with_band(preds, q, pd.DataFrame(
            {GROUP: ["CISO"], "date": pd.to_datetime(["2024-02-07"]), "cluster": [1]}),
            {"1": "conventional"})


# ---------------------------------------------------------------------------
# regimes against zones
# ---------------------------------------------------------------------------
def test_regime_by_zone_counts_zone_days():
    labels = pd.DataFrame({
        GROUP: ["CISO"] * 3 + ["BPAT"] * 2,
        "date": pd.to_datetime(["2024-01-0" + str(i) for i in (1, 2, 3, 1, 2)]),
        "cluster": [2, 2, 1, 0, 0],
    })
    ct = RS.regime_by_zone(labels, {"0": "winter-shaped", "1": "conventional", "2": "DUCK"})
    assert ct.loc["CISO", "DUCK"] == 2 and ct.loc["BPAT", "winter-shaped"] == 2


def test_within_zone_deviations_are_centred_on_each_zone():
    """The point of the within-zone check: geography cannot carry the contrast.

    Each zone's count-weighted deviations must sum to zero, or the baseline is not that
    zone's own mean and the check does not do what it claims.
    """
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    panel = pd.DataFrame({
        GROUP: np.repeat(["CISO", "BPAT"], 200 * 24),
        TIME: np.tile(pd.date_range("2024-01-01", periods=200 * 24, freq="h"), 2),
    })
    panel["temp"] = rng.normal(15, 8, len(panel))
    panel["cloud_cover"] = rng.uniform(0, 100, len(panel))
    panel["ghi"] = rng.uniform(0, 400, len(panel))
    labels = pd.DataFrame({
        GROUP: np.repeat(["CISO", "BPAT"], 200),
        "date": np.tile(dates, 2),
        "cluster": np.tile(rng.integers(0, 3, 200), 2),
    })
    within = RS.regime_weather_within_zone(
        panel, labels, {"0": "winter-shaped", "1": "conventional", "2": "DUCK"})
    for zone, d in within.groupby("zone"):
        for col in ("temp_vs_zone_mean", "cloud_cover_vs_zone_mean", "ghi_vs_zone_mean"):
            weighted = float((d[col] * d["zone_days"]).sum() / d["zone_days"].sum())
            assert weighted == pytest.approx(0.0, abs=1e-6), f"{zone}/{col}"


# ---------------------------------------------------------------------------
# one integration test against the committed artifacts
# ---------------------------------------------------------------------------
def test_the_module_reproduces_the_notebooks_per_horizon_table():
    """The end-to-end version of the guard, on the real holdout."""
    for name in ("panel.parquet", "results.json", "holdout_predictions_by_horizon.parquet"):
        if not (_PROC / name).exists():
            pytest.skip(f"{name} has not been produced")
    ledger = json.loads((_PROC / "results.json").read_text())
    _, guard = RS.build_scored_frames(
        pd.read_parquet(_PROC / "panel.parquet"), ledger["nb04"]["cut"],
        pd.read_parquet(_PROC / "holdout_predictions_by_horizon.parquet"))
    RS.check_against_ledger(guard, ledger["nb04"]["per_horizon_skill"])
