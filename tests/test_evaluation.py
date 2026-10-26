"""
test_evaluate.py — the comparison machinery, checked against answers known from
theory rather than from this codebase.

The recurring failure mode in this module is a test that assumes independent
observations applied to autocorrelated data. Several tests below therefore plant
autocorrelation deliberately and assert that the naive method is fooled and the
blocked one is not. If those ever start agreeing, the blocking has been dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluation as E


# --------------------------------------------------------------------------
# Skill
# --------------------------------------------------------------------------
def test_skill_score_signs():
    assert E.skill_score(75, 100) == pytest.approx(0.25)
    assert E.skill_score(100, 100) == pytest.approx(0.0)
    assert E.skill_score(150, 100) == pytest.approx(-0.5)


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
def _ar1(n, rho, scale, rng):
    """An AR(1) error series — the shape real forecast errors actually have."""
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, scale)
    return e


def test_dm_finds_no_difference_between_identical_forecasts():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 2000)
    out = E.diebold_mariano(e, e.copy())
    assert out["dm"] == pytest.approx(0.0, abs=1e-9)
    assert out["p"] == pytest.approx(1.0)
    assert out["better"] == "tie"


def test_dm_detects_a_genuinely_better_forecast_and_names_it():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 1.0, 3000)
    bad = rng.normal(0, 1.6, 3000)
    out = E.diebold_mariano(good, bad)
    assert out["better"] == "a"
    assert out["dm"] < 0, "a negative statistic must mean A is better"
    assert out["p"] < 0.001


def test_dm_hac_variance_grows_with_the_horizon():
    """The whole reason DM exists: correlated errors need a wider standard error.

    With autocorrelated loss differentials, summing autocovariances out to lag h-1
    must inflate the variance, which shrinks |DM| and raises the p-value. A test
    that ignored the correlation would claim more significance than the data holds.
    """
    rng = np.random.default_rng(2)
    a = _ar1(4000, 0.9, 1.0, rng)
    b = a + rng.normal(0.15, 0.3, 4000)      # b is slightly worse, correlated with a
    h1 = E.diebold_mariano(a, b, horizon=1)
    h24 = E.diebold_mariano(a, b, horizon=24)
    assert h24["hac_variance"] > h1["hac_variance"]
    assert abs(h24["dm"]) < abs(h1["dm"])
    assert h24["p"] > h1["p"], "ignoring autocorrelation must look more significant"


def test_dm_small_sample_correction_is_conservative():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 300)
    b = a + rng.normal(0.1, 0.5, 300)
    raw = E.diebold_mariano(a, b, horizon=24, small_sample=False)
    hln = E.diebold_mariano(a, b, horizon=24, small_sample=True)
    assert abs(hln["dm"]) <= abs(raw["dm"]) + 1e-12
    assert hln["p"] >= raw["p"]


def test_dm_rejects_mismatched_or_tiny_inputs():
    with pytest.raises(ValueError, match="same length"):
        E.diebold_mariano(np.zeros(50), np.zeros(40))
    with pytest.raises(ValueError, match="at least 10"):
        E.diebold_mariano(np.zeros(5), np.zeros(5))


# --------------------------------------------------------------------------
# Blocked Wilcoxon
# --------------------------------------------------------------------------
def test_blocked_wilcoxon_is_not_fooled_by_autocorrelation():
    """The test this module exists to get right.

    Errors are identical in distribution but arrive in day-long runs. Treating the
    hours as independent finds a "significant" difference that is not there; blocking
    by day does not. If this ever fails, the blocking has stopped working and every
    model comparison in notebook 04 is overstated.
    """
    rng = np.random.default_rng(4)
    n_days, per_day = 60, 24
    blocks = np.repeat(np.arange(n_days), per_day)
    # One shared shock per day: within a day every hour moves together.
    day_shock = rng.normal(0, 1.0, n_days)
    a = np.abs(np.repeat(day_shock, per_day) + rng.normal(0, 0.05, n_days * per_day))
    b = a + np.repeat(rng.normal(0.02, 0.9, n_days), per_day)   # no systematic edge

    blocked = E.blocked_wilcoxon(a, b, blocks)
    assert blocked["n_blocks"] == n_days

    from scipy import stats
    naive_p = stats.wilcoxon(np.abs(a) - np.abs(b)).pvalue
    assert naive_p < blocked["p"], (
        "the unblocked test should look more significant than the blocked one; "
        f"naive {naive_p:.2e} vs blocked {blocked['p']:.2e}"
    )


def test_blocked_wilcoxon_still_finds_a_real_difference():
    rng = np.random.default_rng(5)
    blocks = np.repeat(np.arange(80), 24)
    a = np.abs(rng.normal(0, 1.0, 80 * 24))
    b = a + 0.9                                  # b is worse in every single hour
    out = E.blocked_wilcoxon(a, b, blocks)
    assert out["better"] == "a"
    assert out["p"] < 0.001
    assert out["wins_a"] == 80


def test_blocked_wilcoxon_refuses_too_few_blocks():
    with pytest.raises(ValueError, match="blocks"):
        E.blocked_wilcoxon(np.arange(10.0), np.arange(10.0) + 1,
                           np.repeat(np.arange(2), 5))


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------
def test_bh_matches_a_hand_worked_example():
    """m=5, alpha=0.05. Thresholds are i/m*alpha = .01 .02 .03 .04 .05.

    Sorted p: .001 .008 .039 .041 .900
    .001<=.01 yes; .008<=.02 yes; .039<=.03 no; .041<=.04 no; .9<=.05 no.
    Step-up: the largest i that passes is 2, so the first TWO are rejected.
    """
    p = [0.900, 0.008, 0.041, 0.001, 0.039]
    out = E.benjamini_hochberg(p, alpha=0.05)
    assert list(out["reject"]) == [False, True, False, True, False]


def test_bh_is_less_conservative_than_bonferroni():
    p = [0.001, 0.012, 0.02, 0.03, 0.04]
    bh = E.benjamini_hochberg(p, alpha=0.05)["reject"].sum()
    bonferroni = sum(x <= 0.05 / len(p) for x in p)
    assert bh > bonferroni


def test_bh_adjusted_p_values_are_monotone_and_bounded():
    rng = np.random.default_rng(6)
    p = rng.uniform(0, 1, 40)
    out = E.benjamini_hochberg(p).sort_values("p")
    assert out["p_adjusted"].is_monotonic_increasing
    assert (out["p_adjusted"] <= 1.0).all() and (out["p_adjusted"] >= 0).all()
    assert (out["p_adjusted"] >= out["p"] - 1e-12).all()


def test_bh_rejects_nothing_when_everything_is_null():
    rng = np.random.default_rng(7)
    assert E.benjamini_hochberg(rng.uniform(0.3, 1.0, 20))["reject"].sum() == 0


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_truth_and_is_reproducible():
    rng = np.random.default_rng(8)
    v = rng.normal(5.0, 1.0, 3000)
    a = E.bootstrap_ci(v, n_boot=400)
    b = E.bootstrap_ci(v, n_boot=400)
    assert a["lo"] < 5.0 < a["hi"]
    assert a["estimate"] == pytest.approx(v.mean())
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"]), "must be deterministic"


def test_block_bootstrap_is_wider_than_the_naive_one_on_correlated_data():
    """Same data, same statistic; ignoring the blocks buys a falsely tight interval."""
    rng = np.random.default_rng(9)
    blocks = np.repeat(np.arange(50), 24)
    v = np.repeat(rng.normal(0, 1.0, 50), 24) + rng.normal(0, 0.02, 50 * 24)
    naive = E.bootstrap_ci(v, n_boot=500)
    blocked = E.bootstrap_ci(v, n_boot=500, blocks=blocks)
    assert (blocked["hi"] - blocked["lo"]) > 3 * (naive["hi"] - naive["lo"])


# --------------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------------
def test_learning_curve_takes_the_most_recent_slice_and_never_the_validation_block():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=2000, freq="h"),
        "y": np.arange(2000, dtype=float),
    })
    seen = {}

    def fit_predict(train, valid):
        seen[len(train)] = (train["timestamp"].min(), train["timestamp"].max())
        assert train["timestamp"].max() < valid["timestamp"].min(), "leak across the cut"
        return 1.0, 1.0

    out = E.chronological_learning_curve(fit_predict, frame, sizes=(0.25, 0.5, 1.0))
    assert list(out["fraction"]) == [0.25, 0.5, 1.0]
    assert out["n_rows"].is_monotonic_increasing
    # Smaller slices must START LATER — that is what "most recent" means.
    starts = out.sort_values("fraction")["train_start"].tolist()
    assert starts[0] > starts[1] > starts[2]


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------
def test_ablation_ranks_families_by_damage_done_and_keeps_the_sign():
    scores = {"lag": 900.0, "calendar": 620.0, "weather": 590.0, "zone": 598.0}
    out = E.ablation_by_family(lambda f: scores[f], list(scores), baseline=600.0)
    assert out.iloc[0]["family"] == "lag"
    assert out.iloc[0]["delta_vs_full"] == pytest.approx(300.0)
    # A family the model is BETTER without must keep its negative sign, not be clipped.
    assert out.set_index("family").loc["weather", "delta_vs_full"] == pytest.approx(-10.0)


def test_dm_on_a_time_major_panel_needs_the_groups():
    """Six autocorrelated series interleaved hour by hour, as the holdout is sorted. Without
    the groups, lags 1..23 rows reach only about four hours into each series and the HAC
    variance is too small; with them the lag products stay inside each series."""
    rng = np.random.default_rng(0)
    n, k = 2000, 6
    loss_gap = np.zeros((n, k))
    for j in range(k):
        e = rng.normal(size=n)
        for t in range(1, n):
            e[t] = 0.9 * e[t - 1] + e[t]
        loss_gap[:, j] = e
    a = np.abs(loss_gap).ravel() + 0.05          # row order: time-major
    b = np.full_like(a, a.mean())
    groups = np.tile(np.arange(k), n)
    times = np.repeat(np.arange(n), k)
    naive = E.diebold_mariano(a, b, horizon=24)
    grouped = E.diebold_mariano(a, b, horizon=24, groups=groups, times=times)
    assert grouped["hac_variance"] > 1.5 * naive["hac_variance"]
    assert abs(grouped["dm"]) < abs(naive["dm"])
