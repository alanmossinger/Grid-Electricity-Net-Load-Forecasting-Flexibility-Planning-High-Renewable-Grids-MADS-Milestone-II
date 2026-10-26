"""Contract tests for src/explain.py.

These pin a correction as much as a contract. The scaffold's reading rule for
Notebook 07 -- "where grouped >> sum of singles, the features are redundant
substitutes" -- does not hold: measured on exact and near-duplicate pairs across
RandomForest, HistGradientBoosting and Ridge, the ratio is ~1.00 every time, and
`test_grouped_over_sum_ratio_does_not_detect_redundancy` fails if that ever stops
being true. What actually identifies a substitute is a free rider -- a column
scoring ~0 alone inside a family that matters -- and that is pinned separately.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from src import explain as X


def _twin_frame(n=600, seed=0):
    """`a` and `a_twin` are near-identical and jointly drive y; `noise` does not."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({
        "net_load_lag24": a,
        "net_load_lag24_rel": a + rng.normal(scale=0.01, size=n),
        "hour_sin": rng.normal(size=n),
    }), 3.0 * a + rng.normal(scale=0.1, size=n)


def _fit(Xf, y):
    return RandomForestRegressor(n_estimators=30, random_state=0).fit(Xf, y)


def test_grouped_is_at_least_the_largest_single():
    """Permuting a superset cannot help the model.

    This is the property that actually holds. The scaffold's rule -- grouped >>
    sum of singles -- does NOT hold and is pinned as false in the test below.
    """
    Xf, y = _twin_frame()
    m = _fit(Xf, y)
    s = X.permutation_single(m, Xf.copy(), y, n_repeats=3, seed=1)
    g = X.permutation_grouped(m, Xf.copy(), y, n_repeats=3, seed=1)
    gap = X.redundancy_gap(s, g)
    lag = gap.loc[gap["family"] == "lag"].iloc[0]
    assert lag["mae_increase_mw"] >= lag["max_single_mw"] * 0.98


def test_grouped_over_sum_ratio_does_not_detect_redundancy():
    """Pins the correction: the ratio sits at ~1 even for exact duplicates.

    If a future change makes this ratio diverge from 1 for substitutes, the
    scaffold's original reading rule would be back in play and notebook 07's
    prose -- which states the opposite -- would need revisiting. Failing here is
    the intended way to find that out.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=1200)
    Xf = pd.DataFrame({"net_load_lag24": a, "net_load_lag24_rel": a,
                       "hour_sin": rng.normal(size=1200)})
    y = 3.0 * a + rng.normal(scale=0.1, size=1200)
    m = _fit(Xf, y)
    s = X.permutation_single(m, Xf.copy(), y, n_repeats=5, seed=1)
    g = X.permutation_grouped(m, Xf.copy(), y, n_repeats=5, seed=1)
    gap = X.redundancy_gap(s, g)
    ratio = gap.loc[gap["family"] == "lag", "ratio_grouped_over_sum"].iloc[0]
    assert 0.9 <= ratio <= 1.1, f"ratio {ratio} — see redundancy_gap's docstring"


def test_free_rider_is_the_diagnostic_that_works():
    """An exactly duplicated column under boosting scores ~0 alone and is free."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    a = rng.normal(size=1500)
    Xf = pd.DataFrame({"net_load_lag24": a, "net_load_lag24_rel": a,
                       "hour_sin": rng.normal(size=1500)})
    y = 3.0 * a + rng.normal(scale=0.1, size=1500)
    m = HistGradientBoostingRegressor(random_state=0, max_iter=120).fit(Xf, y)
    s = X.permutation_single(m, Xf.copy(), y, n_repeats=5, seed=1)
    g = X.permutation_grouped(m, Xf.copy(), y, n_repeats=5, seed=1)
    gap = X.redundancy_gap(s, g)
    lag = gap.loc[gap["family"] == "lag"].iloc[0]
    assert lag["n_free_riders"] >= 1
    assert lag["min_single_mw"] < 0.05 * lag["mae_increase_mw"]


def test_family_block_moves_under_one_row_order():
    """Independent shuffling would break the within-family correlation too.

    Permuting the block with one order must leave `corr(a, a_twin)` intact, so a
    second call with the same seed reproduces exactly and the measured importance
    is attributable to the family/target link alone.
    """
    Xf, y = _twin_frame()
    m = _fit(Xf, y)
    a = X.permutation_grouped(m, Xf.copy(), y, n_repeats=3, seed=7)
    b = X.permutation_grouped(m, Xf.copy(), y, n_repeats=3, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_permutation_leaves_the_input_frame_unmodified():
    Xf, y = _twin_frame()
    before = Xf.copy(deep=True)
    m = _fit(Xf, y)
    X.permutation_single(m, Xf, y, n_repeats=2, seed=0)
    X.permutation_grouped(m, Xf, y, n_repeats=2, seed=0)
    pd.testing.assert_frame_equal(Xf, before)


def test_irrelevant_feature_scores_near_zero():
    Xf, y = _twin_frame()
    m = _fit(Xf, y)
    s = X.permutation_single(m, Xf.copy(), y, n_repeats=3, seed=2)
    noise = s.loc[s["feature"] == "hour_sin", "mae_increase_mw"].iloc[0]
    lag = s.loc[s["feature"] == "net_load_lag24", "mae_increase_mw"].iloc[0]
    assert abs(noise) < abs(lag)


def test_worst_day_picks_the_injected_day_and_needs_a_full_day():
    ts = pd.date_range("2024-01-01", periods=24 * 4, freq="h")
    frame = pd.DataFrame({"timestamp": list(ts) * 2,
                          "zone": ["A"] * len(ts) + ["B"] * len(ts)})
    y = np.zeros(len(frame))
    pred = np.zeros(len(frame))
    bad = (frame["zone"] == "B") & (frame["timestamp"].dt.date
                                    == pd.Timestamp("2024-01-03").date())
    pred[bad.to_numpy()] = 500.0
    w = X.worst_day(frame, y, pred)
    assert w["zone"] == "B" and w["date"] == "2024-01-03"
    assert w["n_hours"] == 24
    assert w["mae_mw"] == pytest.approx(500.0)


def test_worst_day_ignores_a_short_but_terrible_partial_day():
    """A three-hour day with a huge error must not outrank a full bad day."""
    ts = list(pd.date_range("2024-01-01", periods=24, freq="h"))
    ts += list(pd.date_range("2024-01-02", periods=3, freq="h"))
    frame = pd.DataFrame({"timestamp": ts, "zone": ["A"] * len(ts)})
    y = np.zeros(len(frame))
    pred = np.where(frame["timestamp"].dt.day == 2, 9999.0, 100.0)
    w = X.worst_day(frame, y, pred)
    assert w["date"] == "2024-01-01" and w["n_hours"] == 24


def test_surrogate_rules_are_readable_and_count_the_leaves():
    rng = np.random.default_rng(0)
    Xf = pd.DataFrame({"f0": rng.normal(size=300), "f1": rng.normal(size=300)})
    labels = (Xf["f0"] > 0).astype(int)
    t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xf, labels)
    rules = X.surrogate_rules(t, Xf.columns)
    assert len(rules) == t.get_n_leaves()
    assert {"rule", "predicts", "purity", "n"} <= set(rules.columns)
    assert all(isinstance(r, str) and r for r in rules["rule"])
    assert rules["n"].sum() == len(Xf)


def test_grouped_uses_pipeline_families_so_lag_and_rel_stay_together():
    from src import features as P
    assert P.feature_family("net_load_lag24") == P.feature_family("net_load_lag24_rel")


def test_ale_recovers_a_known_linear_slope():
    """On y = 2x with an uncorrelated companion, ALE must have slope ~2."""
    rng = np.random.default_rng(0)
    Xf = pd.DataFrame({"temp": rng.uniform(-3, 3, 2000),
                       "hour_sin": rng.normal(size=2000)})
    y = 2.0 * Xf["temp"]
    m = _fit(Xf, y)
    a = X.ale_1d(m, Xf, "temp", n_bins=12)
    slope = np.polyfit(a["x"], a["ale"], 1)[0]
    assert slope == pytest.approx(2.0, rel=0.15)
    assert a["ale"].abs().min() < a["ale"].abs().max()


def test_ale_is_centred_and_covers_every_row():
    rng = np.random.default_rng(1)
    Xf = pd.DataFrame({"temp": rng.normal(size=1500),
                       "hour_sin": rng.normal(size=1500)})
    y = Xf["temp"] ** 2
    m = _fit(Xf, y)
    a = X.ale_1d(m, Xf, "temp", n_bins=10)
    assert a["n"].sum() == len(Xf)
    weighted = float((a["ale"] * a["n"]).sum() / a["n"].sum())
    assert abs(weighted) < 1e-6


def test_ale_refuses_a_constant_feature():
    Xf = pd.DataFrame({"temp": np.ones(200), "hour_sin": np.arange(200.0)})
    m = _fit(Xf, Xf["hour_sin"])
    with pytest.raises(ValueError, match="too few distinct"):
        X.ale_1d(m, Xf, "temp")
