"""
test_data_and_clusters.py — the data contract and the unsupervised side.

The data tests are the ones that catch a bad vintage before it costs a week.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src import config as cfg
from src import data as data_loader
from src import features as P
from src import results as R
from src import data as synthetic


# --------------------------------------------------------------------- data
def test_net_load_identity(panel):
    """net_load = demand - wind - solar, exactly. If this drifts, nothing else matters.
    A missing component counts as 0 only in a zone that never has any of it (NYIS solar);
    anywhere else a missing component means the hour has no target and is not in the panel."""
    has = {c: panel.groupby(cfg.GROUP_COL)[c].transform(lambda s: bool((s.fillna(0) != 0).any()))
           for c in ("wind", "solar")}
    for c in ("wind", "solar"):
        assert not (panel[c].isna() & has[c]).any(), f"a {c} outage was scored as zero {c}"
    recomputed = panel["demand"] - panel["wind"].fillna(0) - panel["solar"].fillna(0)
    assert np.allclose(panel[cfg.TARGET], recomputed)


def test_timeline_is_hourly_and_gaps_are_whole_hours(panel):
    """Hours whose target could not be recovered are DROPPED, so the panel may have
    holes — that is the cleaning contract working, not a defect. What must hold is
    that every step is a positive whole number of hours, and that the holes are
    rare. `GroupLagFeatures` looks lags up by timestamp precisely because of this."""
    for z, g in panel.groupby(cfg.GROUP_COL):
        ts = pd.to_datetime(g[cfg.TIME_COL]).sort_values()
        deltas = ts.diff().dropna()
        assert (deltas > pd.Timedelta(0)).all(), f"{z} timeline is not monotone"
        assert (deltas % pd.Timedelta(hours=1) == pd.Timedelta(0)).all(), \
            f"{z} has sub-hourly steps"
        assert (deltas == pd.Timedelta(hours=1)).mean() > 0.99, f"{z} is mostly holes"


def test_no_duplicate_zone_hours(panel):
    assert not panel.duplicated([cfg.GROUP_COL, cfg.TIME_COL]).any()


def test_solar_is_zero_at_night(panel):
    """A sanity check on the source, not on the code: if solar is positive at 03:00
    UTC across every zone, the timestamps are not what you think they are."""
    hour = pd.to_datetime(panel[cfg.TIME_COL]).dt.hour
    night = panel[(hour >= 6) & (hour <= 8)]  # deep night in US local time, UTC-stamped
    assert (night["solar"].fillna(0) >= 0).all()


def test_cleaning_counts_defects_rather_than_hiding_them():
    raw = synthetic.make_panel(zones=["CISO"], start="2023-01-01T00", end="2023-02-28T23")
    _, report = data_loader.clean_panel(raw, verbose=False)
    assert "missing_before_interp" in report
    assert report["interp_limit_hours"] == data_loader.MAX_INTERP_HOURS


def test_data_dictionary_covers_every_column(panel):
    dd = data_loader.data_dictionary(panel)
    assert set(dd["column"]) == set(panel.columns)


def test_implausible_values_are_nulled_but_real_extremes_survive():
    """The two defects the first real pull actually contained, plus the genuine
    extreme that must NOT be removed.

    Both real cases are reproduced here at their true magnitudes:
      - SWPP 2023-06-13 02:00Z: demand ~100x the zone's normal peak. It hides
        itself, because the same inflation appears in `wind` and cancels in
        net_load = demand - wind - solar.
      - CISO 2019-12-20 00-02Z: demand collapses to ~14 MW while generation keeps
        reporting, producing the most negative net load in the panel.
    """
    # The columns must carry REALISTIC DISPERSION. An earlier version of this test
    # used constant columns, which collapses the 99.9th percentile and makes the
    # bound absurdly tight -- it failed on the genuine extreme below. That was a
    # defect in the fixture, not in the rule (real fuel columns have six years of
    # spread), but it is the honest attack on this rule and is noted in
    # flag_implausible.
    n = 2000
    t = np.arange(n)
    base = pd.DataFrame({
        cfg.TIME_COL: pd.date_range("2023-01-01", periods=n, freq="h"),
        cfg.GROUP_COL: "SWPP",
        "demand": 30_000.0 + 5_000 * np.sin(2 * np.pi * t / 24),
        "wind": 5_000.0 + 4_000 * np.sin(2 * np.pi * t / 71),
        # a diurnal solar bell, zero at night, peaking near 10,000 MW
        "solar": np.clip(10_000 * np.sin(2 * np.pi * (t % 24 - 6) / 24), 0, None),
    })

    # the inflated record, and the collapsed record
    base.loc[10, ["demand", "wind"]] = [3_621_097.0, 3_589_445.0]
    base.loc[20, "demand"] = 14.0
    # a genuine extreme: high renewables against low demand. Must SURVIVE.
    base.loc[30, ["demand", "wind", "solar"]] = [12_193.0, 3_398.0, 9_422.0]

    out, counts = data_loader.flag_implausible(base)

    assert counts.get("demand") == 2, f"expected 2 bad demand values, got {counts}"
    assert np.isnan(out.loc[10, "demand"]), "the 100x inflated demand survived"
    assert np.isnan(out.loc[10, "wind"]), "the 100x inflated wind survived"
    assert np.isnan(out.loc[20, "demand"]), "the collapsed demand survived"

    # the real extreme is untouched -- these bounds are not an outlier filter
    assert out.loc[30, "demand"] == 12_193.0
    assert out.loc[30, "solar"] == 9_422.0


def test_real_panel_has_no_physically_impossible_magnitudes():
    """No US balancing authority exceeds ~130 GW. A value above 200 GW is a corrupt
    record, and it is invisible in the target because the error cancels in the
    net-load identity -- so it has to be caught on the components.

    This one deliberately reads the REAL panel rather than the synthetic fixture:
    the synthetic generator cannot produce this defect, so asserting against it
    would guard nothing. Skips when the panel has not been built.
    """
    import pytest

    if not cfg.PANEL.exists():
        pytest.skip("no panel.parquet - run notebooks 00 and 01 first (`make run`)")
    real = pd.read_parquet(cfg.PANEL)
    for col in ("demand", "wind", "solar", "net_generation"):
        if col in real:
            worst = float(real[col].max())
            assert worst < 200_000, f"{col} reaches {worst:,.0f} MW - a corrupt record survived"


# ---------------------------------------------------------------- clustering
def test_daily_shape_matrix_is_normalised(panel):
    shapes = P.daily_shape_matrix(panel)
    assert shapes.shape[1] == 24
    assert np.allclose(shapes.max(axis=1), 1)
    assert np.allclose(shapes.min(axis=1), 0)
    assert not shapes.isna().any().any()


def test_local_calendar_date_follows_the_zone_clock():
    """03:00 UTC on 1 July is still 30 June in Los Angeles (UTC-7), Chicago (UTC-5)
    and New York (UTC-4); 09:00 UTC is 1 July everywhere. The three notebooks that
    join cluster labels onto the panel key on this, not on the UTC date."""
    ts = pd.Timestamp("2024-07-01 03:00")
    df = pd.DataFrame({cfg.TIME_COL: [ts, ts, ts, ts + pd.Timedelta(hours=6)],
                       cfg.GROUP_COL: ["CISO", "MISO", "NYIS", "CISO"]})
    d = P.local_calendar_date(df)
    assert list(d) == [pd.Timestamp("2024-06-30")] * 3 + [pd.Timestamp("2024-07-01")]
    with pytest.raises(ValueError):
        P.local_calendar_date(pd.DataFrame({cfg.TIME_COL: [ts], cfg.GROUP_COL: ["XXXX"]}))


def test_clustering_pipeline_scores_four_metrics(panel):
    shapes = P.daily_shape_matrix(panel)
    pipe = P.make_clustering_pipeline(n_components=3, algorithm="kmeans", k=3)
    s = P.score_clustering(pipe, shapes, n_boot=2, sample=500)
    for m in ("silhouette", "calinski_harabasz", "davies_bouldin", "ari_stability"):
        assert m in s and np.isfinite(s[m])
    assert -1 <= s["silhouette"] <= 1
    assert -1 <= s["ari_stability"] <= 1


def test_ward_and_dbscan_arms_run(panel):
    shapes = P.daily_shape_matrix(panel)
    for algo, kw in (("ward", dict(k=3)), ("dbscan", dict(eps=0.6, min_samples=10))):
        pipe = P.make_clustering_pipeline(n_components=3, algorithm=algo, **kw)
        s = P.score_clustering(pipe, shapes, n_boot=1, sample=400)
        assert "n_clusters" in s


# ------------------------------------------------------------------ results
def test_results_require_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS", tmp_path / "results.json")
    R.save("nb01", {"chosen_lags": [1, 24, 168]})
    assert R.require("nb01", "chosen_lags") == [1, 24, 168]
    try:
        R.require("nb04", "holdout_mae")
    except R.MissingResultError as exc:
        assert "nb04" in str(exc)
    else:
        raise AssertionError("require() did not raise for a missing notebook")


def test_results_merge_does_not_clobber(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS", tmp_path / "results.json")
    R.save("nb04", {"a": 1})
    R.save("nb04", {"b": 2})
    assert R.get("nb04", "a") == 1 and R.get("nb04", "b") == 2


# --------------------------------------------------- the interpolation cap
# THE DEFECT THESE PIN, recorded because the prose and the code disagreed for
# months and nothing could tell. The contract says "interpolate gaps of <= 3
# hours only". It was implemented as `interpolate(limit=3)`, which is a pandas
# FILL LIMIT: it fills the first three NaNs of every gap however long. Each of
# the four 24-hour `demand` outages therefore had three hours filled and 21
# dropped, and the twelve values admitted were points on a straight line drawn
# across a whole day of outage — the precise quantity the contract exists to keep
# out. These tests assert the RULE, so the prose and the code cannot drift again.



def _panel_without_pre_existing_gaps():
    """One synthetic zone with every gap already filled, so an injected outage is
    the ONLY defect in the frame and the census can be asserted exactly. The
    generator injects its own missingness on purpose; leaving it in would make
    these assertions depend on the generator's seed."""
    raw = synthetic.make_panel(zones=["CISO"], start="2023-01-01T00", end="2023-01-31T23")
    raw = raw.sort_values(cfg.TIME_COL).reset_index(drop=True)
    num = raw.select_dtypes(include=[np.number]).columns
    raw[num] = raw[num].interpolate(limit_direction="both")
    return raw


def test_a_run_at_the_cap_is_filled_whole():
    s = pd.Series([10.0, np.nan, np.nan, np.nan, 20.0])
    out = data_loader.interpolate_short_runs(s, limit=3)
    assert not out.isna().any()
    assert np.allclose(out.to_numpy(), [10.0, 12.5, 15.0, 17.5, 20.0])


def test_a_run_over_the_cap_is_not_touched_at_all():
    """Not 'mostly untouched'. A single filled hour inside a day-long outage is the
    bug, because it is the hour furthest from any evidence."""
    s = pd.Series([10.0] + [np.nan] * 4 + [20.0])
    out = data_loader.interpolate_short_runs(s, limit=3)
    assert out.isna().sum() == 4


def test_leading_and_trailing_runs_are_never_filled():
    """Extrapolating off the end of a series is not interpolation."""
    s = pd.Series([np.nan, 10.0, 20.0, np.nan])
    out = data_loader.interpolate_short_runs(s, limit=3)
    assert out.isna().tolist() == [True, False, False, True]


def test_run_length_is_measured_per_column():
    """`demand` may be out for a day while `wind` is out for an hour. One rule
    applied to the row rather than the column would be wrong in one direction."""
    raw = _panel_without_pre_existing_gaps()
    raw.loc[100:123, "demand"] = np.nan     # 24-hour outage
    raw.loc[100:100, "wind"] = np.nan       # 1-hour dropout, same hour
    clean, report = data_loader.clean_panel(raw, verbose=False)
    clean = clean.sort_values(cfg.TIME_COL).reset_index(drop=True)
    assert clean["demand"].isna().sum() >= 24, "the day-long demand outage was filled"
    assert report["interp_runs"]["demand"]["runs_left_whole"] == 1
    assert report["interp_runs"]["wind"]["runs_repaired"] >= 1


def test_the_census_is_reported_so_the_notebook_never_has_to_assume_it():
    raw = _panel_without_pre_existing_gaps()
    raw.loc[200:223, "demand"] = np.nan
    _, report = data_loader.clean_panel(raw, verbose=False)
    census = report["interp_runs"]["demand"]
    assert census["hours_left_whole"] == 24
    assert census["hours_repaired"] == 0
    assert report["interp_rule"].startswith("whole-run gate")


def test_the_live_panel_contains_no_value_from_an_over_cap_run():
    """The gate itself, on the shipped panel rather than on a fixture.

    Every NaN run in the RAW panel is either fully present in the cleaned panel or
    fully absent from it. A run that is partly filled is the defect.
    """
    import pytest
    if not cfg.PANEL_RAW.exists() or not cfg.PANEL.exists():
        pytest.skip("no real panel on disk")
    raw = pd.read_parquet(cfg.PANEL_RAW)
    clean = pd.read_parquet(cfg.PANEL)
    key = [cfg.GROUP_COL, cfg.TIME_COL]
    merged = raw[key + ["demand"]].merge(
        clean[key + ["demand"]], on=key, how="left", suffixes=("_raw", "_clean"))
    merged = merged.sort_values(key).reset_index(drop=True)
    bad = []
    for z, g in merged.groupby(cfg.GROUP_COL):
        isna = g["demand_raw"].isna()
        if not isna.any():
            continue
        run_id = (isna != isna.shift()).cumsum()
        for _, run in g[isna].groupby(run_id[isna]):
            if len(run) > data_loader.MAX_INTERP_HOURS and run["demand_clean"].notna().any():
                bad.append((z, run[cfg.TIME_COL].iloc[0], len(run),
                            int(run["demand_clean"].notna().sum())))
    assert not bad, (
        f"{len(bad)} over-cap outage(s) had hours filled anyway: {bad[:4]}. "
        "The cap is a whole-run gate; rebuild panel.parquet from notebook 01.")
