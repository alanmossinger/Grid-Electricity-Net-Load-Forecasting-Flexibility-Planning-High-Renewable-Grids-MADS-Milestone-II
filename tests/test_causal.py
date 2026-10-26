"""
test_causal.py — the backdoor machinery, checked against structures whose answers
are known from theory rather than from this codebase.

Every test here is a claim you will make out loud in the oral exam. The four
canonical structures — confounder, collider, mediator, M-bias — are the whole of the
concept, and if the implementation gets any of them wrong the adjustment set it
computes for the real DAG is worthless.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import causal as C


# --------------------------------------------------------------------------
# The four canonical structures
# --------------------------------------------------------------------------
def test_confounder_must_be_adjusted():
    """T <- C -> Y. The fork is open until you condition on C."""
    g = C.build_dag([("C", "T", ""), ("C", "Y", "")])
    assert not C.is_valid_adjustment_set(g, set(), "T", "Y")
    assert C.is_valid_adjustment_set(g, {"C"}, "T", "Y")
    assert len(C.backdoor_paths(g, "T", "Y")) == 1


def test_collider_must_NOT_be_adjusted():
    """T -> K <- Y. Closed by default; conditioning on K opens it.

    This is the one that inverts intuition, and the reason `draw_dag` marks
    colliders explicitly.
    """
    g = C.build_dag([("T", "K", ""), ("Y", "K", "")])
    # K is a descendant of T, so the criterion must reject it on clause 1 alone.
    assert C.is_valid_adjustment_set(g, set(), "T", "Y")
    assert not C.is_valid_adjustment_set(g, {"K"}, "T", "Y")


def test_collider_not_downstream_of_treatment_still_opens_a_path():
    """T <- A -> K <- B -> Y: conditioning on the collider K creates association."""
    g = C.build_dag([("A", "T", ""), ("A", "K", ""), ("B", "K", ""), ("B", "Y", "")])
    assert C.is_valid_adjustment_set(g, set(), "T", "Y"), "M-bias: empty set is fine"
    assert not C.is_valid_adjustment_set(g, {"K"}, "T", "Y"), \
        "conditioning on the M-collider must OPEN the path"
    assert C.is_valid_adjustment_set(g, {"K", "A"}, "T", "Y"), \
        "adding A re-blocks the path opened by K"


def test_mediator_must_not_be_adjusted_for_a_total_effect():
    """T -> M -> Y. M is a descendant of T; adjusting removes the effect itself."""
    g = C.build_dag([("T", "M", ""), ("M", "Y", "")])
    assert C.is_valid_adjustment_set(g, set(), "T", "Y")
    assert not C.is_valid_adjustment_set(g, {"M"}, "T", "Y")


def test_conditioning_on_a_colliders_descendant_also_opens_it():
    """The rule is descendants too, not only the collider itself."""
    g = C.build_dag([("A", "T", ""), ("A", "K", ""), ("B", "K", ""),
                     ("B", "Y", ""), ("K", "D", "")])
    assert C.is_valid_adjustment_set(g, set(), "T", "Y")
    assert not C.is_valid_adjustment_set(g, {"D"}, "T", "Y")


# --------------------------------------------------------------------------
# Agreement with an independent implementation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("given", [
    frozenset(), frozenset({"season"}), frozenset({"demand"}),
    frozenset({"zone_char"}), frozenset({"zone_char", "demand_growth"}),
    frozenset({"net_load"}), frozenset({"temp", "cloud_cover"}),
])
def test_path_logic_agrees_with_networkx_d_separation(given):
    """Our open/closed verdict must match networkx's d-separation on the real DAG.

    Two independent implementations of the same rule. If they disagree, at least
    one is wrong and no adjustment set computed here can be trusted.
    """
    g = C.build_dag()
    ours_open = bool(C.open_backdoor_paths(g, given))
    all_paths_blocked = not any(
        C.path_is_open(g, p, given)
        for p in C._undirected_paths(g, C.TREATMENT, C.OUTCOME)
    )
    nx_dsep = nx.is_d_separator(g, {C.TREATMENT}, {C.OUTCOME}, set(given))
    assert all_paths_blocked == nx_dsep, (
        f"given={set(given)}: our d-separation says {all_paths_blocked}, "
        f"networkx says {nx_dsep}"
    )
    # A backdoor path being open implies the two are not d-separated.
    if ours_open:
        assert not nx_dsep


# --------------------------------------------------------------------------
# The project's own DAG
# --------------------------------------------------------------------------
def test_the_dag_is_acyclic_and_every_edge_carries_a_reason():
    g = C.build_dag()
    assert nx.is_directed_acyclic_graph(g)
    for u, v, data in g.edges(data=True):
        assert data.get("why"), f"edge {u}->{v} has no stated reason"


def test_net_load_is_a_collider_of_the_accounting_identity():
    """net_load = demand - wind - solar, so it has three parents by definition.

    This is the concrete answer to the oral question in the notebook header: the
    prediction target IS a collider, which is exactly why it may be regressed ON in
    Notebook 04 and must never be conditioned on here.
    """
    g = C.build_dag()
    assert "net_load" in C.colliders_in(g)
    assert set(g.predecessors("net_load")) == {"demand", "solar_output", "wind_output"}


def test_adjusting_for_the_outcomes_own_ancestor_chain_is_rejected():
    """net_load sits between the treatment and the outcome — never admissible."""
    g = C.build_dag()
    assert not C.is_valid_adjustment_set(g, {"net_load"})


def test_every_weather_variable_appears_in_the_dag():
    """All four ERA5 columns must be represented, not just temperature."""
    g = C.build_dag()
    for w in ("temp", "humidity", "wind_speed", "cloud_cover"):
        assert w in g.nodes, f"{w} is missing from the DAG"
        assert g.out_degree(w) >= 1, f"{w} is in the DAG but influences nothing"


def test_a_minimal_adjustment_set_exists_and_is_observable():
    """Identification is worth nothing if the set cannot be measured here."""
    g = C.build_dag()
    sets = C.minimal_adjustment_sets(g)
    assert sets, "no admissible adjustment set was found"
    for z in sets:
        assert C.is_valid_adjustment_set(g, z)
        for node in z:
            assert C.OBSERVED.get(node, False), f"{node} is not observed"


def test_unobserved_nodes_are_excluded_only_when_asked():
    g = C.build_dag()
    obs = C.minimal_adjustment_sets(g, observed_only=True)
    alln = C.minimal_adjustment_sets(g, observed_only=False)
    assert all(all(C.OBSERVED.get(n, False) for n in z) for z in obs)
    assert len(alln) >= len(obs)


# ==========================================================================
# Difference-in-differences
# ==========================================================================
# Every test below plants a known effect in synthetic data and checks the estimator
# recovers it. That is the only way to test an estimator: real data has no known
# answer, so a test against it can only assert that today's number equals
# yesterday's, which catches a refactor and nothing else.

def _did_panel(effect=5.0, treated_drift=0.0, n_days=400, seed=0):
    """A two-group panel with a planted post-period effect on the treated group.

    `treated_drift` plants a violation of parallel trends — a per-period divergence
    present before treatment as well as after — so the pre-trend diagnostics can be
    tested against a case where they SHOULD fire.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit, treated in [("T1", 1), ("C1", 0), ("C2", 0)]:
        for day in range(n_days):
            post = int(day >= n_days // 2)
            y = (10 * treated                     # a permanent level gap: DiD removes it
                 + 0.02 * day                     # a common time trend: DiD removes it too
                 + treated * treated_drift * day  # a DIFFERENTIAL trend: it does not
                 + effect * treated * post        # the thing to recover
                 + rng.normal(0, 0.5))
            rows.append({"unit": unit, "treated": treated, "day": day, "post": post,
                         "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                         "month": (day % 365) // 30 + 1, "y": y})
    return pd.DataFrame(rows)


def test_did_2x2_recovers_a_planted_effect_through_a_level_gap_and_a_common_trend():
    """The whole point of the second difference, in one assertion."""
    panel = _did_panel(effect=5.0)
    out = C.did_2x2(panel, "y")
    assert out["did"] == pytest.approx(5.0, abs=0.15)
    # and the naive post-period comparison does NOT recover it — it sees the level gap
    naive = out["treated_post"] - out["control_post"]
    assert naive > 12.0, "the naive contrast should be badly wrong; that is the point"


def test_did_2x2_is_literally_the_difference_of_two_differences():
    panel = _did_panel(effect=3.0, seed=1)
    out = C.did_2x2(panel, "y")
    assert out["did"] == pytest.approx(out["delta_treated"] - out["delta_control"])


def test_did_2x2_refuses_an_empty_cell():
    """A treated group never observed before treatment is not a DiD, it is a mean."""
    panel = _did_panel()
    broken = panel[~((panel["treated"] == 1) & (panel["post"] == 0))]
    with pytest.raises(ValueError, match="empty cell"):
        C.did_2x2(broken, "y")


def test_did_is_biased_by_a_differential_pre_trend():
    """The assumption is load-bearing: break it and the estimate breaks with it.

    This is the test that justifies the parallel-trends plot existing at all.
    """
    panel = _did_panel(effect=0.0, treated_drift=0.01)
    out = C.did_2x2(panel, "y")
    assert out["did"] > 1.0, "a pre-existing divergence must show up as a fake effect"


def test_deseasonalise_fits_on_the_mask_only_and_cannot_absorb_the_effect():
    """Fitting seasonal norms on the full sample would eat a seasonal treatment effect.

    Here the planted effect lands only in month 7. Pre-period-only norms leave it
    intact; full-sample norms remove most of it. If this test ever flips, the mask
    argument has been dropped somewhere and every DiD number moves.
    """
    rng = np.random.default_rng(3)
    n = 24
    rows = []
    for unit in ("A", "B"):
        for i in range(n * 12):
            month = i % 12 + 1
            post = int(i >= n * 6)
            y = month * 1.0 + (8.0 if (unit == "A" and post and month == 7) else 0.0)
            rows.append({"unit": unit, "month": month, "post": post,
                         "y": y + rng.normal(0, 0.01)})
    f = pd.DataFrame(rows)
    july_a = (f["unit"] == "A") & (f["month"] == 7) & (f["post"] == 1)

    pre_only = C.deseasonalise(f, "y", "unit", "month", fit_mask=f["post"] == 0)
    full = C.deseasonalise(f, "y", "unit", "month")
    assert pre_only[july_a].mean() == pytest.approx(8.0, abs=0.1)
    assert full[july_a].mean() < 5.0, "full-sample norms must be shown to absorb the effect"


def test_event_study_frame_puts_minus_one_immediately_before_the_event():
    f = pd.DataFrame({"date": pd.to_datetime(
        ["2022-06-30", "2022-07-24", "2022-09-30", "2022-10-01", "2021-07-24"])})
    rel = C.event_study_frame(f, "date", "2022-07-24", freq="Q")
    assert list(rel) == [-1, 0, 0, 1, -4]


# ==========================================================================
# Propensity matching
# ==========================================================================
def _confounded_sample(n=4000, effect=2.0, seed=0):
    """x confounds: it raises both the chance of treatment and the outcome.

    The naive difference in means is therefore biased upward, and matching on x
    should remove most of that bias. Known truth: effect = 2.0.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)                       # a covariate that does nothing
    p = 1 / (1 + np.exp(-(1.5 * x)))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 3 * x + effect * t + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "noise": noise, "T": t, "y": y})


def test_smd_is_zero_for_identical_groups_and_one_for_a_one_sd_gap():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    x = np.r_[a, b]
    t = np.r_[np.ones(20000), np.zeros(20000)]
    assert C.standardised_mean_difference(x, t) == pytest.approx(0.0, abs=0.03)
    assert C.standardised_mean_difference(np.r_[a + 1, b], t) == pytest.approx(1.0, abs=0.03)


def test_matching_removes_confounding_bias_and_recovers_the_effect():
    """With replacement, the planted effect comes back essentially exactly."""
    f = _confounded_sample(effect=2.0)
    naive = f.loc[f["T"] == 1, "y"].mean() - f.loc[f["T"] == 0, "y"].mean()
    res = C.propensity_match(f, "T", ["x", "noise"], replace=True)
    att = C.att_from_pairs(res["frame"], "y", res["pairs"], n_boot=200)
    assert naive > 3.5, "the naive contrast must be visibly confounded"
    assert att["att"] == pytest.approx(2.0, abs=0.15)


def test_matching_improves_balance_on_the_confounder():
    f = _confounded_sample()
    bal = C.propensity_match(f, "T", ["x", "noise"], replace=True)["balance"]
    assert abs(bal.loc["x", "smd_before"]) > 0.5
    assert abs(bal.loc["x", "smd_after"]) < 0.1
    assert bool(bal.loc["x", "balanced"])


def test_matching_without_replacement_leaves_residual_bias_when_controls_run_out():
    """The measured cost of `replace=False`, pinned so it cannot be forgotten.

    This is not a bug being tolerated. A finite control pool runs out, the caliper
    strands the treated units at the extremes, and the surviving matched sample is no
    longer the treated population. The notebook's within-zone design has 891 treated
    days chasing 1,100 controls, so it takes `replace=True` and pays for it with a
    clustered bootstrap instead. If this test ever starts passing at the tolerances
    of the one above, the trade-off has changed and that paragraph needs rewriting.
    """
    f = _confounded_sample(effect=2.0)
    res = C.propensity_match(f, "T", ["x", "noise"], replace=False)
    att = C.att_from_pairs(res["frame"], "y", res["pairs"], n_boot=200)
    assert res["matched_share"] < 0.75, "the caliper should strand a visible share"
    assert abs(res["balance"].loc["x", "smd_after"]) > 0.1
    assert att["att"] > 2.15, "and the residual imbalance should bias the ATT upward"


def test_matching_without_replacement_uses_each_control_once():
    f = _confounded_sample(n=1500)
    res = C.propensity_match(f, "T", ["x"], replace=False)
    controls = [c for _, c in res["pairs"]]
    assert len(controls) == len(set(controls))


def test_a_tighter_caliper_never_produces_more_pairs():
    f = _confounded_sample(n=1500)
    wide = C.propensity_match(f, "T", ["x"], caliper=0.5)["n_matched"]
    tight = C.propensity_match(f, "T", ["x"], caliper=0.01)["n_matched"]
    assert tight <= wide


def test_a_positivity_violation_shows_up_as_a_collapsed_match():
    """When a covariate separates the groups perfectly, there is nothing to match to.

    This is the failure the project's own within-zone design hits on `demand_trend`,
    and the notebook reports it rather than tuning it away. The estimator must make it
    visible — a near-empty matched sample — not paper over it with a wider caliper.
    """
    rng = np.random.default_rng(0)
    n = 1000
    t = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    f = pd.DataFrame({"sep": t * 50.0 + rng.normal(0, 1, n), "T": t,
                      "y": rng.normal(0, 1, n)})
    res = C.propensity_match(f, "T", ["sep"])
    assert res["matched_share"] < 0.10, (
        f"perfect separation should strand the treated units; "
        f"matched {res['matched_share']:.0%}"
    )


def test_matching_is_deterministic():
    f = _confounded_sample(n=1200)
    a = C.propensity_match(f, "T", ["x", "noise"])["pairs"]
    b = C.propensity_match(f, "T", ["x", "noise"])["pairs"]
    assert a == b


# ==========================================================================
# Sensitivity
# ==========================================================================
def test_partial_r2_matches_its_definition():
    assert C.partial_r2(2.0, 96) == pytest.approx(4 / 100)


def test_robustness_value_is_exactly_the_confounder_that_kills_the_estimate():
    """RV_1 is defined so that a confounder that strong causes bias = the estimate.

    Feeding RV back into the bias formula must return the estimate itself. This is
    the strongest available check on both functions at once: they are each other's
    inverse, and they were derived independently from the paper.
    """
    for t_stat, dof, se in [(3.0, 500, 0.4), (12.0, 10_000, 0.01), (2.1, 60, 1.7)]:
        rv = C.robustness_value(t_stat, dof)
        bias = C.ovb_bias(se, dof, rv, rv)
        assert bias == pytest.approx(abs(t_stat) * se, rel=1e-6)


def test_robustness_value_rises_with_the_strength_of_the_finding():
    assert C.robustness_value(2.0, 1000) < C.robustness_value(10.0, 1000)


def test_robustness_value_for_halving_is_smaller_than_for_erasing():
    assert C.robustness_value(6.0, 1000, q=0.5) < C.robustness_value(6.0, 1000, q=1.0)


def test_robustness_value_stays_in_the_unit_interval():
    for t_stat in (0.0, 0.5, 5.0, 500.0):
        assert 0.0 <= C.robustness_value(t_stat, 300) <= 1.0


def test_ovb_bias_explodes_as_the_confounder_determines_the_treatment():
    """Positivity, arriving through the algebra instead of through the overlap plot."""
    weak = C.ovb_bias(1.0, 500, 0.3, 0.10)
    strong = C.ovb_bias(1.0, 500, 0.3, 0.99)
    assert strong > 10 * weak


def test_ovb_bias_is_zero_when_the_confounder_touches_only_one_side():
    """A variable related to the treatment but not the outcome is not a confounder."""
    assert C.ovb_bias(1.0, 500, 0.0, 0.9) == pytest.approx(0.0)
    assert C.ovb_bias(1.0, 500, 0.9, 0.0) == pytest.approx(0.0)


def test_rosenbaum_p_value_is_non_decreasing_in_gamma():
    rng = np.random.default_rng(0)
    d = rng.normal(1.0, 1.0, 300)
    tab = C.rosenbaum_bounds(d, gammas=(1.0, 1.5, 2.0, 4.0, 8.0))
    upper = tab[tab["bound"] == "upper"].sort_values("gamma")["p"].values
    assert np.all(np.diff(upper) >= -1e-12)


def test_rosenbaum_at_gamma_one_matches_the_ordinary_signed_rank_test():
    """Gamma = 1 is the no-hidden-bias case, so it must reproduce the textbook test."""
    from scipy import stats

    rng = np.random.default_rng(1)
    d = rng.normal(0.4, 1.0, 400)
    ours = C.rosenbaum_bounds(d, gammas=(1.0,)).query("bound == 'upper'")["p"].iloc[0]
    theirs = stats.wilcoxon(d, alternative="greater").pvalue
    assert ours == pytest.approx(theirs, abs=5e-3)


def test_rosenbaum_gamma_critical_grows_with_the_size_of_the_effect():
    rng = np.random.default_rng(2)
    weak = C.rosenbaum_gamma_critical(rng.normal(0.1, 1.0, 400))
    strong = C.rosenbaum_gamma_critical(rng.normal(1.5, 1.0, 400))
    assert strong > weak >= 1.0


def test_rosenbaum_gamma_is_one_when_the_result_was_never_significant():
    rng = np.random.default_rng(4)
    assert C.rosenbaum_gamma_critical(rng.normal(0.0, 1.0, 200)) == 1.0


def test_design_matrix_refuses_a_column_listed_as_both_continuous_and_categorical():
    """Overlap silently built duplicate one-hot columns and failed downstream.

    The error named `month_2` without saying why, which cost a notebook run on
    2026-08-22. Refuse it where the cause is visible.
    """
    import pandas as pd
    import pytest as _pytest

    from src import causal as _C

    frame = pd.DataFrame({"temp": [1.0, 2.0], "month": [1, 2], "t": [0, 1]})
    with _pytest.raises(ValueError, match="BOTH covariates and categoricals"):
        _C._design_matrix(frame, ["temp", "month"], ("month",))
    # the correct split still works
    out = _C._design_matrix(frame, ["temp"], ("month",))
    assert "temp" in out.columns and out.shape[0] == 2
