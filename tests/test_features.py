"""
test_pipeline.py — the pipeline fits, predicts, tunes, and keeps its contract.

The step names and attribute names asserted here are what the notebooks reach
for. Renaming a step without updating these tests breaks notebook 07 silently.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src import config as cfg
from src import features as P


def test_step_names_are_the_contract():
    pipe = P.make_supervised_pipeline()
    # 'ablate' was inserted in notebook 04 (2026-08-14) for the feature-family ablation.
    # It is a passthrough unless `drop_family` is set, and it sits immediately BEFORE
    # 'select' so that an ablated arm gets its MI ranking refitted on the surviving
    # columns rather than inheriting the full model's ranking with holes punched in it.
    # 'derive' was inserted 2026-08-23 with the fifteen-column weather set. It sits
    # AFTER 'weather' on purpose: hdd/cdd and the clear-sky index are functions of
    # the same-hour weather, so when a notebook injects forecast error they must see
    # the degraded values. Before 'weather' they would leave a clean copy of the
    # temperature in the design matrix and silently undo the sensitivity.
    # 'swap' was inserted 2026-08-23 with the operational-weather re-scoring. It sits
    # between 'weather' and 'derive': after 'lag' so the lag columns keep their
    # OBSERVED values (a lag L >= h is knowable at the origin and is not a forecast),
    # and before 'derive' so hdd/cdd and the clear-sky index are built from the
    # forecast weather rather than from reanalysis the operator did not have.
    assert [n for n, _ in pipe.steps] == [
        "calendar", "lag", "relative", "weather", "swap", "derive", "encode",
        "ablate", "select", "impute", "model"
    ]
    assert [n for n, _ in P.make_supervised_pipeline(derive_weather=False).steps] == [
        "calendar", "lag", "relative", "weather", "swap", "derive", "encode",
        "ablate", "select", "impute", "model"
    ], "switching the derivations off must keep the step, as a passthrough"
    names_ = [n for n, _ in pipe.steps]
    assert names_.index("ablate") == names_.index("select") - 1
    # 'relative' consumes the lag/rolling columns, so it cannot precede 'lag'.
    names = [n for n, _ in pipe.steps]
    assert names.index("relative") == names.index("lag") + 1
    # 'weather' must follow 'lag' — not necessarily immediately, but after. The
    # exogenous lag refers to a time at or before the forecast origin and is
    # OBSERVED, so it must be built before any forecast error is injected into the
    # same-hour column.
    names = [n for n, _ in pipe.steps]
    assert names.index("weather") > names.index("lag")


def test_fit_predict_and_selected_features(panel):
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,))
    pipe.fit(panel, panel[cfg.TARGET])
    pred = pipe.predict(panel)
    assert len(pred) == len(panel)
    assert np.isfinite(pred).all()
    assert isinstance(pipe.named_steps["select"].selected_, list)
    assert len(pipe.named_steps["select"].selected_) > 3


def test_transform_matrix_matches_selected_names(panel):
    """notebook 07 applies SHAP to pipe[:-1].transform(X) with these column names."""
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,))
    pipe.fit(panel, panel[cfg.TARGET])
    X = pipe[:-1].transform(panel)
    assert X.shape[1] == len(pipe.named_steps["select"].selected_)


def test_cross_val_score_runs_and_beats_the_mean(panel):
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,))
    cv = P.make_ts_cv(n_splits=3, gap=24)
    mae = -cross_val_score(pipe, panel, panel[cfg.TARGET], cv=cv,
                           scoring="neg_mean_absolute_error")
    assert len(mae) == 3
    naive = mean_absolute_error(panel[cfg.TARGET],
                                np.full(len(panel), panel[cfg.TARGET].mean()))
    assert mae.mean() < naive, "the pipeline is not beating a constant prediction"


def test_every_encoding_fits(panel):
    for enc in ("target", "onehot", "ordinal"):
        pipe = P.make_supervised_pipeline(lags=(1,), roll_windows=(24,), encoding=enc)
        pipe.fit(panel, panel[cfg.TARGET])
        assert np.isfinite(pipe.predict(panel)).all()


def test_ridge_tolerates_the_nans_via_imputer(panel):
    """Ridge cannot take NaN. The imputer step is why the ladder works at all."""
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,), model=Ridge())
    pipe.fit(panel, panel[cfg.TARGET])
    assert np.isfinite(pipe.predict(panel)).all()


def test_search_space_keys_all_exist_on_the_pipeline():
    """A typo in a search-space key makes RandomizedSearchCV fail at fit time,
    30 minutes into a run. Catch it in one second instead."""
    pipe = P.make_supervised_pipeline()
    valid = set(pipe.get_params(deep=True))
    unknown = [k for k in P.SUPERVISED_SEARCH_SPACE if k not in valid]
    assert not unknown, f"unknown pipeline parameters: {unknown}"


def test_randomized_search_completes(panel):
    cv = P.make_ts_cv(n_splits=2, gap=24)
    search = RandomizedSearchCV(
        P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,)),
        {"model__learning_rate": [0.05, 0.1], "select__k": ["all", 10]},
        n_iter=2, cv=cv, scoring="neg_mean_absolute_error",
        random_state=cfg.SEED, n_jobs=1,
    )
    search.fit(panel, panel[cfg.TARGET])
    assert search.best_score_ < 0


def test_feature_family_covers_the_real_feature_names(panel):
    pipe = P.make_supervised_pipeline(lags=(1, 24), roll_windows=(24,))
    pipe.fit(panel, panel[cfg.TARGET])
    fams = {P.feature_family(c) for c in pipe.named_steps["select"].selected_}
    assert "other" not in fams or len(fams) > 1
    assert {"lag", "calendar"} <= fams


# ------------------------------------------------------------------ kernel
def test_every_notebook_declares_the_project_kernel():
    """The kernel name is project-specific, and all five places agree.

    `ipykernel install --user --name python3` writes a kernel that OUTRANKS the
    interpreter's own default, so claiming that name would make every other
    notebook on this machine - other courses, other repos - silently launch this
    project's virtualenv. The only symptom would be an ImportError somewhere
    unrelated, which is close to undebuggable.

    Jupyter chooses the kernel by its NAME. A notebook that declares `python3`
    opens in JupyterLab on whatever interpreter that name points to - usually one
    without the pinned packages - whatever its display name says. VS Code's
    Jupyter extension rewrites the metadata to `python3` when a notebook is saved
    there; run_notebooks.py writes the registered spec back after every run.
    """
    import json
    from pathlib import Path

    assert cfg.KERNEL_NAME != "python3", "the project kernel must not claim the default name"

    nbs = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert nbs, "no notebooks found - run `make run`"
    for p in nbs:
        ks = json.loads(p.read_text(encoding="utf-8"))["metadata"].get("kernelspec", {})
        name, display = ks.get("name"), ks.get("display_name") or ""
        assert name == cfg.KERNEL_NAME and display == cfg.KERNEL_DISPLAY, (
            f"{p.name} declares kernel name={name!r} display={display!r}, not the spec "
            f"`make setup` registers ({cfg.KERNEL_NAME!r}); re-run it with run_notebooks.py "
            f"or select that kernel before saving."
        )


# --------------------------------------------------------------------------
# Feature-family ablation (notebook 04)
# --------------------------------------------------------------------------
def test_drop_family_removes_exactly_that_family(panel):
    """Every dropped column must belong to the named family, and none may survive."""
    built = P.Pipeline([("calendar", P.CalendarFeatures()),
                        ("lag", P.GroupLagFeatures(lags=(24, 168), roll_windows=(24,)))])
    X = built.fit(panel).transform(panel)
    step = P.DropFeatureFamily(family="calendar").fit(X)
    out = step.transform(X)
    assert step.dropped_, "nothing was dropped"
    assert all(P.feature_family(c) == "calendar" for c in step.dropped_)
    survivors = [c for c in out.columns if c not in P.DropFeatureFamily.PROTECTED]
    assert not any(P.feature_family(c) == "calendar" for c in survivors)


def test_drop_family_never_removes_the_columns_the_frame_needs(panel):
    built = P.Pipeline([("calendar", P.CalendarFeatures()),
                        ("lag", P.GroupLagFeatures(lags=(24,), roll_windows=(24,)))])
    X = built.fit(panel).transform(panel)
    for fam in ("calendar", "lag", "rolling", "weather", "zone"):
        try:
            out = P.DropFeatureFamily(family=fam).fit(X).transform(X)
        except ValueError:
            continue                      # family absent from this configuration
        for col in P.DropFeatureFamily.PROTECTED:
            assert col in out.columns, f"dropping {fam} removed {col}"


def test_drop_family_none_is_a_passthrough(panel):
    built = P.Pipeline([("calendar", P.CalendarFeatures()),
                        ("lag", P.GroupLagFeatures(lags=(24,), roll_windows=(24,)))])
    X = built.fit(panel).transform(panel)
    out = P.DropFeatureFamily(family=None).fit(X).transform(X)
    assert list(out.columns) == list(X.columns)


def test_drop_family_refuses_a_family_that_matches_nothing(panel):
    """A silent no-op would report 'this family is worth 0 MW' — a false finding.

    'lag', not 'rolling': the raw panel ships a `ramp` column, which `feature_family`
    maps to 'rolling', so 'rolling' is present before the lag step has run. The
    selector drops `ramp` later anyway (it contains net_load(t)), but it is present
    here, and a test that assumed otherwise would be testing the wrong thing.
    """
    import pytest
    built = P.Pipeline([("calendar", P.CalendarFeatures())])
    X = built.fit(panel).transform(panel)
    assert "lag" not in {P.feature_family(c) for c in X.columns}
    with pytest.raises(ValueError, match="matched no columns"):
        P.DropFeatureFamily(family="lag").fit(X)


def test_ablated_pipeline_fits_and_differs_from_the_full_one(panel):
    """End to end: the ablated arm must actually train and actually score differently."""
    from sklearn.linear_model import Ridge
    y = panel[cfg.TARGET]
    full = P.make_supervised_pipeline(cfg.NUMERIC_FEATURES, lags=(24, 168),
                                      model=Ridge()).fit(panel, y)
    cut = P.make_supervised_pipeline(cfg.NUMERIC_FEATURES, lags=(24, 168),
                                     model=Ridge(), drop_family="lag").fit(panel, y)
    import numpy as np
    assert not np.allclose(full.predict(panel), cut.predict(panel))


def test_predicting_on_a_row_slice_is_not_the_same_as_slicing_the_prediction(panel):
    """The lag step resolves history INSIDE the frame it is handed.

    Hand it a single day and every lookup at t-24/t-48/t-168 falls outside that
    frame, is imputed, and the output collapses toward the global mean. This cost
    a wrong `worst_hour_error_mw` in notebook 07 on 2026-08-22: the sliced call
    invented a 64,410 MW error on a day whose real MAE was 19,496 MW.

    The contract this pins: predict once on the whole frame, then index the
    RESULT. If a future change ever makes the two agree, this test failing is the
    signal that the caveat can be dropped.
    """
    import numpy as np

    from src import config as cfg
    from src import features as P

    pipe = P.make_supervised_pipeline(cfg.NUMERIC_FEATURES, lags=(24, 48))
    pipe.fit(panel, panel[cfg.TARGET])

    full = pipe.predict(panel)
    zone = panel[cfg.GROUP_COL].iloc[0]
    day = panel[cfg.TIME_COL].dt.normalize().iloc[-100]
    mask = ((panel[cfg.GROUP_COL] == zone)
            & (panel[cfg.TIME_COL].dt.normalize() == day)).to_numpy()
    assert mask.sum() >= 12, "need a populated day to make the point"

    sliced = pipe.predict(panel.loc[mask])
    indexed = full[mask]

    assert not np.allclose(sliced, indexed), (
        "predicting a slice matched slicing the prediction; the lag-reconstruction "
        "hazard this test documents may have been fixed -- check before relying on it")

    # HOW BADLY it degrades depends on the panel, and this fixture understates it.
    # On the synthetic 3-zone panel every zone sits at a similar level, so imputing
    # a lag lands near the truth and the two predictions stay within a few hundred
    # MW. On the real 6-zone panel, levels span 6,000 to 90,000 MW, so the same
    # imputation collapses the prediction toward the global mean -- measured at
    # 26,551-29,189 MW for a day that ran 60,305-93,429. The invariant worth
    # pinning is that the two disagree AT ALL; the magnitude is data-dependent and
    # is deliberately not asserted here.
    assert np.max(np.abs(sliced - indexed)) > 0


def test_no_family_is_called_renewables_because_none_of_them_is(panel):
    """The family holding demand, coal, gas, nuclear and oil was named "renewables".

    Notebook 07 measured it at 3,945.7 MW of permutation importance with `demand_lag24`
    alone contributing 3,261.6 -- 82.7% -- so every reading of that bar as a solar
    or wind result was a reading of demand history. The membership is deliberate
    (every exogenous column admissible only as a lag); the name was the defect.
    """
    from src import config as cfg
    from src import features as P

    pipe = P.make_supervised_pipeline(cfg.NUMERIC_FEATURES, lags=(24, 48))
    pipe.fit(panel, panel[cfg.TARGET])
    fams = {P.feature_family(c) for c in pipe.named_steps["select"].selected_}

    assert "renewables" not in fams, (
        "a family is called 'renewables' again; it holds demand and five thermal "
        "fuels, so the name asserts something false about every figure using it")
    assert P.feature_family("demand_lag24") == "exogenous"
    assert P.feature_family("coal_lag24") == "exogenous"
    assert P.feature_family("solar_lag24") == "exogenous"
    # and the grouping itself is unchanged by the rename
    assert P.feature_family("demand_lag24") == P.feature_family("solar_lag24")
