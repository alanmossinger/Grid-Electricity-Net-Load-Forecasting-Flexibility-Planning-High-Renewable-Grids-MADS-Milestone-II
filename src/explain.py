"""notebook 07 — explainability helpers.

The reusable half of notebook 07. Three things live here because they are easy to
get subtly wrong and a test is cheaper than an argument at the viva:

1. **Grouped permutation**, which answers "how much does the model lean on this
   whole family" rather than on one column at a time. Note that the scaffold's
   stated reading rule -- redundancy shows up as grouped >> the sum of the
   singles -- is **false**, and `redundancy_gap` documents the measurement that
   disproves it along with the diagnostic that does work. Permutation happens at
   predict time against a fitted model, so nothing is ever "recovered through the
   twin"; the trees are already built.

2. **The block must be permuted with ONE row order, not one per column.** Shuffling
   each column of a family independently destroys the correlation *within* the
   family as well as the relationship to the target, which measures something
   nobody asked about. A single permutation applied to the whole block preserves
   the joint distribution inside the family and breaks only the link to `y`.

3. **Permutation is run on the TRANSFORMED matrix**, against the fitted final
   estimator, not on the raw frame through the whole pipeline. Two reasons: the
   lag step reconstructs history by timestamp lookup on every call, so permuting
   through the pipeline costs a full rebuild per repeat; and the names that come
   back then match `select.selected_`, which is what every SHAP and importance
   plot in notebook 07 is labelled with.

None of this measures causation. Permutation says how much the MODEL leans on a
column, which is a fact about the model and not about the grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src import config as cfg
from src import features as P

__all__ = [
    "transformed_holdout",
    "permutation_single",
    "permutation_grouped",
    "redundancy_gap",
    "ale_1d",
    "worst_day",
    "surrogate_rules",
]


def transformed_holdout(pipe, frame):
    """The matrix the final estimator actually sees, with `selected_` as columns.

    Everything except the last step is applied, so the result is what the model
    was fitted on -- lagged, selected, imputed -- and its column names are the
    ones notebook 07 must label its plots with.
    """
    X = pipe[:-1].transform(frame)
    names = list(pipe.named_steps["select"].selected_)
    if isinstance(X, pd.DataFrame):
        return X.loc[:, names] if list(X.columns) != names else X
    return pd.DataFrame(np.asarray(X), columns=names, index=getattr(frame, "index", None))


def _mae(model, X, y):
    return float(mean_absolute_error(y, model.predict(X)))


def permutation_single(model, X, y, n_repeats: int = 5, seed: int = cfg.SEED):
    """Per-feature permutation importance, in MW of MAE increase.

    Implemented directly rather than through `sklearn.inspection` so that the
    single and grouped paths share one baseline, one RNG discipline and one unit.
    A difference between the two columns then cannot be an artefact of two
    different implementations.
    """
    X = pd.DataFrame(X)
    rng = np.random.default_rng(seed)
    base = _mae(model, X, y)
    rows = []
    for col in X.columns:
        deltas = []
        original = X[col].to_numpy(copy=True)
        for _ in range(n_repeats):
            X[col] = rng.permutation(original)
            deltas.append(_mae(model, X, y) - base)
        X[col] = original
        rows.append({"feature": col, "family": P.feature_family(str(col)),
                     "mae_increase_mw": float(np.mean(deltas)),
                     "sd_mw": float(np.std(deltas))})
    out = pd.DataFrame(rows).sort_values("mae_increase_mw", ascending=False)
    out.attrs["baseline_mae"] = base
    return out.reset_index(drop=True)


def permutation_grouped(model, X, y, n_repeats: int = 5, seed: int = cfg.SEED,
                        families=None):
    """Permutation importance for whole feature families, in MW of MAE increase.

    The whole block moves under ONE permutation per repeat -- see the module
    docstring. `families` may be passed to override the mapping; by default every
    column is classified with `pipeline.feature_family`, which groups `x_lag24`
    with `x` and `lag_k` with `lag_k_rel`.
    """
    X = pd.DataFrame(X)
    fam = dict(families) if families is not None else {
        c: P.feature_family(str(c)) for c in X.columns}
    rng = np.random.default_rng(seed)
    base = _mae(model, X, y)
    rows = []
    for name in sorted(set(fam.values())):
        cols = [c for c in X.columns if fam[c] == name]
        if not cols:
            continue
        original = X[cols].to_numpy(copy=True)
        deltas = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))          # ONE order for the block
            X[cols] = original[order]
            deltas.append(_mae(model, X, y) - base)
        X[cols] = original
        rows.append({"family": name, "n_features": len(cols),
                     "mae_increase_mw": float(np.mean(deltas)),
                     "sd_mw": float(np.std(deltas))})
    out = pd.DataFrame(rows).sort_values("mae_increase_mw", ascending=False)
    out.attrs["baseline_mae"] = base
    return out.reset_index(drop=True)


def redundancy_gap(single: pd.DataFrame, grouped: pd.DataFrame,
                   free_rider_share: float = 0.05) -> pd.DataFrame:
    """Grouped importance beside the singles, with the redundancy diagnostic.

    **The ratio of grouped to the sum of singles does not detect redundancy, and
    the scaffold's reading rule -- "where grouped >> sum of singles, the features
    are redundant substitutes" -- is wrong.** Measured on exact and near-duplicate
    pairs across RandomForest, HistGradientBoosting and Ridge, the ratio is ~1.00
    in every case. The reason is that permutation happens at PREDICT time against
    a fitted model: nothing can be "recovered through the twin", because the trees
    are already built. If the fit leaned on one twin and ignored the other, the
    ignored one scores ~0 alone and the grouped figure simply equals the used
    one's -- so grouped and the sum agree.

    What *does* reveal a substitute is a **free rider**: a feature whose single
    importance is negligible while the family it belongs to matters a great deal.
    On an exact-duplicate pair under HistGradientBoosting the singles are
    [3.308, 0.000] -- the second column is free, and dropping it would cost
    nothing, which is the actionable claim. `n_free_riders` counts members scoring
    below `free_rider_share` of the family's grouped importance.

    The ratio is still reported, because it is worth showing that it is flat.
    """
    grp = single.groupby("family")["mae_increase_mw"]
    stats = pd.DataFrame({
        "sum_of_singles_mw": grp.sum(),
        "max_single_mw": grp.max(),
        "min_single_mw": grp.min(),
    })
    out = grouped.merge(stats, left_on="family", right_index=True, how="left")
    out[["sum_of_singles_mw", "max_single_mw", "min_single_mw"]] = out[
        ["sum_of_singles_mw", "max_single_mw", "min_single_mw"]].fillna(0.0)
    out["ratio_grouped_over_sum"] = np.where(
        out["sum_of_singles_mw"].abs() > 1e-9,
        out["mae_increase_mw"] / out["sum_of_singles_mw"], np.nan)

    riders = []
    for _, r in out.iterrows():
        members = single.loc[single["family"] == r["family"], "mae_increase_mw"]
        thresh = free_rider_share * max(r["mae_increase_mw"], 0.0)
        riders.append(int((members < thresh).sum()))
    out["n_free_riders"] = riders
    return out


def ale_1d(model, X, feature, n_bins: int = 20):
    """First-order accumulated local effects for one feature.

    PDP averages the model over the MARGINAL distribution of the other features,
    which means it evaluates points that never occur -- hour 20 with peak solar,
    January with 35 degrees. Under the correlations in this panel that is not a
    small objection. ALE instead asks a local question: inside a narrow bin of the
    feature, how much does the prediction move when the feature is nudged from the
    bin's lower edge to its upper edge, holding that row's own other features
    fixed? Those local differences are accumulated across bins.

    Returned centred on zero, so the curve is read as a deviation from the average
    prediction and is directly comparable with a centred PDP.
    """
    X = pd.DataFrame(X)
    x = X[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"{feature!r} has too few distinct values for ALE")
    idx = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)

    effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        rows = np.flatnonzero(idx == b)
        if rows.size == 0:
            continue
        lo = X.iloc[rows].copy()
        hi = lo.copy()
        lo[feature] = edges[b]
        hi[feature] = edges[b + 1]
        effects[b] = float(np.mean(model.predict(hi) - model.predict(lo)))
        counts[b] = rows.size

    acc = np.concatenate([[0.0], np.cumsum(effects)])
    centres = (edges[:-1] + edges[1:]) / 2
    mid = (acc[:-1] + acc[1:]) / 2
    offset = float(np.sum(mid * counts) / counts.sum()) if counts.sum() else 0.0
    return pd.DataFrame({"x": centres, "ale": mid - offset, "n": counts.astype(int)})


def worst_day(frame, y_true, y_pred, time_col: str = cfg.TIME_COL,
              group_col: str = cfg.GROUP_COL):
    """The (zone, date) with the largest mean absolute error.

    Returned as a dict rather than a tuple so the saved record says which zone and
    which date without positional guessing. `n_hours` is included because a "worst
    day" with four hours in it is not a worst day.
    """
    d = pd.DataFrame({
        group_col: np.asarray(frame[group_col]),
        "date": pd.to_datetime(np.asarray(frame[time_col])).normalize(),
        "abs_err": np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)),
    })
    g = d.groupby([group_col, "date"])["abs_err"].agg(["mean", "size"])
    g = g[g["size"] >= 24]
    if g.empty:
        g = d.groupby([group_col, "date"])["abs_err"].agg(["mean", "size"])
    row = g["mean"].idxmax()
    return {"zone": str(row[0]), "date": pd.Timestamp(row[1]).strftime("%Y-%m-%d"),
            "mae_mw": float(g.loc[row, "mean"]), "n_hours": int(g.loc[row, "size"])}


def surrogate_rules(tree, feature_names, max_rules: int = 12, X=None, y=None):
    """Readable if/then rules from a shallow decision tree.

    A surrogate explains the MODEL, not the data: its accuracy against the model's
    own labels is the only thing that licenses reading its rules, which is why the
    caller is expected to report that accuracy beside them.

    PURITY. With class_weight='balanced' the tree's own node values are WEIGHTED
    counts, so their share is not the share of rows in the leaf. Pass the rows the tree
    was fitted on (`X`, `y`) and purity is the plain share of them that carry the leaf's
    label; without them it falls back to the tree's values.
    """
    t = tree.tree_
    names = list(feature_names)
    out = []
    leaf_of = tree.apply(np.asarray(X)) if X is not None else None
    yv = np.asarray(y) if y is not None else None

    def walk(node, depth, path):
        if len(out) >= max_rules:
            return
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            label = int(np.argmax(counts))
            cls = tree.classes_[label] if hasattr(tree, "classes_") else label
            if leaf_of is not None and yv is not None:
                in_leaf = yv[leaf_of == node]
                share = float(np.mean(in_leaf == cls)) if len(in_leaf) else float("nan")
            else:
                share = float(counts[label] / counts.sum()) if counts.sum() else float("nan")
            out.append({"rule": " AND ".join(path) if path else "(root)",
                        "predicts": cls.item() if hasattr(cls, "item") else cls, "purity": share,
                        "n": int(t.n_node_samples[node])})
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], depth + 1, path + [f"{f} <= {thr:,.3g}"])
        walk(t.children_right[node], depth + 1, path + [f"{f} > {thr:,.3g}"])

    walk(0, 0, [])
    return pd.DataFrame(out)
