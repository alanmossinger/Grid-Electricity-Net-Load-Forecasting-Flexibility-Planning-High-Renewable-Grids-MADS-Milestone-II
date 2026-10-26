"""
causal.py — the DAG, and the backdoor machinery that reads an adjustment set off it.

WHY THIS IS CODE AND NOT A PICTURE. The plan allows a hand-drawn DAG committed as a
PNG. A picture cannot be checked against the regression printed underneath it, and
nothing catches the two drifting apart. Here the graph is an edge list, the
adjustment set is COMPUTED from it by the backdoor criterion, and the notebook draws
the same object it reasons over. The diagram and the analysis cannot disagree.

WHAT THIS DOES NOT BUY. A computed adjustment set is exactly as good as the edge
list, and the edge list is an assumption — a claim about how the world works, not a
finding from the data. Every edge below therefore carries its reason in the third
field, so the assumption is attackable rather than hidden. That is the honest
position: the DAG is the argument, and the reader is invited to break it.

THE THREE STRUCTURES, because the whole module is an application of them:

    chain     A -> B -> C     B blocks the path when conditioned on
    fork      A <- B -> C     B blocks the path when conditioned on (a confounder)
    collider  A -> B <- C     B blocks the path by DEFAULT, and conditioning on B
                              (or on any descendant of B) OPENS it

The collider rule is the one that inverts the usual intuition: adding a control can
create an association that was not there. `sensitivity` and the notebook's non-energy
example exist to make that concrete.

WHAT ELSE IS IN HERE. The DAG says which variables are admissible; it does not
produce an estimate. The second half of this module holds the three estimators the
notebook compares, plus the two things that keep them honest:

    did_2x2 / event_study_frame        difference-in-differences, and the
                                       parallel-trends evidence for it
    propensity_match / balance_table   matching, and the love plot behind it
    robustness_value / ovb_bias        how strong an omitted cause would have to be
    rosenbaum_bounds                   the same question for the matched pairs

Every one of them is a function of an ASSUMPTION as well as of the data, and each
docstring names which assumption. That is the whole content of the notebook: an
observational estimate is a number plus the conditions under which it means what it
appears to mean, and quoting the number without the conditions is the error the
notebook exists to avoid.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# The DAG: (parent, child, why this edge is claimed)
# --------------------------------------------------------------------------
# Read this as the causal story of the two datasets. Weather is upstream of
# everything in the power system and is caused by none of it — that exogeneity is
# what makes any causal claim here possible at all.
NETLOAD_EDGES = [
    # --- season drives all four weather variables ------------------------
    ("season", "temp", "the annual insolation cycle"),
    ("season", "cloud_cover", "storm tracks and convective season move with the year"),
    ("season", "humidity", "absolute humidity rises with summer temperature"),
    ("season", "wind_speed", "the pressure-gradient climatology is seasonal"),

    # --- weather internals ------------------------------------------------
    ("cloud_cover", "temp", "cloud suppresses the daytime maximum"),
    ("humidity", "cloud_cover", "moisture is the raw material for cloud"),

    # --- weather -> electricity demand -----------------------------------
    ("temp", "demand", "heating below the comfort band, cooling above it"),
    ("humidity", "demand", "humid heat raises cooling load at the same temperature"),
    ("wind_speed", "demand", "wind chill raises winter heating demand"),

    # --- weather -> renewable output --------------------------------------
    ("cloud_cover", "solar_output", "cloud attenuates the irradiance reaching a panel"),
    ("temp", "solar_output", "panel efficiency falls as cell temperature rises"),
    ("wind_speed", "wind_output", "the turbine power curve"),

    # --- structure and policy ---------------------------------------------
    ("zone_char", "solar_capacity", "irradiance, land, market design and policy"),
    ("zone_char", "load_shape", "climate and industrial mix set the daily shape"),
    ("demand_growth", "demand", "more customers and more load per customer"),
    ("demand_growth", "solar_capacity", "growing systems procure more of everything"),
    ("load_shape", "demand", "the shape is a component of the level"),

    # --- capacity -> output ------------------------------------------------
    ("solar_capacity", "solar_output", "you cannot generate from a panel you did not build"),

    # --- the accounting identity ------------------------------------------
    # net_load = demand - wind - solar. These three edges are DEFINITIONAL, which
    # is why net_load is a collider and why conditioning on it is fatal.
    ("demand", "net_load", "definitional: net_load = demand - wind - solar"),
    ("solar_output", "net_load", "definitional, with a negative sign"),
    ("wind_output", "net_load", "definitional, with a negative sign"),
    ("net_load", "evening_ramp", "the ramp is a function of the net-load path"),
]

# Whether the project actually measures each node. An adjustment set that is valid
# but contains an unobserved node is not a usable one, and the distinction is the
# difference between "identified in principle" and "estimable here".
OBSERVED = {
    "temp": True, "humidity": True, "wind_speed": True, "cloud_cover": True,
    "season": True,            # month / day-of-year
    "demand": True, "net_load": True, "solar_output": True, "wind_output": True,
    "evening_ramp": True,
    "zone_char": True,         # proxied by zone identity (fixed effects)
    "demand_growth": True,     # proxied by a per-zone demand trend
    "load_shape": False,       # not measured directly; only its consequences
    "solar_capacity": False,   # NOT measured — proxied by realised solar share
}

TREATMENT = "solar_capacity"
OUTCOME = "evening_ramp"


def build_dag(edges=None) -> nx.DiGraph:
    """The DAG as a networkx object, with the reason kept on each edge."""
    g = nx.DiGraph()
    for parent, child, why in (edges or NETLOAD_EDGES):
        g.add_edge(parent, child, why=why)
    if not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        raise ValueError(f"the graph is not acyclic; found {cycle}")
    return g


# --------------------------------------------------------------------------
# Backdoor machinery
# --------------------------------------------------------------------------
def _undirected_paths(g: nx.DiGraph, a: str, b: str):
    """Every simple path between a and b IGNORING edge direction.

    Confounding travels along paths regardless of which way the arrows point, so
    the search has to be undirected even though the graph is not.
    """
    return nx.all_simple_paths(g.to_undirected(as_view=True), a, b)


def is_collider(g: nx.DiGraph, path, i: int) -> bool:
    """Is path[i] a collider on this path?  ...-> path[i] <- ...  (both arrows in)"""
    if i == 0 or i == len(path) - 1:
        return False
    return g.has_edge(path[i - 1], path[i]) and g.has_edge(path[i + 1], path[i])


def path_is_open(g: nx.DiGraph, path, given=frozenset()) -> bool:
    """Is this path d-connected given the conditioning set?

    A path is open when EVERY node on it is open:
      * a non-collider is open unless it is conditioned on
      * a collider is CLOSED unless the collider itself, or one of its
        descendants, is conditioned on
    """
    given = set(given)
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider(g, path, i):
            opened = node in given or any(
                d in given for d in nx.descendants(g, node)
            )
            if not opened:
                return False           # collider blocks by default
        else:
            if node in given:
                return False           # chain or fork blocked by conditioning
    return True


def backdoor_paths(g: nx.DiGraph, treatment=TREATMENT, outcome=OUTCOME):
    """Paths from treatment to outcome that leave the treatment through an ARROW IN.

    Those are the paths that carry confounding: they are not effects of the
    treatment, so any association they transmit is not causal.
    """
    out = []
    for path in _undirected_paths(g, treatment, outcome):
        if len(path) > 1 and g.has_edge(path[1], path[0]):
            out.append(tuple(path))
    return out


def open_backdoor_paths(g: nx.DiGraph, given=frozenset(),
                        treatment=TREATMENT, outcome=OUTCOME):
    """The backdoor paths still transmitting association given a conditioning set."""
    return [p for p in backdoor_paths(g, treatment, outcome)
            if path_is_open(g, p, given)]


def is_valid_adjustment_set(g: nx.DiGraph, given, treatment=TREATMENT,
                            outcome=OUTCOME) -> bool:
    """The backdoor criterion, both halves.

    Z is admissible when
      1. no member of Z is a descendant of the treatment, and
      2. Z blocks every backdoor path from treatment to outcome.

    Clause 1 is the one people drop. Controlling for a consequence of the treatment
    removes part of the very effect being estimated, and if that consequence is also
    a collider it manufactures a new association on top.
    """
    given = set(given)
    if given & nx.descendants(g, treatment):
        return False
    return not open_backdoor_paths(g, given, treatment, outcome)


def minimal_adjustment_sets(g: nx.DiGraph, treatment=TREATMENT, outcome=OUTCOME,
                            candidates=None, observed_only: bool = True,
                            max_size: int = 4):
    """All admissible adjustment sets up to `max_size`, keeping only the minimal ones.

    `observed_only` restricts candidates to nodes this project actually measures —
    the difference between "identified in principle" and "estimable with this data".
    """
    if candidates is None:
        candidates = [
            n for n in g.nodes
            if n not in (treatment, outcome)
            and n not in nx.descendants(g, treatment)
            and (not observed_only or OBSERVED.get(n, False))
        ]
    found = []
    for size in range(0, max_size + 1):
        for combo in combinations(sorted(candidates), size):
            z = frozenset(combo)
            if any(prev <= z for prev in found):     # a subset already works
                continue
            if is_valid_adjustment_set(g, z, treatment, outcome):
                found.append(z)
    return found


def describe_paths(g: nx.DiGraph, paths, given=frozenset()):
    """Human-readable path listing, marking colliders and open/closed status."""
    given = set(given)
    lines = []
    for p in paths:
        marks = []
        for i, node in enumerate(p):
            tag = node
            if is_collider(g, p, i):
                tag += " [COLLIDER]"
            if node in given:
                tag += " {adjusted}"
            marks.append(tag)
        status = "OPEN" if path_is_open(g, p, given) else "blocked"
        lines.append(f"  [{status:7s}] " + "  ->  ".join(marks))
    return "\n".join(lines) if lines else "  (none)"


def colliders_in(g: nx.DiGraph):
    """Every node with two or more parents — the colliders of the graph."""
    return sorted(n for n in g.nodes if g.in_degree(n) >= 2)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
# Hand-placed rather than spring-laid-out: a force-directed layout of this graph is
# unreadable and, worse, moves every time the edge list changes, so two runs of the
# notebook would produce two different pictures of the same claim. Columns are
# causal depth, left to right.
DAG_POS = {
    "season": (0.0, 3.2), "zone_char": (0.0, 1.1), "demand_growth": (0.0, -0.6),
    "humidity": (1.15, 4.3), "wind_speed": (1.15, 2.2),
    "cloud_cover": (2.3, 3.6), "temp": (2.3, 2.4),
    "load_shape": (1.6, 0.5), "solar_capacity": (1.6, -1.7),
    "solar_output": (3.5, 1.3), "wind_output": (3.5, 3.2), "demand": (3.5, -0.4),
    "net_load": (4.8, 1.0),
    "evening_ramp": (6.0, 1.0),
}

WEATHER_NODES = ("temp", "humidity", "wind_speed", "cloud_cover")


def draw_dag(ax, g: nx.DiGraph, adjust=frozenset(), treatment=TREATMENT,
             outcome=OUTCOME, pos=None):
    """Draw the DAG, colouring by the role each node plays in the identification.

    Roles are computed, not hardcoded: colliders come from in-degree, the adjustment
    set is whatever was passed in. If the edge list changes, the picture changes with
    it, which is the entire reason the DAG lives in code.
    """
    pos = pos or DAG_POS
    colliders = set(colliders_in(g))
    style = {
        "treatment": ("#3fa34d", "treatment"),
        "outcome": ("#c1121f", "outcome"),
        "adjusted": ("#2b7bba", "in the adjustment set"),
        "collider": ("#f2c14e", "collider — do NOT condition"),
        "weather": ("#8ecae6", "weather (exogenous)"),
        "other": ("#d9d9d9", "other"),
    }

    def role(n):
        if n == treatment:
            return "treatment"
        if n == outcome:
            return "outcome"
        if n in adjust:
            return "adjusted"
        if n in colliders:
            return "collider"
        if n in WEATHER_NODES:
            return "weather"
        return "other"

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="0.55", arrows=True,
                           arrowsize=13, width=1.2, node_size=2600,
                           connectionstyle="arc3,rad=0.06")
    for n in g.nodes:
        r = role(n)
        edge = "0.15" if OBSERVED.get(n, False) else "#c1121f"
        width = 1.4 if OBSERVED.get(n, False) else 2.4
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=[n], node_size=2600,
                               node_color=style[r][0], edgecolors=edge,
                               linewidths=width, node_shape="o")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7.6,
                            labels={n: n.replace("_", "\n") for n in g.nodes})

    import matplotlib.pyplot as plt
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, mfc=c, mec="0.15", label=lab)
               for c, lab in style.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=11, mfc="w",
                              mec="#c1121f", mew=2.4, label="NOT observed"))
    # Lower RIGHT: the lower-left corner holds the demand_growth node, and the
    # legend was drawn over its label in both panels of 03_dag.png.
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_axis_off()
    return ax


# ==========================================================================
# Difference-in-differences
# ==========================================================================
# THE IDENTIFYING ASSUMPTION, stated before any code: PARALLEL TRENDS. Absent the
# treatment, treated and control units would have MOVED BY THE SAME AMOUNT. Note
# what it does not require — it does not require them to sit at the same LEVEL,
# which is why zone fixed effects are free here and why comparing ERCO with NYIS is
# not automatically absurd despite ERCO being seven times the size.
#
# It is also NOT TESTABLE. The treated unit's counterfactual post-period trend is
# never observed. What CAN be examined is whether the groups moved together BEFORE
# the treatment, which is evidence about the assumption rather than a test of it.
# That distinction is the most examinable thing about DiD, and `event_study_frame`
# exists to produce the evidence, not a verdict.

def did_2x2(frame, outcome: str, treated_col: str = "treated",
            post_col: str = "post") -> dict:
    """The four cell means, and the DiD computed from them by hand.

    WHY BY HAND, when the interaction regression gives the same number. Because the
    regression gives it as one coefficient among twenty and you cannot watch it
    arrive. Here the estimate is visibly

        (treated_after - treated_before) - (control_after - control_before)

    which is reproducible on paper under examination. The notebook fits the
    regression as well and asserts the two agree; if they ever disagree, the
    specification is wrong, not the arithmetic.

    The second subtraction is the one that does the work. The first difference
    removes everything permanently different about the treated unit; the second
    removes everything that happened to both groups over the same period. What
    survives is what happened to the treated unit ALONE, in that period, and the
    parallel-trends assumption is the claim that nothing else did.
    """
    cells = frame.groupby([treated_col, post_col])[outcome].mean().unstack()
    for i in (0, 1):
        for j in (0, 1):
            missing = (i not in cells.index or j not in cells.columns
                       or not np.isfinite(cells.loc[i, j]))
            if missing:
                # `unstack` fills an absent combination with NaN rather than dropping
                # the row, so checking the axes alone lets an empty cell through and
                # the DiD comes back as NaN with no explanation.
                raise ValueError(
                    f"the 2x2 has an empty cell: {treated_col}={i}, {post_col}={j}. "
                    "A DiD needs both groups observed in both periods."
                )
    d_treated = float(cells.loc[1, 1] - cells.loc[1, 0])
    d_control = float(cells.loc[0, 1] - cells.loc[0, 0])
    n = lambda a, b: int(((frame[treated_col] == a) & (frame[post_col] == b)).sum())
    return {
        "treated_pre": float(cells.loc[1, 0]), "treated_post": float(cells.loc[1, 1]),
        "control_pre": float(cells.loc[0, 0]), "control_post": float(cells.loc[0, 1]),
        "delta_treated": d_treated, "delta_control": d_control,
        "did": d_treated - d_control,
        "n": {"treated_pre": n(1, 0), "treated_post": n(1, 1),
              "control_pre": n(0, 0), "control_post": n(0, 1)},
    }


def deseasonalise(frame, outcome: str, unit: str, season: str, fit_mask=None):
    """Subtract each unit's own seasonal mean, estimated on `fit_mask` only.

    WHY THIS IS NEEDED HERE. A DiD assumes the groups share their time shocks. Six
    balancing authorities in three climates do not: ERCO peaks in August on air
    conditioning and BPAT peaks in December on electric heat, so a "common year
    effect" is not common. Left uncorrected, the event-study coefficients oscillate
    on a four-quarter cycle and the pre-period looks non-parallel for a reason that
    has nothing to do with solar.

    WHY THE MASK MATTERS, and this is the whole subtlety. The seasonal means are
    fitted on the PRE-period only. Fit them on the full sample and the seasonal norm
    absorbs part of the treatment effect: if the effect is concentrated in summer, a
    full-sample July mean is partly the effect, and subtracting it removes what you
    are trying to measure. Pre-period-only makes the correction a fact about the
    world before the treatment, so it cannot launder the effect away.

    This is the same discipline as fitting a scaler on the training fold only, and
    that is not a coincidence — both are "never let a parameter learn from the thing
    you are about to measure".
    """
    mask = (pd.Series(True, index=frame.index) if fit_mask is None
            else pd.Series(np.asarray(fit_mask), index=frame.index))
    norms = frame[mask].groupby([unit, season])[outcome].mean().rename("_norm")
    joined = frame[[unit, season]].merge(norms, left_on=[unit, season],
                                         right_index=True, how="left")
    return frame[outcome] - joined["_norm"].values


def event_study_frame(frame, time: str, t0, freq: str = "Q", clip=(-12, 9)):
    """Period index relative to the treatment date; -1 is the last pre-period.

    Periods far from the event are CLIPPED into end bins rather than dropped, so the
    tails become one coefficient apiece instead of a row of coefficients each fitted
    on a handful of days. Binning is a variance decision and it is visible in the
    plot; dropping those periods would silently change the sample instead.
    """
    ts = pd.to_datetime(frame[time])
    t0 = pd.Timestamp(t0)
    f = freq.upper()[:1]
    if f == "Q":
        rel = (ts.dt.year - t0.year) * 4 + (ts.dt.quarter - ((t0.month - 1) // 3 + 1))
    elif f == "M":
        rel = (ts.dt.year - t0.year) * 12 + (ts.dt.month - t0.month)
    elif f in ("Y", "A"):
        rel = ts.dt.year - t0.year
    else:
        raise ValueError(f"freq must be Q, M or Y; got {freq!r}")
    return rel.clip(*clip) if clip else rel


# ==========================================================================
# Propensity score matching
# ==========================================================================
# THE THREE ASSUMPTIONS, stated before the code because the code cannot state them:
#
#   1. CONDITIONAL IGNORABILITY (no unmeasured confounding). Given the covariates,
#      treatment is as good as randomly assigned. Unverifiable from data — which is
#      exactly why `robustness_value` exists two sections below.
#   2. POSITIVITY (overlap). Every covariate stratum contains both treated and
#      control units. This one IS checkable, it is checked here, and in this project
#      it FAILS for one of the covariate sets. A positivity failure is not a nuisance
#      to be smoothed over: it means the question has no answer in this data for part
#      of the population, and the honest response is to name which part.
#   3. SUTVA. One unit's treatment does not affect another's outcome. Weak here:
#      balancing authorities trade across seams, so solar built in ERCO can move net
#      load in SWPP. Stated as a limitation, not solved.
#
# WHY A PROPENSITY SCORE AT ALL. Rosenbaum and Rubin's result is that if treatment is
# ignorable given a vector X, it is ignorable given the single number e(X) = P(T=1|X).
# That collapses matching in many dimensions — where nothing is ever close to anything
# — into matching on one. The score is a BALANCING score, not a prediction, and the
# distinction has a practical edge: a propensity model with excellent AUC is BAD news,
# because it means treatment is nearly determined by X and there is little overlap
# left to exploit. This is the one place in the project where a good classifier is a
# bad sign.

def standardised_mean_difference(x, t) -> float:
    """SMD: the mean gap between groups, in pooled standard-deviation units.

    WHY NOT A T-TEST FOR BALANCE, which is what people reach for first. A p-value
    conflates the size of the imbalance with the size of the sample. Matching
    discards units, so the p-value rises even when the imbalance is unchanged, and
    the balance table appears to improve because the sample got smaller. The SMD is
    scale-free and sample-size-free, which is the entire point of using it. Under
    ~0.1 is the conventional bar and it is a convention, not a theorem.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t)
    a, b = x[t == 1], x[t == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _design_matrix(frame, covariates, categoricals=()):
    """Numeric design matrix: continuous columns as they are, categoricals one-hot.

    `covariates` must hold ONLY the continuous columns -- `categoricals` is
    appended, not merged. Passing a column in both produced duplicate one-hot
    columns and a downstream error that named `month_2` without saying why, so
    the overlap is now refused here where the cause is visible.
    """
    dup = sorted(set(covariates) & set(categoricals))
    if dup:
        raise ValueError(
            f"{dup} appear in BOTH covariates and categoricals; list continuous "
            "columns in `covariates` and categorical ones in `categoricals` only")
    X = frame[list(covariates) + list(categoricals)].copy()
    if categoricals:
        X = pd.get_dummies(X, columns=list(categoricals), drop_first=True)
    return X.astype(float)


def propensity_scores(frame, treatment: str, covariates, categoricals=(), C: float = 1.0):
    """Fit e(X) = P(T=1 | X) as a Pipeline([StandardScaler, LogisticRegression]).

    WHY A PIPELINE when no cross-validation fold is involved. The scaler learns two
    parameters per column, so by this repository's own rule it belongs inside a
    Pipeline whether or not a fold boundary is nearby; a rule applied conditionally
    is a rule that gets forgotten somewhere it matters.

    WHY THE SCORE IS FIT IN-SAMPLE, AND WHY THAT IS NOT A LEAK. Expect this question,
    because it looks like the thing the rest of the project forbids. A propensity
    score is not a forecast: no claim is made about unseen units, and its job is to
    balance the sample in hand. The ESTIMATED score is known to balance better than
    the true score would, because the estimate absorbs the chance imbalance of this
    particular sample — which is precisely the imbalance you want removed. Holding
    out a fold would degrade balance in exchange for protecting an out-of-sample
    claim that is never made.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = _design_matrix(frame, covariates, categoricals)
    t = np.asarray(frame[treatment]).astype(int)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("logit", LogisticRegression(max_iter=5000, C=C))]).fit(X, t)
    return pipe.predict_proba(X)[:, 1], X, pipe


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def greedy_match(ps, t, caliper: float = 0.2, replace: bool = False):
    """1:1 nearest-neighbour matching on the LOGIT of the propensity score.

    WHY THE LOGIT AND NOT THE SCORE ITSELF. Distances near 0 and 1 are compressed on
    the probability scale: 0.01 and 0.02 are 0.01 apart and a factor of two apart in
    odds, so matching on the raw score calls them a near-perfect pair. Matching on
    the logit is the standard fix, and the caliper is conventionally defined against
    the logit's standard deviation.

    THE CALIPER is `caliper` x SD(logit(e)), conventionally 0.2. It is a bias-variance
    dial left in the open: tighten it and the surviving pairs are better matched but
    fewer, which raises variance and — more importantly — silently redefines the
    population the estimate refers to. Any treated unit with no control inside the
    caliper is DROPPED, and the count is reported, because an estimate that discarded
    a third of the treated units answers a different question from the one asked.

    WITHOUT REPLACEMENT by default, treated units taken in descending order of score.
    Hardest-to-match-first is the standard greedy order: it spends scarce high-score
    controls on the units that need them, where an arbitrary order strands the
    extremes. Greedy is not optimal — optimal full matching does better — and the
    reason it is used here is that it can be explained in two sentences under
    examination, which this repository weights above sophistication.

    THE COST OF `replace=False`, MEASURED, because it is not a free default. On the
    synthetic benchmark in `tests/test_causal.py` — a planted effect of 2.0 with a
    strong confounder — matching without replacement recovers 2.39 and leaves the
    confounder at SMD 0.19, while matching WITH replacement recovers 2.03 and leaves
    it at SMD 0.005. The reason is that a finite control pool runs out: the caliper
    strands 40% of the treated units, and the ones stranded are systematically the
    extreme ones, so the surviving matched sample is not the treated population.

    So the choice is not "independence is better than balance". It is: without
    replacement buys independent pairs and pays in bias whenever the control pool is
    thin; with replacement buys balance and pays in an effective sample size smaller
    than the pair count, which makes a naive pair bootstrap too confident. When you
    take `replace=True`, quote the CLUSTERED bootstrap from `att_from_pairs` and
    report how many times the most-reused control appears. Notebook 03 uses the
    default instead: it matches pooled across zones, without replacement, and
    quotes the plain pair bootstrap, a lower bound on the uncertainty (see
    `att_from_pairs`).
    """
    lg = _logit(ps)
    t = np.asarray(t).astype(int)
    width = float(caliper * np.std(lg, ddof=1))
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return [], width

    # Hardest first, and `stable` so the tie order is the row order — the run has to
    # reproduce exactly, and argsort's default is not stable.
    order = treated_idx[np.argsort(-lg[treated_idx], kind="stable")]
    c_sorted = control_idx[np.argsort(lg[control_idx], kind="stable")]
    c_vals = lg[c_sorted]
    used = np.zeros(len(c_sorted), dtype=bool)

    pairs = []
    for ti in order:
        pos = int(np.searchsorted(c_vals, lg[ti]))
        lo, hi = pos - 1, pos
        best, best_d = -1, np.inf
        # Walk outwards from the insertion point, always advancing the nearer side.
        # Candidates therefore arrive in increasing distance order, so the first
        # unused one is the nearest unused one and the loop can stop there.
        while lo >= 0 or hi < len(c_vals):
            left_gap = abs(c_vals[lo] - lg[ti]) if lo >= 0 else np.inf
            right_gap = abs(c_vals[hi] - lg[ti]) if hi < len(c_vals) else np.inf
            if left_gap <= right_gap:
                cand, gap = lo, left_gap
                lo -= 1
            else:
                cand, gap = hi, right_gap
                hi += 1
            if gap > width:
                break                     # nothing closer remains inside the caliper
            if replace or not used[cand]:
                best, best_d = cand, gap
                break
        if best >= 0:
            pairs.append((int(ti), int(c_sorted[best])))
            if not replace:
                used[best] = True
    return pairs, width


def balance_table(frame, treatment: str, covariates, pairs):
    """SMD for every covariate, before matching and after — the love plot's data.

    Read it row by row, not as an average: one covariate at |SMD| = 0.4 is not
    excused by nine at 0.02. And note what the table cannot tell you — balance on the
    covariates you MEASURED says nothing about the ones you did not, which is the
    entire reason a sensitivity analysis has to follow it.
    """
    t_all = np.asarray(frame[treatment]).astype(int)
    ti = np.array([p[0] for p in pairs], dtype=int)
    ci = np.array([p[1] for p in pairs], dtype=int)
    t_matched = np.r_[np.ones(len(ti)), np.zeros(len(ci))]
    rows = []
    for c in covariates:
        x = np.asarray(frame[c], dtype=float)
        before = standardised_mean_difference(x, t_all)
        after = (standardised_mean_difference(np.r_[x[ti], x[ci]], t_matched)
                 if len(pairs) else float("nan"))
        rows.append({"covariate": c, "smd_before": before, "smd_after": after,
                     "improved": bool(abs(after) < abs(before)) if len(pairs) else False,
                     "balanced": bool(abs(after) < 0.1) if len(pairs) else False})
    return pd.DataFrame(rows).set_index("covariate")


def att_from_pairs(frame, outcome: str, pairs, n_boot: int = 2000,
                   seed: int = 42, cluster=None) -> dict:
    """ATT as the mean within-pair outcome difference, with a bootstrap CI.

    THE ESTIMAND IS THE ATT, NOT THE ATE, and the difference is not pedantry here.
    Every treated unit is kept (subject to the caliper) and controls are chosen to
    resemble them, so the number answers "what did the treatment do to the units that
    got it". The treated units in this project are zone-days in sunny, solar-building
    territories. An ATE would be a claim about building the same solar in the Pacific
    Northwest, and nothing in this design supports it.

    THE CI IS A LOWER BOUND ON THE UNCERTAINTY, for three reasons worth saying out
    loud rather than burying in a footnote: it treats the propensity model as fixed
    when it was estimated from the same data; pairs are not independent when a control
    is reused; and adjacent days are autocorrelated. `cluster` resamples whole blocks
    (pass a zone-month key) rather than individual pairs, which addresses the third
    and is the version to quote.
    """
    ti = np.array([p[0] for p in pairs], dtype=int)
    ci = np.array([p[1] for p in pairs], dtype=int)
    y = np.asarray(frame[outcome], dtype=float)
    diff = y[ti] - y[ci]
    rng = np.random.default_rng(seed)

    if cluster is None:
        draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                          for _ in range(n_boot)])
    else:
        key = np.asarray(cluster)[ti]
        groups = [np.where(key == k)[0] for k in pd.unique(key)]
        draws = np.empty(n_boot)
        for b in range(n_boot):
            picked = rng.integers(0, len(groups), len(groups))
            draws[b] = diff[np.concatenate([groups[i] for i in picked])].mean()
    return {"att": float(diff.mean()), "n_pairs": int(len(diff)),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "se": float(draws.std(ddof=1)),
            "diffs": diff}


def propensity_match(frame, treatment: str, covariates, categoricals=(),
                     caliper: float = 0.2, replace: bool = False, C: float = 1.0) -> dict:
    """The whole matching step: score, overlap diagnostics, match, balance.

    The diagnostics come back alongside the pairs on purpose. A matched sample handed
    over without its overlap statistics and its balance table is not a result, it is
    a claim.
    """
    f = frame.reset_index(drop=True)
    ps, X, pipe = propensity_scores(f, treatment, covariates, categoricals, C=C)
    t = np.asarray(f[treatment]).astype(int)
    pairs, width = greedy_match(ps, t, caliper=caliper, replace=replace)
    lo = float(max(ps[t == 1].min(), ps[t == 0].min()))
    hi = float(min(ps[t == 1].max(), ps[t == 0].max()))
    numeric = [c for c in list(covariates) + list(categoricals)
               if pd.api.types.is_numeric_dtype(f[c])]
    return {
        "frame": f, "ps": ps, "pairs": pairs, "caliper_width": width,
        "model": pipe, "design_columns": list(X.columns),
        "n_treated": int(t.sum()), "n_control": int((1 - t).sum()),
        "n_matched": len(pairs),
        "matched_share": float(len(pairs) / max(int(t.sum()), 1)),
        "common_support": (lo, hi),
        "treated_outside_support": int(((ps[t == 1] < lo) | (ps[t == 1] > hi)).sum()),
        "control_outside_support": int(((ps[t == 0] < lo) | (ps[t == 0] > hi)).sum()),
        "balance": balance_table(f, treatment, numeric, pairs),
    }


# ==========================================================================
# Sensitivity to an unobserved confounder
# ==========================================================================
# THE PROBLEM THIS SOLVES, and it is the problem that makes observational work
# observational. Every estimate above rests on "no unmeasured confounding", and no
# amount of data checks that assumption, because the offending variable is by
# definition not in the data. The DAG cannot help either: it is the same assumption,
# drawn.
#
# WHAT CAN BE DONE INSTEAD. Stop asking "is there a confounder?" — unanswerable — and
# ask "HOW STRONG would one have to be?". That question has an exact answer as a
# function of the estimate itself, and the answer is a number a domain expert can
# argue with: if overturning the finding requires an omitted cause three times more
# predictive of the ramp than temperature is, someone who knows power systems can say
# whether such a thing plausibly exists.
#
# Two implementations, because they answer the question for two different estimators:
#   robustness_value / ovb_bias   for a regression coefficient  (Cinelli & Hazlett)
#   rosenbaum_bounds              for matched pairs             (Rosenbaum)
# They are not interchangeable. The first bounds OMITTED VARIABLE BIAS in partial-R2
# units; the second bounds the effect of DIFFERENTIAL ASSIGNMENT ODDS within a matched
# pair. Quoting one as if it were the other is a mistake an examiner may look for.

def partial_r2(t_stat: float, dof: int) -> float:
    """Partial R^2 of the treatment with the outcome, from the t-statistic alone.

        R2 = t^2 / (t^2 + dof)

    The share of the outcome variation left over after the other covariates that the
    treatment explains. It is the natural currency for this whole section, because it
    lets the strength of a confounder you cannot see be quoted on the same scale as
    covariates you can.
    """
    t2 = float(t_stat) ** 2
    return float(t2 / (t2 + dof))


def ovb_bias(se: float, dof: int, r2_yz_dx: float, r2_dz_x: float) -> float:
    """The largest bias a confounder of the given strength could cause.

        |bias| = se * sqrt(dof) * sqrt( R2_yz.dx * R2_dz.x / (1 - R2_dz.x) )

    Cinelli & Hazlett (2020), "Making Sense of Sensitivity". The two arguments are
    the confounder's partial R^2 with the OUTCOME (given treatment and covariates)
    and with the TREATMENT (given covariates). The formula is a worst case over the
    confounder's sign and functional form, which is what makes it usable: no
    distributional assumption about the missing variable is needed.

    Read the denominator. As a confounder approaches determining the treatment,
    R2_dz.x -> 1 and the bound explodes — that is the same fact positivity states in
    the matching section, arriving here through the algebra instead.
    """
    r2_yz_dx, r2_dz_x = float(r2_yz_dx), float(r2_dz_x)
    if not (0 <= r2_yz_dx < 1 and 0 <= r2_dz_x < 1):
        raise ValueError("partial R^2 arguments must lie in [0, 1)")
    return float(se * np.sqrt(dof) * np.sqrt(r2_yz_dx * r2_dz_x / (1 - r2_dz_x)))


def robustness_value(t_stat: float, dof: int, q: float = 1.0,
                     alpha: float = 1.0) -> float:
    """RV_q: how strong a confounder must be to reduce the estimate by 100q %.

    Stated as a partial R^2 that the confounder would need with BOTH the treatment
    and the outcome. RV = 0.30 reads as: an unobserved cause would have to explain
    30% of the residual variation in the treatment AND 30% of what is left in the
    outcome to destroy the finding. RV = 0.02 reads as: almost anything would do it.

    q = 1 asks what it takes to drive the estimate to zero. q = 0.5 asks what it
    takes to halve it — often the more honest question, since a business case rarely
    survives a halving either.

    alpha < 1 additionally requires the estimate to be pushed out of significance at
    that level; alpha = 1 (the default) ignores significance and asks only about the
    point estimate, which is the version to quote when the sample is large enough
    that significance is not the binding constraint. It is not, in this project:
    twelve thousand zone-days make almost anything significant, which is exactly why
    the notebook argues about magnitudes rather than p-values.
    """
    from scipy import stats

    f = abs(float(t_stat)) / np.sqrt(dof)
    fq = q * f
    if alpha >= 1.0:
        fqa = fq
        f_crit = 0.0
    else:
        f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
        fqa = fq - f_crit
    if fqa <= 0:
        return 0.0
    rv = 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    if f_crit > 0 and fq > 1 / f_crit:
        rv = (fq ** 2 - f_crit ** 2) / (1 + fq ** 2)
    return float(np.clip(rv, 0.0, 1.0))


def bias_contour(se: float, dof: int, estimate: float, grid=None):
    """The grid behind the sensitivity contour plot: adjusted estimate over (R2d, R2y).

    Returns (r2_d_axis, r2_y_axis, adjusted) where `adjusted` is the point estimate
    after removing the worst-case bias a confounder at that grid point could cause.
    The contour at zero is the set of confounders that would exactly explain away the
    finding; the robustness value is where that contour crosses the diagonal.

    A contour plot is the right display because the question has two dimensions and a
    single number has to pick a point on it. RV picks the diagonal; the plot shows
    every other trade — a confounder weakly related to treatment but strongly to the
    outcome, and the reverse — which is where domain knowledge actually enters.
    """
    grid = np.linspace(0.0, 0.9, 100) if grid is None else np.asarray(grid, dtype=float)
    r2_d, r2_y = np.meshgrid(grid, grid, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = se * np.sqrt(dof) * np.sqrt(r2_y * r2_d / (1 - r2_d))
    adjusted = float(estimate) - np.sign(float(estimate)) * bias
    return grid, grid, adjusted


def rosenbaum_bounds(differences, gammas=(1.0, 1.2, 1.5, 2.0, 3.0, 5.0)):
    """Upper-bound p-values for matched pairs under hidden bias of strength Gamma.

    THE MODEL. Matching equalises the covariates you measured. Suppose it did not
    equalise something you did not measure, so that within a matched pair one unit's
    odds of being treated were up to Gamma times the other's. Gamma = 1 is the
    randomised ideal. Gamma = 2 says the assignment odds could differ twofold for
    reasons invisible to the design. For each Gamma this returns the WORST-CASE
    p-value a Wilcoxon signed-rank test could take under that much hidden bias.

    HOW TO REPORT IT. Quote the Gamma at which the result stops being significant.
    "The finding survives to Gamma = 1.6" means an unobserved confounder would have to
    shift within-pair treatment odds by 60% to overturn it. That is a statement a
    domain expert can accept or reject, which a p-value of 0.001 is not.

    WHAT IT IS NOT. It is not a probability that a confounder exists, and a large
    Gamma is not proof of causation. It is a bound on how fragile the inference is,
    and — a point examiners like — Gamma bounds the ASSOCIATION with treatment only;
    a confounder must also predict the outcome to do damage, which is why the
    Cinelli-Hazlett version above, which takes both, is the stricter statement.
    """
    from scipy import stats

    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]          # zero differences carry no rank information
    if len(d) < 2:
        raise ValueError("need at least two non-zero pair differences")
    ranks = stats.rankdata(np.abs(d))
    stat = float(ranks[d > 0].sum())
    total, total_sq = float(ranks.sum()), float((ranks ** 2).sum())

    rows = []
    for g in gammas:
        p_plus = g / (1.0 + g)                # worst case: treated unit favoured
        p_minus = 1.0 / (1.0 + g)             # best case, for the two-sided reading
        for label, p in (("upper", p_plus), ("lower", p_minus)):
            mean = p * total
            var = p * (1 - p) * total_sq
            z = (stat - mean) / np.sqrt(var)
            rows.append({"gamma": float(g), "bound": label, "z": float(z),
                         "p": float(stats.norm.sf(z))})
    return pd.DataFrame(rows)


def rosenbaum_gamma_critical(differences, alpha: float = 0.05,
                             gmax: float = 20.0, tol: float = 1e-3) -> float:
    """The largest Gamma at which the upper-bound p-value is still below `alpha`.

    Returned as 1.0 when the result is not significant even under no hidden bias —
    which is a meaningful answer, not a failure, and the notebook must print it as
    such rather than hiding the row.
    """
    lo, hi = 1.0, float(gmax)
    p_at = lambda g: float(rosenbaum_bounds([*differences], gammas=(g,))
                           .query("bound == 'upper'")["p"].iloc[0])
    if p_at(1.0) >= alpha:
        return 1.0
    if p_at(hi) < alpha:
        return float(gmax)                    # survives everything on the grid
    while hi - lo > tol:                      # p is monotone in Gamma, so bisect
        mid = (lo + hi) / 2
        if p_at(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return float(lo)
