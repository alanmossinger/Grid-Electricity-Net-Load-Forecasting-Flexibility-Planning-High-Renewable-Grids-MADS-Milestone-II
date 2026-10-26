"""
uncertainty.py — making a quantile forecast mean what it says.

WHY THIS MODULE EXISTS. Notebook 06 shipped a failed goal, G3: five independently
fitted quantile models delivered **81.1% coverage at a nominal 90%**, the median
sat at **0.540** instead of 0.500, and the quantiles **crossed on 8.5% of holdout
rows** — one row in twelve was not an interval at all. Everything priced in Notebook
8 is sized off that distribution, so this is not presentation polish: an interval
8.9 points too narrow under-prices reserve, and the money number inherits the
error.

Three defects, three named fixes, in the order they must be applied:

  1. CROSSING   -> monotone rearrangement (Chernozhukov, Fernandez-Val and
                   Galichon 2010). Sort each row's quantile predictions. It
                   cannot make any individual quantile's loss worse, which is the
                   part that makes it free rather than a trade.
  2. COVERAGE   -> conformalized quantile regression (Romano, Patterson and
                   Candes 2019). Correct the fitted band by one number learned on
                   a held-out calibration block, chosen so that the corrected band
                   covers 1 - alpha of it.

                   ADAPTIVITY, STATED PRECISELY, because the loose version of this
                   claim is wrong and it is measurable. Split conformal on absolute
                   residuals produces a CONSTANT half-width: every hour of the year
                   gets the same band, so a duck evening and a flat night are
                   reported as equally uncertain. Additive CQR does better — the
                   wide rows stay wider — but a scalar margin added to every row
                   COMPRESSES the ratio: a band 10x wider than another comes out
                   only ~1.5x wider once both have 1,600 MW added to each side.
                   The scaled variant (`cqr_margin_scaled`) multiplies instead of
                   adds and preserves the ratio exactly. Both are implemented,
                   both are scored, and the notebook picks on Winkler rather than
                   on which one sounds better.
  3. MEDIAN BIAS-> the same conformal shift applied one-sidedly, level by level.

WHAT CONFORMAL PREDICTION ACTUALLY PROMISES, said out loud because the guarantee
is weaker than it sounds and an examiner will ask. It is MARGINAL coverage,
averaged over all rows, under EXCHANGEABILITY of the calibration and test scores.
This is a time series with drift and a holdout that starts after the calibration
block, so exchangeability does not hold strictly. That is why coverage is
reported per horizon and per regime as well as overall: a method that hits 90%
on average and 70% on duck evenings has not solved the operator's problem, and
only the conditional table can show that.

THE CALIBRATION BLOCK IS CHRONOLOGICAL, never random. A random split would let
the calibration quantile see hours after the ones it is calibrating, and would
report a coverage that cannot be reproduced at an operating desk.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from src import config as cfg


# ---------------------------------------------------------------------------
# 1. monotone rearrangement
# ---------------------------------------------------------------------------
def rearrange(Q: pd.DataFrame | np.ndarray, levels=None):
    """Sort each row's quantile predictions so they are non-decreasing in level.

    THE ARGUMENT FOR WHY THIS IS FREE. A predicted quantile curve that decreases
    somewhere is not a distribution function, so it cannot be right. Chernozhukov,
    Fernandez-Val and Galichon (2010) show that the sorted (rearranged) curve is
    weakly closer to the true curve than the unsorted one in every L^p norm — the
    monotone projection cannot move you away from a monotone target. So the
    rearranged quantiles are no worse in pinball loss on every level, and the
    crossing rate goes to exactly zero. There is no trade to argue about; the only
    reason not to do it would be not knowing it exists.

    WHAT IT DOES NOT FIX. Rearrangement is a statement about ORDER, not about
    LEVEL. A set of quantiles that is uniformly too narrow stays uniformly too
    narrow after sorting, and coverage does not move much. That is what step 2 is
    for, and conflating the two is the commonest way to claim calibration from a
    one-line fix.
    """
    if isinstance(Q, pd.DataFrame):
        cols = list(Q.columns) if levels is None else list(levels)
        out = np.sort(Q[cols].to_numpy(), axis=1)
        return pd.DataFrame(out, index=Q.index, columns=cols)
    return np.sort(np.asarray(Q, dtype=float), axis=1)


def crossing_rate(Q: pd.DataFrame | np.ndarray, levels=None) -> float:
    """Share of rows on which some higher quantile is predicted below a lower one."""
    if isinstance(Q, pd.DataFrame):
        cols = sorted(Q.columns) if levels is None else list(levels)
        A = Q[cols].to_numpy()
    else:
        A = np.asarray(Q, dtype=float)
    return float((np.diff(A, axis=1) < 0).any(axis=1).mean())


# ---------------------------------------------------------------------------
# 2. conformalized quantile regression
# ---------------------------------------------------------------------------
def conformal_quantile_index(n: int, level: float) -> int:
    """The order statistic that carries the finite-sample guarantee.

    The (n+1) rather than n is the whole difference between a proof and an
    approximation: the guarantee is about the rank of the TEST score among the
    n calibration scores plus itself, so the quantile is taken at
    ceil((n+1)(1-alpha)) / n. At n in the tens of thousands it moves the answer by
    a hair; it is still the reason the statement is a theorem.
    """
    return int(min(max(ceil((n + 1) * level), 1), n)) - 1


def cqr_margin(y_cal, lo_cal, hi_cal, alpha: float = 0.10) -> float:
    """The CQR correction: one number, added outward to both ends of the band.

    The conformity score is E_i = max(lo_i - y_i, y_i - hi_i): how far outside the
    fitted band the observation fell, NEGATIVE when it fell inside (and then the
    band is shrunk rather than widened, which is why CQR can end up sharper than
    the model it corrects, not only wider).

    Returns the ceil((n+1)(1-alpha)) order statistic of those scores. The
    corrected band [lo - margin, hi + margin] then covers at least 1 - alpha of
    exchangeable future rows.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    scores = np.maximum(np.asarray(lo_cal, dtype=float) - y_cal,
                        y_cal - np.asarray(hi_cal, dtype=float))
    scores = np.sort(scores)
    return float(scores[conformal_quantile_index(len(scores), 1 - alpha)])


def apply_cqr(lo, hi, margin: float):
    """The corrected band. Kept separate from `cqr_margin` so the margin can be
    fitted once on the calibration block and applied to any number of test blocks
    — refitting it on the block being scored is the leak this signature prevents."""
    return np.asarray(lo, dtype=float) - margin, np.asarray(hi, dtype=float) + margin


def cqr_margin_scaled(y_cal, lo_cal, hi_cal, alpha: float = 0.10,
                      floor: float = 1e-6) -> float:
    """CQR with a MULTIPLICATIVE correction, so the band's shape is preserved exactly.

    The conformity score is normalised by the fitted width,
    E_i = max(lo_i - y_i, y_i - hi_i) / (hi_i - lo_i), so the correction scales
    each row's band rather than padding it. A row the model called 10x more
    uncertain than another is still 10x wider afterwards — which additive CQR
    does not preserve, and which constant-width split conformal destroys outright.

    The price, stated because it is real: the guarantee is the same marginal one,
    and the multiplicative form is more fragile where the fitted width is near
    zero. `floor` keeps the division defined; rows at the floor are exactly the
    rows where the quantile model claimed certainty it did not have.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    lo_cal = np.asarray(lo_cal, dtype=float)
    hi_cal = np.asarray(hi_cal, dtype=float)
    width = np.maximum(hi_cal - lo_cal, floor)
    scores = np.sort(np.maximum(lo_cal - y_cal, y_cal - hi_cal) / width)
    return float(scores[conformal_quantile_index(len(scores), 1 - alpha)])


def apply_cqr_scaled(lo, hi, margin: float):
    """Widen (or narrow) each band by `margin` times its own width."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    return lo - margin * width, hi + margin * width


def conformal_level_shift(y_cal, pred_cal, level: float) -> float:
    """One-sided conformal correction for a SINGLE quantile level.

    CQR above calibrates a band. This calibrates one level, which is what the
    median needs: the score is the signed residual s_i = y_i - qhat_i, and the
    shift is its `level` quantile with the same (n+1) correction, so that
    P(y <= qhat + shift) ~ level. Applied at 0.5 it removes the median bias
    directly; applied at every reported level it is what makes the calibration
    curve in figure `06_calibration.png` lie on the diagonal rather than above it.

    WHY BOTH ARE HERE AND WHICH IS PRIMARY. The two-sided CQR margin carries the
    named finite-sample interval guarantee and is what the 90% band ships with.
    The per-level shifts calibrate the whole curve including the median. They do
    not have to agree, and where they disagree the two-sided one is quoted for the
    band because it is the one with the theorem attached.
    """
    resid = np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float)
    resid = np.sort(resid)
    return float(resid[conformal_quantile_index(len(resid), level)])


def calibrate_levels(y_cal, Q_cal: pd.DataFrame, Q_test: pd.DataFrame) -> tuple:
    """Per-level conformal shifts fitted on calibration, applied to test, rearranged.

    Returns (corrected test quantiles, {level: shift}). Rearrangement runs LAST
    because nothing guarantees the shifts are monotone in the level, and a
    correction that fixed coverage while reintroducing crossings would have
    traded one defect for the other.
    """
    shifts = {q: conformal_level_shift(y_cal, Q_cal[q].to_numpy(), float(q))
              for q in Q_cal.columns}
    out = pd.DataFrame({q: Q_test[q].to_numpy() + shifts[q] for q in Q_test.columns},
                       index=Q_test.index)
    return rearrange(out), shifts


# ---------------------------------------------------------------------------
# 2b. Mondrian (group-conditional) conformal — the margin the operator holds
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. Pooled CQR passed the 90% gate on 2026-08-29 while covering
# 99.25% of BPAT and 80.24% of ERCO: the guarantee is MARGINAL, so a band far too
# wide in one zone and far too narrow in another averages to a number that
# describes no zone in the panel. The median had the same shape with the sign
# flipped — too low in BPAT, too high in CISO — so no single additive shift could
# remove it and the median test was left deliberately red.
#
# Mondrian conformal (Vovk 2012; Romano, Barber, Sabatti and Candes 2020, where it
# is "conformal prediction with equalized coverage") computes the correction
# SEPARATELY within each cell of a taxonomy. The guarantee becomes
# group-conditional: 1 - alpha within every group, under exchangeability within
# that group. That is the margin an operator actually stands at, because reserve
# is held per balancing authority, not per fleet.
#
# WHICH TAXONOMY, and why not the obvious other one. Zone. It is known at
# forecast time. The regime label is NOT: notebook 05 assigns it to a day from that
# day's realised shape, so calibrating on it at h=24 would condition the band on
# a fact about the future. Regime stays a diagnostic table; zone is a
# calibration cell.
#
# THE PRICE. Each cell calibrates on its own rows only — about a sixth of the
# block here, ~8,900 rows a zone — so each margin is a noisier order statistic.
# The finite-sample slack is 1/(n+1), one part in nine thousand: negligible. A
# cell below `min_rows`, or a group at test time that was never calibrated, falls
# back to the POOLED margin and says so in the returned map, so the fallback is
# recorded rather than silent.
POOLED = "__pooled__"


def _cells(groups, min_rows: int):
    groups = np.asarray(groups)
    counts = pd.Series(groups).value_counts()
    return groups, [g for g, n in counts.items() if n >= min_rows]


def mondrian_margins(y_cal, lo_cal, hi_cal, groups_cal, alpha: float = 0.10,
                     min_rows: int = 200, scaled: bool = False) -> dict:
    """{group: CQR margin fitted on that group's calibration rows} plus POOLED.

    `scaled=True` fits the multiplicative margin instead of the additive one; the
    grouping logic is identical. Groups thinner than `min_rows` are not given a
    key, so `apply_mondrian` routes them to POOLED.
    """
    fn = cqr_margin_scaled if scaled else cqr_margin
    y_cal = np.asarray(y_cal, dtype=float)
    lo_cal = np.asarray(lo_cal, dtype=float)
    hi_cal = np.asarray(hi_cal, dtype=float)
    groups, cells = _cells(groups_cal, min_rows)
    out = {POOLED: fn(y_cal, lo_cal, hi_cal, alpha=alpha)}
    for g in cells:
        m = groups == g
        out[g] = fn(y_cal[m], lo_cal[m], hi_cal[m], alpha=alpha)
    return out


def apply_mondrian(lo, hi, groups, margins: dict, scaled: bool = False):
    """Each row corrected by its own group's margin. Returns (lo, hi, used) where
    `used` is the margin actually applied per row — a row on the POOLED fallback is
    visible there rather than indistinguishable from a calibrated one."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    groups = np.asarray(groups)
    used = np.array([margins.get(g, margins[POOLED]) for g in groups], dtype=float)
    if scaled:
        width = hi - lo
        return lo - used * width, hi + used * width, used
    return lo - used, hi + used, used


def mondrian_level_shifts(y_cal, Q_cal: pd.DataFrame, groups_cal,
                          min_rows: int = 200) -> dict:
    """{group: {level: one-sided shift}} plus POOLED — `conformal_level_shift`
    fitted within each cell. This is what a sign-flipping median bias needs: BPAT
    can be shifted up while CISO is shifted down."""
    y_cal = np.asarray(y_cal, dtype=float)
    groups, cells = _cells(groups_cal, min_rows)
    out = {POOLED: {q: conformal_level_shift(y_cal, Q_cal[q].to_numpy(), float(q))
                    for q in Q_cal.columns}}
    for g in cells:
        m = groups == g
        out[g] = {q: conformal_level_shift(y_cal[m], Q_cal[q].to_numpy()[m], float(q))
                  for q in Q_cal.columns}
    return out


def calibrate_levels_by_group(y_cal, Q_cal: pd.DataFrame, Q_test: pd.DataFrame,
                              groups_cal, groups_test, min_rows: int = 200) -> tuple:
    """Group-conditional version of `calibrate_levels`: per-level shifts fitted
    within each calibration cell, applied to the matching test rows, rearranged
    last for the same reason as the pooled version. Returns (corrected test
    quantiles, {group: {level: shift}})."""
    shifts = mondrian_level_shifts(y_cal, Q_cal, groups_cal, min_rows)
    groups_test = np.asarray(groups_test)
    out = {}
    for q in Q_test.columns:
        s = np.array([shifts.get(g, shifts[POOLED])[q] for g in groups_test], dtype=float)
        out[q] = Q_test[q].to_numpy() + s
    return rearrange(pd.DataFrame(out, index=Q_test.index)), shifts


def median_bias_by(y, med, groups) -> pd.DataFrame:
    """Empirical P(y <= median) within each group. The pooled 0.537 hid a BPAT at
    0.426 and a CISO at 0.602; this is the table that shows it."""
    y = np.asarray(y, dtype=float)
    below = y <= np.asarray(med, dtype=float)
    df = pd.DataFrame({"group": np.asarray(groups), "below": below})
    out = df.groupby("group").agg(rows=("below", "size"), p_below=("below", "mean"))
    return out.reset_index()


# ---------------------------------------------------------------------------
# 3. scoring — coverage, sharpness, and the score that prices the trade
# ---------------------------------------------------------------------------
def coverage(y, lo, hi) -> float:
    y = np.asarray(y, dtype=float)
    return float(((y >= np.asarray(lo)) & (y <= np.asarray(hi))).mean())


def coverage_by(y, lo, hi, groups, min_rows: int = 30) -> pd.DataFrame:
    """Coverage within each group — the conditional table marginal coverage hides.

    A method can hit 90% overall while covering 70% of duck evenings and 98% of
    nights. The operator only ever stands in one of those cells at a time, so the
    marginal number is the least useful one to quote alone.

    Groups thinner than `min_rows` are reported with their count rather than
    dropped: a coverage of 1.000 on eleven rows is not evidence and should not be
    silently deleted either.
    """
    y = np.asarray(y, dtype=float)
    inside = (y >= np.asarray(lo)) & (y <= np.asarray(hi))
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    df = pd.DataFrame({"group": np.asarray(groups), "inside": inside, "width": width})
    out = df.groupby("group").agg(rows=("inside", "size"),
                                  coverage=("inside", "mean"),
                                  mean_width=("width", "mean"))
    out["thin"] = out["rows"] < min_rows
    return out.reset_index()


def pinball(y, pred, level: float) -> float:
    """The loss the quantile models are trained on, and the one rearrangement
    provably cannot worsen."""
    e = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(level * e, (level - 1) * e)))


def winkler(y, lo, hi, alpha: float = 0.10) -> float:
    """Width plus a penalty for how far an observation lands outside.

    Coverage and width cannot be traded off by eye — a wider interval always
    covers more — so a method cannot win here by being vague.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    pen = np.where(y < lo, 2 / alpha * (lo - y),
                   np.where(y > hi, 2 / alpha * (y - hi), 0.0))
    return float(np.mean((hi - lo) + pen))


def median_bias(y, med) -> float:
    """Empirical P(y <= median forecast). 0.500 is the promise; 0.540 was delivered."""
    return float((np.asarray(y, dtype=float) <= np.asarray(med, dtype=float)).mean())


def interval_report(y, lo, hi, alpha: float = 0.10) -> dict:
    return {"coverage": coverage(y, lo, hi),
            "mean_width": float(np.mean(np.asarray(hi) - np.asarray(lo))),
            "winkler": winkler(y, lo, hi, alpha)}


# ---------------------------------------------------------------------------
# 4. confidence interval vs prediction interval — and which one sizes reserve
# ---------------------------------------------------------------------------
# The plan asks for both to be implemented and the distinction written out,
# because conflating them is the single cheapest way to under-buy reserve.
#
#   A CONFIDENCE interval covers the MEAN of a quantity. Its half-width shrinks
#   like 1/sqrt(n): with 47,000 holdout hours the mean forecast error is known to
#   within a few tens of megawatts.
#   A PREDICTION interval covers the NEXT OBSERVATION. Its half-width does not
#   shrink with n at all — it is the spread of the errors themselves, thousands
#   of megawatts here — because no amount of history makes tomorrow's wind
#   deterministic (the aleatoric half of the opening distinction).
#
# Reserve is bought against the next hour, not against the average hour, so it
# is sized from the PREDICTION interval. Sizing it from the confidence interval
# would understate it by roughly sqrt(n) — a factor the notebook quotes from the
# data rather than from this comment.
def confidence_interval(x, level: float = 0.90) -> tuple:
    """Two-sided CI for the MEAN of x: mean +/- t * s / sqrt(n)."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, df=n - 1)
    m = x.mean()
    return float(m - t * se), float(m + t * se)


def prediction_interval(x, level: float = 0.90, method: str = "empirical") -> tuple:
    """Two-sided interval for the NEXT observation of x.

    `empirical` (default) takes the (1-level)/2 and (1+level)/2 sample quantiles
    and makes no shape assumption. `gaussian` is mean +/- z * s, kept so the
    notebook can show what the shape assumption costs: forecast errors here are
    heavier-tailed than normal, and a Gaussian band is too narrow exactly where
    the reserve is spent.
    """
    x = np.asarray(x, dtype=float)
    a = (1 - level) / 2
    if method == "gaussian":
        from scipy import stats
        z = stats.norm.ppf(1 - a)
        return float(x.mean() - z * x.std(ddof=1)), float(x.mean() + z * x.std(ddof=1))
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def ci_vs_pi(x, level: float = 0.90) -> dict:
    """Both intervals on the same errors, their half-widths, and the ratio — the
    factor by which reserve would be understated if it were sized from the
    confidence interval."""
    ci = confidence_interval(x, level)
    pi = prediction_interval(x, level)
    pg = prediction_interval(x, level, "gaussian")
    hw = lambda iv: (iv[1] - iv[0]) / 2
    return {"n": int(len(x)), "ci": ci, "pi": pi, "pi_gaussian": pg,
            "ci_half_width": hw(ci), "pi_half_width": hw(pi),
            "pi_gaussian_half_width": hw(pg),
            "ratio_pi_to_ci": hw(pi) / hw(ci) if hw(ci) > 0 else float("inf"),
            "sqrt_n": float(np.sqrt(len(x)))}


def overlap_vs_paired(a, b, blocks=None, level: float = 0.90) -> dict:
    """Overlapping confidence intervals do NOT imply a non-significant difference.

    Two methods scored on the SAME rows have errors that move together — a hard
    hour is hard for both — so the variance of their difference is far smaller
    than the variance of either. Each mean's CI can overlap the other's while the
    paired difference is many standard errors from zero.

    `blocks`: hourly rows are autocorrelated, so a paired test on 47,000 rows
    overstates its own evidence. Pass a block label (zone x date here) and the
    test runs on block means of the difference — fewer, nearly independent
    units, which is the conservative and honest count.
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    hourly_a, hourly_b = float(a.mean()), float(b.mean())
    if blocks is not None:
        # the two levels go on the SAME unit as their difference: block means. Row-level
        # intervals for the levels beside a block-level interval for the difference would
        # compare a too-narrow interval with an honest one
        g = np.asarray(blocks)
        a = pd.Series(a).groupby(g).mean().to_numpy()
        b = pd.Series(b).groupby(g).mean().to_numpy()
    d = a - b
    ci_a, ci_b = confidence_interval(a, level), confidence_interval(b, level)
    ci_d = confidence_interval(d, level)
    if d.std(ddof=1) == 0:        # identical on every unit: no test, no NaN in the ledger
        t_stat, t_p, w_p = 0.0, 1.0, 1.0
    else:
        t = stats.ttest_1samp(d, 0.0)
        t_stat, t_p = float(t.statistic), float(t.pvalue)
        try:
            w_p = float(stats.wilcoxon(d).pvalue)
        except ValueError:        # all differences zero
            w_p = 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_a_hourly": hourly_a, "mean_b_hourly": hourly_b,
            "ci_a": ci_a, "ci_b": ci_b,
            "cis_overlap": bool(ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]),
            "mean_diff": float(d.mean()), "ci_diff": ci_d,
            "n_units": int(len(d)),
            "paired_t": t_stat, "paired_p": t_p,
            "wilcoxon_p": w_p,
            "diff_excludes_zero": bool(ci_d[0] > 0 or ci_d[1] < 0)}


# ---------------------------------------------------------------------------
# the calibration split
# ---------------------------------------------------------------------------
def chronological_calibration_split(frame: pd.DataFrame, frac: float = 0.20,
                                    time_col: str = cfg.TIME_COL):
    """(fit block, calibration block) cut by TIME, never at random.

    The calibration block is the tail of dev — the rows closest to the holdout,
    which is the closest this data comes to satisfying exchangeability with it.
    A random split would additionally leak hours from after the calibrated ones
    into the calibration quantile.
    """
    cut = frame[time_col].quantile(1 - frac)
    return frame[frame[time_col] <= cut], frame[frame[time_col] > cut], cut
