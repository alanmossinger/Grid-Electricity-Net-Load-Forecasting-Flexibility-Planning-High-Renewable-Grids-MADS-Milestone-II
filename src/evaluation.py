"""Evaluation: skill scores, Diebold-Mariano and Wilcoxon tests with Benjamini-Hochberg, bootstrap
intervals, learning curves, ablation, and the per-region skill table.
"""
from __future__ import annotations


# ---- Scores and tests --------------------------------------------------------------------------

import numpy as np
import pandas as pd

from src import config as cfg


# ===========================================================================
# 1. Skill
# ===========================================================================
def skill_score(error_model: float, error_reference: float) -> float:
    """1 - model/reference. Positive means better than the reference.

    Quoted alongside raw MAE throughout, because a MAE is uninterpretable without
    knowing what is easy. 600 MW is excellent on a 50 GW zone and catastrophic on a
    6 GW one, and the skill score is what makes the six zones comparable.
    """
    if error_reference == 0:
        return float("nan")
    return float(1.0 - error_model / error_reference)


# ===========================================================================
# 2. Diebold-Mariano
# ===========================================================================
def diebold_mariano(errors_a, errors_b, horizon: int = 1, loss: str = "abs",
                    small_sample: bool = True, groups=None, times=None) -> dict:
    """Test whether two forecast error series differ, allowing for autocorrelation.

    A PANEL OF SEVERAL SERIES. The autocovariance at lag k pairs each hour with the hour
    k before it IN THE SAME SERIES. Passed rows of six zones sorted by time, a row lag of
    k reaches k/6 hours back, into other zones, so the HAC sum out to h-1 rows covers
    about four hours instead of a day and the statistic comes out too large. Give
    `groups` (the zone of each row) and `times`: the rows are put in (group, time) order
    and lag products are summed within each group only.

    THE STATISTIC. With loss differential d_t = L(e_a,t) - L(e_b,t),

        DM = mean(d) / sqrt( V_hac(d) / n ),
        V_hac = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k

    A NEGATIVE statistic means model A has the smaller loss.

    WHY THE HAC VARIANCE, and this is the whole point of the test. An h-step-ahead
    forecast error is theoretically an MA(h-1) process: consecutive forecasts made
    from consecutive origins share most of their information, so their errors are
    correlated by construction. Using the ordinary variance would treat 45,000
    correlated hours as 45,000 independent ones and understate the standard error by
    a large factor. Summing autocovariances out to lag h-1 is what the test exists to
    do, and it is why DM is the domain-correct comparison rather than a paired t-test
    with a forecasting label on it.

    THE SMALL-SAMPLE CORRECTION is Harvey, Leybourne and Newbold (1997): the raw
    statistic over-rejects, more so at long horizons, so it is scaled and compared to
    a t distribution rather than a normal. Left on by default; the flag exists so the
    difference can be shown rather than asserted.

    WHAT IT STILL WILL NOT DO. It says nothing about magnitude. A DM p-value of 1e-30
    is compatible with a 2 MW improvement on a 20,000 MW load. Always read it beside
    `bootstrap_ci`.
    """
    from scipy import stats

    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"error series must be the same length: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 10:
        raise ValueError(f"need at least 10 paired errors, got {n}")
    g = None
    if groups is not None:
        g = np.asarray(groups)[ok]
        t = np.asarray(times)[ok] if times is not None else np.arange(n)
        order = np.lexsort((t, g))
        a, b, g = a[order], b[order], g[order]

    if loss == "abs":
        d = np.abs(a) - np.abs(b)
    elif loss == "sq":
        d = a ** 2 - b ** 2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")

    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(dc @ dc) / n

    # Two forecasts that are identical hour by hour give a loss differential that is
    # identically zero, so the statistic is 0/0. That is a tie, not a missing value,
    # and returning nan here would propagate silently into the comparison table.
    # It is a real case: the ladder compares a model against itself as a control.
    if gamma0 == 0.0:
        return {"dm": 0.0, "p": 1.0, "n": int(n), "mean_loss_diff": float(dbar),
                "hac_variance": 0.0, "hac_degenerate": False,
                "better": "tie" if dbar == 0 else ("a" if dbar < 0 else "b")}
    v = gamma0
    for k in range(1, int(horizon)):
        if k >= n:
            break
        if g is None:
            gk = float(dc[k:] @ dc[:-k]) / n
        else:
            same = g[k:] == g[:-k]
            gk = float(dc[k:][same] @ dc[:-k][same]) / n
        v += 2.0 * gk
    if v <= 0:
        # A negative HAC variance is possible with truncated sums and means the
        # correction has broken down. Fall back rather than return a complex number,
        # and say so in the payload so the notebook can report it.
        v, degenerate = gamma0, True
    else:
        degenerate = False

    dm = dbar / np.sqrt(v / n)
    h = int(horizon)
    if small_sample:
        adj = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        dm = dm * adj
        p = 2 * stats.t.sf(abs(dm), df=n - 1)
    else:
        p = 2 * stats.norm.sf(abs(dm))
    return {"dm": float(dm), "p": float(p), "n": int(n),
            "mean_loss_diff": float(dbar), "hac_variance": float(v),
            "hac_degenerate": bool(degenerate),
            "better": "a" if dbar < 0 else ("b" if dbar > 0 else "tie")}


# ===========================================================================
# 3. Blocked paired Wilcoxon
# ===========================================================================
def blocked_wilcoxon(errors_a, errors_b, blocks, loss: str = "abs") -> dict:
    """Paired signed-rank test on BLOCK-mean losses rather than hourly ones.

    WHY PAIRED. Both models forecast the same hours from the same data, so the pairing
    removes the enormous hour-to-hour variation in how hard the target is. An unpaired
    test would be comparing two noisy distributions when the question is about their
    difference.

    WHY BLOCKED, which is the part usually skipped. Wilcoxon assumes independent pairs.
    Hourly forecast errors are strongly autocorrelated — a bad day is bad for
    twenty-four consecutive hours — so 45,000 hourly pairs are nothing like 45,000
    independent ones. Averaging within a block (pass a zone-day key) and testing the
    block means restores approximate independence, at the honest cost of a much
    smaller n. If the result survives that, it is a result.

    WHY DISTRIBUTION-FREE. Absolute errors are strongly right-skewed; a paired t-test
    on them leans on a normality assumption that is visibly false here.
    """
    from scipy import stats

    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    key = np.asarray(blocks)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, key = a[ok], b[ok], key[ok]
    la = np.abs(a) if loss == "abs" else a ** 2
    lb = np.abs(b) if loss == "abs" else b ** 2

    frame = pd.DataFrame({"block": key, "a": la, "b": lb})
    g = frame.groupby("block", sort=True)[["a", "b"]].mean()
    d = (g["a"] - g["b"]).to_numpy()
    d = d[np.isfinite(d)]
    if len(d) < 6:
        raise ValueError(f"only {len(d)} blocks; the test would prove nothing")

    nz = d[d != 0]
    stat, p = stats.wilcoxon(nz) if len(nz) else (np.nan, 1.0)
    return {"stat": float(stat), "p": float(p), "n_blocks": int(len(d)),
            "median_block_diff": float(np.median(d)),
            "wins_a": int((d < 0).sum()), "wins_b": int((d > 0).sum()),
            "better": "a" if np.median(d) < 0 else ("b" if np.median(d) > 0 else "tie")}


# ===========================================================================
# 4. Multiple comparisons
# ===========================================================================
def benjamini_hochberg(pvalues, alpha: float = 0.05) -> pd.DataFrame:
    """Control the false DISCOVERY rate across a family of tests.

    WHY NOT BONFERRONI, which is the reflex. Bonferroni controls the probability of
    ANY false positive, which is the right target when a single false claim is
    catastrophic. Comparing five forecasting models is not that situation: ten tests
    at alpha = 0.05 give roughly a 40% chance of at least one spurious "significant"
    result, but a spurious result here costs a paragraph, not a life. BH controls the
    expected PROPORTION of rejections that are false, which is far less conservative
    and is the standard choice for a screen of this kind.

    Returns the ordered table so the notebook can show the comparison being made:
    p_(i) is rejected when p_(i) <= i/m * alpha.
    """
    p = np.asarray(list(pvalues), dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresh
    cutoff = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:cutoff]] = True
    # Step-up adjusted p-values, monotone by construction.
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return pd.DataFrame({"p": p, "p_adjusted": out, "reject": reject,
                         "rank": np.argsort(order, kind="stable") + 1})


# ===========================================================================
# 5. Effect size
# ===========================================================================
def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 blocks=None, seed: int = cfg.SEED) -> dict:
    """A confidence interval for an effect size, resampling BLOCKS when given.

    Pass `blocks` (a zone-day key) for anything computed on hourly data. Resampling
    individual hours pretends they are independent and returns an interval several
    times too narrow — the same error `blocked_wilcoxon` exists to avoid, in interval
    form. The moving-block idea is the same: resample whole days, keep their internal
    correlation intact.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    if blocks is None:
        draws = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    else:
        key = np.asarray(blocks)[np.isfinite(np.asarray(values, dtype=float))]
        groups = [np.where(key == k)[0] for k in pd.unique(key)]
        draws = np.empty(n_boot)
        for i in range(n_boot):
            picked = rng.integers(0, len(groups), len(groups))
            draws[i] = stat(v[np.concatenate([groups[j] for j in picked])])
    return {"estimate": float(stat(v)),
            "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
            "se": float(draws.std(ddof=1)), "n_boot": int(n_boot)}


# ===========================================================================
# 6. Learning curves — chronological, because sklearn's shuffles
# ===========================================================================
def chronological_learning_curve(fit_predict, frame, sizes=(0.2, 0.4, 0.6, 0.8, 1.0),
                                 time_col: str = cfg.TIME_COL,
                                 val_fraction: float = 0.15) -> pd.DataFrame:
    """Training-set-size curve that takes the MOST RECENT slice at each size.

    WHY NOT `sklearn.model_selection.learning_curve`. It subsamples the training set
    at random. On a time series that produces training sets scattered across the whole
    timeline, which (a) leaks across the validation boundary and (b) answers a
    question nobody asked — "what if I had a random 20% of history?" — instead of the
    real one, "how far back does it pay to keep data?". The plan calls it invalid and
    it is.

    THE CHOICE THIS MAKES, and it is a choice worth defending. Taking the most recent
    slice confounds SIZE with RECENCY: the 20% curve point is both small and recent.
    The alternative — oldest-first — confounds size with staleness instead, and is
    worse, because it is not the decision anyone faces. Nobody chooses whether to keep
    their oldest data; they choose how far back to go from today.

    `fit_predict(train_frame, valid_frame) -> (train_error, valid_error)`.
    """
    f = frame.sort_values(time_col).reset_index(drop=True)
    stamps = pd.to_datetime(f[time_col])
    cut = stamps.quantile(1 - val_fraction)
    train_all, valid = f[stamps <= cut], f[stamps > cut]
    ts_train = pd.to_datetime(train_all[time_col])

    rows = []
    for s in sizes:
        start = ts_train.quantile(1 - s)          # most recent fraction s
        part = train_all[ts_train >= start]
        tr_err, va_err = fit_predict(part, valid)
        rows.append({"fraction": float(s), "n_rows": int(len(part)),
                     "train_start": part[time_col].min(),
                     "train_mae": float(tr_err), "valid_mae": float(va_err)})
    return pd.DataFrame(rows)


# ===========================================================================
# 7. Ablation by feature family
# ===========================================================================
def ablation_by_family(score_without, families, baseline: float) -> pd.DataFrame:
    """Score the pipeline with each feature FAMILY removed, against the full model.

    WHY FAMILIES AND NOT SINGLE FEATURES. `net_load_lag1` and `net_load_lag1_rel` carry
    almost the same information. Drop either alone and the model recovers through its
    twin, so both look worthless; drop the family and the truth appears. Notebook 02
    measured that correlation directly (the eleven lowest-MI features are jointly worth
    46.92 MW while individually near-zero), so single-feature ablation is known to be
    misleading on this feature set rather than suspected to be.

    A POSITIVE delta means removing the family made the error WORSE, i.e. the family
    was earning its place. A negative delta means the model was better without it,
    which happens and should be reported rather than rounded to zero.
    """
    rows = []
    for fam in families:
        mae = score_without(fam)
        rows.append({"family": fam, "mae": float(mae),
                     "delta_vs_full": float(mae - baseline)})
    return (pd.DataFrame(rows).sort_values("delta_vs_full", ascending=False)
            .reset_index(drop=True))


# ---- Per-region skill and the duck shape -------------------------------------------------------

import pandas as pd

from src import models as BL
from src import config as cfg

REGION_ZONES = ["CISO", "ERCO", "MISO", "SWPP", "BPAT", "NYIS"]
NAME = {"CISO": "California ISO", "ERCO": "ERCOT (Texas)", "MISO": "Midcontinent ISO",
        "SWPP": "Southwest Power Pool", "BPAT": "Bonneville (Pacific NW)", "NYIS": "New York ISO"}
VALID_DAYS = 90
H = 24


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def local(frame: pd.DataFrame, zone: str) -> pd.DataFrame:
    d = frame[frame[cfg.GROUP_COL] == zone].copy()
    loc = d[cfg.TIME_COL].dt.tz_localize("UTC").dt.tz_convert(cfg.ZONE_TZ[zone])
    d["lh"], d["yr"] = loc.dt.hour, loc.dt.year
    return d


def duck_shape(panel: pd.DataFrame, L: dict) -> tuple[pd.DataFrame, dict]:
    """The median 2024 day per zone, in local time, and the numbers that describe it."""
    rows, profiles = [], {}
    for z in REGION_ZONES:
        d = local(panel, z)
        y = d[d["yr"] == 2024]
        prof = y.groupby("lh")[["demand", "net_load", "solar", "wind", "hydro"]].median()
        profiles[z] = prof
        net = prof["net_load"]
        trough_h = int(net.idxmin())
        trough_index = 100 * net.min() / net.mean()
        # steepest three-hour rise of the median day and the hour it ends: the evening
        # solar collapse in California and Texas, the morning demand ramp elsewhere
        rise = net.diff(3).dropna()
        ramp_mw, ramp_end = float(rise.max()), int(rise.idxmax())
        gen = y[["solar", "wind", "hydro", "net_generation"]].sum()
        rows.append(dict(
            zone=z, name=NAME[z],
            mean_demand_2024=float(y["demand"].mean()), mean_net_load_2024=float(y["net_load"].mean()),
            solar_pct_gen_2024=100 * gen["solar"] / gen["net_generation"],
            wind_pct_gen_2024=100 * gen["wind"] / gen["net_generation"],
            hydro_pct_gen_2024=100 * gen["hydro"] / gen["net_generation"],
            midday_solar_share_of_demand=100 * prof.loc[13, "solar"] / prof.loc[13, "demand"],
            # wind at night (h0-5) against wind at midday (h11-16), on the median day
            night_to_day_wind=float(prof.loc[0:5, "wind"].mean() / max(prof.loc[11:16, "wind"].mean(), 1.0)),
            trough_hour_local=trough_h, trough_index=trough_index,
            steepest_rise_3h_mw=ramp_mw, rise_end_hour=ramp_end,
            rise_pct_of_mean=100 * ramp_mw / net.mean(),
        ))
    out = pd.DataFrame(rows).set_index("zone")
    # The ledger's own trough hour (notebook 00, raw panel) must agree with the cleaned one.
    for z in REGION_ZONES:
        assert out.loc[z, "trough_hour_local"] == L["nb00"]["duck_curve"]["trough_hour_local_by_zone_2024"][z], z
        assert close(out.loc[z, "trough_index"], L["nb00"]["duck_curve"]["trough_index_by_zone_2024"][z], 1.5), z
    return out, profiles


def per_zone_skill(panel: pd.DataFrame, L: dict) -> pd.DataFrame:
    """Rebuild notebook 04's h=24 benchmark table and split it by zone."""
    df = panel.sort_values([cfg.GROUP_COL, cfg.TIME_COL]).reset_index(drop=True)
    cut = cfg.HOLDOUT_CUT
    dev, hold = df[df[cfg.TIME_COL] <= cut], df[df[cfg.TIME_COL] > cut].copy()
    coefs = BL.fit_renewable_legs(dev)
    coefs_sel = BL.fit_renewable_legs(dev, valid_days=VALID_DAYS)
    chosen = BL.select_leg_methods(dev, coefs_sel, valid_days=VALID_DAYS)
    bl = hold.copy()
    bl["naive168"] = BL.lag_by_time(bl, cfg.TARGET, 168)
    bl["df_composite"] = (bl["demand_forecast"] - BL.lag_by_time(bl, "wind", 24)
                          - BL.lag_by_time(bl, "solar", 24))
    bl["df_composite_v2"] = BL.composite_v2_selected(bl, coefs, chosen, lag_hours=H)["df_composite_v2"]
    pred = pd.read_parquet(cfg.DATA_PROC / "holdout_predictions_by_horizon.parquet")
    pred = pred[pred["horizon"] == H][["timestamp", "zone", "y_pred", "y_true"]]
    bl = bl.merge(pred, on=["timestamp", "zone"], how="left")
    ok = (bl["df_composite_v2"].notna() & bl["naive168"].notna() & bl["y_pred"].notna()
          & bl[cfg.TARGET].notna() & bl["df_composite"].notna())
    s = bl[ok].copy()
    assert (s["y_true"] - s[cfg.TARGET]).abs().max() < 1e-6, "holdout truth mismatch"
    ph = L["nb04"]["per_horizon_skill"]["24"]
    assert len(s) == int(ph["n"]), (len(s), ph["n"])

    def mae(frame, col):
        return float((frame[cfg.TARGET] - frame[col]).abs().mean())

    pooled = {c: mae(s, c) for c in ("y_pred", "naive168", "df_composite", "df_composite_v2")}
    assert close(pooled["y_pred"], ph["model_mae"], 0.5)
    assert close(pooled["naive168"], ph["naive_mae"], 0.5)
    assert close(pooled["df_composite_v2"], ph["composite_v2_mae"], 0.5)
    assert close(pooled["df_composite"], L["nb04"]["baselines"]["df_composite_mae"], 0.5)

    rows = []
    for z, g in s.groupby(cfg.GROUP_COL):
        m = {c: mae(g, c) for c in ("y_pred", "naive168", "df_composite", "df_composite_v2")}
        rows.append(dict(
            zone=z, n_scored=len(g), model_mae=m["y_pred"], naive_mae=m["naive168"],
            df_composite_mae=m["df_composite"], composite_v2_mae=m["df_composite_v2"],
            skill_vs_naive=1 - m["y_pred"] / m["naive168"],
            skill_vs_df_composite=1 - m["y_pred"] / m["df_composite"],
            skill_vs_v2=1 - m["y_pred"] / m["df_composite_v2"],
        ))
    out = pd.DataFrame(rows).set_index("zone")
    hz = L["nb04"]["holdout_by_zone"]
    # The notebook's per-zone MAE is on all 7,896 holdout rows per zone; the skill table is
    # on the rows every benchmark can reach. The two differ most for MISO, whose July data
    # gap (report section on failure 1) is exactly the set of rows no benchmark can score,
    # so its common-row MAE sits ~110 MW below its all-row MAE. Both are reported.
    for z in REGION_ZONES:
        assert close(out.loc[z, "model_mae"], hz[z]["mae"], 150), (z, out.loc[z, "model_mae"], hz[z]["mae"])
    return out

