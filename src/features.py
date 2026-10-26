"""Feature pipeline: calendar, lags by horizon, rolling statistics, weather, zone encoding, the
mutual-information selector, the chronological CV split and the clustering pipeline.
"""
from __future__ import annotations


# ---- Pipeline ----------------------------------------------------------------------------------

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from src import config as cfg

# ===========================================================================
# 1. Calendar features
# ===========================================================================


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Hour, day-of-week, month and their cyclical encodings.

    WHY CYCLICAL: hour 23 and hour 0 are one hour apart, but as integers they are
    23 apart. A linear model reads that gap as a cliff; a tree has to spend splits
    reconstructing the wrap. sin/cos at the right period puts them adjacent.

    WHY IT IS A PIPELINE STEP even though it learns nothing: uniformity. Every
    feature definition lives in one object graph, so no notebook can quietly use
    a different `hour`. It is stateless, so it is also the one step that could
    safely sit outside — which is the answer to the obvious oral-exam question.
    """

    def __init__(self, time_col: str = cfg.TIME_COL, add_cyclical: bool = True,
                 add_holiday_proxy: bool = True):
        self.time_col = time_col
        self.add_cyclical = add_cyclical
        self.add_holiday_proxy = add_holiday_proxy

    def fit(self, X, y=None):
        self.feature_names_out_ = None
        return self

    def transform(self, X):
        X = X.copy()
        ts = pd.to_datetime(X[self.time_col])
        X["hour"] = ts.dt.hour
        X["dayofweek"] = ts.dt.dayofweek
        X["month"] = ts.dt.month
        X["dayofyear"] = ts.dt.dayofyear
        X["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        if self.add_cyclical:
            X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
            X["doy_sin"] = np.sin(2 * np.pi * X["dayofyear"] / 365.25)
            X["doy_cos"] = np.cos(2 * np.pi * X["dayofyear"] / 365.25)
            X["dow_sin"] = np.sin(2 * np.pi * X["dayofweek"] / 7)
            X["dow_cos"] = np.cos(2 * np.pi * X["dayofweek"] / 7)
        if self.add_holiday_proxy:
            # Not a holiday calendar — a proxy, and labelled as one. US holidays
            # cluster at end-December and around 4 July; both behave like Sundays
            # on the load curve. A real calendar (`holidays` package) is the named
            # upgrade in docs/decision_log.md.
            X["is_yearend"] = ((ts.dt.month == 12) & (ts.dt.day >= 24)).astype(int)
        return X


# ===========================================================================
# 2. Lags and rolling statistics — the leakage-critical step
# ===========================================================================


class GroupLagFeatures(BaseEstimator, TransformerMixin):
    """Per-zone lags and rolling statistics of the target, with a history buffer.

    `fit` remembers the last `buffer_` rows of each zone in the TRAINING window.
    `transform` prepends that buffer to the rows it is given, computes the
    features on the concatenation, then returns only the requested rows. Result:
    the first row of a validation fold has a real 168-hour lag instead of NaN,
    and that lag is built from training-window observations only.

    WHY LAGGING THE TARGET IS NOT LEAKAGE: every lag is >= 1 hour, so no row ever
    sees its own value. At operational prediction time the operator does know
    yesterday's realised net load. `min(lags) >= 1` is asserted in `fit` rather
    than trusted.

    WHY ROLLING WINDOWS END AT THE FORECAST ORIGIN: `rolling(24).mean()` includes
    the current row, so the window has to exclude it. But excluding only the
    current row is right at h=1 and wrong at every longer horizon — see
    `_roll_shift` below, which is the horizon-correct version and was a live
    leak until notebook 04.
    """

    def __init__(self, lags=cfg.LAGS, roll_windows=cfg.ROLL_WINDOWS,
                 target=cfg.TARGET, group_col=cfg.GROUP_COL, time_col=cfg.TIME_COL,
                 exog_lags=tuple(cfg.EXOG_LAG_FEATURES),
                 roll_ends_at_origin: bool = True):
        self.lags = lags
        self.roll_windows = roll_windows
        self.target = target
        self.group_col = group_col
        self.time_col = time_col
        self.exog_lags = exog_lags
        # False reproduces the pre-notebook 04 leak, on purpose. It is a parameter rather
        # than deleted history because "how much was that leak worth?" is a question
        # with a number, and the only way to get the number is to be able to run the
        # broken arm. Notebook 02 learned this the hard way: its first leakage experiment
        # could not produce a non-zero answer because it had no true control arm.
        # Anything other than True is a control arm and must never ship a result.
        self.roll_ends_at_origin = roll_ends_at_origin

    # -- helpers ------------------------------------------------------------
    @property
    def _lags(self):
        return tuple(int(x) for x in self.lags)

    @property
    def _rolls(self):
        return tuple(int(x) for x in self.roll_windows)

    @property
    def _exog_lag(self):
        """The exogenous lag, chosen so it is observable at the forecast origin.

        THE BUG THIS REPLACES (fixed 2026-08-13). This was hardcoded to 24 for every
        horizon. The project's rule is that a lag L is admissible only if L >= h,
        enforced for the target by `cfg.LAGS_BY_HORIZON` and tested. The exogenous
        lag was exempt from both. At h=48, `temp_lag24` is the temperature 24 hours
        before the target — which is 24 hours AFTER an origin at t-48, and therefore
        not knowable when the forecast is made. At h=72 it is 48 hours after. The
        h=1 and h=24 pipelines were fine; the two long horizons were reading the
        future, by the same definition the target lags are protected against.

        THE RULE. The horizon is `min(lags)` — that is what `lags_for(h)` guarantees.
        Round it UP to a whole number of days, so the lag stays aligned to the same
        hour of day and keeps the diurnal comparison that made 24 the right choice
        in the first place: h=1 and h=24 -> 24, h=48 -> 48, h=72 -> 72.
        """
        horizon = min(self._lags)
        return int(np.ceil(horizon / 24.0) * 24)

    @property
    def _roll_shift(self):
        """Extra hours the rolling window must be pushed back, beyond excluding `t`.

        THE BUG THIS REPLACES (fixed 2026-08-14, notebook 04). `closed='left'` makes the
        window `[t-W, t)` — the W hours ending at t-1. That is exactly right at h=1,
        where the operator standing at origin t-1 does know net load at t-1. It is
        wrong at every longer horizon, and `src/config.py` has carried a comment
        admitting so since notebook 01: at h=24 the window still ends at t-1, handing the
        model twenty-three hours of net load that do not exist yet at an origin of
        t-24. Measured on CISO, the reported roll24 mean was 18,810 MW where the
        admissible window gives 16,505.

        Nothing caught it because the leakage test only asserted that the window
        excludes the CURRENT hour, which it always did, and because every notebook
        before this one ran at h=1, where the two definitions coincide.

        THE RULE. The window must end at the forecast origin, t-h. The existing
        computation evaluated at time `t - (h-1)` yields `[t-h-W+1, t-h]`, which is
        the W hours ending at the origin. So the shift is h-1 — and it is ZERO at
        h=1, which is why notebook 02's numbers are unchanged by this fix.
        """
        if not self.roll_ends_at_origin:
            return 0                      # the control arm — see __init__
        return max(0, min(self._lags) - 1)

    def _needed_cols(self, X):
        cols = [self.time_col, self.group_col, self.target]
        cols += [c for c in (self.exog_lags or ()) if c in X.columns]
        return [c for c in dict.fromkeys(cols) if c in X.columns]

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y=None):
        if min(self._lags) < 1:
            raise ValueError(
                f"lags must all be >= 1 (got {self._lags}); a lag of 0 is the target itself."
            )
        # The rolling window now reaches back W + (h-1) hours, so the history buffer
        # has to cover that too or the first rows of a validation fold lose it.
        self.buffer_ = max(
            max(self._lags),
            (max(self._rolls) + self._roll_shift) if self._rolls else 1,
        )
        cols = self._needed_cols(X)
        # Keep the tail of each zone's training window — nothing else is needed,
        # and keeping the whole training set here would make the fitted pipeline
        # enormous when pickled.
        self.history_ = (
            X[cols]
            .sort_values([self.group_col, self.time_col])
            .groupby(self.group_col, group_keys=False)
            .tail(self.buffer_)
            .reset_index(drop=True)
            .copy()
        )
        self.train_max_ts_ = pd.to_datetime(X[self.time_col]).max()
        return self

    def transform(self, X):
        if not hasattr(self, "history_"):
            raise RuntimeError("GroupLagFeatures.transform called before fit.")
        X = X.copy()
        cols = self._needed_cols(X)

        # Order marker so the original row order and index survive the sort.
        X["__pos__"] = np.arange(len(X))
        frame = X[cols + ["__pos__"]].copy()

        # Only buffer rows STRICTLY older than the block being transformed —
        # otherwise transforming the training set itself duplicates rows.
        hist = self.history_.copy()
        hist = hist[pd.to_datetime(hist[self.time_col]) < pd.to_datetime(frame[self.time_col]).min()]
        if len(hist):
            hist = hist.assign(__pos__=np.nan)
            frame = pd.concat([hist[cols + ["__pos__"]], frame], ignore_index=True)

        frame = frame.sort_values([self.group_col, self.time_col], kind="mergesort")
        frame[self.time_col] = pd.to_datetime(frame[self.time_col])

        # ------------------------------------------------------------------
        # Lags are computed on the TIMESTAMP, not on row position.
        #
        # `shift(24)` reaches back 24 ROWS. The cleaning contract drops hours
        # whose target could not be recovered (outages longer than 3 hours), so
        # rows are not guaranteed contiguous, and 24 rows can be 31 hours. The
        # error is invisible: the column is populated, the model trains, and the
        # lag simply means something slightly different on either side of every
        # outage. Looking the value up at `timestamp - L hours` is exact whether
        # or not the grid has holes. `tests/test_leakage.py` punctures a panel
        # deliberately and checks this.
        # ------------------------------------------------------------------
        keys = pd.MultiIndex.from_arrays(
            [frame[self.group_col].to_numpy(), frame[self.time_col].to_numpy()]
        )
        feats = {}
        for col in [self.target] + [c for c in (self.exog_lags or ()) if c in frame.columns]:
            src = pd.Series(frame[col].to_numpy(), index=keys)
            src = src[~src.index.duplicated(keep="last")]
            lag_list = self._lags if col == self.target else (self._exog_lag,)
            for L in lag_list:
                want = pd.MultiIndex.from_arrays([
                    frame[self.group_col].to_numpy(),
                    (frame[self.time_col] - pd.Timedelta(hours=int(L))).to_numpy(),
                ])
                feats[f"{col}_lag{L}"] = pd.Series(
                    src.reindex(want).to_numpy(), index=frame.index
                )

        # Rolling windows use a TIME-based window with closed='left', which excludes
        # the current hour, and are then looked up at `t - (h-1)` so the window ends
        # at the forecast ORIGIN rather than at t-1. See `_roll_shift`. The lookup is
        # by timestamp for the same reason the lags are: rows are not contiguous.
        if self._rolls:
            shift = self._roll_shift
            ind = frame.set_index(self.time_col)
            want_roll = pd.MultiIndex.from_arrays([
                frame[self.group_col].to_numpy(),
                (frame[self.time_col] - pd.Timedelta(hours=int(shift))).to_numpy(),
            ])
            for W in self._rolls:
                roll = ind.groupby(self.group_col)[self.target].rolling(
                    f"{int(W)}h", closed="left", min_periods=max(2, W // 4)
                )
                for stat in ("mean", "std"):
                    vals = getattr(roll, stat)().reset_index(level=0, drop=True)
                    src = pd.Series(vals.to_numpy(), index=keys)
                    src = src[~src.index.duplicated(keep="last")]
                    feats[f"{self.target}_roll{W}_{stat}"] = pd.Series(
                        src.reindex(want_roll).to_numpy(), index=frame.index
                    )

        # Ramp: the hour-over-hour swing the operator actually has to cover.
        if 1 in self._lags and 24 in self._lags:
            feats[f"{self.target}_ramp_1_24"] = (
                feats[f"{self.target}_lag1"] - feats[f"{self.target}_lag24"]
            )

        out = frame.assign(**feats)
        out = out[out["__pos__"].notna()].sort_values("__pos__", kind="mergesort")

        new_cols = list(feats)
        for c in new_cols:
            X[c] = out[c].to_numpy()
        return X.drop(columns="__pos__")


# ===========================================================================
# 3. Categorical encoding of the zone
# ===========================================================================


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Zone -> smoothed mean of the target, shrunk towards the global mean.

        encoded(z) = (n_z * mean_z + m * global_mean) / (n_z + m)

    WHY SMOOTHING: an unsmoothed group mean is a maximum-likelihood estimate with
    no shrinkage, so a small group's encoded value is mostly noise and the model
    treats it as signal. `m` is the number of pseudo-observations of the global
    mean — it is a hyperparameter and it is tuned in notebook 04.

    WHY IT MUST BE INSIDE THE PIPELINE: fitted on the full panel, a zone's value
    contains the mean of its own validation rows. Notebook 02 quantifies the
    resulting optimism in MAE.

    Unseen zone at transform time -> the global mean. Not NaN, and not a crash:
    the global mean is the correct prior for a group you know nothing about.
    """

    def __init__(self, col: str = cfg.GROUP_COL, smoothing: float = 10.0,
                 out_col: str = "zone_te"):
        self.col = col
        self.smoothing = smoothing
        self.out_col = out_col

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SmoothedTargetEncoder needs y; it is a supervised transform.")
        y = pd.Series(np.asarray(y), index=X.index)
        stats = y.groupby(X[self.col]).agg(["mean", "count"])
        self.global_mean_ = float(y.mean())
        m = float(self.smoothing)
        self.mapping_ = (
            (stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m)
        ).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.out_col] = X[self.col].map(self.mapping_).fillna(self.global_mean_)
        return X


class PerZoneRelative(BaseEstimator, TransformerMixin):
    """Scale-free variants: each target-derived feature over its zone's own past week.

    THE PROBLEM. This panel pools six systems spanning an order of magnitude —
    `net_load_lag1` is ~6,000 MW in BPAT and ~73,000 MW in MISO. A tree can cope,
    but only by spending its first splits separating zones before it can learn any
    shape, and "this zone is running 10% above normal" has a different absolute
    value in every zone. A feature that already expresses that is worth more than
    depth spent rediscovering it.

    WHY A RATIO AND NOT A FITTED Z-SCORE. The textbook per-zone normalisation is
    `(x - mu_zone) / sigma_zone`. Two objections, either sufficient. It learns a
    parameter, so it needs per-fold refitting and becomes one more leak surface to
    defend. And it is STATIC: a zone whose load grows across six years is measured
    against a stale mean for the later folds. Dividing by
    `net_load_roll168_mean` — already causal, already shifted, already in the
    pipeline — gives the same scale-free comparison, **learns nothing**, and moves
    with the zone. `fit` is a genuine no-op and that is the point of the design,
    not an oversight.

    The precedent is `net_load_ramp_1_24`, which is likewise a pure row-wise
    function of two lag columns rather than anything estimated.

    WHY THE DENOMINATOR IS SAFE. It is a 168-hour mean of net load. Measured on the
    real panel its minimum is 3,585 MW (BPAT) and it is never <= 0, so the guard
    below never fires — it is there so that a future zone with a genuinely tiny or
    negative weekly mean produces NaN for the imputer rather than an infinity that
    would poison the tree.
    """

    #  net_load_lag24 / net_load_roll24_mean / net_load_roll168_std / net_load_ramp_1_24
    _SRC = r"^{t}_(lag\d+|roll\d+_(mean|std)|ramp_\d+_\d+)$"

    def __init__(self, target: str = cfg.TARGET, denom_window: int = 168,
                 min_denom: float = 1.0):
        self.target = target
        self.denom_window = denom_window
        self.min_denom = min_denom

    @property
    def denom_col_(self) -> str:
        return f"{self.target}_roll{int(self.denom_window)}_mean"

    def source_columns(self, X) -> list:
        """Target-derived columns eligible for a relative variant.

        The denominator itself is excluded: dividing it by itself is the constant 1,
        which carries no information and would be silently kept by a `k='all'`
        selector.
        """
        pat = re.compile(self._SRC.format(t=re.escape(self.target)))
        return [c for c in X.columns if pat.match(c) and c != self.denom_col_]

    def fit(self, X, y=None):
        return self          # deliberately nothing — see the class docstring

    def transform(self, X):
        if self.denom_col_ not in X.columns:
            return X                       # no weekly mean built; nothing to do
        X = X.copy()
        den = X[self.denom_col_]
        den = den.where(den > self.min_denom)          # -> NaN, never inf
        for c in self.source_columns(X):
            X[f"{c}_rel"] = X[c].to_numpy() / den.to_numpy()
        return X


class WeatherForecastNoise(BaseEstimator, TransformerMixin):
    """Degrade contemporaneous weather to what a forecast at this horizon would be.

    WHY THIS EXISTS. The weather features are ERA5 reanalysis — hindsight. Using
    them at h=72 assumes a perfect three-day weather forecast. This step injects the
    error a real forecast would have carried, so notebook 04 can measure what perfect
    foresight was worth instead of quietly banking it.

    WHY ONLY THE CONTEMPORANEOUS COLUMNS. At an origin t-h, everything at or before
    t-h is observed and everything after it is forecast. Since `GroupLagFeatures`
    now picks an exogenous lag L >= h, `temp_lag{L}` refers to a time at or before
    the origin and is genuinely OBSERVED — it must not be degraded. Only weather at
    time t is a forecast. That is why this step sits AFTER 'lag' in the pipeline:
    the lag is built from clean temperature, and only the same-hour column is
    perturbed. Moving it before 'lag' would corrupt an observed feature.

    WHY IT IS OFF BY DEFAULT. `scale=0.0` is a passthrough, so the pipeline behaves
    exactly as before unless a notebook opts in. The number it injects is an assumption
    with four documented failure modes (see `cfg.NWP_TEMP_MAE_C_BY_LEAD_DAY`), and
    an assumption that silently changes a headline score is worse than no assumption
    at all. It is swept, not applied.

    WHY GAUSSIAN AND UNBIASED. Because that is the least this can assume given one
    published MAE per lead day. It is also, stated plainly, the least damaging
    choice: real forecast error is autocorrelated and clusters in frontal passages,
    so i.i.d. noise understates the harm. Do not describe the output as realistic —
    describe it as a lower bound on the harm at an overstated error level.
    """

    def __init__(self, horizon: int = 1, scale: float = None,
                 columns=("temp",), random_state: int = cfg.SEED):
        self.horizon = horizon
        self.scale = scale
        self.columns = columns
        self.random_state = random_state

    @property
    def sigma_(self) -> float:
        """Standard deviation of the injected error, in degrees C. 0 disables."""
        scale = cfg.WEATHER_NOISE_SCALE if self.scale is None else self.scale
        if not scale:
            return 0.0
        lead = cfg.NWP_LEAD_DAY_BY_HORIZON.get(int(self.horizon))
        if not lead:                      # h=1, or a horizon with no published lead
            return 0.0
        mae = cfg.NWP_TEMP_MAE_C_BY_LEAD_DAY[lead]
        return float(mae * cfg.MAE_TO_SIGMA * scale)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sigma = self.sigma_
        if sigma <= 0:
            return X
        X = X.copy()
        # Seeded from (random_state, horizon) so the h=48 and h=72 arms do not get
        # the identical error sequence, which would make their comparison a
        # comparison of one draw rather than of two lead times.
        rng = np.random.RandomState(self.random_state + 1000 * int(self.horizon))
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].to_numpy() + rng.normal(0.0, sigma, size=len(X))
        return X


class WeatherDerivedFeatures(BaseEstimator, TransformerMixin):
    """Physically motivated transforms of the weather columns. Stateless.

    THREE FEATURES, AND AN HONEST ACCOUNT OF WHO EACH HELPS. This is the part an
    examiner should probe, so it is written down rather than implied.

      hdd = max(0, hinge - temp)      heating-degree arm
      cdd = max(0, temp - hinge)      cooling-degree arm
          Load against temperature is a U — heating below a comfort point, cooling
          above it — which is why `temp` correlates near zero with net load and why
          Notebook 02 selects on mutual information instead. Splitting the U at the
          hinge makes each arm monotone.

      wind_cube_capped = min(v100, v_rated)^3 / 1000
          Turbine output goes as the cube of wind speed below rated speed and is
          flat above it. Cubed km/h reaches 10^5, so it is divided by 1000 to keep
          the column on a scale a linear model can be read on.

    WHO THIS HELPS, AND WHO IT DOES NOT. A gradient-boosted tree is INVARIANT to a
    monotone transform of a single feature: it can put a split at 15 C or at
    43.2 km/h by itself, and `min(v, v_rated)^3` is monotone non-decreasing in v.
    So the champion should score the same with these columns as without, to within
    fold noise. THAT IS A PREDICTION, and notebook 04's ablation tests it. If these
    columns turn out to buy the tree something large, suspect a bug before
    celebrating.

    They earn their place on the LINEAR arm and in the explanations. A single
    `temp` coefficient averages the two halves of the U towards zero — one of the
    documented sources of the 22 sign flips in notebook 04's OLS — and an ALE curve
    with a visible hinge is quotable in a way a smooth one is not.
    """

    def __init__(self, hinge_c: float = cfg.TEMP_HINGE_C,
                 rated_kmh: float = cfg.WIND_RATED_KMH):
        self.hinge_c = hinge_c
        self.rated_kmh = rated_kmh

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "temp" in X.columns:
            t = X["temp"].to_numpy(dtype=float)
            X["hdd"] = np.clip(self.hinge_c - t, 0.0, None)
            X["cdd"] = np.clip(t - self.hinge_c, 0.0, None)
        speed_col = "wind_speed_100m" if "wind_speed_100m" in X.columns else "wind_speed"
        if speed_col in X.columns:
            v = np.clip(X[speed_col].to_numpy(dtype=float), 0.0, self.rated_kmh)
            X["wind_cube_capped"] = v ** 3 / 1000.0
        return X


class SolarClearSkyIndex(BaseEstimator, TransformerMixin):
    """Irradiance normalised by what a clear sky would deliver at that place and hour.

    WHY THIS ONE IS DIFFERENT FROM THE TRANSFORMS ABOVE. `hdd`, `cdd` and the
    capped cubic are monotone functions of a single column, so a tree gains nothing
    from them. The clear-sky index is a RATIO between irradiance and a
    zone-month-hour baseline — an interaction the tree would have to reconstruct
    from three columns and many splits. It is the one derived weather feature that
    can plausibly move the champion.

    WHAT IT MEANS. 1.0 is a cloudless hour; 0.3 is heavy cloud; 0.0 is night. Raw
    GHI conflates two entirely different things — "it is December" and "it is
    overcast" — and the solar belly responds to the second one given the first.

    THE BASELINE IS FITTED, SO IT LIVES IN THE PIPELINE. The clear-sky ceiling is
    estimated as a high quantile of observed GHI per (zone, month, hour) on the
    TRAINING FOLD only, and refit inside every fold. A ceiling estimated on the
    full panel would let the validation rows help define their own denominator —
    a small leak, but the same species as every other one this project bans.

    WHY A QUANTILE AND NOT THE MAXIMUM. The maximum of a few hundred observations
    is a single outlier: one hour of cloud enhancement (bright cloud edges focusing
    beam radiation onto a clear patch) can exceed the true clear-sky value by 20%,
    and dividing by it would compress every other hour of the month. p97.5 is the
    ceiling with the outlier trimmed off, and the index is clipped at 1.2 so a
    genuine enhancement stays visible without dominating.

    WHY NOT A SOLAR-POSITION MODEL. A real clear-sky model (Ineichen, Bird) is
    better physics and would need latitude, elevation, turbidity and a solar
    ephemeris. The empirical ceiling gets most of the signal from data already on
    disk, and — the deciding argument for this project — it can be explained in two
    sentences at an oral exam. Preferring the defensible method over the
    sophisticated one is the standing rule here.
    """

    def __init__(self, ghi_col: str = "ghi", group_col: str = cfg.GROUP_COL,
                 quantile: float = 0.975, floor_wm2: float = 20.0,
                 clip_max: float = 1.2, out_col: str = "clearsky_index"):
        self.ghi_col = ghi_col
        self.group_col = group_col
        self.quantile = quantile
        self.floor_wm2 = floor_wm2
        self.clip_max = clip_max
        self.out_col = out_col

    def _keys(self, X):
        return [X[self.group_col], X["month"], X["hour"]]

    def fit(self, X, y=None):
        self.ceiling_ = None
        if self.ghi_col not in X.columns or "hour" not in X.columns:
            return self
        self.ceiling_ = (X.groupby(self._keys(X), observed=True)[self.ghi_col]
                         .quantile(self.quantile))
        self.global_ceiling_ = float(X[self.ghi_col].quantile(self.quantile))
        return self

    def transform(self, X):
        if self.ceiling_ is None or self.ghi_col not in X.columns:
            return X
        X = X.copy()
        idx = pd.MultiIndex.from_arrays(
            [X[self.group_col], X["month"], X["hour"]])
        # A (zone, month, hour) cell absent from the training fold — a fold that
        # covers ten months cannot know December — falls back to the global
        # ceiling rather than to NaN, so the column is never all-missing and the
        # imputer never has to invent the difference between night and cloud.
        ceiling = self.ceiling_.reindex(idx).to_numpy()
        ceiling = np.where(np.isnan(ceiling), self.global_ceiling_, ceiling)
        ghi = X[self.ghi_col].to_numpy(dtype=float)
        # Below the floor the sun is effectively down: the ratio is meaningless
        # (0/0), and 0.0 is the physically right answer for "no sunlight to block".
        idxv = np.where(ceiling >= self.floor_wm2, ghi / np.maximum(ceiling, 1e-9), 0.0)
        X[self.out_col] = np.clip(idxv, 0.0, self.clip_max)
        return X


class WeatherForecastSwap(BaseEstimator, TransformerMixin):
    """Replace the CONTEMPORANEOUS weather with the forecast issued at the origin.

    WHY THIS EXISTS, AND WHY IT SUPERSEDES `WeatherForecastNoise`. Every score this
    project has reported was computed with ERA5 reanalysis in the same-hour weather
    columns — weather as it turned out to be. At h=24, 48 and 72 that hands the model
    a perfect weather forecast. `WeatherForecastNoise` patched the gap by adding
    Gaussian noise calibrated to a published 1994-95 temperature-error curve, for
    temperature only, with four documented ways of being wrong. This transformer does
    the thing the noise was standing in for: it substitutes the value a real model run
    issued N days earlier actually predicted for that hour. No distributional
    assumption, no vintage problem, no missing variables — the error is whatever the
    forecast's error was, including its autocorrelation and its regime dependence,
    which is precisely what i.i.d. noise cannot reproduce.

    WHY IT SITS AFTER 'lag', AND THIS IS THE WHOLE CORRECTNESS ARGUMENT. At an origin
    t-h, everything at or before t-h is OBSERVED and everything after it is forecast.
    `GroupLagFeatures` picks an exogenous lag L >= h, so `temp_lag{L}` refers to a
    time at or before the origin and is genuinely known — it must keep its reanalysis
    value. Only the same-hour columns are forecasts. Placing this step before 'lag'
    would corrupt observed history with forecast error and overstate the penalty; the
    lag columns are built first and are never touched here.

    IT IS A PASSTHROUGH WITHOUT A FRAME, so every existing pipeline behaves exactly
    as before. A notebook opts in by handing it a forecast table.

    WHAT IT REFUSES TO DO. If a row has no forecast for a column, the reanalysis value
    stays and the row is COUNTED in `n_missing_`. It does not silently fall back and
    it does not drop the row: falling back quietly would understate the degradation
    being measured, which is the one direction this measurement must not be wrong in.
    `coverage_` is exposed so a caller can assert on it.
    """

    def __init__(self, forecast=None, columns=None,
                 time_col: str = cfg.TIME_COL, group_col: str = cfg.GROUP_COL):
        self.forecast = forecast
        self.columns = columns
        self.time_col = time_col
        self.group_col = group_col

    def fit(self, X, y=None):
        self.lookup_ = None
        if self.forecast is None:
            return self
        fc = self.forecast.copy()
        fc[self.time_col] = pd.to_datetime(fc[self.time_col])
        cols = list(self.columns) if self.columns is not None else [
            c for c in fc.columns
            if c not in (self.time_col, self.group_col, "lead_day")]
        fc = fc.set_index([self.group_col, self.time_col])
        self.lookup_ = fc[cols]
        self.lookup_ = self.lookup_[~self.lookup_.index.duplicated(keep="last")]
        self.swapped_columns_ = cols
        return self

    def transform(self, X):
        if self.lookup_ is None:
            self.n_missing_, self.coverage_ = 0, 1.0
            return X
        X = X.copy()
        want = pd.MultiIndex.from_arrays(
            [X[self.group_col].to_numpy(),
             pd.to_datetime(X[self.time_col]).to_numpy()])
        aligned = self.lookup_.reindex(want)
        n_missing = 0
        for c in self.swapped_columns_:
            if c not in X.columns:
                continue
            new = aligned[c].to_numpy()
            miss = np.isnan(new)
            n_missing += int(miss.sum())
            X[c] = np.where(miss, X[c].to_numpy(), new)
        self.n_missing_ = n_missing
        denom = max(len(X) * max(len(self.swapped_columns_), 1), 1)
        self.coverage_ = 1.0 - n_missing / denom
        return X


class NoZoneEncoding(BaseEstimator, TransformerMixin):
    """Encode the zone not at all. Exists so the leakage experiment can be honest.

    WHY THIS IS NOT DEAD CODE. Notebook 02 measures what fitting a target encoder on
    the whole panel costs, by scoring the same CV twice. Until 2026-08-13 the
    "broken" arm was built by ADDING a full-panel zone mean to a pipeline that
    still contained a correctly refit `SmoothedTargetEncoder`, so it compared
    in-fold encoding against in-fold encoding plus a redundant copy. The measured
    optimism was 0.0 MW, and it could not have been anything else: the model
    already had the correct column, so the leaked near-duplicate carried no
    information the tree could use.

    A leakage experiment must REPLACE the correct step, not supplement it. This
    passthrough is what lets the broken arm carry the full-panel encoding alone.
    """

    def __init__(self, col: str = cfg.GROUP_COL):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


class OneHotZone(BaseEstimator, TransformerMixin):
    """One-hot the zone. Included so notebook 02 can compare encodings under one CV.

    Nominal and honest, at the cost of one column per zone. With six zones that
    cost is nothing; the comparison exists because with 600 zones it would not be.
    """

    def __init__(self, col: str = cfg.GROUP_COL):
        self.col = col

    def fit(self, X, y=None):
        self.categories_ = sorted(pd.Series(X[self.col]).dropna().unique().tolist())
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.categories_:
            X[f"{self.col}_is_{c}"] = (X[self.col] == c).astype(int)
        return X


class OrdinalZone(BaseEstimator, TransformerMixin):
    """Zone -> integer. Present in order to be argued AGAINST in notebook 02.

    Zone is nominal, so integer coding asserts CISO < ERCO < MISO, an ordering
    that does not exist. It often still scores well with trees, because a tree can
    isolate any integer with enough splits — which is why "it scored well" is not
    a defence of it.
    """

    def __init__(self, col: str = cfg.GROUP_COL, out_col: str = "zone_ord"):
        self.col = col
        self.out_col = out_col

    def fit(self, X, y=None):
        self.mapping_ = {c: i for i, c in enumerate(sorted(pd.Series(X[self.col]).dropna().unique()))}
        return self

    def transform(self, X):
        X = X.copy()
        X[self.out_col] = X[self.col].map(self.mapping_).fillna(-1).astype(int)
        return X


# ===========================================================================
# 4. Feature selection
# ===========================================================================


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Keep the top-k numeric features by mutual information with the target.

    WHY MI AND NOT CORRELATION: hour-of-day has near-zero linear correlation with
    net load and enormous predictive value, because the relationship is periodic.
    Pearson would drop it. MI measures any dependence, monotone or not.

    WHY IT IS A PIPELINE STEP: the ranking is estimated from data. Selecting
    features once on the full panel and then cross-validating is a textbook
    leak — the validation rows helped choose the columns.

    Cost control: MI is a k-NN estimator and is O(n log n) per feature with a
    large constant. It is estimated on a random subsample (`mi_sample`); the
    ranking is stable long before the estimate is precise.

    LEAKAGE GUARD: `DROP_ALWAYS` removes the identifier columns and — critically —
    any raw column that shares a term with the target. `demand`, `wind` and
    `solar` define net_load exactly (net_load = demand - wind - solar), so leaving
    any of them in makes the problem arithmetic rather than forecasting.

    That guard was NINE COLUMNS TOO NARROW until 2026-08-13. The notebook 02 audit
    (notebook 02 §2.15) found that the per-fuel generation columns sum to the
    target and reconstruct it at r = 0.9997, reaching the same failure by a route
    the original list did not cover. They are excluded here via
    `cfg.REALISED_GENERATION_COLS`. Do not re-inline that list: the tests read the
    same constant, so the pipeline and its guard cannot drift apart.
    """

    DROP_ALWAYS = (
        cfg.TIME_COL,
        cfg.GROUP_COL,
        cfg.TARGET,
        "demand",
        "wind",
        "solar",
        "net_generation",
        "total_interchange",
        "demand_forecast",
        "date",
        # `ramp` is net_load(t) - net_load(t-1). It CONTAINS the current target,
        # so a model given it is reading its own answer. It exists for notebook 01
        # profiling and for the flexibility framing, and it is never a feature.
        "ramp",
        "hour",   # re-derived by CalendarFeatures; the raw copy would duplicate it
        # Same-hour realised generation by fuel, and the share that is
        # algebraically the target. See cfg.REALISED_GENERATION_COLS for the two
        # independent reasons and the measured correlations.
        *cfg.REALISED_GENERATION_COLS,
        "renewable_share",
    )

    def __init__(self, k="all", mi_sample: int = 20_000, random_state: int = cfg.SEED):
        self.k = k
        self.mi_sample = mi_sample
        self.random_state = random_state

    def _candidates(self, X):
        num = X.select_dtypes(include=[np.number])
        cands = [c for c in num.columns if c not in self.DROP_ALWAYS]
        # A column that is entirely missing in the training fold is not a feature.
        # It also breaks the contract downstream: SimpleImputer silently DROPS
        # all-NaN columns, so the transformed matrix would have fewer columns than
        # `selected_` names and every SHAP plot in notebook 07 would be mislabelled by
        # one position. EIA returns an empty `coal` series for hydro-dominated BAs,
        # so this is a real case, not a hypothetical one.
        return [c for c in cands if not X[c].isna().all()]

    def fit(self, X, y=None):
        cands = self._candidates(X)
        if not cands:
            raise ValueError("MutualInfoSelector: no numeric candidate features left.")
        if self.k == "all" or int(self.k) >= len(cands):
            self.selected_ = cands
            self.scores_ = None
            return self

        # Gaps are filled with each column's median, as the imputer after this step
        # fills them for the model. Dropping incomplete rows instead removed whole
        # zones: a fuel a zone never burns (oil in BPAT, ERCO and MISO) is missing on
        # every one of its rows, so the ranking was computed on three zones of six.
        Xc = X[cands].fillna(X[cands].median())
        if y is None:
            raise ValueError("MutualInfoSelector needs y.")
        yv = pd.Series(np.asarray(y), index=X.index)
        mask = yv.notna()
        Xc, yv = Xc[mask], yv[mask]
        if len(Xc) > self.mi_sample:
            idx = np.random.RandomState(self.random_state).choice(
                len(Xc), self.mi_sample, replace=False
            )
            Xc, yv = Xc.iloc[idx], yv.iloc[idx]
        if len(Xc) < 50:  # degenerate fold — keep everything rather than guess
            self.selected_, self.scores_ = cands, None
            return self
        mi = mutual_info_regression(Xc.to_numpy(), yv.to_numpy(),
                                    random_state=self.random_state)
        order = np.argsort(mi)[::-1][: int(self.k)]
        self.scores_ = dict(zip(cands, mi.round(4).tolist()))
        self.selected_ = [cands[i] for i in sorted(order)]
        return self

    def transform(self, X):
        missing = [c for c in self.selected_ if c not in X.columns]
        if missing:
            raise KeyError(f"MutualInfoSelector: columns missing at transform: {missing}")
        return X[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


# ===========================================================================
# 5. Cross-validation for a PANEL
# ===========================================================================


class PanelTimeSeriesSplit(BaseCrossValidator):
    """Expanding-window CV that splits on unique timestamps, not on rows.

    `gap` is in TIME STEPS (hours), which is the unit the leakage argument is
    stated in: the longest rolling window is 168 hours, so 168 hours of the
    timeline after each training block are discarded. sklearn's TimeSeriesSplit
    counts rows, and on a six-zone panel `gap=168` would silently mean 28 hours.

    Yields positional integer indices, so `df.iloc[tr]` works as expected.
    """

    def __init__(self, n_splits: int = cfg.N_SPLITS, gap: int = cfg.CV_GAP,
                 time_col: str = cfg.TIME_COL, max_train_size: int | None = None):
        self.n_splits = n_splits
        self.gap = gap
        self.time_col = time_col
        self.max_train_size = max_train_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if self.time_col not in getattr(X, "columns", []):
            raise ValueError(
                f"PanelTimeSeriesSplit needs a DataFrame with a '{self.time_col}' column. "
                f"Pass the panel itself as X — the pipeline consumes the frame, not a matrix."
            )
        ts = pd.to_datetime(X[self.time_col]).to_numpy()
        stamps = np.unique(ts)
        n = len(stamps)
        fold = n // (self.n_splits + 1)
        if fold <= self.gap:
            raise ValueError(
                f"{n} timestamps is too few for {self.n_splits} splits with gap={self.gap}. "
                f"Reduce N_SPLITS or CV_GAP in src/config.py, or widen the date range."
            )
        pos = pd.Series(np.arange(len(ts)))
        for i in range(1, self.n_splits + 1):
            train_end = fold * i
            val_start = train_end + self.gap
            val_end = min(train_end + fold, n)
            if val_start >= val_end:
                continue
            train_start = 0
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            tr_stamps = stamps[train_start:train_end]
            va_stamps = stamps[val_start:val_end]
            tr = pos[np.isin(ts, tr_stamps)].to_numpy()
            va = pos[np.isin(ts, va_stamps)].to_numpy()
            yield tr, va


def make_ts_cv(n_splits: int = cfg.N_SPLITS, gap: int = cfg.CV_GAP, **kw):
    """The one cross-validator. Every notebook uses this, so scores are comparable."""
    return PanelTimeSeriesSplit(n_splits=n_splits, gap=gap, **kw)


def with_history(pipe: Pipeline, observed: pd.DataFrame) -> Pipeline:
    """Give a fitted pipeline the lag history of `observed`, and change nothing else.

    The lag step's buffer is the tail of the window the pipeline was FITTED on, which
    is right only when the block being forecast starts the next hour. Notebook 06
    fits its calibrated models on the first 80% of dev and forecasts the holdout,
    which starts a year later: without this, every lag of the holdout's first
    week reaches into hours the buffer does not hold, is imputed, and the first day
    is forecast almost without regard to the region. At forecast time those hours
    are simply the recent past, so the buffer is rebuilt from `observed` (the whole
    dev block). Every fitted parameter - encoder, selector, imputer, model - stays
    exactly as fitted; only the buffer changes, and `transform` still drops any
    buffer row that is not strictly older than the block it is given.
    """
    lag = pipe.named_steps["lag"]
    lag.history_ = clone(lag).fit(observed).history_
    return pipe


# ===========================================================================
# 6. The supervised pipeline
# ===========================================================================


def make_supervised_pipeline(numeric_cols=cfg.NUMERIC_FEATURES, lags=cfg.LAGS,
                             roll_windows=cfg.ROLL_WINDOWS, model=None,
                             # 'onehot' is the notebook 02 decision, 2026-08-13: scored
                             # 579.63 MW against target encoding's 583.73, better in
                             # 5 of 5 paired folds. Target encoding compresses zone
                             # into a near-restatement of lag1 (r = +0.91) and loses
                             # the identity a tree can otherwise use. See
                             # Notebook 02 section 2.7 and the decision log.
                             encoding: str = "onehot", k="all",
                             smoothing: float = 10.0,
                             weather_noise_scale: float = None,
                             per_zone_relative: bool = True,
                             roll_ends_at_origin: bool = True,
                             exog_lags=None,
                             derive_weather: bool = True,
                             weather_forecast=None,
                             weather_forecast_columns=None,
                             drop_family: str | None = None) -> Pipeline:
    """Assemble the supervised pipeline.

    Steps, in order, and why each is inside:
      calendar  stateless, but kept inside so one definition of `hour` exists
      lag       fits a history buffer from the training window only
      encode    fits group means from the training fold's y
      select    fits an MI ranking from the training fold
      impute    fits medians from the training fold (Ridge cannot take NaN;
                HistGradientBoosting can, but the pipeline must work for both)
      model     the estimator

    `numeric_cols` is carried on the pipeline for notebook 07's feature grouping and
    does NOT build features -- the pipeline hands every frame column to `select`
    and lets DROP_ALWAYS filter. `exog_lags` is the list that does decide which
    columns get a lagged copy. They are not the feature set either: the feature
    set is what `select` keeps.
    """
    if model is None:
        model = HistGradientBoostingRegressor(random_state=cfg.SEED)

    encoders = {
        "target": SmoothedTargetEncoder(smoothing=smoothing),
        "onehot": OneHotZone(),
        "ordinal": OrdinalZone(),
        # Not an encoding choice to compare in 2.7 — the control arm for the
        # leakage experiment in 2.5. See NoZoneEncoding.
        "none": NoZoneEncoding(),
    }
    if encoding not in encoders:
        raise ValueError(f"encoding must be one of {sorted(encoders)}, got {encoding!r}")

    steps = [
        ("calendar", CalendarFeatures()),
        # exog_lags is passed EXPLICITLY rather than left to the transformer's
        # default. It was defaulted for the whole project and nobody noticed the
        # model was running on one exogenous column; a parameter that matters
        # this much belongs in the signature where an ablation can reach it.
        ("lag", GroupLagFeatures(lags=tuple(lags), roll_windows=tuple(roll_windows),
                                 exog_lags=tuple(exog_lags if exog_lags is not None
                                                 else cfg.EXOG_LAG_FEATURES),
                                 roll_ends_at_origin=roll_ends_at_origin)),
        # Consumes the lag/rolling columns, so it must follow 'lag'. A passthrough
        # when switched off, which is how notebook 02 measures whether it earns its keep.
        ("relative", PerZoneRelative() if per_zone_relative else "passthrough"),
        # AFTER 'lag' deliberately: the lag must be built from observed weather.
        # A passthrough unless a notebook sets weather_noise_scale. See the class.
        ("weather", WeatherForecastNoise(horizon=min(int(l) for l in lags),
                                         scale=weather_noise_scale)),
        # AFTER 'weather' on purpose: hdd/cdd and the clear-sky index are functions
        # of the same-hour weather columns, so if a notebook is injecting forecast
        # error they must see the degraded values. Computing them before the noise
        # step would leave a clean copy of the temperature in the design matrix and
        # silently undo the sensitivity being measured.
        # BETWEEN 'weather' and 'derive', and both neighbours are load-bearing.
        # After 'lag' (and therefore after 'weather'), so the lag columns keep their
        # observed values -- a lag L >= h is knowable at the origin and is not a
        # forecast. Before 'derive', so hdd, cdd and the clear-sky index are computed
        # from the FORECAST weather rather than from the reanalysis the operator did
        # not have. Passthrough unless a notebook hands it a forecast table.
        ("swap", WeatherForecastSwap(forecast=weather_forecast,
                                     columns=weather_forecast_columns)),
        ("derive", Pipeline([("physical", WeatherDerivedFeatures()),
                             ("clearsky", SolarClearSkyIndex())])
                   if derive_weather else "passthrough"),
        ("encode", encoders[encoding]),
        # A passthrough unless notebook 04 is ablating. It sits BEFORE `select` so the MI
        # ranking is refitted on what survives rather than inherited with holes.
        ("ablate", DropFeatureFamily(family=drop_family)),
        ("select", MutualInfoSelector(k=k)),
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ]
    pipe = Pipeline(steps)
    # Carried so notebook 07 can group features by family without re-deriving them.
    pipe.numeric_cols_ = list(numeric_cols)
    return pipe


# Two-stage search: this locates the region (RandomizedSearchCV), then notebook 04 §4.4
# refines it with a narrow GridSearchCV. Ranges are deliberately wide — an
# optimum sitting on a boundary means the range was too narrow, not that you won.
SUPERVISED_SEARCH_SPACE = {
    "lag__lags": [(1, 24, 168), (1, 2, 24, 168), (1, 24, 48, 168), (1, 24, 168, 336)],
    "lag__roll_windows": [(24,), (24, 168), (6, 24, 168)],
    # `encode__smoothing` was REMOVED on 2026-08-13 for two independent reasons.
    # It only exists on SmoothedTargetEncoder, and the notebook 02 encoding comparison
    # chose one-hot — so on the shipped pipeline the key does not resolve at all.
    # And it was never a real axis: section 2.11 measured smoothing at m=10 moving
    # the encoded value by at most 8.31 MW against zone means of 5,710-63,821 MW,
    # because with 44,000+ rows per zone m in {1, 10, 50} are the same encoder to
    # four significant figures. Searching it would have spent grid cells to
    # rediscover that the formula behaves correctly at high cardinality.
    # RANGES SET FROM MEASURED VALIDATION CURVES, 2026-08-20. The previous ranges
    # put the optimum ON AN EDGE for four axes at once, which means the search was
    # reporting where the box stopped rather than where the model is best:
    #     max_iter          searched [200, 400, 600]     -> reported winner 1200
    #     min_samples_leaf  searched [20, 50, 100]        -> won at 100, top
    #     max_depth         searched [None, 4, 6, 10]     -> won at 10, UNCHECKED
    #     select__k         searched ["all", 12, 18, 25]  -> won at max, UNCHECKED
    # The boundary check only covered `numeric_axes`, so the last two escaped it.
    #
    # Each range below is centred so the measured optimum is interior. Spans are
    # against a 26.0 MW fold noise floor; see reports/figures/
    # 04_hyperparam_validation_curves.png and 04_hyperparam_search_box.png.
    #
    # "all" stays first: it wins by 796 MW over k=8 and is the structural ceiling,
    # which is notebook 02's never-flattening forward curve showing up again.
    "select__k": ["all", 45, 35, 25],
    # U-shaped, minimum at 0.02-0.03; 0.3 costs 186 MW. Matters.
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    # U-shaped with the minimum at 800, MEASURED: 400 -> 2,216.8, 800 -> 2,211.2,
    # 1200 -> 2,213.5, 2000 -> 2,222.6, 4000 -> 2,244.7. More boosting rounds
    # eventually overfit, which is also why the old escalation to 2400 bought
    # 0.0 MW.
    #
    # CAPPED AT 1200, not 2000, and this is a compute decision made on evidence
    # rather than a shortcut. A fit at max_iter=2000 costs ~2.5x one at 800, and
    # the sweep already established that region is WORSE. Searching it spends
    # hours to rediscover a measured negative: the run that reached it sat in the
    # search cell for over two hours without finishing. 1200 keeps the optimum of
    # 800 interior, which is the only property the boundary check needs.
    "model__max_iter": [400, 800, 1200],
    # Flat past 8: the wide sweep measured 2,208.5 MW at depth 8 against 2,213.2 at
    # None (unlimited), a 4.7 MW difference inside a 26.0 MW noise floor. `None` is
    # dropped because it is the single largest COST driver in the search -- an
    # unlimited-depth booster at max_iter=2000 dominated the run that timed out at
    # 7200s -- and it buys nothing measurable. 14 is the top of the range and is
    # itself 6 MW worse than 8, so the optimum stays interior without it.
    "model__max_depth": [4, 6, 8, 10, 14],
    # Span 8.7 MW against a 26.0 MW noise floor: this axis does not matter. Kept
    # narrow and cheap rather than dropped, so the ablation can still see it.
    "model__min_samples_leaf": [5, 20, 50, 100],
    # Span 8.5 MW -- also immaterial. Extended to 20.0 only so the swept optimum
    # is interior and the boundary check stops flagging noise as a boundary.
    "model__l2_regularization": [0.0, 0.1, 1.0, 5.0, 20.0],
}


class DropFeatureFamily(BaseEstimator, TransformerMixin):
    """Remove one whole feature family, for the notebook 04 ablation.

    WHY A PIPELINE STEP AND NOT A COLUMN LIST BUILT OUTSIDE. The columns to drop do
    not exist until the calendar and lag steps have run, and the set changes with the
    lag configuration. Deciding them outside would mean maintaining a second, silently
    divergent copy of what the pipeline actually produces. Placed immediately before
    `select`, the MI selector then ranks whatever survives, so the ablated model gets a
    genuinely refitted feature selection rather than the full model's ranking with
    holes in it.

    WHY FAMILIES AND NOT SINGLE FEATURES: see `evaluate.ablation_by_family`. Briefly,
    `net_load_lag24` and `net_load_lag24_rel` carry nearly the same information, so
    dropping either alone measures the twin's ability to cover for it, not the value
    of the information.

    `family=None` is a passthrough, which is what the un-ablated baseline arm uses so
    that both arms travel through an identical pipeline shape.
    """

    # Never dropped whatever family they map to: the frame has to keep travelling.
    PROTECTED = (cfg.TIME_COL, cfg.GROUP_COL, cfg.TARGET)

    def __init__(self, family: str | None = None):
        self.family = family

    def fit(self, X, y=None):
        if self.family is None:
            self.dropped_ = []
        else:
            self.dropped_ = [c for c in X.columns
                             if c not in self.PROTECTED and feature_family(c) == self.family]
            if not self.dropped_:
                raise ValueError(
                    f"family {self.family!r} matched no columns; the ablation would "
                    f"silently compare a model against itself. Families present: "
                    f"{sorted({feature_family(c) for c in X.columns})}"
                )
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])


_DERIVATION = re.compile(r"(_lag\d+|_roll\d+_(?:mean|std)|_rel)$")

# Everything downstream of the weather data, whatever derivation is applied to it.
_WEATHER_BASES = frozenset(list(cfg.WEATHER_FEATURES)
                           + list(cfg.DERIVED_WEATHER_FEATURES))
# Exogenous drivers admissible ONLY as lags -- generation by fuel, demand, and the
# renewable share -- grouped as one family.
#
# NAMED FOR WHAT IT CONTAINS. This was `_RENEWABLE_BASES` and `feature_family`
# returned "renewables" for it, which was false: the set holds `demand`, `coal`,
# `gas`, `nuclear`, `oil` and `other` alongside wind and solar. Notebook 07 measured
# the family at 3,945.7 MW of permutation importance with `demand_lag24` alone
# contributing 3,261.6 of it -- 82.7% -- so every reading of that bar as a
# solar-or-wind result was a reading of demand history. The membership was always
# deliberate; only the label was wrong, so this rename changes no measured value.
_EXOGENOUS_BASES = frozenset(["wind", "solar", "demand", "renewable_share",
                              *cfg.EXOG_FUEL_LAGS])

#: Retained so an external caller importing the old name still resolves.
_RENEWABLE_BASES = _EXOGENOUS_BASES


def base_variable(name: str) -> str:
    """`cloud_cover_lag24` -> `cloud_cover`; `net_load_lag24_rel` -> `net_load`.

    Derivations are stripped repeatedly because they stack: PerZoneRelative
    appends `_rel` to a column that already carries `_lag24`.
    """
    prev = None
    while prev != name:
        prev = name
        name = _DERIVATION.sub("", name)
    return name


def feature_family(name: str) -> str:
    """Map a feature name to its family, for grouped ablation and grouped SHAP.

    Families answer "how much does the model lean on this whole group", which
    single-feature permutation cannot: with correlated columns each single score
    is about the column, not the information. Note that the gap between a family
    and the sum of its members does NOT measure redundancy -- measured, that ratio
    is flat at 0.71-1.13 across all six families, and `explain.redundancy_gap`
    documents why and what to use instead.

    FAMILIES ARE NAMED FOR WHAT THEY CONTAIN. `exogenous` holds generation by
    fuel, demand and renewable share -- it is not a renewables family, and was
    wrongly called one until 2026-08-23.

    CLASSIFY BY THE BASE VARIABLE, NOT BY THE SUFFIX. The previous version tested
    `"_lag" in name` first, so once EXOG_LAG_FEATURES widened beyond `temp` every
    lagged weather and renewable column was classified as target history -- and
    `startswith("temp")` was the only weather test, so humidity, wind_speed and
    cloud_cover landed in "other", which notebook 04's ablation never scored. The
    weather family was one column out of eighteen, and 4.10's "weather costs
    202 MW" was measuring temperature alone. `cloud_cover` drives the midday
    solar belly, so the duck curve's own weather driver was invisible.

    Grouping `x_lag24` with `x` also satisfies the standing requirement that
    `lag_k` be grouped with `lag_k_rel` before any importance ranking is read.
    """
    if name.startswith(("hour", "dow", "doy", "month", "is_")) or name in (
            "dayofweek", "dayofyear"):
        return "calendar"
    if name.startswith("zone") or "cluster" in name:
        return "zone"
    if "ramp" in name:                        # raw `ramp` carries net_load(t)
        return "rolling"

    base = base_variable(name)
    if base == cfg.TARGET:
        # The RAW target column is not a feature of any family -- it is the thing
        # being predicted, and DROP_ALWAYS removes it. Only its DERIVATIONS are
        # features. Returning "lag" for it would make DropFeatureFamily('lag')
        # match on a frame that has no lag columns at all.
        if name == base:
            return "other"
        return "rolling" if "roll" in name else "lag"
    if base in _WEATHER_BASES:
        return "weather"
    if base in _EXOGENOUS_BASES:
        return "exogenous"
    return "other"


# ===========================================================================
# 7. The unsupervised side
# ===========================================================================


def daily_shape_matrix(df: pd.DataFrame, target: str = cfg.TARGET,
                       group_col: str = cfg.GROUP_COL,
                       time_col: str = cfg.TIME_COL) -> pd.DataFrame:
    """One row per zone-day, 24 columns, each row min-max normalised to [0, 1].

    WHY NORMALISE WITHIN THE DAY: it makes this SHAPE clustering rather than
    LEVEL clustering. Without it, k-means mostly recovers "big zone / small zone",
    which you already knew and which is not a regime.

    WHY THE HOURS ARE LOCAL: the panel is UTC, but the six zones span four time
    zones, and a daily shape is a fact about the SUN -- the belly at local noon,
    the ramp at local sunset. On a UTC index, hour 21 is 13:00 in CISO and 16:00 in
    NYIS, so the same column holds different times of day for different zones. That
    was the original implementation and it inverted the reading of the regimes:
    the high-solar CISO cluster looked like an evening-minimum day when its
    minimum is at local midday. The spring-forward DST day (23 local hours) is
    dropped by the same rule that drops incomplete days; the autumn fall-back day
    (25 local hours) is kept, its repeated hour averaged by pivot_table. The
    notebook reports the net number of zone-days lost.

    Days with missing hours are dropped rather than filled — an incomplete day has
    an incomplete shape, and interpolating one would invent the belly of the duck
    curve. The count of dropped days is reported in the notebook.
    """
    d = df[[time_col, group_col, target]].copy()
    d[time_col] = pd.to_datetime(d[time_col])

    # LOCAL TIME, PER ZONE. The panel is UTC because EIA is UTC, but a daily SHAPE
    # is a fact about the sun: the solar belly sits at local noon and the ramp at
    # local sunset. Indexing by UTC hour compares 13:00 in California with 16:00 in
    # New York as though they were the same feature, which smears exactly the
    # structure this matrix exists to expose. The DATE is taken locally too -- a
    # local day that straddles two UTC dates would otherwise be split in half.
    local = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    for zone, tz in cfg.ZONE_TZ.items():
        m = (d[group_col] == zone).to_numpy()
        if not m.any():
            continue
        local[m] = (d.loc[m, time_col].dt.tz_localize("UTC").dt.tz_convert(tz)
                    .dt.tz_localize(None))
    unmapped = int(local.isna().sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} rows have a zone missing from cfg.ZONE_TZ: "
            f"{sorted(set(d.loc[local.isna(), group_col]))}. A shape matrix cannot "
            f"be built on UTC hours -- see the docstring.")
    d["date"] = local.dt.date
    d["hour"] = local.dt.hour
    wide = d.pivot_table(index=[group_col, "date"], columns="hour", values=target)
    wide = wide.dropna(axis=0, how="any")
    wide = wide.reindex(columns=range(24))
    rng = wide.max(axis=1) - wide.min(axis=1)
    wide = wide[rng > 0]                      # a flat day has no shape to normalise
    shapes = wide.sub(wide.min(axis=1), axis=0).div(
        wide.max(axis=1) - wide.min(axis=1), axis=0
    )
    shapes.columns = [int(c) for c in shapes.columns]
    return shapes


def local_calendar_date(df: pd.DataFrame, group_col: str = cfg.GROUP_COL,
                        time_col: str = cfg.TIME_COL) -> pd.Series:
    """The LOCAL calendar date of every row, as a tz-naive midnight Timestamp.

    WHY THIS EXISTS: `daily_shape_matrix` keys every zone-day by its LOCAL date,
    because a daily shape is a fact about the sun. Anything that joins those
    labels back onto the hourly panel -- the downstream-utility arms in notebook
    05, the coverage-by-regime table in notebook 06, the surrogate tree in
    notebook 07 -- must key on the same clock, or the first five to eight hours of
    every UTC day are handed the label of the wrong local day. Three cells did
    exactly that with `.dt.normalize()` / `.dt.date` on the UTC timestamp, and the
    merge succeeded silently every time. One helper, used everywhere, is the fix.
    """
    t = pd.to_datetime(df[time_col])
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for zone, tz in cfg.ZONE_TZ.items():
        m = (df[group_col] == zone).to_numpy()
        if m.any():
            out[m] = (t[m].dt.tz_localize("UTC").dt.tz_convert(tz)
                      .dt.tz_localize(None).dt.normalize())
    if out.isna().any():
        raise ValueError(
            f"{int(out.isna().sum())} rows have a zone missing from cfg.ZONE_TZ: "
            f"{sorted(set(df.loc[out.isna(), group_col]))}")
    return out


def make_clustering_pipeline(n_components: int = 5, algorithm: str = "kmeans",
                             k: int = 4, eps: float = 0.5,
                             min_samples: int = 10) -> Pipeline:
    """PCA -> clusterer, as a Pipeline so `n_components` is a tunable axis.

    WHY PCA FIRST: the 24 hourly columns are heavily correlated, so Euclidean
    distance in 24-d is dominated by the shared diurnal component that every day
    has. Projecting first lets the distance see what differs between days.

    The step names 'pca' and 'clust' are contractual — the notebooks read
    `named_steps['clust'].labels_`.
    """
    algos = {
        "kmeans": KMeans(n_clusters=k, n_init=10, random_state=cfg.SEED),
        "ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "dbscan": DBSCAN(eps=eps, min_samples=min_samples),
    }
    if algorithm not in algos:
        raise ValueError(f"algorithm must be one of {sorted(algos)}, got {algorithm!r}")
    return Pipeline([
        ("pca", PCA(n_components=n_components, random_state=cfg.SEED)),
        ("clust", algos[algorithm]),
    ])


def score_clustering(pipe: Pipeline, X: pd.DataFrame, n_boot: int = 5,
                     sample: int = 3000, random_state: int = cfg.SEED) -> dict:
    """Fit and score a clustering pipeline on four metrics.

    silhouette          cohesion vs separation, in [-1, 1], higher better
    calinski_harabasz   between/within variance ratio, higher better — but it is
                        not comparable across k on its own (its (n-k)/(k-1)
                        factor falls with k; on this project's shapes CH falls
                        as k rises), which is why it is never used alone
    davies_bouldin      average worst-case cluster similarity, LOWER better
    ari_stability       mean adjusted Rand index between the full-data labels and
                        labels from refits on bootstrap subsamples, on the shared
                        rows. This is the metric most projects skip and the one
                        that says whether the partition survives resampling at
                        all: above ~0.7 is structure, near zero is an artefact.

    Silhouette is O(n^2), so it is computed on a subsample; the subsample is
    seeded, so the comparison across grid cells is like-for-like.
    """
    Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    pipe = clone(pipe)
    pipe.fit(Xv)
    labels = np.asarray(pipe.named_steps["clust"].labels_)
    Z = pipe.named_steps["pca"].transform(Xv)

    out = {"n_clusters": int(len(set(labels[labels >= 0])))}
    valid = out["n_clusters"] > 1
    rs = np.random.RandomState(random_state)

    if valid:
        idx = rs.choice(len(Z), min(sample, len(Z)), replace=False)
        if len(set(labels[idx])) > 1:
            out["silhouette"] = float(silhouette_score(Z[idx], labels[idx]))
        else:
            out["silhouette"] = float("nan")
        out["calinski_harabasz"] = float(calinski_harabasz_score(Z, labels))
        out["davies_bouldin"] = float(davies_bouldin_score(Z, labels))
    else:
        out.update(silhouette=float("nan"), calinski_harabasz=float("nan"),
                   davies_bouldin=float("nan"))

    # --- ARI stability -----------------------------------------------------
    aris = []
    n = len(Xv)
    for b in range(n_boot):
        sub = np.random.RandomState(random_state + b).choice(n, int(0.8 * n), replace=False)
        try:
            p2 = clone(pipe).fit(Xv[sub])
            l2 = np.asarray(p2.named_steps["clust"].labels_)
            aris.append(adjusted_rand_score(labels[sub], l2))
        except Exception:
            continue
    out["ari_stability"] = float(np.mean(aris)) if aris else float("nan")
    out["noise_fraction"] = float((labels < 0).mean())
    return out

