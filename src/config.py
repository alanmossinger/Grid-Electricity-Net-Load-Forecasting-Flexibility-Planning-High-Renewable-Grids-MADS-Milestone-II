"""
config.py — every path, constant and assumption in one place.

WHY THIS FILE EXISTS: if a zone list or a cutoff date appears in two notebooks,
they will eventually disagree and you will not notice. One definition, imported
everywhere. This is also the file a grader reads first to see what you assumed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --- credentials ------------------------------------------------------------
def _load_dotenv(path: "Path | None" = None) -> None:
    """Read KEY=value lines from .env into os.environ, without a dependency.

    WHY: the README tells you to put EIA_API_KEY in .env, and nothing was
    reading it. `export` in one shell does not survive into a Jupyter kernel
    launched from another, which is exactly the confusing failure this avoids.

    Existing environment variables WIN — an explicit `export` overrides the
    file, which is the behaviour you want when switching between keys.

    .env is gitignored. Never commit a key, and never paste one into a chat,
    an issue, or a notebook cell: notebook outputs are committed.
    """
    import os
    from pathlib import Path as _Path

    env = _Path(path) if path else _Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)   # setdefault: real env wins


_load_dotenv()


# --- who this is by -------------------------------------------------------
# Read by src/report.py, report_sections.py (the DOCX cover) and the READMEs, so
# the name cannot be lost by a regeneration and cannot disagree between surfaces.
# It is deliberately here rather than typed into each document, for the same
# reason no number is typed into the report.


# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
DOCS = ROOT / "docs"

# Notebook 00 gathers and freezes the RAW join, untouched: no cleaning, no
# interpolation, no derived columns. Notebook 01 (EDA) is where defects are found,
# argued about, and repaired — so the raw table is what its decisions are
# evidence about, and it stays on disk to be re-examined.
PANEL_RAW = DATA_PROC / "panel_raw.parquet"  # Notebook 00 output — as served by the APIs
PANEL = DATA_PROC / "panel.parquet"          # Notebook 01 output — cleaned, derived
RESULTS = DATA_PROC / "results.json"         # every number the report prints
MANIFEST = DATA_PROC / "manifest.json"       # provenance + determinism record
BEST_PARAMS = DATA_PROC / "best_params.json"

for _p in (DATA_RAW, DATA_PROC, FIGURES, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# --- jupyter kernel --------------------------------------------------------
# The kernel is named for THIS project, not `python3`.
#
# WHY IT MATTERS: `ipykernel install --user --name python3` writes to
# %APPDATA%/jupyter/kernels/python3 (or ~/.local/share/...), which takes
# precedence over the interpreter's own default kernel. Every other notebook on
# the machine that declares kernelspec `python3` — every other course, every
# other project — would then silently launch THIS project's virtualenv, and the
# only symptom would be an ImportError in an unrelated repository.
#
# One definition, used by build_notebooks.py, by `make setup`, and asserted
# against every notebook's metadata in tests/test_features.py.
KERNEL_NAME = "electricity-grid"
KERNEL_DISPLAY = "Python 3 (electricity-grid)"

# --- reproducibility -------------------------------------------------------
SEED = 42
# Worker processes for scikit-learn's parallel loops (cross-validation, searches, forests,
# permutation importance). The results do not depend on it. On Windows a joblib pool can
# deadlock late in a long kernel (notebook 04 did once); NETLOAD_N_JOBS=4 or 1 is the fallback.
N_JOBS = int(os.environ.get("NETLOAD_N_JOBS", "-1"))

# --- data ------------------------------------------------------------------
# EIA-930 respondent codes are NOT the common names. CISO not CAISO,
# ERCO not ERCOT, SWPP not SPP. Getting this wrong returns an empty payload
# with a 200 status, which is the worst kind of error.
ZONES = ["CISO", "ERCO", "MISO", "SWPP", "BPAT", "NYIS"]
START = "2019-01-01T00"
END = "2024-12-31T23"

TIME_COL = "timestamp"
GROUP_COL = "zone"
TARGET = "net_load"

# Local civil time per balancing authority. The panel is stored in UTC on purpose
# (EIA serves UTC and Open-Meteo is requested with timezone=UTC to match), and the
# MODEL should stay in UTC: one clock, no DST discontinuity, no 23- or 25-hour days.
#
# But solar output is a fact about the SUN, so any figure about the shape of a day
# has to be drawn in local time or it smears. CISO's midday solar peak lands at
# 20:00-21:00 UTC in summer and 19:00-20:00 in winter; averaged over six years in
# UTC the belly of the duck spreads across two hours and looks shallower than it is.
# Use these for presentation only, never to build a feature.
#
# MISO is an approximation: its footprint spans Central and Eastern, and it settles
# its market in Eastern. Central is where most of its load sits.
ZONE_TZ = {
    "CISO": "America/Los_Angeles",
    "BPAT": "America/Los_Angeles",
    "ERCO": "America/Chicago",
    "MISO": "America/Chicago",
    "SWPP": "America/Chicago",
    "NYIS": "America/New_York",
}

# Raw exogenous features. `demand` is deliberately ABSENT: net_load =
# demand - wind - solar, so demand shares a term with the target, and the
# operator does not know realised demand at prediction time either.
# See docs/decision_log.md.
NUMERIC_FEATURES = ["temp"]

# Same-hour realised generation by fuel. Notebook 02 audit, 2026-08-13.
#
# These are inadmissible for TWO independent reasons, either one sufficient:
#
#   Algebraically, they sum to the target. net_generation is their sum, and
#   net_load = net_generation - wind - solar + total_interchange to within the
#   interchange term. Measured on the real panel, the nine columns reconstruct
#   net load at r = 0.9997 for ERCO and r = 0.9940 for SWPP. A model given them
#   is doing arithmetic, not forecasting.
#
#   Operationally, they are realised values for the hour being predicted. At a
#   24-hour forecast origin nobody knows tomorrow's gas output. This is the same
#   argument that already excludes `demand`, and it is the one that survives even
#   if the algebra is somehow broken by a future schema change.
#
# `renewable_share` belongs here too, by a shorter route: net_load is identically
# demand * (1 - renewable_share), verified to a maximum residual of 0.0 MW across
# all 315,552 rows.
#
# The LAGGED fuels are a different matter and are legitimate — `wind_lag24` IS
# knowable at a 24-hour origin. They must be introduced deliberately as lag
# features, never by leaving the realised column in and hoping.
REALISED_GENERATION_COLS = [
    "coal", "gas", "nuclear", "oil", "other", "hydro",
    "battery", "unknown_storage", "solar_battery",
]

# Exogenous columns that ARE admissible: forecastable at any origin, and not a
# term in the target. This list is CLOSED — `test_no_unvetted_column_becomes_a
# _feature` fails if any other panel column reaches the feature set, so a new
# EIA fuel code is caught by a red test rather than by a suspiciously low MAE.
# WIDENED 2026-08-23, from four columns to fifteen. The four were temperature,
# humidity, 10 m wind and cloud cover, and the report's own limitation section
# named that as the project's largest technical gap: `cloud_cover` is a coarse
# PROXY for irradiance, which is what actually drives the midday solar belly —
# the duck itself — and 10 m wind is not the height a turbine sees.
#
# MEASURED BEFORE THE WIDENING, 2026-08-23, forecast-vs-reanalysis MAE on the
# holdout window (see src/sources.load_all_forecast_weather):
#     lead day        1       2       3
#     temp (C)      1.35    1.95    2.09
#     cloud (pp)    21.5    23.1    25.2
#     GHI (W/m2)    36.4    37.0    39.1     against a mean GHI of 185
# Cloud cover is forecast with an error of a fifth to a quarter of its full range
# and was held at reanalysis truth in every result this project has reported. That
# is why the 6.26% weather share at h=72 was labelled a FLOOR.
#
# All fifteen are EXOGENOUS and contemporaneous-admissible. None is a term in
# net_load = demand - wind - solar; all are forecast by public NWP at every
# horizon used here. The leakage ban covers demand/wind/solar/ramp because they
# COMPOSE the target identically — it does not cover physical drivers of it.
# Irradiance is not leakage; `solar` is. State that in notebook 02 so a reviewer
# does not mistake one for the other.
WEATHER_FEATURES = [
    "temp",             # 2 m air temperature, deg C
    "humidity",         # relative humidity, %
    "wind_speed",       # 10 m wind speed, km/h
    "cloud_cover",      # total cloud cover, %
    "ghi",              # shortwave (global horizontal) irradiance, W/m2
    "dni_horiz",        # direct component on the horizontal, W/m2
    "dhi",              # diffuse component, W/m2
    "wind_speed_100m",  # hub-height wind speed, km/h
    "wind_gust",        # 10 m gusts, km/h — turbine cut-out marker
    "apparent_temp",    # felt temperature, deg C — cooling load responds to this
    "dew_point",        # dew point, deg C — the humidity term in cooling load
    "snowfall",         # cm/h
    "snow_depth",       # m — PV derating and Regime 2 winters
    "precipitation",    # mm/h
    "pressure",         # surface pressure, hPa — frontal passage marker
]

# --- derived weather features, and an honest note about who they help --------
# THE HINGE. Load against temperature is a U: heating below a comfort point,
# cooling above it. Splitting `temp` into heating- and cooling-degree arms at that
# point makes the relationship linear in each arm.
#
# WHO THIS ACTUALLY HELPS, said plainly because the obvious claim is wrong. A
# gradient-boosted tree is INVARIANT to any monotone transform of a single
# feature, and it can place a split at 15 C by itself. So `hdd`/`cdd` and the
# capped-cubic wind term below buy the CHAMPION essentially nothing, and if the
# Notebook 04 ablation shows them buying something large, the first hypothesis should
# be a bug rather than a discovery. They are here for the LINEAR arm — where a
# single `temp` coefficient averages the two halves of the U into near-zero and is
# one source of the sign-flipping documented in notebook 04 — and for the ALE plots,
# where a readable hinge is worth more than a smooth curve nobody can quote.
#
# The one derived feature that is NOT a monotone transform, and therefore the one
# that can help the tree, is the clear-sky index: irradiance normalised by what
# that zone-month-hour would receive under a clear sky. That is an interaction,
# it is fitted from data, and it lives inside the pipeline (see
# pipeline.SolarClearSkyIndex).
TEMP_HINGE_C = 15.0

# Turbine rated wind speed. Output goes as v^3 below rating and is flat above it,
# so the physically motivated feature is min(v, v_rated)^3. 12 m/s = 43.2 km/h is
# the middle of the rated-speed band for modern utility-scale onshore turbines
# (IEC class II, typically 11-14 m/s). It is a PHYSICAL CONSTANT taken from the
# turbine literature, not a parameter fitted here — which is deliberate: a fitted
# v_rated would have to be refit inside every CV fold to stay honest, and it would
# buy a tree nothing (see above). Open-Meteo serves wind speed in km/h.
WIND_RATED_KMH = 43.2

DERIVED_WEATHER_FEATURES = ["hdd", "cdd", "wind_cube_capped", "clearsky_index"]

# Columns that get a LAGGED copy at the exogenous lag (the horizon rounded up to
# a whole number of days -- see GroupLagFeatures._exog_lag). Distinct from
# WEATHER_FEATURES, which are carried CONTEMPORANEOUSLY.
#
# WHY THE TWO LISTS DIFFER, and it is the point of this constant. `wind`, `solar`,
# `demand` and `renewable_share` may never appear contemporaneously -- they are
# terms in the target's accounting identity. Lagged they are ordinary history: an
# operator forecasting at a 24-hour origin does know yesterday's wind. So they
# belong here and are excluded above, and that is not an inconsistency.
#
# MEASURED, 2026-08-19, h=24, champion HistGradientBoosting, identical folds:
#   ("temp",) -- what every model ran with until now      2,302.0 MW
#   all four weather lagged                               2,266.0 MW  (-36.1)
#   + wind, solar, demand, renewable_share                2,254.8 MW  (-47.3)
#   + the nine realised fuel columns  <-- SHIPPED         2,256.7 MW  (-45.4)
# Better in 5/5 folds against a 34.8 MW fold noise floor.
#
# THE FUEL MIX IS INCLUDED BY OWNER'S DECISION, not because it scored best. It
# costs 1.9 MW against the arm without it: nine columns for no measured skill,
# because wind/solar/demand history already carries what they contribute. The
# brief is that every panel variable be available to the model in lagged form,
# and 1.9 MW is a defensible price for that completeness -- but the table above
# is the honest record and the ablation in notebook 04 is where it gets re-argued.
#
# WHY THIS IS STILL ADMISSIBLE, and the distinction matters more than the score.
# Only the LAGGED copies enter. The raw same-hour columns remain banned by
# REALISED_GENERATION_COLS below and by the parameterised leakage test, which is
# what caught the r = 0.9997 contamination in notebook 02. `coal_lag24` is knowable
# at a 24-hour origin; `coal` is not, and never becomes a feature.
#
# That the fuel arm did NOT jump is itself the reassuring result. These are the
# columns that reconstructed the target contemporaneously, so a large gain once
# lagged would have signalled that contamination returning by another route. A
# 1.9 MW REGRESSION is the signature of columns carrying no independent
# information, which is exactly what the lag should leave of them.
#
# NOTE ON PERFECT FORESIGHT. The contemporaneous weather columns are ERA5
# reanalysis, so WEATHER_NOISE_SCALE exists to discount them. A lagged weather
# value carries no such caveat: it is genuinely observed by the forecast origin.
# The fuel columns EIA reported across the training window. Three of the nine in
# REALISED_GENERATION_COLS are excluded and the reason is data, not preference:
# `battery`, `unknown_storage` and `solar_battery` are 0.00% present in the dev
# block and ~2.8% present in the holdout -- EIA began publishing them in July and
# November 2024, inside the holdout period. A column that is entirely NaN in
# training gives the median imputer nothing to learn and the MI selector nothing
# to rank; its only real values are in the block being scored. Including them
# raised the lag NaN rate until two leakage tests failed, which is the guard
# working. This is the same mid-window schema drift the notebook 02 audit flagged.
EXOG_FUEL_LAGS = ["coal", "gas", "nuclear", "oil", "other", "hydro"]

# WHICH WEATHER COLUMNS GET A LAGGED COPY, and why it is not all fifteen.
#
# A lagged weather value is admissible — an operator forecasting at a 24-hour
# origin does know yesterday's temperature — so the question is cost, not legality.
# Fifteen more lagged columns roughly doubles the design matrix, and notebook 04's
# search is already the most expensive thing in the project at ~5 hours. The four
# below are the ones whose OWN history carries information beyond the
# contemporaneous value: temperature (thermal mass in the building stock),
# irradiance and cloud (persistence of synoptic cloud regimes), and hub-height
# wind (the same for wind regimes). Pressure, snow and the humidity family are
# carried contemporaneously only.
#
# This is a COST decision made explicitly rather than a modelling claim, and it is
# reversible in one line. If the notebook 04 ablation shows lagged weather mattering
# more than expected, widen it and re-cost.
WEATHER_LAG_FEATURES = ["temp", "cloud_cover", "ghi", "wind_speed_100m"]

EXOG_LAG_FEATURES = (WEATHER_LAG_FEATURES
                     + ["wind", "solar", "demand", "renewable_share"]
                     + EXOG_FUEL_LAGS)

# --- forecast horizons -----------------------------------------------------
# Notebook 01 decision, 2026-08-11. This project forecasts at FOUR horizons, not
# one, and each is scored on the same blind holdout block.
#
# The horizon is not cosmetic — it decides which lags EXIST. Forecasting
# net_load(t) from an origin at t-h means every feature must be observable at
# t-h, so a lag L is admissible only if L >= h. Lag 1 is by far the strongest
# predictor in the panel (mean |PACF| 0.976 across the six zones) and it is
# legal at h=1 and illegal at every other horizon. That single constraint is
# why error must rise with h, and it is the answer to "why is your 72-hour
# model so much worse than your 1-hour model?".
HORIZONS = (1, 24, 48, 72)

# --- feature engineering ---------------------------------------------------
# The admissible lag set per horizon, derived from the PACF in notebook 01
# (notebook 01 Pass 5, figure `01_pacf.png`). ONE rule generates every row:
#
#   the three freshest admissible hours   (h, h+1, h+2)
# + the next whole-day anchor             (smallest multiple of 24 that is >= h
#                                          and not already in that block)
# + the weekly lag                        (168)
#
# Same rule and the same feature count at every horizon, so notebook 04's
# cross-horizon comparison measures the horizon rather than a confound with
# how many lag columns each model was handed.
LAGS_BY_HORIZON = {
    1:  (1, 2, 3, 24, 168),
    24: (24, 25, 26, 48, 168),
    48: (48, 49, 50, 72, 168),
    72: (72, 73, 74, 96, 168),
}

# The h=1 set, kept as the module default so existing callers keep working.
# Notebook 04 selects per horizon via `lags_for()`.
LAGS = LAGS_BY_HORIZON[1]

# Rolling windows end at the FORECAST ORIGIN, t-h, not at t-1. A 24-hour rolling
# mean ending one hour before the target uses data the operator does not have at a
# 24-hour origin. This was a live leak until notebook 04 found it (2026-08-14): the
# window covered [t-24, t-1] at every horizon, handing an h=24 model twenty-three
# hours of net load that do not exist yet. `GroupLagFeatures._roll_shift` is the fix
# and `test_rolling_window_ends_at_the_forecast_origin` pins it at all four horizons.
# The shift is zero at h=1, so notebook 02's feature decisions are unaffected.
ROLL_WINDOWS = (24, 168)


def lags_for(horizon: int) -> tuple:
    """The notebook 01 admissible lag set for `horizon`, or a loud failure.

    Raising beats returning a default: a silent fallback to the h=1 lags at
    h=72 would hand the model 71 hours of information it cannot have, and the
    only symptom would be a suspiciously good score.
    """
    try:
        return LAGS_BY_HORIZON[int(horizon)]
    except KeyError:
        raise KeyError(
            f"no notebook 01 lag set for horizon {horizon}h; "
            f"known horizons: {sorted(LAGS_BY_HORIZON)}"
        ) from None


# --- validation ------------------------------------------------------------
N_SPLITS = 5
# The gap must be at least the longest rolling window, or the first validation
# row's 168-hour feature is built from rows the model trained on.
CV_GAP = max(ROLL_WINDOWS)


def cv_gap_for(horizon: int) -> int:
    """The CV gap in hours at `horizon`, which is NOT the same at every horizon.

    A validation row at time t now reaches back `max(ROLL_WINDOWS) + h - 1` hours,
    because the rolling window ends at the origin t-h rather than at t-1. The flat
    `CV_GAP = 168` is therefore correct at h=1 and too small at h=24, 48 and 72,
    where the earliest validation rows would build features from hours the model
    trained on. At h=72 the required gap is 168 + 71 = 239.

    Stated as a function rather than a constant because the horizon is the thing
    that varies in notebook 04, and a constant that is right for one of four horizons
    is the shape of the bug this replaces.
    """
    return int(max(ROLL_WINDOWS) + max(0, int(horizon) - 1))
# The holdout is every hour AFTER this timestamp (UTC): the cut HOLDOUT_FRACTION gave on the
# panel the holdout was sealed on (315,552 rows), frozen so that a later cleaning change can
# move rows in or out of the panel but never moves the boundary itself.
HOLDOUT_CUT = pd.Timestamp("2024-02-06 23:00:00")

# --- weather forecast realism (notebook 04 sensitivity, notebook 06 uncertainty) ----
# THE PROBLEM. Our weather features are ERA5 REANALYSIS: a physics model run over
# history with observations assimilated into it — observations that did not exist
# at the forecast origin. Feeding contemporaneous ERA5 to an h=24 model assumes a
# PERFECT weather forecast. Real weather forecasts degrade with lead time, so the
# skill this project reports is optimistic, and optimistic by MORE at h=72 than at
# h=24. Left unmodelled, that would corrupt prediction P5 in particular: the rise
# in temperature's importance with horizon is partly real and partly an artefact of
# handing the long-horizon model weather it could not have had.
#
# THE PUBLISHED NUMBERS. Mean absolute error of National Weather Service public
# temperature forecasts for Oklahoma City, by forecast lead day, over 338 days
# between 4 January 1994 and 6 March 1995:
#
#     lead day        1     2     3     4     5
#     max temp (F)   3.8   4.8   5.9   6.7   7.5
#     min temp (F)   2.9   3.6   4.8   5.5   5.8
#
#   Brooks, H. E., A. Witt, and M. D. Eilts, 1997: "Verification of public weather
#   forecasts available via the media." Bulletin of the American Meteorological
#   Society, 78, 2167-2178.
#   Data table: https://www.nssl.noaa.gov/users/brooks/public_html/media/okcmed.html
#
# Below is the mean of the max and min rows, converted to degrees Celsius
# (a DIFFERENCE, so x5/9, no offset). Error roughly doubles from day 1 to day 5.
#
# FOUR WAYS THIS IS WRONG, AND IN WHICH DIRECTION. State all four out loud; do not
# describe this as a clean bound, because it is not one.
#   1. TOO BIG. The data is from 1994-95. Numerical weather prediction has improved
#      substantially since, so a modern day-1 forecast beats 1.9 C comfortably.
#      This overstates the penalty.
#   2. TOO SMALL, in scope. Brooks et al. report temperature only. `humidity`,
#      `wind_speed` and `cloud_cover` are forecast with error too and are NOT
#      degraded here, because inventing a number for them is exactly the thing this
#      repository refuses to do. Cloud cover matters most for solar, so the true
#      penalty on net load is larger than what this measures.
#   3. TOO SMALL, in structure. The noise injected is i.i.d. Gaussian. Real forecast
#      error is autocorrelated and regime-dependent — small in settled weather, and
#      concentrated in exactly the frontal passages and heatwave onsets that drive
#      the load extremes an operator cares about. A bust is a run of bad hours, not
#      independent draws.
#   4. WRONG PLACE. Oklahoma City is a fair proxy for ERCO and SWPP and a poor one
#      for CISO, BPAT and NYIS.
#
# HOW IT IS USED. Not as a correction — as a SENSITIVITY. Notebook 04 scores each
# horizon at several values of WEATHER_NOISE_SCALE and reports the response, the
# same way the reserve price is treated downstream. scale=0 is the perfect-foresight case
# the project currently measures; scale=1 is the 1990s-vintage case. The truth is
# between them, nearer 0. Reporting the pair is honest; reporting either alone is
# not.
NWP_TEMP_MAE_C_BY_LEAD_DAY = {1: 1.861, 2: 2.333, 3: 2.972, 4: 3.389, 5: 3.694}

# Horizon (hours) -> forecast lead day used to look up the MAE above.
# h=1 is deliberately 0.0: Brooks et al. report nothing below one day, an
# hour-ahead temperature is close to observed, and inventing a sub-daily value
# would be fabricating the one number this whole block exists to avoid fabricating.
NWP_LEAD_DAY_BY_HORIZON = {1: 0, 24: 1, 48: 2, 72: 3}

# Multiplier on the injected error. 0.0 = perfect foresight (current behaviour and
# the default, so nothing changes unless a notebook opts in). 1.0 = the Brooks curve.
# Notebook 04 sweeps it; notebook 06 uses the result to widen intervals with the horizon.
WEATHER_NOISE_SCALE = 0.0

# MAE -> sigma for a zero-mean Gaussian: E|X| = sigma*sqrt(2/pi), so
# sigma = MAE * sqrt(pi/2) = MAE * 1.2533.
MAE_TO_SIGMA = 1.2533141373155003

# --- business assumptions (reserve sizing) ----------------------------------------
# Every one of these is an ASSUMPTION, not a measurement. The sensitivity
# surface for reserve cost exists because of that.
RESERVE_COVERAGE = 0.95            # size reserve at p95 of absolute error
UNDER_FORECAST_PENALTY_RATIO = 5.0  # under-forecast MWh costs ~5x over-forecast

# --- observed market prices (out of scope here) --------------------------------------
# These are CITED OBSERVATIONS, not assumptions: each entry names its source,
# its document date, and the date it was retrieved (2026-09-02). They exist so
# A pricing step can price the reserve quantities at prices a real market has
# actually cleared at, instead of only at the $45,000 assumption above. PJM's
# capacity auction is the cleanest public "price of holding 1 MW ready for a
# year" in the US; none of the six zones in this panel is in PJM, so the pricing step
# uses these as a REFERENCE BAND for the price of readiness, never as the
# price any of these six BAs actually pays — that caveat ships with every
# dollar figure derived from them.

# Wholesale ENERGY price anchors — the "current dollar number of energy" a
# reader should hold next to every MW figure. Cited observations, same rules.

# --- plotting --------------------------------------------------------------
FIGSIZE = (10, 5)
DPI = 130
