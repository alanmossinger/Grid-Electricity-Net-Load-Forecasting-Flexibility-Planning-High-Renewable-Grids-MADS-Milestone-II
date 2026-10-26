# Methodology

How the project is built, in the order the notebooks run. Numbers live in `data/processed/results.json` and the
notebooks; this page explains the design.

## 1. Data (notebooks 00–01)

- **Sources.** EIA-930 hourly demand, the operator's day-ahead demand forecast and generation by fuel for six
  balancing authorities (CISO, ERCO, MISO, SWPP, BPAT, NYIS); Open-Meteo ERA5 weather at one population-weighted
  point per region, plus the archived weather forecasts for the holdout year; ancillary-service prices for the
  reserve application. Window: 2019-01-01 to 2024-12-31, hourly, UTC.
- **Target.** Net load = demand − wind − solar, derived per row. If a region normally reports wind or solar and
  that column is missing in an hour, the target is left missing and the row is dropped, never filled with zero.
- **Cleaning contract** (`src/data.py`): impossible values nulled, small negative solar readings clipped, gaps of
  up to 3 hours interpolated, longer outages left missing. Every step is counted in the ledger.
- **What the EDA decides.** Partial autocorrelation sets the lag sets per horizon; the weather columns show
  physical structure (U-shaped temperature response, wind power curve, irradiance); the generation mix shows why
  the problem grows every year.

## 2. Features and the pipeline (notebook 02, `src/features.py`)

One scikit-learn pipeline: calendar features (cyclical hour, weekday, season), target lags by horizon,
rolling statistics, weather and derived weather, zone encoding, a mutual-information selector, then the model.

| Horizon | Lags of net load (hours) |
|---|---|
| 1 h | 1, 2, 3, 24, 168 |
| 24 h | 24, 25, 26, 48, 168 |
| 48 h | 48, 49, 50, 72, 168 |
| 72 h | 72, 73, 74, 96, 168 |

Every lag is at least the horizon: the model only sees what is known when the forecast is issued.

## 3. Evaluation protocol

- **Chronological holdout.** All hours after 2024-02-06 23:00 (about 15% of the panel) are held out and scored
  once, at the end.
- **Cross-validation.** Five expanding-window folds on the development block, with a gap between training and
  validation, used for every model choice and every hyperparameter.
- **Leakage rules**, each with a test in `tests/test_leakage.py`:
  1. rolling windows end at the forecast origin, not at the hour before the target;
  2. same-hour generation columns that add up to the target are excluded;
  3. imputation, encoding and selection are fitted inside each fold;
  4. CV folds are chronological, with a gap in hours that covers the longest look-back.
- **Benchmarks** (`src/models.py`), all scored on exactly the same rows as the model: seasonal naive (same hour
  one week earlier); the EIA composite (the EIA day-ahead demand forecast minus yesterday's wind and solar); and a
  strengthened composite whose wind and solar legs are predicted from the weather.
- **Metrics.** MAE in MW as the headline (it is what reserve is sized from), skill = 1 − MAE_model / MAE_benchmark,
  paired tests on blocked errors (Wilcoxon, Diebold–Mariano, Benjamini–Hochberg).

## 4. Supervised learning (notebook 04)

A ladder from linear regression (with its assumption checks) through random forest to histogram gradient
boosting; a two-stage hyperparameter search over the whole pipeline; one champion per horizon; ablation by
feature family, learning curves, failure analysis and rolling-origin refits that cover the winter the holdout
misses.

## 5. Unsupervised learning (notebook 05)

Each zone-day becomes a 24-hour shape normalised within the day. PCA (5 components) feeds k-means, Ward and DBSCAN
over a grid of settings; the choice uses stability under resampling (ARI) together with silhouette,
Davies–Bouldin and Calinski–Harabasz. The chosen partition (k-means, k = 3) is checked against the zones and the
weather, and tested as an extra feature for the forecaster.

## 6. Uncertainty (notebook 06)

Quantile gradient boosting gives the raw band. Conformalised quantile regression calibrates it, **per region**
(Mondrian), on a chronological calibration block; the band is checked per region, per horizon and per regime, and
re-scored with the weather forecasts an operator would actually have held.

## 7. Explainability and causal questions (notebooks 07, 03)

Permutation importance (single and grouped), SHAP, partial dependence and ALE explain the champion. The causal
notebook draws a DAG, derives the adjustment set, and runs difference-in-differences and propensity matching on
whether more solar causes steeper evening ramps, with sensitivity analysis.

## 8. The decision (notebook 08)

The calibrated band sizes reserve; conditioning it on the regime and hour moves reserve to the hours where the
risk is, at equal overall risk. Notebook 08 also collects every result into the summary tables the report uses.
