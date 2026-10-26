# Data

Everything the notebooks need is committed (about 85 MB), so the project runs offline.

## `raw/` — pulls, cached by `src/sources.py` (24 MB)

| Files | Source | Content |
|---|---|---|
| `eia930_<ZONE>_2019_2024.parquet` | EIA-930 Hourly Electric Grid Monitor, EIA Open Data API v2 (free key) | demand, day-ahead demand forecast, net generation, interchange, generation by fuel |
| `weather_<ZONE>_2019_2024_v2.parquet` | Open-Meteo historical API (ERA5 reanalysis, CC-BY-4.0) | 15 hourly weather variables at one point per region |
| `weather_<ZONE>_2019_2024.parquet` | Open-Meteo, earlier variable set | kept for the record; not used by the pipeline |
| `weatherfc_<ZONE>_2024-02-01_2024-12-31_v2.parquet` | Open-Meteo previous-runs API | the weather forecasts as issued, holdout year |
| `ancillary_caiso_…`, `ancillary_miso_…` | CAISO OASIS, MISO market reports | hourly reserve prices (reserve valuation in notebook 08) |

`<ZONE>` is one of CISO, ERCO, MISO, SWPP, BPAT, NYIS.

## `processed/` — written by the notebooks (61 MB)

| File | Written by | Content |
|---|---|---|
| `results.json` | every notebook | the results ledger: every number the notebooks store, per notebook |
| `panel_raw.parquet` | 00 | the merged raw panel (6 zones × 52,608 hours) |
| `panel.parquet` | 01 | the cleaned panel with the net-load target |
| `manifest.json` | 00, 01 | provenance: row counts, hashes, cleaning counts |
| `weather_forecast_holdout.parquet` | 00 | operational weather forecasts aligned to the holdout |
| `ancillary_prices.parquet` | 00 | hourly reserve prices |
| `best_params.json` | 04 | the tuned pipeline per horizon |
| `holdout_predictions_by_horizon.parquet` | 04 | held-out predictions of every model and benchmark |
| `linear/*.csv` | 04 | OLS coefficient tables and VIF |
| `unsupervised/*.parquet` | 05 | daily shapes, the clustering grid, regime labels |
| `holdout_quantiles.parquet`, `uncertainty/*.parquet` | 06 | calibrated quantiles and interval diagnostics |

## `reference/` and `sample/`

`reference/us_states_20m.geojson` (US state boundaries for the region map); `sample/` holds the first 100 rows of
both panels as CSV for a quick look.

To pull fresh data, add an EIA key to `.env` (see `.env.example`); Open-Meteo needs no key.
