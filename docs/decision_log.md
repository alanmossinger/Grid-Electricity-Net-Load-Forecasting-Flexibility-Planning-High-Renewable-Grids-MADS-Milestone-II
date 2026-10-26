# Decision log

Decisions that changed what the code does, each with the reason it was made, the
evidence it was made on, and the upgrade it defers. Four places in the source refer
here for their justification — `src/config.py`, `src/features.py`,
`src/sources.py` and notebook 02 §2.15 — so this file is the target of those
references and not a summary of them.

The rule this file exists to serve: a defect found by an audit, measured, and fixed is
stronger evidence of rigour than a defect that was never there. Entries are kept even
when they record a mistake — especially then.

---

## D1 · `demand` is not a feature

**Decided** at the outset · **code** `src/config.py` (`EXCLUDED_FEATURES`)

`net_load = demand − wind − solar`, so `demand` shares a term with the target. Two
independent reasons, either sufficient:

- *Algebraically*, a model given realised demand is doing arithmetic, not forecasting.
- *Operationally*, an operator does not know realised demand at a 24-hour forecast
  origin either. This reason survives any future schema change.

`net_generation` and `total_interchange` are excluded on the same grounds.
**Lagged demand is admissible and is used** — `demand_lag24` is knowable at the origin.

---

## D2 · The same-hour fuel columns were leaking, and the leak was measured before it was fixed

**Decided** 2026-08-13, from the notebook 02 audit · **code** `src/config.py`
(`REALISED_GENERATION_COLS`) · **narrated** notebook 02 §2.15

The nine same-hour generation-by-fuel columns and `renewable_share` were reaching the
feature set. They are inadmissible for the two reasons of D1:

- the nine columns reconstruct net load at **r = 0.9997** (ERCO) and **r = 0.9940**
  (SWPP) on the real panel;
- `net_load ≡ demand × (1 − renewable_share)`, verified to a maximum residual of
  **0.0 MW** across all 315,552 rows;
- and at a 24-hour origin nobody knows tomorrow's gas output.

**What was done rather than only fixed.** The reconstruction was *measured* first —
notebook 02's `02_leakage_fuel_columns` figure is that measurement — and only then
removed, because "we found a leak" is a weaker claim than "we found a leak and here is
what it was worth". Any CV score produced before the fix was treated as contaminated and
never written to `results.json`. The separate rolling-feature leak of notebook 04 §4.2
(`04_leak_cost`) was priced the same way.

**The deliberate part.** The *lagged* fuels are legitimate (`wind_lag24` is knowable at
the origin) and are introduced explicitly through `exog_lags`, never by leaving a
realised column in and hoping.

**The guard.** The admissible list is closed: `test_no_unvetted_column_becomes_a_feature`
fails if any other panel column reaches the feature set, so the next EIA fuel code is
caught by a red test rather than by a suspiciously low MAE.

---

## D3 · The weather feature set was widened from four columns to fifteen

**Decided** 2026-08-23 · **code** `src/config.py` (admissible exogenous columns)

The original four were temperature, humidity, 10 m wind and cloud cover, and the
report's own limitations section named that as the project's largest technical gap:
`cloud_cover` is a coarse *proxy* for irradiance, which is what actually drives the
midday belly — the duck itself — and 10 m wind is not the height a turbine sees.

Measured *before* the widening, forecast-versus-reanalysis MAE on the holdout window
(`src/sources.load_all_forecast_weather`):

| lead day | 1 | 2 | 3 |
|---|---|---|---|
| temp (°C) | 1.35 | 1.95 | 2.09 |
| cloud (pp) | 21.5 | 23.1 | 25.2 |
| GHI (W/m²) | 36.4 | 37.0 | 39.1 (against a mean GHI of 185) |

The widened set is admissible by the D1 test: forecastable at any origin, and not a
term in the target.

---

## D4 · One weather point per balancing authority — a stated limitation, not an oversight

**Decided** at the outset · **code** `src/sources.py` · **documented**
`docs/data_dictionary.md` §"Weather points"

One coordinate per BA under-represents temperature spread across a large footprint.
MISO spans Manitoba to Louisiana; one point in Minneapolis summarises a fraction of its
load. The coordinates are population- and load-weighted judgement calls, written down
with their reasoning so the choice is visible and challengeable.

**Deferred upgrade:** 3–4 metro points per BA combined with a load-weighted mean. Not
done here; named so that the limitation is a choice on the record rather than a gap
discovered by a reader.

---

## D5 · The holiday feature is a proxy, and is labelled as one

**Decided** at the outset · **code** `src/features.py` (`add_holiday_proxy`)

`is_yearend` flags 24–31 December only. US holidays cluster at end-December and around
4 July, and both behave like Sundays on the load curve, so a proxy captures most of the
effect at no dependency cost.

**Deferred upgrade:** a real calendar via the `holidays` package. The feature is named
`is_yearend` rather than `is_holiday` precisely so the gap cannot be misread.

---

## D6 · Weather is requested in UTC, and every join is checked for it

**Decided** at the outset · **code** `src/sources.py`

EIA-930 periods are UTC, so the weather module requests `timezone=UTC` explicitly.
Mixing local and UTC across the `(zone, timestamp)` join produces a shift of one to
several hours that raises nothing: the join succeeds, the row count is right, and
temperature simply stops predicting anything.

**This defect did occur, elsewhere.** The daily regime labels of notebook 05 are keyed
by *local* date and were merged onto the *UTC* date, silently, for the same reason. It
was found in review, fixed with one shared helper (`features.local_calendar_date`) used
by notebooks 05, 06 and 07, and given a unit test. The corrected downstream-utility
numbers are in `reports/GUIDE_supervised_unsupervised.md` §B.6. The lesson is the one
already in the module docstring: a timezone defect is invisible at the join and only
visible in the metric.

**Local time is for presentation only.** `ZONE_TZ` exists to draw the duck curve —
averaged in UTC over six years the belly spreads across two hours and looks shallower
than it is — and never to build a feature. MISO's entry is an approximation: its
footprint spans Central and Eastern, it settles its market in Eastern, and Central is
where most of its load sits.

---

## Decisions recorded elsewhere, deliberately

- **Which covariates a causal estimate may condition on**, and why the project plan's
  proposed set is inadmissible — `docs/adjustment_set.md`, generated by notebook 03
  from `src/causal.py`.
- **Every modelling decision that has a number attached** — the notebook that computed
  it, with the number in `data/processed/results.json`. This log records the decisions
  that do *not* reduce to a ledger entry.
