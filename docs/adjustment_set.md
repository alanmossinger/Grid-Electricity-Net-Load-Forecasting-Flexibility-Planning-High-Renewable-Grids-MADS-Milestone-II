# 05 — The adjustment set implied by the DAG

## The graph

| parent | child | why this edge is claimed |
|---|---|---|
| `cloud_cover` | `solar_output` | cloud attenuates the irradiance reaching a panel |
| `cloud_cover` | `temp` | cloud suppresses the daytime maximum |
| `demand` | `net_load` | definitional: net_load = demand - wind - solar |
| `demand_growth` | `demand` | more customers and more load per customer |
| `demand_growth` | `solar_capacity` | growing systems procure more of everything |
| `humidity` | `cloud_cover` | moisture is the raw material for cloud |
| `humidity` | `demand` | humid heat raises cooling load at the same temperature |
| `load_shape` | `demand` | the shape is a component of the level |
| `net_load` | `evening_ramp` | the ramp is a function of the net-load path |
| `season` | `cloud_cover` | storm tracks and convective season move with the year |
| `season` | `humidity` | absolute humidity rises with summer temperature |
| `season` | `temp` | the annual insolation cycle |
| `season` | `wind_speed` | the pressure-gradient climatology is seasonal |
| `solar_capacity` | `solar_output` | you cannot generate from a panel you did not build |
| `solar_output` | `net_load` | definitional, with a negative sign |
| `temp` | `demand` | heating below the comfort band, cooling above it |
| `temp` | `solar_output` | panel efficiency falls as cell temperature rises |
| `wind_output` | `net_load` | definitional, with a negative sign |
| `wind_speed` | `demand` | wind chill raises winter heating demand |
| `wind_speed` | `wind_output` | the turbine power curve |
| `zone_char` | `load_shape` | climate and industrial mix set the daily shape |
| `zone_char` | `solar_capacity` | irradiance, land, market design and policy |

## Backdoor paths

- total backdoor paths: **56**
- open with nothing adjusted: **2**
- colliders in the graph: `cloud_cover`, `demand`, `net_load`, `solar_capacity`, `solar_output`, `temp`

## Admissible adjustment sets (observed nodes only)

- { `demand_growth`, `zone_char` }
- { `cloud_cover`, `demand`, `temp`, `wind_output` }
- { `cloud_cover`, `demand`, `temp`, `wind_speed` }
- { `demand`, `humidity`, `temp`, `wind_speed` }

## What is NOT admissible, and why

| set | admissible | reason |
|---|---|---|
| { `demand` } | **no** | `demand` is a collider (parents: demand_growth, humidity, load_shape, temp, wind_speed). Conditioning on it opens 48 backdoor paths, against 2 when nothing is adjusted. |
| { `temp`, `season`, `demand` } | **no** | the set the project plan proposes for propensity matching. It conditions on the `demand` collider without blocking the paths that opens. |
| { `net_load` } | **no** | `net_load` is both a descendant of the treatment and the collider of the accounting identity `net_load = demand − wind − solar`. |

## Where notebook 04's predictive feature set departs from this, deliberately

notebook 04 regresses **on** `net_load` and uses `demand`-derived history, both of which are inadmissible here. That is not an inconsistency: the two notebooks have different targets. Prediction estimates `E[y | features]`, where any variable carrying information is admissible and a collider is simply a useful covariate. Causal identification estimates an effect, where conditioning on a collider opens a non-causal path. Same data, different estimand, different admissible set.

## The three assumptions this rests on

1. **Parallel trends** — treated and control zones would have moved together absent the treatment. Checked as a plot in 3.4; believed or not on the evidence there.
2. **No unmeasured confounding** — the graph above is complete. This is unverifiable from data; 3.5 reports how strong an omitted cause would have to be to overturn the estimate.
3. **Positivity** — every covariate stratum contains both treated and control units. Checked by the overlap plot in 3.5.
