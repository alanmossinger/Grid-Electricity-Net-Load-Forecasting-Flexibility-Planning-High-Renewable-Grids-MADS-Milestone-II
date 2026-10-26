"""Data sources: EIA-930 (demand, generation, day-ahead forecast), Open-Meteo weather (ERA5 and the
archived forecasts), and ancillary-service reserve prices (CAISO, ERCOT, MISO). Pulled files are cached
under data/raw/, so the pipeline runs without network access.
"""
from __future__ import annotations


# ---- EIA-930 -----------------------------------------------------------------------------------

import time
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

BASE = "https://api.eia.gov/v2/electricity/rto"
REGION_URL = f"{BASE}/region-data/data/"
FUEL_URL = f"{BASE}/fuel-type-data/data/"

PAGE = 5000            # the API's hard cap per request
MAX_RETRIES = 4
BACKOFF = 3.0          # seconds, doubled per retry

# EIA-930 respondent codes. The comment is the common name people say out loud —
# keeping both here is what stops the wrong one being typed into a facet.
RESPONDENTS = {
    "CISO": "California ISO (CAISO)",
    "ERCO": "ERCOT",
    "MISO": "Midcontinent ISO",
    "SWPP": "Southwest Power Pool (SPP)",
    "BPAT": "Bonneville Power Administration",
    "NYIS": "New York ISO (NYISO)",
}

# The full fuel-code vocabulary, confirmed against the API's own facet metadata:
#   https://api.eia.gov/v2/electricity/rto/fuel-type-data/facet/fueltype/
# NOT guessed from the data. An unmapped code lands in the panel under its raw EIA
# name, undocumented and mislabelled, which is how BAT/UES/SNB first appeared here.
#
# The storage and hybrid codes are recent: EIA began publishing BAT and UES for ERCO
# in Nov 2024 and SNB for BPAT in Jul 2024, i.e. inside this project's window.
FUEL_MAP = {
    # conventional generation
    "SUN": "solar", "WND": "wind", "WAT": "hydro", "NUC": "nuclear",
    "NG": "gas", "COL": "coal", "OIL": "oil", "OTH": "other",
    "GEO": "geothermal", "UNK": "unknown_fuel",
    # storage — these are NOT generation and must never be added to renewables.
    # UES in particular is reported NEGATIVE while charging (it is load, not supply).
    "BAT": "battery", "PS": "pumped_storage",
    "OES": "other_storage", "UES": "unknown_storage",
    # hybrid renewable-plus-storage. IMPORTANT: SNB is solar that is NOT counted in
    # SUN, and WNB is wind that is NOT counted in WND. See the note in
    # data.derive_net_load about what this means for the target.
    "SNB": "solar_battery", "WNB": "wind_battery",
}
REGION_MAP = {
    "D": "demand", "DF": "demand_forecast", "NG": "net_generation",
    "TI": "total_interchange",
}


class EIAError(RuntimeError):
    """Raised when the API returns something unusable — loudly, and early."""


def _request(url: str, params: dict) -> dict:
    """One GET with retries. 429 and 5xx are retried; 4xx is not (it will not fix itself)."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise EIAError(f"network error after {MAX_RETRIES} attempts: {exc}") from exc
            time.sleep(BACKOFF * 2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF * 2 ** attempt)
            continue
        if r.status_code in (401, 403):
            raise EIAError(
                "EIA rejected the API key (HTTP %s). Check EIA_API_KEY in .env — "
                "register free at https://www.eia.gov/opendata/" % r.status_code
            )
        raise EIAError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
    raise EIAError("unreachable")


def _paginate(url: str, params: dict, label: str, verbose: bool = True) -> pd.DataFrame:
    """Walk `offset` until the API stops handing out full pages."""
    rows, offset, total = [], 0, None
    while True:
        p = dict(params, offset=offset, length=PAGE)
        payload = _request(url, p).get("response", {})
        chunk = payload.get("data", [])
        if total is None:
            total = payload.get("total")
            total = int(total) if total is not None else None
        rows.extend(chunk)
        if verbose:
            print(f"    {label}: {len(rows):>7,}" + (f" / {total:,}" if total else ""), end="\r")
        if len(chunk) < PAGE:
            break
        offset += PAGE
        time.sleep(0.2)          # be a polite client
    if verbose:
        print()
    if total is not None and len(rows) < total:
        raise EIAError(
            f"{label}: pagination stopped at {len(rows)} of {total} rows. "
            f"Re-run; if it repeats, narrow the date range and pull in two passes."
        )
    return pd.DataFrame(rows)


def _fetch(url: str, zone: str, start: str, end: str, api_key: str,
           type_facet: str, label: str, verbose: bool = True) -> pd.DataFrame:
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": zone,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    df = _paginate(url, params, label=label, verbose=verbose)
    if df.empty:
        raise EIAError(
            f"{label}: empty payload for respondent '{zone}'. The respondent code is "
            f"almost certainly wrong — valid codes are {sorted(RESPONDENTS)} "
            f"(CISO not CAISO, ERCO not ERCOT, SWPP not SPP). A bad code returns "
            f"HTTP 200 with no rows."
        )
    df = df.rename(columns={type_facet: "series"})
    # The explicit cast. See the module docstring, point 3.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df[cfg.TIME_COL] = pd.to_datetime(df["period"], utc=True, errors="coerce").dt.tz_localize(None)
    return df[[cfg.TIME_COL, "series", "value"]]


def load_zone(zone: str, start: str = cfg.START, end: str = cfg.END,
              api_key: str | None = None, cache: bool = True,
              verbose: bool = True) -> pd.DataFrame:
    """Hourly panel for ONE balancing authority: demand, forecast, and fuel mix.

    Returns one row per timestamp, wide, with the columns of REGION_MAP and
    FUEL_MAP that the API actually returned.

    Smoke-test one zone and one year before the full pull — the full pull is
    several hundred requests:

        python -c "import os; from src.sources import load_zone; \\
            print(load_zone('CISO','2023-01-01T00','2023-12-31T23').shape)"
    """
    import os

    if zone not in RESPONDENTS:
        raise EIAError(f"'{zone}' is not an EIA-930 respondent code. Valid: {sorted(RESPONDENTS)}")

    # The cache is checked BEFORE the key: a zone that was already pulled needs no
    # credential to be read back, so the committed data/raw/ files make every notebook
    # runnable without an EIA account. The key is only required for a fresh pull.
    cache_path = Path(cfg.DATA_RAW) / f"eia930_{zone}_{start[:4]}_{end[:4]}.parquet"
    if cache and cache_path.exists():
        if verbose:
            print(f"  {zone}: cached ({cache_path.name})")
        return pd.read_parquet(cache_path)

    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise EIAError(
            "No EIA_API_KEY. Register free at https://www.eia.gov/opendata/ and put "
            "EIA_API_KEY=... in .env at the repository root."
        )

    if verbose:
        print(f"  {zone}  ({RESPONDENTS[zone]})  {start} -> {end}")

    region = _fetch(REGION_URL, zone, start, end, api_key, "type",
                    label=f"{zone} region-data", verbose=verbose)
    fuel = _fetch(FUEL_URL, zone, start, end, api_key, "fueltype",
                  label=f"{zone} fuel-type", verbose=verbose)

    reg_w = (region.pivot_table(index=cfg.TIME_COL, columns="series", values="value",
                                aggfunc="mean")
             .rename(columns=REGION_MAP))
    fue_w = (fuel.pivot_table(index=cfg.TIME_COL, columns="series", values="value",
                              aggfunc="mean")
             .rename(columns=FUEL_MAP))
    # `NG` means two different things in the two endpoints — net generation in
    # region-data, natural gas in fuel-type-data. Renaming BEFORE the join is what
    # stops one silently overwriting the other.
    out = reg_w.join(fue_w, how="outer").sort_index().reset_index()
    out[cfg.GROUP_COL] = zone

    if cache:
        out.to_parquet(cache_path, index=False)
    if verbose:
        print(f"  {zone}: {len(out):,} hours, columns {sorted(c for c in out.columns)}")
    return out


def load_all(zones=cfg.ZONES, start: str = cfg.START, end: str = cfg.END,
             api_key: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Every zone, stacked long-ways. Cached per zone, so an interrupted pull resumes."""
    frames = [load_zone(z, start, end, api_key, verbose=verbose) for z in zones]
    return pd.concat(frames, ignore_index=True)


# ---- Open-Meteo weather ------------------------------------------------------------------------

import time
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# --- the variable set, and why each one is here ----------------------------
# Added 2026-08-23. Until then only temperature, humidity, 10 m wind and cloud
# cover were pulled, and the report's own limitation section named that as the
# largest technical gap: cloud cover is a PROXY for the thing that actually drives
# the solar belly, and 10 m wind is not the height a turbine sees.
#
#   shortwave_radiation      global horizontal irradiance — the physical driver of
#                            PV output. Cloud cover is a coarse summary of it.
#   direct_radiation         the beam component; tracking and concentrating plant
#   diffuse_radiation        responds to it differently from the diffuse part.
#   wind_speed_100m          hub height. Turbine power goes as v^3 below rating,
#                            and the 10 m value understates shear over rough terrain.
#   wind_gusts_10m           turbine cut-out events — the steep-ramp alert feature.
#   apparent_temperature     what cooling load responds to: temperature after
#   dew_point_2m             humidity, wind and radiation. ERCO and MISO summers are
#                            humid and the dry-bulb reading understates the load.
#   snowfall, snow_depth     Regime 2 winters, and PV derating under snow cover.
#   precipitation            minor, and hydro-adjacent for BPAT.
#   surface_pressure         frontal passage marker; cheap, and it costs one column.
#
# THE ADMISSIBILITY ARGUMENT, which must be stated where a reviewer will look for
# it. Every one of these is EXOGENOUS: none is a term in net_load = demand - wind
# - solar, and all are forecast by public NWP at every horizon this project uses.
# The leakage ban applies to demand, wind, solar and ramp because they COMPOSE the
# target identically — not to physical drivers of it. Irradiance is not leakage;
# `solar` is.
WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_100m",
    "wind_gusts_10m",
    "apparent_temperature",
    "dew_point_2m",
    "snowfall",
    "snow_depth",
    "precipitation",
    "surface_pressure",
]

# API name -> the column name used everywhere in this project.
RENAME = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "ghi",
    "direct_radiation": "dni_horiz",
    "diffuse_radiation": "dhi",
    "wind_speed_100m": "wind_speed_100m",
    "wind_gusts_10m": "wind_gust",
    "apparent_temperature": "apparent_temp",
    "dew_point_2m": "dew_point",
    "snowfall": "snowfall",
    "snow_depth": "snow_depth",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
}

# The cache filename carries this. WHY: the v1 cache holds four columns under the
# same (zone, years) name. Without a version in the path, a v2 pull would find the
# v1 file, return it, and the eleven new columns would silently not exist — the
# join would still succeed and `temp` would still be 100% populated, so every
# guard in the project would pass.
WEATHER_CACHE_SCHEMA = "v2"

# (lat, lon, why this point)
BA_COORDS = {
    "CISO": (34.05, -118.24, "Los Angeles — the largest load centre inside CAISO; "
                             "the LA basin drives the summer cooling peak."),
    "ERCO": (32.78, -96.80, "Dallas-Fort Worth — the largest ERCOT metro; Houston is "
                            "comparable and more humid, and is the named sensitivity check."),
    "MISO": (44.98, -93.27, "Minneapolis — MISO's winter-peaking northern load. The "
                            "weakest of the six choices: MISO spans to Louisiana."),
    "SWPP": (35.47, -97.52, "Oklahoma City — central to the SPP footprint and close to "
                            "the wind corridor that sets its net load."),
    "BPAT": (45.52, -122.68, "Portland — BPA's dominant load centre; hydro and "
                             "Columbia Gorge wind sit upstream of it."),
    "NYIS": (40.71, -74.01, "New York City — NYISO Zone J is the load concentration; "
                            "upstate is hydro-rich and much colder."),
}


class WeatherError(RuntimeError):
    pass


def _to_date(stamp: str) -> str:
    """'2019-01-01T00' -> '2019-01-01' (the archive API takes dates, not hours)."""
    return str(stamp)[:10]


def load_zone_weather(zone: str, start: str = cfg.START, end: str = cfg.END,
                      cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """The full ERA5 variable set for one BA, UTC-stamped, ready to join on
    (zone, timestamp). See WEATHER_HOURLY_VARS for what is pulled and why."""
    if zone not in BA_COORDS:
        raise WeatherError(f"no coordinates for '{zone}'. Add it to BA_COORDS with a reason.")
    lat, lon, _why = BA_COORDS[zone]

    cache_path = (Path(cfg.DATA_RAW)
                  / f"weather_{zone}_{start[:4]}_{end[:4]}_{WEATHER_CACHE_SCHEMA}.parquet")
    if cache and cache_path.exists():
        if verbose:
            print(f"  {zone}: weather cached ({cache_path.name})")
        return pd.read_parquet(cache_path)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": _to_date(start),
        "end_date": _to_date(end),
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": "UTC",          # see the module docstring. Do not change this.
    }
    for attempt in range(4):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=120)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise WeatherError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as exc:
            if attempt == 3:
                raise WeatherError(f"network error: {exc}") from exc
            time.sleep(5 * (attempt + 1))
    else:
        raise WeatherError("Open-Meteo rate limit not cleared after 4 attempts.")

    hourly = r.json().get("hourly", {})
    if not hourly.get("time"):
        raise WeatherError(f"{zone}: no hourly block returned. Check the date range.")

    cols = {cfg.TIME_COL: pd.to_datetime(hourly["time"])}
    for api_name in WEATHER_HOURLY_VARS:
        if api_name not in hourly:
            raise WeatherError(
                f"{zone}: Open-Meteo returned no `{api_name}`. A variable the project "
                f"asks for and does not get must fail loudly here — silently missing "
                f"columns are how a feature set shrinks without anyone noticing.")
        cols[RENAME[api_name]] = hourly[api_name]
    df = pd.DataFrame(cols)
    df[cfg.GROUP_COL] = zone
    df = df.dropna(subset=[cfg.TIME_COL])

    if cache:
        df.to_parquet(cache_path, index=False)
    if verbose:
        print(f"  {zone}: {len(df):,} hourly weather rows "
              f"({df[cfg.TIME_COL].min()} -> {df[cfg.TIME_COL].max()}, UTC)")
    return df


def load_all_weather(zones=cfg.ZONES, start: str = cfg.START, end: str = cfg.END,
                     verbose: bool = True) -> pd.DataFrame:
    """One call per zone. Six calls total."""
    frames = []
    for z in zones:
        frames.append(load_zone_weather(z, start, end, verbose=verbose))
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# the forecast AS ISSUED — what an operator actually held at the origin
# ---------------------------------------------------------------------------
# Horizon -> which previous run stands in for it. h=1 is deliberately absent: an
# hour-ahead weather forecast is close enough to observation that ERA5 is the
# honest stand-in, and the API's shortest lead is a whole day. Inventing a
# sub-daily degradation would be fabricating the number this whole exercise
# exists to stop fabricating.
FORECAST_LEAD_BY_HORIZON = {24: 1, 48: 2, 72: 3}

# The variables re-pulled as forecasts. Not all fifteen: the previous-runs API
# serves the operational model's own fields, and the four that matter for net load
# are temperature, cloud, irradiance and hub-height wind. Anything not re-pulled
# keeps its reanalysis value, which makes the measured degradation a LOWER bound
# on that column's contribution — stated rather than glossed.
FORECAST_VARS = ["temperature_2m", "cloud_cover", "shortwave_radiation",
                 "wind_speed_100m", "relative_humidity_2m", "wind_speed_10m"]


def load_zone_forecast_weather(zone: str, start: str, end: str,
                               leads=(1, 2, 3), cache: bool = True,
                               verbose: bool = True) -> pd.DataFrame:
    """Weather for one BA as FORECAST `leads` days ahead, one column set per lead.

    Returns a long frame with a `lead_day` column, so a caller can select the lead
    matching the horizon it is scoring. Columns are named exactly as the archive
    loader names them, so a scoring pass can swap one frame for the other without
    touching the pipeline.
    """
    if zone not in BA_COORDS:
        raise WeatherError(f"no coordinates for '{zone}'.")
    lat, lon, _why = BA_COORDS[zone]

    cache_path = (Path(cfg.DATA_RAW)
                  / f"weatherfc_{zone}_{_to_date(start)}_{_to_date(end)}_{WEATHER_CACHE_SCHEMA}.parquet")
    if cache and cache_path.exists():
        if verbose:
            print(f"  {zone}: forecast weather cached ({cache_path.name})")
        return pd.read_parquet(cache_path)

    fields = []
    for v in FORECAST_VARS:
        fields.append(v)
        fields += [f"{v}_previous_day{d}" for d in leads]

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": _to_date(start), "end_date": _to_date(end),
        "hourly": ",".join(fields),
        "timezone": "UTC",          # same trap as the archive loader. Do not change.
    }
    for attempt in range(4):
        try:
            r = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=180)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise WeatherError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as exc:
            if attempt == 3:
                raise WeatherError(f"network error: {exc}") from exc
            time.sleep(5 * (attempt + 1))
    else:
        raise WeatherError("Open-Meteo rate limit not cleared after 4 attempts.")

    hourly = r.json().get("hourly", {})
    if not hourly.get("time"):
        raise WeatherError(f"{zone}: no hourly block from the previous-runs API.")

    stamps = pd.to_datetime(hourly["time"])
    frames = []
    for d in leads:
        block = {cfg.TIME_COL: stamps, "lead_day": d}
        for v in FORECAST_VARS:
            key = f"{v}_previous_day{d}"
            if key not in hourly:
                raise WeatherError(
                    f"{zone}: no `{key}`. The lead is either out of range for this "
                    f"date window or the variable is not archived — do not fall back "
                    f"to the analysis value, that would hide the degradation being "
                    f"measured.")
            block[RENAME[v]] = hourly[key]
        frames.append(pd.DataFrame(block))

    df = pd.concat(frames, ignore_index=True)
    df[cfg.GROUP_COL] = zone
    df = df.dropna(subset=[cfg.TIME_COL])

    if cache:
        df.to_parquet(cache_path, index=False)
    if verbose:
        got = df.groupby("lead_day")[RENAME["temperature_2m"]].apply(lambda s: s.notna().mean())
        print(f"  {zone}: {len(df):,} forecast rows, temp coverage by lead "
              f"{ {int(k): round(float(v), 3) for k, v in got.items()} }")
    return df


def load_all_forecast_weather(zones=cfg.ZONES, start: str = "2024-02-01",
                              end: str = cfg.END, leads=(1, 2, 3),
                              verbose: bool = True) -> pd.DataFrame:
    """One call per zone over the holdout window. Six calls."""
    frames = []
    for z in zones:
        frames.append(load_zone_forecast_weather(z, start, end, leads=leads,
                                                 verbose=verbose))
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)


def coord_table() -> pd.DataFrame:
    """The coordinate choices and their justifications — goes straight into the report."""
    return pd.DataFrame(
        [(z, lat, lon, why) for z, (lat, lon, why) in BA_COORDS.items()],
        columns=["zone", "lat", "lon", "why this point"],
    )


# ---- Reserve prices ----------------------------------------------------------------------------

#!/usr/bin/env python3

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

OASIS = "http://oasis.caiso.com/oasisapi/SingleZip"
UA = {"User-Agent": "MADS-Milestone2 research (net-load flexibility study)"}
PRICE_CACHE_SCHEMA = "v1"

# CAISO clears these regionally. The plain regions (AS_CAISO, AS_NP26, AS_SP26) read
# zero because the binding price sits in the EXPANDED regions -- verified on two
# separate days before this was written. Using the plain region would have produced a
# price of exactly zero for every hour, which is the kind of wrong that looks tidy.
SYSTEM_REGION = "AS_CAISO_EXP"

# Reserve analogues. RU/RD are regulation (a different service, faster) and RMU/RMD are
# mileage; neither is what a day-ahead headroom requirement buys.
RESERVE_PRODUCTS = ("SR", "NR")

POLITE_SECONDS = 5.0        # CAISO asks callers not to exceed roughly this rate


def _one_day(day: pd.Timestamp) -> pd.DataFrame:
    """One operating day of DAM ancillary clearing prices, tidy."""
    # The operating day runs from local midnight to local midnight: 07:00 UTC in summer
    # (PDT) but 08:00 UTC in winter (PST). A fixed 07:00Z fetched the wrong hours in winter.
    t0 = pd.Timestamp(day.date()).tz_localize("America/Los_Angeles").tz_convert("UTC")
    t1 = (pd.Timestamp(day.date()) + pd.Timedelta(days=1)).tz_localize("America/Los_Angeles").tz_convert("UTC")
    start, end = t0.strftime("%Y%m%dT%H:%M-0000"), t1.strftime("%Y%m%dT%H:%M-0000")
    r = requests.get(OASIS, headers=UA, timeout=180, params={
        "queryname": "PRC_AS", "startdatetime": start, "enddatetime": end,
        "version": "1", "market_run_id": "DAM",
        "anc_type": "ALL", "anc_region": "ALL", "resultformat": "6"})
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise RuntimeError(f"OASIS returned no zip for {day.date()}: {r.text[:200]}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df = df[df["XML_DATA_ITEM"].astype(str).str.contains("CLR_PRC", na=False)]
    df = df[(df["ANC_REGION"] == SYSTEM_REGION)
            & (df["ANC_TYPE"].isin(RESERVE_PRODUCTS))]
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "product", "price_usd_per_mw_h"])
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True,
                                    format="mixed"),
        "product": df["ANC_TYPE"].astype(str),
        "price_usd_per_mw_h": pd.to_numeric(df["MW"], errors="coerce"),
    })
    return out.dropna().sort_values(["timestamp", "product"]).reset_index(drop=True)


def load_caiso_as_prices(start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """Hourly CAISO day-ahead reserve clearing prices over [start, end).

    One request per operating day, because the API silently truncates longer windows.
    Cached per (start, end) so a re-run costs nothing.
    """
    cache_path = (Path(cfg.DATA_RAW)
                  / f"ancillary_caiso_{start[:10]}_{end[:10]}_{PRICE_CACHE_SCHEMA}.parquet")
    if cache and cache_path.exists():
        return _tidy_hours(pd.read_parquet(cache_path), start, end)

    days = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    frames, missing = [], []
    for i, day in enumerate(days):
        try:
            d = _one_day(day)
        except Exception as exc:                       # noqa: BLE001
            missing.append((str(day.date()), f"{type(exc).__name__}: {exc}"))
            d = pd.DataFrame()
        if d.empty:
            missing.append((str(day.date()), "empty"))
        else:
            frames.append(d)
        if i + 1 < len(days):
            time.sleep(POLITE_SECONDS)

    if not frames:
        raise RuntimeError("CAISO returned nothing for the whole window")
    out = pd.concat(frames, ignore_index=True)

    # The silent-truncation guard. A short frame here means the API capped the window
    # or the fetch died part-way; either way the price series would be wrong and quiet.
    covered = out["timestamp"].dt.floor("D").nunique()
    if covered < 0.95 * len(days):
        raise RuntimeError(
            f"CAISO price series covers {covered} of {len(days)} days "
            f"({len(missing)} gaps, first few: {missing[:5]}). Refusing to return a "
            f"short series -- pricing reserve on it would understate every hour it "
            f"does not contain.")

    out.attrs["missing_days"] = missing
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return _tidy_hours(out, start, end)


def _tidy_hours(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """One price per hour and product, inside [start, end). The cache committed in
    data/raw was fetched with a fixed 07:00Z day boundary, which repeats an hour at the
    November clock change and reaches into the day before the window in winter."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    keep = (ts >= lo) & (ts < hi) & ~df.assign(timestamp=ts).duplicated(["timestamp", "product"])
    return df[keep.to_numpy()].reset_index(drop=True)


def hourly_reserve_price(start: str, end: str, product: str = "SR",
                         cache: bool = True) -> pd.Series:
    """One hourly $/MW-h series for the named reserve product, indexed by UTC hour."""
    df = load_caiso_as_prices(start, end, cache=cache)
    s = (df[df["product"] == product]
         .set_index("timestamp")["price_usd_per_mw_h"]
         .sort_index())
    return s[~s.index.duplicated(keep="first")]


def implied_annual_usd_per_mw(start: str, end: str, product: str = "SR",
                              cache: bool = True) -> float:
    """The observed price expressed in the units the report's assumption uses.

    Mean $/MW-h across the window x 8,760, so it can be set beside the assumed
    $/MW-year directly. This is the number that says whether the assumption was
    reasonable for THIS operator, rather than for a market it does not trade in.
    """
    s = hourly_reserve_price(start, end, product=product, cache=cache)
    return float(s.mean() * 8760.0)

# ===========================================================================
# ERCOT — the market that matters most here, and the one behind a login
# ===========================================================================
# ERCOT carries the largest share of the fleet reserve requirement, so its price is
# the one worth having. Unlike CAISO it is not open: every call needs BOTH a
# subscription key and a bearer ID token minted from an account username and password.
# Tokens last one hour and cannot be refreshed, so long fetches re-mint mid-flight.
#
# UNTESTED AGAINST THE LIVE API. This was written from ERCOT's published specification
# without credentials to exercise it. Treat the first successful pull as an
# investigation -- check the product codes and the row count per day before believing
# any number that comes out of it.

ERCOT_TOKEN_URL = ("https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
                   "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token")
ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
ERCOT_AS_URL = ("https://api.ercot.com/api/public-reports/"
                "np4-188-cd/dam_clear_price_for_cap")

# ERCOT's reserve-family products. ECRS was introduced partly to cover ramp and
# net-load uncertainty, which is the closest match in this market to what this project
# sizes; RRS is the established responsive reserve. REGUP/REGDN are regulation -- a
# faster, different service -- and are deliberately excluded, for the same reason
# CAISO's RU/RD are.
ERCOT_RESERVE_PRODUCTS = ("ECRS", "RRS", "NSPIN")

_ERCOT_TOKEN: dict = {"value": None, "expires": 0.0}


def _ercot_credentials() -> tuple[str, str, str]:
    import os
    key = os.environ.get("ERCOT_SUBSCRIPTION_KEY")
    user = os.environ.get("ERCOT_USERNAME")
    pwd = os.environ.get("ERCOT_PASSWORD")
    if not (key and user and pwd):
        raise RuntimeError(
            "No ERCOT credentials. Register free at https://apiexplorer.ercot.com/ "
            "(Sign Up, then Products -> Public API -> Subscribe, then Profile -> Show "
            "to copy the Primary key) and put ERCOT_SUBSCRIPTION_KEY, ERCOT_USERNAME "
            "and ERCOT_PASSWORD in .env at the repository root. .env is gitignored.")
    return key, user, pwd


def _ercot_id_token() -> str:
    """Mint (or reuse) an ID token. Valid one hour; there is no refresh."""
    now = time.time()
    if _ERCOT_TOKEN["value"] and now < _ERCOT_TOKEN["expires"] - 120:
        return _ERCOT_TOKEN["value"]
    _, user, pwd = _ercot_credentials()
    r = requests.post(ERCOT_TOKEN_URL, timeout=60, data={
        "grant_type": "password", "username": user, "password": pwd,
        "client_id": ERCOT_CLIENT_ID, "response_type": "id_token",
        "scope": f"openid {ERCOT_CLIENT_ID} offline_access"})
    if not r.ok:
        raise RuntimeError(
            f"ERCOT rejected the sign-in (HTTP {r.status_code}). Check ERCOT_USERNAME "
            f"and ERCOT_PASSWORD in .env; the account must be the one that owns the "
            f"subscription. Body: {r.text[:200]}")
    tok = r.json().get("id_token")
    if not tok:
        raise RuntimeError(f"ERCOT returned no id_token: {list(r.json())}")
    _ERCOT_TOKEN.update(value=tok, expires=now + 3600.0)
    return tok


def load_ercot_as_prices(start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """Hourly ERCOT day-ahead market clearing prices for capacity over [start, end).

    Returns the same tidy shape as the CAISO loader -- timestamp, product,
    price_usd_per_mw_h -- so the two can be concatenated and handled identically.
    """
    cache_path = (Path(cfg.DATA_RAW)
                  / f"ancillary_ercot_{start[:10]}_{end[:10]}_{PRICE_CACHE_SCHEMA}.parquet")
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    key, _, _ = _ercot_credentials()
    rows, page, pages = [], 1, None
    while pages is None or page <= pages:
        r = requests.get(ERCOT_AS_URL, timeout=180, params={
            "deliveryDateFrom": start[:10], "deliveryDateTo": end[:10],
            "page": page, "size": 5000,
        }, headers={"Ocp-Apim-Subscription-Key": key,
                    "Authorization": f"Bearer {_ercot_id_token()}", **UA})
        if r.status_code in (401, 403):
            raise RuntimeError(
                f"ERCOT refused the request (HTTP {r.status_code}). Both a valid "
                f"subscription key AND a bearer token are required; a key alone is not "
                f"enough. Body: {r.text[:200]}")
        r.raise_for_status()
        payload = r.json()
        fields = [f["name"] for f in payload.get("fields", [])]
        rows.extend(payload.get("data", []))
        meta = payload.get("_meta", {})
        pages = meta.get("totalPages", 1)
        page += 1
        time.sleep(0.5)

    if not rows:
        raise RuntimeError("ERCOT returned no rows for the window")
    raw = pd.DataFrame(rows, columns=fields)

    # Column names are taken from the response rather than assumed, because this has
    # not been exercised against the live API.
    def _col(*cands):
        for c in cands:
            for f in raw.columns:
                if f.lower() == c.lower():
                    return f
        raise RuntimeError(f"ERCOT response has no column like {cands}; got "
                           f"{list(raw.columns)}")

    dcol, hcol = _col("deliveryDate"), _col("hourEnding")
    acol, pcol = _col("ancillaryType", "AncillaryType"), _col("MCPC", "mcpc")
    hours = pd.to_numeric(raw[hcol].astype(str).str.slice(0, 2), errors="coerce")
    ts = (pd.to_datetime(raw[dcol]) + pd.to_timedelta(hours - 1, unit="h"))
    out = pd.DataFrame({
        "timestamp": ts.dt.tz_localize(cfg.ZONE_TZ["ERCO"], ambiguous="NaT",
                                       nonexistent="NaT").dt.tz_convert("UTC"),
        "product": raw[acol].astype(str).str.upper(),
        "price_usd_per_mw_h": pd.to_numeric(raw[pcol], errors="coerce"),
    }).dropna()
    out = out[out["product"].isin(ERCOT_RESERVE_PRODUCTS)]
    out = out.sort_values(["timestamp", "product"]).reset_index(drop=True)

    days = pd.date_range(start, end, freq="D", inclusive="left")
    covered = out["timestamp"].dt.floor("D").nunique()
    if covered < 0.95 * len(days):
        raise RuntimeError(
            f"ERCOT price series covers {covered} of {len(days)} days. Refusing to "
            f"return a short series -- pricing reserve on it would understate every "
            f"hour it does not contain.")
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out

# ===========================================================================
# MISO — open, no key, and the second-largest reserve requirement in the panel
# ===========================================================================
# Reached for after ERCOT's portal blocked access at the network level. MISO publishes
# one CSV per operating day with no key and no login, and it carries 6,807 MW of the
# fleet requirement against ERCOT's 7,300 -- so it is very nearly as informative.
#
# TWO TRAPS, both explicit in the file and both silent if ignored:
#
#   1. "All Hours-Ending are Eastern Standard Time (EST)" -- a FIXED -5 offset, not
#      America/New_York. Parsing it as a DST-aware zone shifts every summer hour by
#      one, which would misalign the price against the reserve requirement for more
#      than half the holdout and produce a plausible, wrong, seasonal pattern.
#   2. Columns are HOUR ENDING. HE 1 covers 00:00-01:00, so the interval START is
#      HE - 1. Off-by-one here would shift the whole series against the evening ramp,
#      which is the one part of the day this project cares most about.
#
# The file also carries a per-node section thousands of rows long after the system
# block; only the "MISO Wide" rows are the system clearing price.

MISO_URL = "https://docs.misoenergy.org/marketreports/{ymd}_asm_exante_damcp.csv"
MISO_EST = "Etc/GMT+5"          # fixed -5, deliberately NOT America/New_York

# GENSPIN is spinning reserve and GENSUPP supplemental -- the same pair of products
# CAISO calls SR and NR. Regulation (REG) is excluded for the same reason it is at
# CAISO: it is a faster, different service.
MISO_PRODUCTS = {"GENSPINMCP": "SPIN", "GENSUPPMCP": "SUPP"}
MISO_POLITE_SECONDS = 1.0


def _miso_one_day(day: pd.Timestamp) -> pd.DataFrame:
    """One operating day of MISO system-wide ancillary clearing prices, tidy."""
    r = requests.get(MISO_URL.format(ymd=day.strftime("%Y%m%d")),
                     headers=UA, timeout=120)
    if r.status_code == 404:
        return pd.DataFrame(columns=["timestamp", "product", "price_usd_per_mw_h"])
    r.raise_for_status()

    rows = []
    for line in r.text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 27 or parts[0] != "MISO Wide":
            continue                       # skip the per-node section entirely
        product = MISO_PRODUCTS.get(parts[2])
        if product is None:
            continue
        for he in range(1, 25):
            val = parts[2 + he]
            if val in ("", "-"):
                continue
            # HOUR ENDING -> interval start is HE - 1, in fixed EST.
            ts = (day.tz_localize(None) + pd.Timedelta(hours=he - 1))
            rows.append((ts, product, float(val)))

    if not rows:
        return pd.DataFrame(columns=["timestamp", "product", "price_usd_per_mw_h"])
    out = pd.DataFrame(rows, columns=["timestamp", "product", "price_usd_per_mw_h"])
    out["timestamp"] = (out["timestamp"].dt.tz_localize(MISO_EST)
                        .dt.tz_convert("UTC"))
    return out.sort_values(["timestamp", "product"]).reset_index(drop=True)


def load_miso_as_prices(start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """Hourly MISO day-ahead reserve clearing prices over [start, end).

    Same tidy shape as the CAISO loader, and the same refusal to return a short series.
    """
    cache_path = (Path(cfg.DATA_RAW)
                  / f"ancillary_miso_{start[:10]}_{end[:10]}_{PRICE_CACHE_SCHEMA}.parquet")
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    days = pd.date_range(start, end, freq="D", inclusive="left")
    frames, missing = [], []
    for i, day in enumerate(days):
        try:
            d = _miso_one_day(day)
        except Exception as exc:                       # noqa: BLE001
            missing.append((str(day.date()), f"{type(exc).__name__}: {exc}"))
            d = pd.DataFrame()
        if d.empty:
            missing.append((str(day.date()), "empty"))
        else:
            frames.append(d)
        if i + 1 < len(days):
            time.sleep(MISO_POLITE_SECONDS)

    if not frames:
        raise RuntimeError("MISO returned nothing for the whole window")
    out = pd.concat(frames, ignore_index=True)
    covered = out["timestamp"].dt.floor("D").nunique()
    if covered < 0.95 * len(days):
        raise RuntimeError(
            f"MISO price series covers {covered} of {len(days)} days "
            f"({len(missing)} gaps, first few: {missing[:5]}). Refusing to return a "
            f"short series -- pricing reserve on it would understate every hour it "
            f"does not contain.")
    out.attrs["missing_days"] = missing
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out

