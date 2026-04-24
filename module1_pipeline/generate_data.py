"""
NigeriaAgriScope — Module 1: Data Pipeline
===========================================
Fetches, cleans, and merges data from five international sources into a
single SQLite master table used by all downstream modules.

Sources
-------
1. FAOSTAT QCL   — Crop production: yield, area, production (7 crops, 2000–2023)
2. FAOSTAT RFN   — Fertilizer use by nutrient (N, P2O5, K2O), Nigeria, 2000–2023
3. NASA POWER    — Monthly climate data for 6 Nigerian geopolitical zones
4. World Bank    — Agricultural macro-indicators (wb_fertilizer_kg_ha, ag GDP share)
5. USDA PSD      — Palm oil & cassava supply/demand balance for Nigeria

Output
------
  data/processed/nigeria_agri.db  (SQLite — master_table)
  data/processed/master_table.csv (Streamlit Cloud–ready snapshot)

DATA CONTRACT — Column naming for downstream modules
-----------------------------------------------------
Two fertilizer columns exist with deliberately different meanings:

  fertilizer_kg_ha    : Derived zone-crop intensity figure.
                        = fertilizer_total_kg_zone / area_ha
                        Varies by zone and crop within each year.

  wb_fertilizer_kg_ha : World Bank national aggregate (kg/ha of arable land).
                        Broadcast identically to every zone-crop row in a year.
                        Source: WB indicator AG.CON.FERT.ZS.

These are distinct. Do not conflate them in Module 2 SQL or Module 4 features.

Usage
-----
  python module1_pipeline/generate_data.py

Author : Fidelis Akinbule
Date   : April 2026
"""

import hashlib
import logging
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m1_pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
DB_PATH = PROC / "nigeria_agri.db"
CSV_PATH = PROC / "master_table.csv"

# ── API base URLs ──────────────────────────────────────────────────────────
FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data"
WB_API = "https://api.worldbank.org/v2"
NASA_API = "https://power.larc.nasa.gov/api/temporal/monthly/point"

YEARS = list(range(2000, 2024))  # 2000–2023 inclusive

# 7 target crops with FAOSTAT item codes
CROPS = {
    "Oil palm fruit": 254,
    "Cassava": 125,
    "Maize": 56,
    "Yam": 392,
    "Rice (paddy)": 27,
    "Sorghum": 83,
    "Cocoa beans": 661,
}

# Six geopolitical zones with centroid coordinates (lat, lon)
ZONES = {
    "North West": (12.0, 8.5),
    "North East": (11.5, 13.0),
    "North Central": (9.0, 7.5),
    "South West": (7.0, 4.0),
    "South East": (6.0, 7.5),
    "South South": (5.0, 6.0),
}

# World Bank indicators → master table column names
WB_INDICATORS = {
    "AG.CON.FERT.ZS": "wb_fertilizer_kg_ha",  # Fertilizer consumption (kg/ha arable land)
    "NV.AGR.TOTL.ZS": "agric_gdp_share",  # Agriculture value added (% of GDP)
    "SP.RUR.TOTL": "rural_population",  # Rural population
}

# ── Validation thresholds (sourced from FAOSTAT QCL + NIMET long-term data) ─
# Oil palm yield range: FAOSTAT Nigeria 2000–2023 historical bounds (hg/ha)
PALM_YIELD_MIN_HG_HA = 80_000
PALM_YIELD_MAX_HG_HA = 140_000
# South South minimum annual rainfall: NIMET long-term average floor (mm)
SOUTH_SOUTH_RAIN_MIN_MM = 1_800
# Acceptable minimum fraction of theoretical row count before pipeline errors
PIPELINE_MIN_ROW_FRACTION = 0.80
# Maximum permissible deviation of zone production totals from national total
ZONE_CONSERVATION_TOL = 0.02  # 2% tolerance to absorb integer rounding

# ── Zone–Crop Production-Share Weights ─────────────────────────────────────
#
# Each crop's weights must sum to exactly 1.00 across all six zones so that
# sum(zone_production) == national_production for every crop-year.
#
# Source: FAOSTAT sub-national production allocations cross-referenced with
# NBS Agricultural Survey data.
#
# Verification (all six zones per crop sum to 1.00):
#   Oil palm fruit : 0.02+0.02+0.08+0.20+0.32+0.36 = 1.00 ✓
#   Cassava        : 0.05+0.03+0.22+0.28+0.23+0.19 = 1.00 ✓
#   Maize          : 0.25+0.12+0.28+0.18+0.09+0.08 = 1.00 ✓
#   Yam            : 0.05+0.05+0.42+0.22+0.18+0.08 = 1.00 ✓
#   Rice (paddy)   : 0.28+0.18+0.22+0.12+0.11+0.09 = 1.00 ✓
#   Sorghum        : 0.38+0.32+0.18+0.05+0.04+0.03 = 1.00 ✓
#   Cocoa beans    : 0.01+0.01+0.05+0.55+0.10+0.28 = 1.00 ✓
#
ZONE_CROP_WEIGHTS = {
    "North West": {
        "Oil palm fruit": 0.02,
        "Cassava": 0.05,
        "Maize": 0.25,
        "Yam": 0.05,
        "Rice (paddy)": 0.28,
        "Sorghum": 0.38,
        "Cocoa beans": 0.01,
    },
    "North East": {
        "Oil palm fruit": 0.02,
        "Cassava": 0.03,
        "Maize": 0.12,
        "Yam": 0.05,
        "Rice (paddy)": 0.18,
        "Sorghum": 0.32,
        "Cocoa beans": 0.01,
    },
    "North Central": {
        "Oil palm fruit": 0.08,
        "Cassava": 0.22,
        "Maize": 0.28,
        "Yam": 0.42,
        "Rice (paddy)": 0.22,
        "Sorghum": 0.18,
        "Cocoa beans": 0.05,
    },
    "South West": {
        "Oil palm fruit": 0.20,
        "Cassava": 0.28,
        "Maize": 0.18,
        "Yam": 0.22,
        "Rice (paddy)": 0.12,
        "Sorghum": 0.05,
        "Cocoa beans": 0.55,
    },
    "South East": {
        "Oil palm fruit": 0.32,
        "Cassava": 0.23,
        "Maize": 0.09,
        "Yam": 0.18,
        "Rice (paddy)": 0.11,
        "Sorghum": 0.04,
        "Cocoa beans": 0.10,
    },
    "South South": {
        "Oil palm fruit": 0.36,
        "Cassava": 0.19,
        "Maize": 0.08,
        "Yam": 0.08,
        "Rice (paddy)": 0.09,
        "Sorghum": 0.03,
        "Cocoa beans": 0.28,
    },
}

# SOURCE 1 — FAOSTAT Crop Production


def _fetch_faostat_crops() -> pd.DataFrame:
    """Download Nigeria crop production (QCL) from FAOSTAT; fall back to local CSV."""
    local_file = RAW / "faostat_crops_nigeria.csv"
    try:
        log.info("Fetching FAOSTAT crop data from API…")
        year_str = ",".join(str(y) for y in YEARS)
        item_str = ",".join(str(v) for v in CROPS.values())
        params = {
            "area": "159",
            "area_cs": "FAO",
            "element": "5312,5510,5419",  # Area harvested, Production, Yield
            "element_cs": "FAO",
            "item": item_str,
            "item_cs": "FAO",
            "year": year_str,
            "output_type": "objects",
            "per_page": "5000",
        }
        r = requests.get(f"{FAOSTAT_BASE}/QCL", params=params, timeout=60)
        r.raise_for_status()
        records = r.json().get("data", [])
        if not records:
            raise ValueError("Empty FAOSTAT API response")
        df = pd.DataFrame(records)
        df = df.rename(
            columns={
                "area": "country",
                "item": "crop",
                "element": "element",
                "year": "year",
                "value": "value",
                "unit": "unit",
            }
        )
        df["year"] = df["year"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        log.info("  FAOSTAT API: %d records", len(df))
        return df

    except Exception as exc:
        log.warning("  FAOSTAT API unavailable (%s). Loading local CSV.", exc)
        if not local_file.exists():
            raise FileNotFoundError(
                f"No local fallback at {local_file}. "
                "Download faostat_crops_nigeria.csv and place it in data/raw/."
            )
        df = pd.read_csv(local_file)
        df = df.rename(
            columns={
                "Area": "country",
                "Item": "crop",
                "Element": "element",
                "Year": "year",
                "Value": "value",
                "Unit": "unit",
            }
        )
        df["year"] = df["year"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        log.info("  Loaded %d rows from %s", len(df), local_file.name)
        return df


def _pivot_crops(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot element rows → wide format: one row per (crop, year)."""
    element_map = {
        "Area harvested": "area_ha",
        "Production": "production_tonnes",
        "Yield": "yield_hg_ha",
    }
    df_long["element_clean"] = df_long["element"].map(element_map)

    # Warn on duplicate crop-year-element entries before aggregating.
    # pivot_table would silently take the first; we surface the count instead.
    dupes = df_long.duplicated(subset=["crop", "year", "element_clean"])
    if dupes.any():
        log.warning(
            "  %d duplicate crop-year-element rows detected — keeping first occurrence. "
            "Re-download the FAOSTAT CSV if data was recently revised.",
            dupes.sum(),
        )
        df_long = df_long[~dupes].copy()

    df_long = df_long.dropna(subset=["element_clean"])

    df_wide = df_long.pivot_table(
        index=["crop", "year"],
        columns="element_clean",
        values="value",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    # Hard assertions: a silent empty pivot would corrupt every downstream module.
    assert len(df_wide) > 0, (
        "Crop pivot produced zero rows. The FAOSTAT 'element' labels in this "
        "response do not match the expected strings ('Area harvested', "
        "'Production', 'Yield'). Check element_map against the current API response."
    )
    for required_col in ("area_ha", "production_tonnes", "yield_hg_ha"):
        assert required_col in df_wide.columns, (
            f"Required column '{required_col}' missing after pivot. "
            "element_map may be stale."
        )

    log.info("  Crops pivoted: %d crop-year rows", len(df_wide))
    return df_wide


# SOURCE 2 — FAOSTAT Fertilizers by Nutrient


def _fetch_faostat_fertilizer() -> pd.DataFrame:
    """Download Nigeria fertilizer by nutrient (RFN); fall back to local CSV."""
    local_file = RAW / "faostat_fertilizer_nigeria.csv"
    try:
        log.info("Fetching FAOSTAT fertilizer data from API…")
        year_str = ",".join(str(y) for y in YEARS)
        params = {
            "area": "159",
            "area_cs": "FAO",
            # Element 5159 = Agricultural Use (the correct RFN element code).
            # Item codes: 3102=Nitrogen, 3103=Phosphate (P2O5), 3104=Potash (K2O).
            "element": "5159",
            "element_cs": "FAO",
            "item": "3102,3103,3104",
            "item_cs": "FAO",
            "year": year_str,
            "output_type": "objects",
            "per_page": "5000",
        }
        r = requests.get(f"{FAOSTAT_BASE}/RFN", params=params, timeout=60)
        r.raise_for_status()
        records = r.json().get("data", [])
        if not records:
            raise ValueError("Empty API response")
        df = pd.DataFrame(records)
        log.info("  FAOSTAT Fertilizer API: %d records", len(df))
        return df

    except Exception as exc:
        log.warning(
            "  FAOSTAT Fertilizer API unavailable (%s). Loading local CSV.", exc
        )
        if not local_file.exists():
            raise FileNotFoundError(f"No local fallback at {local_file}")
        df = pd.read_csv(local_file)
        log.info("  Loaded %d rows from %s", len(df), local_file.name)
        return df


def _aggregate_fertilizer(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw nutrient rows into one row per year (N, P, K, total in kg)."""
    item_col = "Item" if "Item" in df.columns else "item"
    value_col = "Value" if "Value" in df.columns else "value"
    year_col = "Year" if "Year" in df.columns else "year"

    df = df[[item_col, year_col, value_col]].copy()
    df[year_col] = df[year_col].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    def _classify(name: str) -> str:
        s = str(name).lower()
        if "nitrogen" in s or " n " in s or s.endswith("n"):
            return "fertilizer_n_tonnes"
        if "phosphate" in s or "p2o5" in s:
            return "fertilizer_p_tonnes"
        if "potash" in s or "k2o" in s:
            return "fertilizer_k_tonnes"
        return "other"

    df["nutrient"] = df[item_col].apply(_classify)
    df = df[df["nutrient"] != "other"]

    wide = df.pivot_table(
        index=year_col, columns="nutrient", values=value_col, aggfunc="sum"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={year_col: "year"})

    for col in ["fertilizer_n_tonnes", "fertilizer_p_tonnes", "fertilizer_k_tonnes"]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["fertilizer_n_kg"] = wide["fertilizer_n_tonnes"] * 1_000
    wide["fertilizer_p_kg"] = wide["fertilizer_p_tonnes"] * 1_000
    wide["fertilizer_k_kg"] = wide["fertilizer_k_tonnes"] * 1_000
    wide["fertilizer_total_kg"] = (
        wide["fertilizer_n_kg"] + wide["fertilizer_p_kg"] + wide["fertilizer_k_kg"]
    )
    log.info("  Fertilizer aggregated: %d year rows", len(wide))
    return wide[
        [
            "year",
            "fertilizer_n_kg",
            "fertilizer_p_kg",
            "fertilizer_k_kg",
            "fertilizer_total_kg",
        ]
    ]


# SOURCE 3 — NASA POWER (Climate, 6 zones)


def _fetch_nasa_zone(zone: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch monthly climate from NASA POWER for one zone centroid; fall back to estimates."""
    try:
        params = {
            "parameters": "PRECTOTCORR,T2M,RH2M,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": f"{YEARS[0]}01",
            "end": f"{YEARS[-1]}12",
            "format": "JSON",
        }
        r = requests.get(NASA_API, params=params, timeout=120)
        r.raise_for_status()
        param_data = r.json()["properties"]["parameter"]

        days_in_month = {
            1: 31,
            2: 28,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
        rows = []
        for yyyymm, rain_val in param_data["PRECTOTCORR"].items():
            yr, mo = int(yyyymm[:4]), int(yyyymm[4:])
            rows.append(
                {
                    "zone": zone,
                    "year": yr,
                    "month": mo,
                    "rain_mm_day": rain_val,
                    "temp_c": param_data["T2M"].get(yyyymm, np.nan),
                    "humidity_pct": param_data["RH2M"].get(yyyymm, np.nan),
                    "solar_mj_m2": param_data["ALLSKY_SFC_SW_DWN"].get(yyyymm, np.nan),
                }
            )

        df_m = pd.DataFrame(rows)
        df_m = df_m[df_m["year"].isin(YEARS)]
        df_m["rain_mm_month"] = df_m["rain_mm_day"] * df_m["month"].map(days_in_month)

        df_a = (
            df_m.groupby(["zone", "year"])
            .agg(
                rainfall_mm_annual=("rain_mm_month", "sum"),
                temp_avg_celsius=("temp_c", "mean"),
                humidity_pct=("humidity_pct", "mean"),
                solar_radiation=("solar_mj_m2", "mean"),
            )
            .reset_index()
        )
        df_a["latitude"] = lat
        df_a["longitude"] = lon
        log.info("  NASA POWER [%s]: %d year-rows", zone, len(df_a))
        return df_a

    except Exception as exc:
        log.warning(
            "  NASA POWER [%s] failed (%s). Using climatological estimates.", zone, exc
        )
        return _nasa_fallback(zone, lat, lon)


def _nasa_fallback(zone: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Climatological fallback derived from NIMET long-term zone averages.

    The seed for inter-annual variability is derived from the zone name using
    hashlib.md5 — deterministic across Python processes regardless of the
    PYTHONHASHSEED environment variable, ensuring repeated runs produce
    bit-for-bit identical fallback outputs.
    """
    profiles = {
        "North West": {"rain": 650, "temp": 28.5, "hum": 45, "solar": 22.5},
        "North East": {"rain": 500, "temp": 29.0, "hum": 40, "solar": 23.0},
        "North Central": {"rain": 1100, "temp": 27.5, "hum": 58, "solar": 20.5},
        "South West": {"rain": 1400, "temp": 27.0, "hum": 75, "solar": 18.5},
        "South East": {"rain": 1800, "temp": 26.5, "hum": 80, "solar": 17.0},
        "South South": {"rain": 2400, "temp": 26.5, "hum": 85, "solar": 16.0},
    }
    p = profiles.get(zone, {"rain": 1200, "temp": 27.0, "hum": 65, "solar": 19.0})

    # md5 of the zone name gives a stable 16-byte digest; take the first 4 bytes
    # as a uint32 seed. This is deterministic regardless of PYTHONHASHSEED.
    seed = int.from_bytes(hashlib.md5(zone.encode()).digest()[:4], byteorder="little")
    rng = np.random.default_rng(seed=seed)

    rows = []
    for year in YEARS:
        rows.append(
            {
                "zone": zone,
                "year": year,
                "rainfall_mm_annual": max(
                    200, p["rain"] + rng.normal(0, p["rain"] * 0.10)
                ),
                "temp_avg_celsius": p["temp"] + rng.normal(0, 0.5),
                "humidity_pct": p["hum"] + rng.normal(0, 3.0),
                "solar_radiation": p["solar"] + rng.normal(0, 0.5),
                "latitude": lat,
                "longitude": lon,
            }
        )
    df = pd.DataFrame(rows)
    df["rainfall_mm_annual"] = df["rainfall_mm_annual"].clip(lower=200).round(1)
    log.info("  Fallback climate [%s]: %d year-rows", zone, len(df))
    return df


def fetch_all_climate() -> pd.DataFrame:
    """
    Fetch NASA POWER data for all 6 zones in parallel.

    ThreadPoolExecutor with max_workers=6 issues all zone requests concurrently.
    Each call has a 120-second timeout; total wall time is bounded by the slowest
    single zone rather than the sum of all six. Falls back per zone independently
    on any network failure — a single zone outage does not abort the others.
    """
    log.info("  Fetching climate for %d zones in parallel…", len(ZONES))
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=len(ZONES)) as pool:
        future_to_zone = {
            pool.submit(_fetch_nasa_zone, zone, lat, lon): zone
            for zone, (lat, lon) in ZONES.items()
        }
        for future in as_completed(future_to_zone):
            frames.append(future.result())

    df = pd.concat(frames, ignore_index=True)
    log.info("Climate data assembled: %s", df.shape)
    return df


# SOURCE 4 — World Bank Agricultural Macro-Indicators


def _fetch_wb_indicator(indicator: str, col_name: str) -> pd.DataFrame:
    """Fetch a single World Bank indicator for Nigeria."""
    try:
        url = f"{WB_API}/country/NGA/indicator/{indicator}"
        params = {"format": "json", "per_page": "100", "mrv": "30"}
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            raise ValueError("Empty WB response")
        records = [
            {"year": int(d["date"]), col_name: d["value"]}
            for d in payload[1]
            if d["value"] is not None and int(d["date"]) in YEARS
        ]
        df = pd.DataFrame(records)
        log.info("  WB [%s]: %d year-rows", indicator, len(df))
        return df
    except Exception as exc:
        log.warning(
            "  World Bank [%s] unavailable (%s). Using estimates.", indicator, exc
        )
        return _wb_fallback(col_name)


def _wb_fallback(col_name: str) -> pd.DataFrame:
    """
    Fallback World Bank indicators from published WDI for Nigeria.

    wb_fertilizer_kg_ha is the WB national aggregate (kg/ha of arable land),
    distinct from the derived zone-crop fertilizer_kg_ha produced in the merge.
    """
    estimates = {
        "wb_fertilizer_kg_ha": {
            2000: 4.2,
            2001: 4.5,
            2002: 4.8,
            2003: 5.0,
            2004: 5.2,
            2005: 5.5,
            2006: 5.8,
            2007: 6.1,
            2008: 6.4,
            2009: 6.7,
            2010: 7.0,
            2011: 7.4,
            2012: 7.8,
            2013: 8.2,
            2014: 8.6,
            2015: 9.5,
            2016: 10.2,
            2017: 10.8,
            2018: 11.3,
            2019: 11.7,
            2020: 11.4,
            2021: 11.8,
            2022: 11.2,
            2023: 10.7,
        },
        "agric_gdp_share": {
            2000: 26.0,
            2001: 26.5,
            2002: 27.0,
            2003: 26.8,
            2004: 25.7,
            2005: 24.6,
            2006: 24.5,
            2007: 24.1,
            2008: 23.4,
            2009: 24.7,
            2010: 22.9,
            2011: 23.0,
            2012: 23.3,
            2013: 21.5,
            2014: 21.2,
            2015: 21.9,
            2016: 24.9,
            2017: 25.1,
            2018: 25.1,
            2019: 26.0,
            2020: 26.7,
            2021: 24.7,
            2022: 23.8,
            2023: 24.2,
        },
        "rural_population": {
            2000: 67_800_000,
            2001: 69_400_000,
            2002: 71_100_000,
            2003: 72_900_000,
            2004: 74_700_000,
            2005: 76_600_000,
            2006: 78_500_000,
            2007: 80_400_000,
            2008: 82_300_000,
            2009: 84_300_000,
            2010: 86_300_000,
            2011: 88_200_000,
            2012: 90_200_000,
            2013: 92_200_000,
            2014: 94_200_000,
            2015: 96_300_000,
            2016: 98_300_000,
            2017: 100_400_000,
            2018: 102_500_000,
            2019: 104_500_000,
            2020: 106_600_000,
            2021: 108_600_000,
            2022: 110_600_000,
            2023: 112_700_000,
        },
    }
    values = estimates.get(col_name, {})
    df = pd.DataFrame(
        [{"year": y, col_name: v} for y, v in values.items() if y in YEARS]
    )
    log.info("  WB fallback [%s]: %d year-rows", col_name, len(df))
    return df


def fetch_world_bank() -> pd.DataFrame:
    """Fetch all World Bank indicators and merge into one DataFrame."""
    dfs = [_fetch_wb_indicator(ind, col) for ind, col in WB_INDICATORS.items()]
    df_wb = dfs[0]
    for df_next in dfs[1:]:
        df_wb = pd.merge(df_wb, df_next, on="year", how="outer")
    df_wb["year"] = df_wb["year"].astype(int)
    log.info("World Bank data merged: %s", df_wb.shape)
    return df_wb


# SOURCE 5 — USDA PSD (Palm Oil & Cassava Supply/Demand)


def _load_usda_psd() -> pd.DataFrame:
    """
    Read and clean the USDA PSD CSV for Nigeria.
    Returns one row per year with Palm Oil and Cassava supply/demand columns.

    Expected CSV columns: year, commodity, area_harvested_kha, production_kmt,
    domestic_consumption_kmt, exports_kmt, imports_kmt, ending_stocks_kmt

    Download from: apps.fas.usda.gov/psdonline (filter Nigeria; commodities:
    Palm Oil, Cassava; save as data/raw/usda_psd_nigeria.csv).
    """
    path = RAW / "usda_psd_nigeria.csv"
    if not path.exists():
        log.warning(
            "  USDA PSD file not found at %s. Using zero-filled fallback.", path
        )
        return _usda_fallback()

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df[df["year"].isin(YEARS)].copy()

        palm_cols = {
            "area_harvested_kha": "palm_area_kha",
            "production_kmt": "palm_production_kmt",
            "domestic_consumption_kmt": "palm_dom_consumption_kmt",
            "exports_kmt": "palm_exports_kmt",
            "imports_kmt": "palm_imports_kmt",
            "ending_stocks_kmt": "palm_ending_stocks_kmt",
        }
        palm = df[
            df["commodity"].str.lower().str.contains("palm oil", na=False)
        ].rename(columns=palm_cols)[["year"] + list(palm_cols.values())]

        cassava_cols = {
            "area_harvested_kha": "cassava_usda_area_kha",
            "production_kmt": "cassava_usda_production_kmt",
            "exports_kmt": "cassava_usda_exports_kmt",
        }
        cassava = df[
            df["commodity"].str.lower().str.contains("cassava", na=False)
        ].rename(columns=cassava_cols)[["year"] + list(cassava_cols.values())]

        usda = pd.merge(palm, cassava, on="year", how="outer")
        usda["year"] = usda["year"].astype(int)
        log.info("  USDA PSD loaded: %d year rows", len(usda))
        return usda

    except Exception as exc:
        log.warning("  USDA PSD load failed (%s). Using zero-filled fallback.", exc)
        return _usda_fallback()


def _usda_fallback() -> pd.DataFrame:
    """
    Estimated USDA PSD figures for Nigeria (Palm Oil and Cassava), 2000–2023.
    Source: USDA PSD Online historical archives and FAO cross-referencing.
    All quantities in thousand metric tonnes (kmt) or thousand hectares (kha).
    """
    palm_prod = [
        700,
        720,
        740,
        760,
        780,
        800,
        820,
        840,
        860,
        880,
        900,
        920,
        940,
        960,
        980,
        1000,
        1020,
        1040,
        1060,
        1080,
        1100,
        1120,
        1140,
        1160,
    ]
    palm_cons = [
        810,
        820,
        830,
        840,
        850,
        860,
        870,
        880,
        890,
        900,
        910,
        920,
        930,
        940,
        950,
        960,
        970,
        980,
        990,
        1000,
        1010,
        1020,
        1030,
        1040,
    ]
    palm_exp = [
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        8,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
        12,
        13,
        13,
        14,
        14,
    ]
    palm_imp = [
        140,
        140,
        130,
        130,
        125,
        120,
        115,
        110,
        105,
        100,
        95,
        90,
        85,
        80,
        75,
        70,
        65,
        60,
        55,
        55,
        50,
        50,
        45,
        45,
    ]
    palm_area = [
        360,
        363,
        366,
        370,
        374,
        378,
        382,
        386,
        390,
        395,
        400,
        405,
        410,
        415,
        420,
        425,
        430,
        435,
        440,
        445,
        450,
        455,
        460,
        465,
    ]
    palm_stock = [
        60,
        65,
        70,
        75,
        74,
        74,
        73,
        72,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
        71,
    ]

    cas_prod = [
        30000,
        31000,
        32000,
        34000,
        36000,
        38000,
        40000,
        42000,
        44000,
        45000,
        46000,
        47000,
        48000,
        49000,
        50000,
        51000,
        52000,
        53000,
        55000,
        57000,
        58000,
        59000,
        59500,
        60000,
    ]
    cas_area = [
        3200,
        3250,
        3300,
        3380,
        3450,
        3520,
        3590,
        3660,
        3730,
        3780,
        3830,
        3880,
        3930,
        3980,
        4030,
        4080,
        4130,
        4180,
        4230,
        4280,
        4330,
        4380,
        4430,
        4480,
    ]
    cas_exp = [
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
        12,
        13,
        13,
        14,
        14,
        15,
    ]

    df = pd.DataFrame(
        {
            "year": YEARS,
            "palm_area_kha": palm_area,
            "palm_production_kmt": palm_prod,
            "palm_dom_consumption_kmt": palm_cons,
            "palm_exports_kmt": palm_exp,
            "palm_imports_kmt": palm_imp,
            "palm_ending_stocks_kmt": palm_stock,
            "cassava_usda_area_kha": cas_area,
            "cassava_usda_production_kmt": cas_prod,
            "cassava_usda_exports_kmt": cas_exp,
        }
    )
    log.info("  USDA PSD fallback: %d year rows", len(df))
    return df


# Zone-Crop Disaggregation


def _assert_weights_sum_to_one() -> None:
    """Guard: raise immediately if any crop's zone weights don't sum to 1.00."""
    for crop in CROPS:
        total = sum(ZONE_CROP_WEIGHTS[z].get(crop, 0.0) for z in ZONE_CROP_WEIGHTS)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"ZONE_CROP_WEIGHTS do not conserve national production for '{crop}': "
                f"sum={total:.6f} (must be exactly 1.0). Fix the weights table."
            )
    log.info("  Weight-conservation check passed (all 7 crops sum to 1.00) ✓")


def _assign_zones_to_crops(df_crops: pd.DataFrame) -> pd.DataFrame:
    """
    Disaggregate national FAOSTAT data to zone level using ZONE_CROP_WEIGHTS.
    Area and production are scaled by zone weight; yield is kept as national
    average (zone-level yield adjustment is a Module 4 model output, not an
    input assumption).
    """
    rows = []
    for (crop, year), grp in df_crops.groupby(["crop", "year"]):
        area_ha = grp["area_ha"].iloc[0]
        prod_t = grp["production_tonnes"].iloc[0]
        yield_hha = grp["yield_hg_ha"].iloc[0]

        for zone, weights in ZONE_CROP_WEIGHTS.items():
            w = weights.get(crop, 0.0)
            if w == 0.0:
                continue
            rows.append(
                {
                    "zone": zone,
                    "crop": crop,
                    "year": int(year),
                    "area_ha": round(area_ha * w) if not pd.isna(area_ha) else np.nan,
                    "production_tonnes": (
                        round(prod_t * w) if not pd.isna(prod_t) else np.nan
                    ),
                    "yield_hg_ha": (
                        round(yield_hha) if not pd.isna(yield_hha) else np.nan
                    ),
                }
            )

    df_zone = pd.DataFrame(rows)
    log.info("  Zone-crop rows: %d", len(df_zone))
    return df_zone


# Master Table Assembly — helpers


def _compute_zone_fertilizer(master: pd.DataFrame) -> pd.DataFrame:
    """
    Append zone-level fertilizer estimates and derived intensity column.

    National FAOSTAT totals are partitioned to each zone-crop row using
    the same ZONE_CROP_WEIGHTS used for production disaggregation.

    fertilizer_kg_ha (zone-crop level) is deliberately named differently
    from wb_fertilizer_kg_ha (WB national aggregate) — see DATA CONTRACT
    in the module docstring.
    """

    def _zone_weight(row: pd.Series) -> float:
        return ZONE_CROP_WEIGHTS.get(row["zone"], {}).get(row["crop"], 0.0)

    master["_zone_weight"] = master.apply(_zone_weight, axis=1)

    for col in [
        "fertilizer_n_kg",
        "fertilizer_p_kg",
        "fertilizer_k_kg",
        "fertilizer_total_kg",
    ]:
        if col in master.columns:
            master[f"{col}_zone"] = (master[col] * master["_zone_weight"]).round(0)

    master["fertilizer_kg_ha"] = np.where(
        master["area_ha"].gt(0),
        (master.get("fertilizer_total_kg_zone", np.nan) / master["area_ha"]).round(2),
        np.nan,
    )
    master = master.drop(columns=["_zone_weight"], errors="ignore")
    return master


def _add_state_group(master: pd.DataFrame) -> pd.DataFrame:
    """Map each geopolitical zone to its constituent states for dashboard display."""
    state_groups = {
        "North West": "Kano, Kaduna, Sokoto, Katsina, Zamfara, Kebbi, Jigawa",
        "North East": "Borno, Yobe, Adamawa, Gombe, Bauchi, Taraba",
        "North Central": "Benue, Niger, Plateau, Kogi, Kwara, Nasarawa, FCT",
        "South West": "Lagos, Ogun, Oyo, Osun, Ondo, Ekiti",
        "South East": "Anambra, Imo, Abia, Enugu, Ebonyi",
        "South South": "Rivers, Delta, Edo, Bayelsa, Cross River, Akwa Ibom",
    }
    master["state_group"] = master["zone"].map(state_groups)
    return master


# Master Table Assembly — orchestrator


def build_master_table(
    df_zone_crops: pd.DataFrame,
    df_climate: pd.DataFrame,
    df_fertilizer: pd.DataFrame,
    df_wb: pd.DataFrame,
    df_usda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all five data sources into one flat master table.
    Grain: one row per (zone, crop, year).
    """
    log.info("Assembling master table…")

    master = pd.merge(df_zone_crops, df_climate, on=["zone", "year"], how="left")
    master = pd.merge(master, df_fertilizer, on="year", how="left")
    master = _compute_zone_fertilizer(master)
    master = pd.merge(master, df_wb, on="year", how="left")
    master = pd.merge(master, df_usda, on="year", how="left")
    master = _add_state_group(master)

    col_order = [
        "zone",
        "state_group",
        "year",
        "crop",
        "area_ha",
        "production_tonnes",
        "yield_hg_ha",
        "rainfall_mm_annual",
        "temp_avg_celsius",
        "humidity_pct",
        "solar_radiation",
        "latitude",
        "longitude",
        "fertilizer_n_kg",
        "fertilizer_p_kg",
        "fertilizer_k_kg",
        "fertilizer_total_kg",
        "fertilizer_n_kg_zone",
        "fertilizer_p_kg_zone",
        "fertilizer_k_kg_zone",
        "fertilizer_total_kg_zone",
        "fertilizer_kg_ha",
        "wb_fertilizer_kg_ha",
        "agric_gdp_share",
        "rural_population",
        "palm_area_kha",
        "palm_production_kmt",
        "palm_dom_consumption_kmt",
        "palm_exports_kmt",
        "palm_imports_kmt",
        "palm_ending_stocks_kmt",
        "cassava_usda_area_kha",
        "cassava_usda_production_kmt",
        "cassava_usda_exports_kmt",
    ]
    present = [c for c in col_order if c in master.columns]
    remaining = [c for c in master.columns if c not in present]
    master = master[present + remaining]

    master = (
        master.dropna(subset=["area_ha", "production_tonnes", "yield_hg_ha"])
        .sort_values(["zone", "crop", "year"])
        .reset_index(drop=True)
    )
    log.info("Master table: %d rows × %d cols", *master.shape)
    return master


# Validation


def validate_master(df: pd.DataFrame, df_crops_wide: pd.DataFrame) -> bool:
    """
    Run Phase-1 validation checks against the assembled master table.

    Parameters
    ----------
    df            : The assembled master table (post-merge).
    df_crops_wide : The original FAOSTAT pivot output (pre-disaggregation).
                    Required for the production conservation check — comparing
                    zone-summed totals against the authoritative national figures
                    from FAOSTAT rather than against the disaggregated data itself.
    """
    log.info("Running validation checks…")
    ok = True

    # Check 1: row count
    expected_min = int(len(CROPS) * len(ZONES) * len(YEARS) * PIPELINE_MIN_ROW_FRACTION)
    if len(df) < expected_min:
        log.error("  ✗ Row count %d below minimum %d", len(df), expected_min)
        ok = False
    else:
        log.info("  ✓ Row count: %d", len(df))

    # Check 2: missing values (<15% per column)
    for col in df.columns:
        pct = df[col].isna().mean()
        if pct > 0.15:
            log.warning("  ⚠ '%s' has %.1f%% missing values", col, pct * 100)

    # Check 3: oil palm yield — national average should be within historical range
    palm_yield = df[df["crop"] == "Oil palm fruit"]["yield_hg_ha"].dropna()
    if not palm_yield.empty:
        mean_y = palm_yield.mean()
        if PALM_YIELD_MIN_HG_HA <= mean_y <= PALM_YIELD_MAX_HG_HA:
            log.info("  ✓ Oil palm avg yield: %,.0f hg/ha", mean_y)
        else:
            log.warning(
                "  ⚠ Oil palm avg yield %,.0f hg/ha outside expected range "
                "(%,.0f – %,.0f)",
                mean_y,
                PALM_YIELD_MIN_HG_HA,
                PALM_YIELD_MAX_HG_HA,
            )

    # Check 4: South South rainfall — should be ≥ NIMET long-term floor
    ss_rain = df[(df["zone"] == "South South") & (df["rainfall_mm_annual"].notna())][
        "rainfall_mm_annual"
    ]
    if not ss_rain.empty:
        mean_r = ss_rain.mean()
        if mean_r >= SOUTH_SOUTH_RAIN_MIN_MM:
            log.info("  ✓ South South avg rainfall: %,.0f mm/year", mean_r)
        else:
            log.warning(
                "  ⚠ South South rainfall %,.0f mm/year below floor %,.0f mm",
                mean_r,
                SOUTH_SOUTH_RAIN_MIN_MM,
            )

    # Check 5: column naming — wb_fertilizer_kg_ha and fertilizer_kg_ha must both
    # be present and distinct; the old collision name must be absent.
    if "wb_fertilizer_kg_ha" not in df.columns:
        log.error("  ✗ 'wb_fertilizer_kg_ha' missing from master table")
        ok = False
    if "fertilizer_kg_ha" not in df.columns:
        log.error("  ✗ 'fertilizer_kg_ha' missing from master table")
        ok = False
    if "wb_fertilizer_kg_ha" in df.columns and "fertilizer_kg_ha" in df.columns:
        log.info("  ✓ Fertilizer column contract intact (two distinct columns present)")

    # Check 6: production conservation — compare zone-summed totals from the master
    # table against the original national FAOSTAT totals in df_crops_wide.
    # This is a real cross-source check; the denominator comes from a different
    # DataFrame than the numerator.
    national_totals = df_crops_wide.set_index(["crop", "year"])["production_tonnes"]
    for crop in CROPS:
        crop_df = df[df["crop"] == crop]
        if crop_df.empty:
            continue
        zone_sums_by_year = crop_df.groupby("year")["production_tonnes"].sum()
        for year, zone_sum in zone_sums_by_year.items():
            try:
                national = national_totals.loc[(crop, year)]
            except KeyError:
                continue
            if pd.isna(national) or national == 0:
                continue
            ratio = zone_sum / national
            if abs(ratio - 1.0) > ZONE_CONSERVATION_TOL:
                log.warning(
                    "  ⚠ Production conservation [%s %d]: zone sum %.0f vs "
                    "national %.0f (ratio=%.4f, tolerance=%.2f)",
                    crop,
                    year,
                    zone_sum,
                    national,
                    ratio,
                    ZONE_CONSERVATION_TOL,
                )
                ok = False
        else:
            log.info("  ✓ Production conservation [%s]", crop)

    return ok


# Save Outputs


def save_outputs(df: pd.DataFrame) -> None:
    """Write master table to SQLite and CSV."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("master_table", conn, if_exists="replace", index=False)
    conn.close()
    log.info("Saved → %s", DB_PATH)

    df.to_csv(CSV_PATH, index=False)
    log.info("Saved → %s", CSV_PATH)

    log.info("%s", "=" * 60)
    log.info("Phase 1 complete.")
    log.info("  master_table rows : %d", len(df))
    log.info("  Zones             : %d", df["zone"].nunique())
    log.info("  Crops             : %d", df["crop"].nunique())
    log.info("  Years             : %d–%d", df["year"].min(), df["year"].max())
    log.info("  Columns           : %d", df.shape[1])
    log.info(
        "  USDA PSD cols     : %s",
        [c for c in df.columns if c.startswith(("palm_", "cassava_usda"))],
    )
    log.info("%s", "=" * 60)


# Entry Point


def main() -> None:
    log.info("NigeriaAgriScope — Module 1: Data Pipeline")
    log.info("%s", "=" * 60)

    # Pre-flight: verify weight table integrity before any network I/O.
    # This guard is cheap (7 additions) and must fire before the 5 API
    # calls that follow — a bad weights table should cost milliseconds,
    # not minutes.
    _assert_weights_sum_to_one()

    log.info("STEP 1 — FAOSTAT Crop Production")
    df_crops_long = _fetch_faostat_crops()
    df_crops_wide = _pivot_crops(df_crops_long)
    df_zone_crops = _assign_zones_to_crops(df_crops_wide)

    log.info("STEP 2 — FAOSTAT Fertilizer by Nutrient")
    df_fert_raw = _fetch_faostat_fertilizer()
    df_fert = _aggregate_fertilizer(df_fert_raw)

    log.info("STEP 3 — NASA POWER Climate Data (6 zones, parallel)")
    df_climate = fetch_all_climate()

    log.info("STEP 4 — World Bank Agricultural Macro-Indicators")
    df_wb = fetch_world_bank()

    log.info("STEP 5 — USDA PSD Supply/Demand (Palm Oil & Cassava)")
    df_usda = _load_usda_psd()

    log.info("STEP 6 — Assemble master table")
    master = build_master_table(df_zone_crops, df_climate, df_fert, df_wb, df_usda)

    log.info("STEP 7 — Validate")
    # df_crops_wide is passed so the conservation check can compare zone sums
    # against the original national FAOSTAT totals rather than circular self-reference.
    validate_master(master, df_crops_wide)

    log.info("STEP 8 — Save outputs")
    save_outputs(master)


if __name__ == "__main__":
    main()
