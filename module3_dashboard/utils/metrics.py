"""
utils/metrics.py
================
Single source of all business logic and KPI derivations for NigeriaAgriScope.
Think of this as the DAX layer — all computed columns and aggregations live here.
Page files import from this module; they never recompute metrics inline.

⚠ DATA CONTRACT (enforced throughout):
  - fertilizer_kg_ha      = zone-crop intensity  (USE for input-efficiency analysis)
  - wb_fertilizer_kg_ha   = WB national aggregate (USE only for macro benchmarks)
  - fertilizer_total_kg   = national total BROADCAST per year — NEVER sum across rows;
                            always aggregate with DISTINCT year or use _zone variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Global yield benchmarks (FAOSTAT world-average, 2018–2022 mean) ───────────
# These are hardcoded constants — not in the DB. Used for benchmark comparisons
# across Page 1 (command centre) and Page 2 (crop performance).
GLOBAL_BENCHMARKS_HG_HA: dict[str, dict[str, float]] = {
    "Oil palm fruit": {
        "World average": 170_000,
        "Malaysia": 245_000,
        "Indonesia": 230_000,
    },
    "Cassava": {"World average": 108_000, "Thailand": 218_000},
    "Maize": {"World average": 56_000, "USA": 115_000},
    "Yam": {"World average": 102_000, "Ghana": 115_000},
    "Rice (paddy)": {"World average": 47_000, "China": 70_000},
    "Sorghum": {"World average": 14_500, "USA": 43_000},
    "Cocoa beans": {"World average": 5_100, "Ivory Coast": 6_200},
}

# Recommended fertilizer kg/ha by crop (agronomic literature stubs — M5 will refine)
RECOMMENDED_FERT_KG_HA: dict[str, float] = {
    "Oil palm fruit": 150.0,
    "Maize": 120.0,
    "Cassava": 80.0,
    "Yam": 90.0,
    "Sorghum": 60.0,
    "Rice (paddy)": 110.0,
    "Cocoa beans": 100.0,
}

# Estimated yield uplift per additional 10 kg/ha fertilizer (agronomic rule of thumb)
# Used to populate the ROI table on Page 3.
_YIELD_UPLIFT_T_PER_10KG: dict[str, float] = {
    "Oil palm fruit": 0.12,
    "Maize": 0.08,
    "Cassava": 0.06,
    "Yam": 0.07,
    "Sorghum": 0.04,
    "Rice (paddy)": 0.09,
    "Cocoa beans": 0.03,
}


# ── 1. Yield in display units ─────────────────────────────────────────────────


def compute_yield_t_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add yield_t_ha column = yield_hg_ha / 10_000.
    Converts from FAOSTAT storage unit (hg/ha) to agronomic display unit (t/ha).
    """
    df = df.copy()
    df["yield_t_ha"] = df["yield_hg_ha"] / 10_000
    return df


# ── 2. National production aggregate ─────────────────────────────────────────


def compute_national_production(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate zone rows to national total.
    Returns a DataFrame with columns: crop, year, national_production_tonnes.

    Summing production_tonnes across all 6 zones per (crop, year) recovers the
    national total because zone shares are designed to sum to 1.00 per crop
    (weight-conservation guard enforced in M1 generate_data.py).
    """
    national = (
        df.groupby(["crop", "year"], as_index=False)["production_tonnes"]
        .sum()
        .rename(columns={"production_tonnes": "national_production_tonnes"})
    )
    return national


# ── 3. Yield gap vs 90th-percentile potential ────────────────────────────────


def compute_yield_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add yield_gap_hg_ha and yield_gap_pct columns.

    Potential yield = 90th percentile of yield_hg_ha per crop across all
    zones and years. Rationale: robust to outlier years; represents a
    domestically achievable ceiling rather than a theoretical maximum.

    yield_gap_pct = (potential - actual) / potential * 100
    A value of 0 means the zone-year is at or above the 90th percentile.
    """
    df = df.copy()
    # Compute per-crop 90th-pct potential (scalar per crop)
    potential = (
        df.groupby("crop")["yield_hg_ha"].quantile(0.90).rename("potential_yield_hg_ha")
    )
    df = df.join(potential, on="crop")
    df["yield_gap_hg_ha"] = (df["potential_yield_hg_ha"] - df["yield_hg_ha"]).clip(
        lower=0
    )
    df["yield_gap_pct"] = (df["yield_gap_hg_ha"] / df["potential_yield_hg_ha"]) * 100
    return df


# ── 4. Year-over-year yield change ────────────────────────────────────────────


def compute_yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add yoy_yield_change_pct: % change in yield_hg_ha vs previous year,
    computed per (zone, crop) pair. First year of each series → NaN.

    Sort by year before shifting so gaps in year sequences don't produce
    spurious large changes. pandas pct_change handles NaN propagation correctly.
    """
    df = df.copy().sort_values(["zone", "crop", "year"])
    df["yoy_yield_change_pct"] = (
        df.groupby(["zone", "crop"])["yield_hg_ha"].pct_change() * 100
    )
    return df


# ── 5. Drought flag ───────────────────────────────────────────────────────────


def compute_drought_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drought_flag: 1 if rainfall_mm_annual < 85% of that zone's
    2000–2023 mean annual rainfall, else 0.

    Zone mean is computed across all years and all crop rows for that zone.
    Because rainfall_mm_annual is broadcast per (zone, year) — same value
    for all crops in a zone-year — the mean is computed on distinct zone-year
    pairs to avoid double-counting the same rainfall value per crop.
    """
    df = df.copy()
    # Compute zone mean from distinct zone-year pairs only
    zone_mean = (
        df.drop_duplicates(subset=["zone", "year"])
        .groupby("zone")["rainfall_mm_annual"]
        .mean()
        .rename("zone_mean_rainfall")
    )
    df = df.join(zone_mean, on="zone")
    df["drought_flag"] = (
        df["rainfall_mm_annual"] < 0.85 * df["zone_mean_rainfall"]
    ).astype(int)
    return df


# ── 6. Fertilizer efficiency ratio ───────────────────────────────────────────


def compute_fertilizer_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a year-level summary DataFrame with fert_efficiency_ratio:
    national_production_tonnes / (fertilizer_total_kg / 1000) per year.

    ⚠ fertilizer_total_kg is BROADCAST per year — the same value appears on
    every zone-crop row for that year. We must deduplicate to year level
    before dividing to avoid n×42 overcounting.

    Returns columns: year, total_production_tonnes, fert_total_kt,
                     fert_efficiency_ratio (t crop per t fertilizer)
    """
    # Step 1: national production per year (sum across all zones and crops)
    prod_yr = (
        df.groupby("year")["production_tonnes"].sum().rename("total_production_tonnes")
    )
    # Step 2: deduplicate fertilizer_total_kg to one row per year
    fert_yr = (
        df.drop_duplicates(subset="year")
        .set_index("year")["fertilizer_total_kg"]
        .rename("fertilizer_total_kg")
    )
    summary = pd.concat([prod_yr, fert_yr], axis=1).reset_index()
    summary["fert_total_kt"] = summary["fertilizer_total_kg"] / 1_000
    # Efficiency: tonnes of all crops produced per tonne of fertiliser applied
    summary["fert_efficiency_ratio"] = (
        summary["total_production_tonnes"] / summary["fert_total_kt"]
    )
    return summary


# ── 7. Nigeria vs global benchmarks ──────────────────────────────────────────


def compute_nigeria_vs_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a crop-level DataFrame with:
      nigeria_avg_yield_hg_ha  — mean yield across all zones and years
      benchmark_gap_pct        — % gap vs world average (negative = below world)

    Uses GLOBAL_BENCHMARKS_HG_HA["World average"] for each crop.
    Crops absent from the benchmarks dict are dropped.
    """
    nigeria_avg = (
        df.groupby("crop")["yield_hg_ha"]
        .mean()
        .rename("nigeria_avg_yield_hg_ha")
        .reset_index()
    )
    nigeria_avg["world_avg_hg_ha"] = nigeria_avg["crop"].map(
        {k: v["World average"] for k, v in GLOBAL_BENCHMARKS_HG_HA.items()}
    )
    nigeria_avg = nigeria_avg.dropna(subset=["world_avg_hg_ha"])
    nigeria_avg["benchmark_gap_pct"] = (
        (nigeria_avg["nigeria_avg_yield_hg_ha"] - nigeria_avg["world_avg_hg_ha"])
        / nigeria_avg["world_avg_hg_ha"]
        * 100
    )
    return nigeria_avg


# ── 8. Zone ranking for a crop ────────────────────────────────────────────────


def compute_zone_ranking(
    df: pd.DataFrame,
    crop: str,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    """
    Return zones ranked by avg yield_t_ha for the given crop and year range.

    Columns: zone, avg_yield_t_ha, rank, performance_tier
    performance_tier: Top (rank 1–2) | Mid (rank 3–4) | Bottom (rank 5–6)
    """
    mask = (
        (df["crop"] == crop)
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    )
    ranked = (
        df.loc[mask]
        .groupby("zone")["yield_hg_ha"]
        .mean()
        .div(10_000)  # → t/ha
        .rename("avg_yield_t_ha")
        .reset_index()
        .sort_values("avg_yield_t_ha", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1
    ranked["performance_tier"] = ranked["rank"].map(
        lambda r: "Top" if r <= 2 else ("Mid" if r <= 4 else "Bottom")
    )
    return ranked


# ── 9. Opportunity score ──────────────────────────────────────────────────────


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opportunity_score_norm (0–100):
      raw score = yield_gap_hg_ha × avg_area_ha (per crop)
      normalised to 0–100 min-max scale

    High score = large yield gap AND large area → highest intervention ROI.
    Requires yield_gap_hg_ha to be present; call compute_yield_gap() first.
    """
    if "yield_gap_hg_ha" not in df.columns:
        df = compute_yield_gap(df)
    df = df.copy()
    avg_area = df.groupby("crop")["area_ha"].mean().rename("avg_area_ha")
    df = df.join(avg_area, on="crop")
    df["opportunity_raw"] = df["yield_gap_hg_ha"] * df["avg_area_ha"]
    min_val, max_val = df["opportunity_raw"].min(), df["opportunity_raw"].max()
    if max_val > min_val:
        df["opportunity_score_norm"] = (
            (df["opportunity_raw"] - min_val) / (max_val - min_val) * 100
        )
    else:
        df["opportunity_score_norm"] = 0.0
    return df


# ── Composed helper: fully enriched DataFrame ────────────────────────────────


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all row-level enrichments in dependency order and return the
    fully enriched DataFrame. Called once in app.py so pages receive a
    pre-enriched df without duplicate computation.

    Enrichments applied:
      yield_t_ha, yield_gap_hg_ha/pct, yoy_yield_change_pct,
      drought_flag, zone_mean_rainfall, opportunity_score_norm
    """
    df = compute_yield_t_ha(df)
    df = compute_yield_gap(df)
    df = compute_yoy_change(df)
    df = compute_drought_flag(df)
    df = compute_opportunity_score(df)
    return df


# ── ROI recommendation table ──────────────────────────────────────────────────


def compute_roi_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ROI recommendation table used on Page 3.

    Columns: zone, crop, current_fert_kg_ha, recommended_fert_kg_ha,
             estimated_yield_uplift_t_ha, priority_tier

    priority_tier logic:
      High   — current < 50% of recommended
      Medium — current 50–80% of recommended
      Low    — current > 80% of recommended
    """
    summary = (
        df.groupby(["zone", "crop"])["fertilizer_kg_ha"]
        .mean()
        .rename("current_fert_kg_ha")
        .reset_index()
    )
    summary["recommended_fert_kg_ha"] = summary["crop"].map(RECOMMENDED_FERT_KG_HA)
    summary = summary.dropna(subset=["recommended_fert_kg_ha"])

    gap_kg = (summary["recommended_fert_kg_ha"] - summary["current_fert_kg_ha"]).clip(
        lower=0
    )
    uplift_per_10 = summary["crop"].map(_YIELD_UPLIFT_T_PER_10KG)
    summary["estimated_yield_uplift_t_ha"] = (gap_kg / 10) * uplift_per_10

    ratio = summary["current_fert_kg_ha"] / summary["recommended_fert_kg_ha"]
    summary["priority_tier"] = pd.cut(
        ratio,
        bins=[-np.inf, 0.5, 0.8, np.inf],
        labels=["High", "Medium", "Low"],
    )
    return summary.sort_values(
        ["priority_tier", "estimated_yield_uplift_t_ha"],
        ascending=[True, False],
    ).reset_index(drop=True)
