"""
NigeriaAgriScope — Module 5: Input Requirements Calculator
===========================================================
Generates an enhanced input requirements reference table by combining M4's
fertilizer optimisation outputs with agronomic standards for seed/seedling
rates, labour requirements, and agrochemical needs per crop-zone-hectare.

Also computes per-hectare production cost estimates in Nigerian Naira using
2024 NBS commodity price benchmarks.

Design decisions
----------------
- M4's input_requirements_reference.csv provides the fertilizer baseline
  (recommended_min/max_kg_ha, PDR). This script reads that file and ADDS
  columns — it never overwrites M4's data.
- Seed/seedling, labour, and agrochemical constants are sourced from:
    * IITA Crop Production Guide for Smallholder Farmers in Nigeria (2021)
    * FAO Fertilizer and Plant Nutrition Bulletin No. 16 (Sub-Saharan Africa)
    * NAERLS/ABU Zaria Farm Management Guide (2022)
- Cost benchmarks are from NBS Agricultural Commodity Price Survey Q3 2024.
  All prices are in Nigerian Naira (₦). Price year is recorded in the output.
- Zone adjustments: input requirements are broadly uniform across zones for
  most crops. Where zone ecology causes meaningful divergence (e.g. rice in
  South South vs North West needs more water management labour), a zone_factor
  is applied to the base labour estimate.

Output
------
  module5_planning/outputs/input_requirements_enhanced.csv
  module5_planning/outputs/chart03_input_cost_breakdown.png

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5_input_calculator")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
M4_REF_CSV = ROOT / "module4_models" / "outputs" / "input_requirements_reference.csv"
OUT_DIR = ROOT / "module5_planning" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENHANCED_CSV = OUT_DIR / "input_requirements_enhanced.csv"
CHART_COST = OUT_DIR / "chart03_input_cost_breakdown.png"

# ── Agronomic reference tables ─────────────────────────────────────────────
#
# All quantities are per HECTARE unless noted.
#
# SEED / PLANTING MATERIAL
# Source: IITA Crop Production Guide (2021), Table 3.1–3.7
#
# seed_kg_ha       : seed weight in kg/ha (for grain crops)
# seedlings_ha     : number of seedlings/cuttings per ha (for vegetative crops)
# seed_unit        : display unit for the UI
# seed_cost_per_unit_naira : cost per kg seed or per 100 seedlings (2024 NBS)

SEED_REFERENCE: dict[str, dict] = {
    "Maize": {
        "seed_kg_ha": 25.0,
        "seedlings_ha": 0,
        "seed_unit": "kg",
        "seed_cost_per_unit_naira": 1_800,  # ₦1,800/kg certified hybrid seed
    },
    "Cassava": {
        "seed_kg_ha": 0,
        "seedlings_ha": 10_000,  # stem cuttings per ha
        "seed_unit": "cuttings (bundles)",
        "seed_cost_per_unit_naira": 8_000,  # ₦8,000 per bundle of ~500 cuttings
    },
    "Yam": {
        "seed_kg_ha": 0,
        "seedlings_ha": 5_000,  # seed yam setts per ha
        "seed_unit": "setts (kg)",
        "seed_cost_per_unit_naira": 650,  # ₦650/kg seed yam (2024)
    },
    "Rice (paddy)": {
        "seed_kg_ha": 60.0,
        "seedlings_ha": 0,
        "seed_unit": "kg",
        "seed_cost_per_unit_naira": 900,  # ₦900/kg certified paddy seed
    },
    "Sorghum": {
        "seed_kg_ha": 10.0,
        "seedlings_ha": 0,
        "seed_unit": "kg",
        "seed_cost_per_unit_naira": 750,  # ₦750/kg improved variety
    },
    "Oil palm fruit": {
        "seed_kg_ha": 0,
        "seedlings_ha": 143,  # palms/ha at 8×8m spacing
        "seed_unit": "seedlings",
        "seed_cost_per_unit_naira": 2_500,  # ₦2,500/tenera seedling (NIFOR)
    },
    "Cocoa beans": {
        "seed_kg_ha": 0,
        "seedlings_ha": 1_111,  # trees/ha at 3×3m spacing
        "seed_unit": "seedlings",
        "seed_cost_per_unit_naira": 350,  # ₦350/seedling (hybrid variety)
    },
}

# LABOUR REQUIREMENTS
# Source: NAERLS/ABU Farm Management Guide (2022), person-days per ha per season
# Base figures; zone_factors applied per crop-zone below.

LABOUR_BASE: dict[str, dict] = {
    "Maize": {
        "land_prep_pd_ha": 8,
        "planting_pd_ha": 5,
        "weeding_pd_ha": 12,  # 2 weeding rounds
        "fertilizing_pd_ha": 4,
        "harvesting_pd_ha": 10,
        "total_pd_ha": 39,
        "daily_wage_naira": 3_500,
    },
    "Cassava": {
        "land_prep_pd_ha": 10,
        "planting_pd_ha": 8,
        "weeding_pd_ha": 15,
        "fertilizing_pd_ha": 4,
        "harvesting_pd_ha": 20,  # labour-intensive root harvest
        "total_pd_ha": 57,
        "daily_wage_naira": 3_500,
    },
    "Yam": {
        "land_prep_pd_ha": 15,  # mound construction
        "planting_pd_ha": 10,
        "weeding_pd_ha": 14,
        "fertilizing_pd_ha": 4,
        "harvesting_pd_ha": 18,
        "total_pd_ha": 61,
        "daily_wage_naira": 3_500,
    },
    "Rice (paddy)": {
        "land_prep_pd_ha": 12,  # includes puddling for lowland rice
        "planting_pd_ha": 8,
        "weeding_pd_ha": 16,
        "fertilizing_pd_ha": 5,
        "harvesting_pd_ha": 15,
        "total_pd_ha": 56,
        "daily_wage_naira": 3_500,
    },
    "Sorghum": {
        "land_prep_pd_ha": 8,
        "planting_pd_ha": 4,
        "weeding_pd_ha": 10,
        "fertilizing_pd_ha": 3,
        "harvesting_pd_ha": 10,
        "total_pd_ha": 35,
        "daily_wage_naira": 3_500,
    },
    "Oil palm fruit": {
        "land_prep_pd_ha": 20,  # land clearing for new planting
        "planting_pd_ha": 8,
        "weeding_pd_ha": 12,
        "fertilizing_pd_ha": 5,
        "harvesting_pd_ha": 25,  # bunch cutting is strenuous
        "total_pd_ha": 70,
        "daily_wage_naira": 4_000,  # skilled harvesting commands premium
    },
    "Cocoa beans": {
        "land_prep_pd_ha": 18,
        "planting_pd_ha": 10,
        "weeding_pd_ha": 14,
        "fertilizing_pd_ha": 4,
        "harvesting_pd_ha": 20,  # pod breaking + fermentation
        "total_pd_ha": 66,
        "daily_wage_naira": 4_000,
    },
}

# Zone labour adjustment factors
# Captures ecological differences: rice in South South needs more water
# management; yam in North Central needs more mound construction work.
# Factor applied to total_pd_ha. Base = 1.0.
# Source: expert adjustment from NAERLS zonal reports.

ZONE_LABOUR_FACTORS: dict[str, dict[str, float]] = {
    "North West": {"Rice (paddy)": 1.15, "Oil palm fruit": 1.10},
    "North East": {"Rice (paddy)": 1.15, "Sorghum": 0.92},
    "North Central": {"Yam": 1.10, "Rice (paddy)": 1.05},
    "South West": {"Cocoa beans": 1.05, "Oil palm fruit": 1.05},
    "South East": {"Yam": 1.08, "Cassava": 1.05},
    "South South": {"Rice (paddy)": 1.20, "Oil palm fruit": 1.08},
}

# AGROCHEMICAL REQUIREMENTS
# Source: IITA Integrated Pest Management Guide Nigeria (2020)
# herbicide_l_ha   : litres of herbicide per ha per season
# pesticide_kg_ha  : kg of pesticide a.i. per ha
# fungicide_kg_ha  : kg of fungicide per ha (cocoa, palm)

AGROCHEMICAL_BASE: dict[str, dict] = {
    "Maize": {
        "herbicide_l_ha": 3.0,
        "pesticide_kg_ha": 0.5,
        "fungicide_kg_ha": 0.0,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 0,
    },
    "Cassava": {
        "herbicide_l_ha": 2.5,
        "pesticide_kg_ha": 0.3,
        "fungicide_kg_ha": 0.0,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 0,
    },
    "Yam": {
        "herbicide_l_ha": 3.0,
        "pesticide_kg_ha": 0.8,
        "fungicide_kg_ha": 0.5,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 4_800,
    },
    "Rice (paddy)": {
        "herbicide_l_ha": 4.0,
        "pesticide_kg_ha": 0.6,
        "fungicide_kg_ha": 0.3,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 4_800,
    },
    "Sorghum": {
        "herbicide_l_ha": 2.0,
        "pesticide_kg_ha": 0.4,
        "fungicide_kg_ha": 0.0,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 0,
    },
    "Oil palm fruit": {
        "herbicide_l_ha": 4.0,
        "pesticide_kg_ha": 0.5,
        "fungicide_kg_ha": 0.5,
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 4_800,
    },
    "Cocoa beans": {
        "herbicide_l_ha": 2.0,
        "pesticide_kg_ha": 0.4,
        "fungicide_kg_ha": 1.2,  # black pod disease management
        "herbicide_cost_per_l_naira": 3_200,
        "pesticide_cost_per_kg_naira": 5_500,
        "fungicide_cost_per_kg_naira": 4_800,
    },
}

# FERTILIZER COST BENCHMARK
# Source: NBS Fertilizer Price Survey Q3 2024
# NPK 15:15:15 blended bag (50kg) — most common in Nigeria
FERTILIZER_COST_PER_KG_NAIRA = 820  # ₦820/kg (₦41,000/50kg bag)
UREA_COST_PER_KG_NAIRA = 750  # ₦750/kg for N-top dressing
PRICE_BENCHMARK_YEAR = 2024


# ── Data loading ──────────────────────────────────────────────────────────


def load_m4_reference() -> pd.DataFrame:
    """Load M4's input_requirements_reference.csv."""
    if not M4_REF_CSV.exists():
        raise FileNotFoundError(
            f"M4 reference CSV not found at {M4_REF_CSV}. "
            "Run module4_models/input_optimizer.py first."
        )
    df = pd.read_csv(M4_REF_CSV)
    log.info(
        "Loaded M4 reference: %d rows × %d cols from %s",
        len(df),
        df.shape[1],
        M4_REF_CSV.name,
    )
    return df


# ── Seed cost computation ─────────────────────────────────────────────────


def compute_seed_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Append seed/planting material requirements and cost per ha."""
    seed_total_cost = []
    seed_qty = []
    seed_units = []

    for _, row in df.iterrows():
        crop = row["crop"]
        ref = SEED_REFERENCE.get(crop, {})

        if ref.get("seed_kg_ha", 0) > 0:
            qty = ref["seed_kg_ha"]
            unit = ref["seed_unit"]
            cost = qty * ref["seed_cost_per_unit_naira"]
        elif ref.get("seedlings_ha", 0) > 0:
            qty = ref["seedlings_ha"]
            unit = ref["seed_unit"]
            # Seedling costs vary: cuttings priced per bundle, palms per seedling
            if crop == "Cassava":
                # 10,000 cuttings ≈ 20 bundles of 500
                cost = (qty / 500) * ref["seed_cost_per_unit_naira"]
            elif crop == "Yam":
                # seed yam quoted per kg; avg sett = 150g
                cost = (qty * 0.15) * ref["seed_cost_per_unit_naira"]
            else:
                cost = qty * ref["seed_cost_per_unit_naira"]
        else:
            qty = 0
            unit = "N/A"
            cost = 0

        seed_total_cost.append(round(cost))
        seed_qty.append(qty)
        seed_units.append(unit)

    df["seed_qty_per_ha"] = seed_qty
    df["seed_unit"] = seed_units
    df["seed_cost_naira_per_ha"] = seed_total_cost
    return df


# ── Labour cost computation ───────────────────────────────────────────────


def compute_labour_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Append labour requirements (person-days/ha) and cost per ha."""
    labour_pd = []
    labour_cost = []

    for _, row in df.iterrows():
        crop = row["crop"]
        zone = row["zone"]
        base = LABOUR_BASE.get(crop, {})

        total_pd = base.get("total_pd_ha", 40)
        zone_factor = ZONE_LABOUR_FACTORS.get(zone, {}).get(crop, 1.0)
        adj_pd = round(total_pd * zone_factor, 1)

        wage = base.get("daily_wage_naira", 3_500)
        cost = round(adj_pd * wage)

        labour_pd.append(adj_pd)
        labour_cost.append(cost)

    df["labour_person_days_per_ha"] = labour_pd
    df["labour_cost_naira_per_ha"] = labour_cost
    return df


# ── Agrochemical cost computation ──────────────────────────────────────────


def compute_agrochemical_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Append agrochemical requirements and cost per ha."""
    agro_cost = []
    herb_l = []
    pest_kg = []
    fung_kg = []

    for _, row in df.iterrows():
        crop = row["crop"]
        ref = AGROCHEMICAL_BASE.get(crop, {})

        h_l = ref.get("herbicide_l_ha", 0)
        p_kg = ref.get("pesticide_kg_ha", 0)
        f_kg = ref.get("fungicide_kg_ha", 0)

        cost = (
            h_l * ref.get("herbicide_cost_per_l_naira", 0)
            + p_kg * ref.get("pesticide_cost_per_kg_naira", 0)
            + f_kg * ref.get("fungicide_cost_per_kg_naira", 0)
        )

        herb_l.append(h_l)
        pest_kg.append(p_kg)
        fung_kg.append(f_kg)
        agro_cost.append(round(cost))

    df["herbicide_l_per_ha"] = herb_l
    df["pesticide_kg_per_ha"] = pest_kg
    df["fungicide_kg_per_ha"] = fung_kg
    df["agrochemical_cost_naira_per_ha"] = agro_cost
    return df


# ── Fertilizer cost computation ────────────────────────────────────────────


def compute_fertilizer_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fertilizer cost using M4's recommended_max_kg_ha and 2024 NBS price.

    Uses recommended_max as the planning quantity (upper end of optimal range).
    NPK blend cost applied to full quantity; urea top-dress accounted for
    N-intensive crops (maize, rice) at 20% of total N requirement as urea.
    """
    fert_cost = []
    for _, row in df.iterrows():
        rec_max = row.get("recommended_max_kg_ha", 0) or 0
        crop = row["crop"]

        # Base NPK cost
        base_cost = rec_max * FERTILIZER_COST_PER_KG_NAIRA

        # N top-dress supplement for maize and rice (20% of total N as urea)
        if crop in ("Maize", "Rice (paddy)"):
            urea_supplement_kg = rec_max * 0.20
            base_cost += urea_supplement_kg * UREA_COST_PER_KG_NAIRA

        fert_cost.append(round(base_cost))

    df["fertilizer_cost_naira_per_ha"] = fert_cost
    df["fertilizer_price_benchmark_year"] = PRICE_BENCHMARK_YEAR
    return df


# ── Total cost and summary ─────────────────────────────────────────────────


def compute_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all cost components into total_input_cost_naira_per_ha."""
    df["total_input_cost_naira_per_ha"] = (
        df["seed_cost_naira_per_ha"]
        + df["labour_cost_naira_per_ha"]
        + df["agrochemical_cost_naira_per_ha"]
        + df["fertilizer_cost_naira_per_ha"]
    )
    # Cost in USD at 2024 CBN reference rate (₦1,580/USD)
    df["total_input_cost_usd_per_ha"] = (
        df["total_input_cost_naira_per_ha"] / 1_580
    ).round(2)
    return df


# ── Chart ─────────────────────────────────────────────────────────────────


def plot_cost_breakdown(df: pd.DataFrame) -> None:
    """
    chart03 — Stacked horizontal bar chart of input cost breakdown per crop.

    Shows national average (mean across zones) cost split into:
    fertilizer / labour / seed / agrochemical.
    Ordered by total cost descending.
    """
    cost_cols = {
        "Fertilizer": "fertilizer_cost_naira_per_ha",
        "Labour": "labour_cost_naira_per_ha",
        "Seed/Planting": "seed_cost_naira_per_ha",
        "Agrochemicals": "agrochemical_cost_naira_per_ha",
    }
    colours = {
        "Fertilizer": "#2E75B6",
        "Labour": "#E05C2A",
        "Seed/Planting": "#70AD47",
        "Agrochemicals": "#FFC000",
    }

    # Aggregate to crop level (mean across zones)
    agg = df.groupby("crop")[list(cost_cols.values())].mean().reset_index()
    agg["total"] = agg[list(cost_cols.values())].sum(axis=1)
    agg = agg.sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    left = np.zeros(len(agg))

    for label, col in cost_cols.items():
        vals = agg[col].values / 1_000  # convert to ₦000
        ax.barh(
            agg["crop"],
            vals,
            left=left / 1_000,
            color=colours[label],
            label=label,
            edgecolor="white",
            linewidth=0.4,
        )
        left += agg[col].values

    # Annotate total cost
    for i, (_, row) in enumerate(agg.iterrows()):
        total_k = row["total"] / 1_000
        ax.text(
            total_k / 1_000 + 2,
            i,
            f"₦{total_k:,.0f}k",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Input Cost (₦ thousands per hectare)", fontsize=11)
    ax.set_title(
        "NigeriaAgriScope — Crop Input Cost Breakdown per Hectare\n"
        f"NBS 2024 Price Benchmarks | Zonal Average | Module 5",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, agg["total"].max() / 1_000 * 1.18)
    fig.text(
        0.99,
        0.01,
        "Source: IITA (2021), NAERLS (2022), NBS Q3 2024 | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_COST, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", CHART_COST.name)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 5: Input Requirements Calculator")
    log.info("=" * 60)

    log.info("STEP 1 — Load M4 input requirements reference")
    df = load_m4_reference()

    log.info("STEP 2 — Compute seed/planting material costs")
    df = compute_seed_costs(df)

    log.info("STEP 3 — Compute labour requirements and costs")
    df = compute_labour_costs(df)

    log.info("STEP 4 — Compute agrochemical costs")
    df = compute_agrochemical_costs(df)

    log.info("STEP 5 — Compute fertilizer costs (NBS 2024 benchmarks)")
    df = compute_fertilizer_costs(df)

    log.info("STEP 6 — Compute total input cost per ha")
    df = compute_total_cost(df)

    log.info("STEP 7 — Save enhanced reference CSV")
    df.to_csv(ENHANCED_CSV, index=False)
    log.info(
        "Enhanced CSV saved → %s  (%d rows × %d cols)",
        ENHANCED_CSV.name,
        len(df),
        df.shape[1],
    )

    log.info("STEP 8 — Generate cost breakdown chart")
    plot_cost_breakdown(df)

    # Summary table
    log.info("=" * 60)
    log.info("COST SUMMARY (national average across zones):")
    summary = (
        df.groupby("crop")["total_input_cost_naira_per_ha"]
        .mean()
        .sort_values(ascending=False)
    )
    for crop, cost in summary.items():
        log.info("  %-20s  ₦%s / ha", crop, f"{cost:,.0f}")

    log.info("=" * 60)
    log.info("Module 5 — input_calculator.py COMPLETE")
    log.info("  input_requirements_enhanced.csv  → %d rows", len(df))
    log.info("  chart03_input_cost_breakdown.png")
    log.info("  Outputs → module5_planning/outputs/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
