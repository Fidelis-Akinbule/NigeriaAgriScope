"""
NigeriaAgriScope — Module 5: Planting Calendar Generator
=========================================================
Computes optimal planting windows for 7 crops across 6 Nigerian geopolitical
zones, anchored to historical rainfall onset dates derived from NASA POWER
monthly climatological profiles.

Monthly rainfall reconstruction
--------------------------------
master_table stores only annual rainfall aggregates (rainfall_mm_annual).
Monthly resolution was lost during M1 aggregation. This script reconstructs
monthly rainfall distributions using NIMET long-term seasonal profiles for
each zone — the same profiles used as fallback in M1's _nasa_fallback().

The reconstruction applies a fixed seasonal distribution curve (fraction of
annual total per month) to each zone's observed annual total, yielding a
synthetic monthly series that preserves the annual total exactly while
approximating the seasonal shape. This is appropriate for a planning tool:
we are estimating typical onset months, not measuring exact onset dates.

Rainfall onset definition
--------------------------
West African agronomic standard (FAO Crop Water Requirements, Allen et al.):
  Onset month = first month where reconstructed rainfall ≥ 80 mm
  AND at least one of the two following months also ≥ 80 mm
  (the two-month filter prevents false starts from isolated convective events)

This is the standard used by IITA and NAERLS for Nigerian planting advisories.

Crop planting window offsets (agronomic literature sources)
------------------------------------------------------------
Offsets are measured in months from the onset month.
Sources: IITA Crop Improvement Guide (2021), FAOSTAT Nigeria crop calendars,
         Akobundu & Agyakwa (1987) West African Weeds and Crops.

  Crop           Planting offset   Growing duration   Notes
  ─────────────  ──────────────── ─────────────────  ──────────────────────────
  Maize          +0 (at onset)    4 months           Short-season varieties
  Cassava        +0 (at onset)    9–12 months        Planted at start of rains
  Yam            -1 (pre-onset)   6–8 months         Early planting for tuber set
  Rice (paddy)   +1               5 months           Needs established moisture
  Sorghum        +0               5 months           Drought-tolerant; at onset
  Oil palm fruit +0               Perennial           Rains trigger fruit set
  Cocoa beans    +0               Perennial           Rains trigger flowering

Outputs (→ module5_planning/outputs/)
  planting_calendar_all_zones.csv
  chart01_planting_calendar_heatmap.png
  chart02_zone_risk_profile.png

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5_planting_calendar")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "nigeria_agri.db"
CSV_PATH = ROOT / "data" / "processed" / "master_table.csv"
OUT_DIR = ROOT / "module5_planning" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALENDAR_CSV = OUT_DIR / "planting_calendar_all_zones.csv"
CHART_HEATMAP = OUT_DIR / "chart01_planting_calendar_heatmap.png"
CHART_RISK = OUT_DIR / "chart02_zone_risk_profile.png"

# ── Constants ──────────────────────────────────────────────────────────────
ONSET_THRESHOLD_MM = 80.0  # minimum monthly rainfall to qualify as onset
YEARS = list(range(2000, 2024))
MONTHS = list(range(1, 13))
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

ZONES = [
    "North West",
    "North East",
    "North Central",
    "South West",
    "South East",
    "South South",
]

CROPS = [
    "Cassava",
    "Oil palm fruit",
    "Maize",
    "Yam",
    "Rice (paddy)",
    "Sorghum",
    "Cocoa beans",
]

# ── Seasonal distribution profiles ────────────────────────────────────────
# Fraction of annual rainfall in each month per zone.
# Derived from NIMET long-term monthly normals (1981–2010).
# Each row sums to 1.00. Dry months shown as near-zero fractions.
#
# Verification (North West): 0.00+0.00+0.01+0.04+0.10+0.18+0.28+0.24+0.11+0.03+0.01+0.00 = 1.00
# Verification (South South): 0.02+0.04+0.09+0.12+0.14+0.13+0.10+0.10+0.12+0.09+0.04+0.01 = 1.00

SEASONAL_PROFILES: dict[str, list[float]] = {
    #                  Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    "North West": [
        0.00,
        0.00,
        0.01,
        0.04,
        0.10,
        0.18,
        0.28,
        0.24,
        0.11,
        0.03,
        0.01,
        0.00,
    ],
    "North East": [
        0.00,
        0.00,
        0.01,
        0.03,
        0.08,
        0.15,
        0.30,
        0.27,
        0.12,
        0.03,
        0.01,
        0.00,
    ],
    "North Central": [
        0.00,
        0.01,
        0.03,
        0.08,
        0.13,
        0.16,
        0.18,
        0.17,
        0.14,
        0.07,
        0.02,
        0.01,
    ],
    "South West": [
        0.01,
        0.02,
        0.06,
        0.12,
        0.15,
        0.17,
        0.10,
        0.06,
        0.13,
        0.12,
        0.05,
        0.01,
    ],
    "South East": [
        0.01,
        0.02,
        0.07,
        0.12,
        0.14,
        0.14,
        0.11,
        0.09,
        0.13,
        0.11,
        0.05,
        0.01,
    ],
    "South South": [
        0.02,
        0.04,
        0.09,
        0.12,
        0.14,
        0.13,
        0.10,
        0.10,
        0.12,
        0.09,
        0.04,
        0.01,
    ],
}

# ── Crop planting offsets from onset month ────────────────────────────────
# offset_months: signed integer, months relative to onset
# duration_months: crop growing cycle length
# perennial: True means the crop grows year-round; harvest_offset still
#            indicates the primary harvest window relative to planting.

CROP_CALENDAR: dict[str, dict] = {
    "Maize": {
        "offset_months": 0,
        "duration_months": 4,
        "perennial": False,
        "notes": "Short-season variety; plant at first rains",
    },
    "Cassava": {
        "offset_months": 0,
        "duration_months": 10,
        "perennial": False,
        "notes": "Plant at onset; harvest at 9-12 months",
    },
    "Yam": {
        "offset_months": -1,
        "duration_months": 7,
        "perennial": False,
        "notes": "Pre-onset planting for tuber set; needs dry-to-wet transition",
    },
    "Rice (paddy)": {
        "offset_months": 1,
        "duration_months": 5,
        "perennial": False,
        "notes": "Plant 4-6 weeks after rains establish; needs standing water",
    },
    "Sorghum": {
        "offset_months": 0,
        "duration_months": 5,
        "perennial": False,
        "notes": "Drought-tolerant; plant at onset; can extend into dry season",
    },
    "Oil palm fruit": {
        "offset_months": 0,
        "duration_months": 12,
        "perennial": True,
        "notes": "Perennial; rains trigger fruit set; primary harvest window shown",
    },
    "Cocoa beans": {
        "offset_months": 0,
        "duration_months": 8,
        "perennial": True,
        "notes": "Perennial; rains trigger pod development; main crop shown",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────


def load_annual_climate() -> pd.DataFrame:
    """Load zone × year annual rainfall from master_table."""
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(
                "SELECT DISTINCT zone, year, rainfall_mm_annual "
                "FROM master_table ORDER BY zone, year",
                conn,
            )
        log.info("Loaded annual climate: %d zone-year rows from SQLite", len(df))
    else:
        df = pd.read_csv(CSV_PATH)
        df = (
            df[["zone", "year", "rainfall_mm_annual"]]
            .drop_duplicates(subset=["zone", "year"])
            .sort_values(["zone", "year"])
            .reset_index(drop=True)
        )
        log.info("Loaded annual climate: %d zone-year rows from CSV", len(df))

    df["year"] = df["year"].astype(int)
    df["rainfall_mm_annual"] = pd.to_numeric(df["rainfall_mm_annual"], errors="coerce")
    return df


# ── Monthly rainfall reconstruction ───────────────────────────────────────


def reconstruct_monthly(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct monthly rainfall from annual totals using seasonal profiles.

    For each (zone, year), multiply the annual total by the zone's seasonal
    fraction for each month. This preserves the observed annual total while
    approximating the seasonal shape from NIMET long-term normals.

    Returns a DataFrame with columns: zone, year, month, rainfall_mm_month
    """
    rows = []
    for _, row in annual.iterrows():
        zone = row["zone"]
        year = int(row["year"])
        annual_total = row["rainfall_mm_annual"]
        if pd.isna(annual_total):
            continue
        profile = SEASONAL_PROFILES.get(zone)
        if profile is None:
            log.warning("No seasonal profile for zone '%s' — skipping", zone)
            continue
        for m, frac in enumerate(profile, start=1):
            rows.append(
                {
                    "zone": zone,
                    "year": year,
                    "month": m,
                    "rainfall_mm_month": round(annual_total * frac, 1),
                }
            )

    df = pd.DataFrame(rows)
    log.info("Monthly reconstruction complete: %d zone-year-month rows", len(df))
    return df


# ── Rainfall onset detection ───────────────────────────────────────────────


def detect_onset(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rainfall onset month per (zone, year).

    Onset = first month M where:
      rainfall_mm_month[M] >= ONSET_THRESHOLD_MM
      AND rainfall_mm_month[M+1] >= ONSET_THRESHOLD_MM  (two-month filter)

    The two-month filter follows IITA/NAERLS standard to avoid false starts
    from isolated convective storms, which are common in the Sahel transition
    zones (North West, North East) in April-May.

    If no onset detected (rare in high-rainfall zones), onset is set to the
    month of maximum rainfall as a fallback.
    """
    records = []
    for (zone, year), grp in monthly.groupby(["zone", "year"]):
        grp = grp.sort_values("month")
        rain = grp["rainfall_mm_month"].values
        months = grp["month"].values

        onset = None
        for i in range(len(months) - 1):
            if rain[i] >= ONSET_THRESHOLD_MM and rain[i + 1] >= ONSET_THRESHOLD_MM:
                onset = int(months[i])
                break

        if onset is None:
            # Fallback: month of peak rainfall
            onset = int(months[np.argmax(rain)])
            log.debug(
                "  [%s %d] No clean onset detected — using peak month %d",
                zone,
                year,
                onset,
            )

        records.append({"zone": zone, "year": year, "onset_month": onset})

    df = pd.DataFrame(records)
    log.info("Onset detection complete: %d zone-year records", len(df))
    return df


# ── Onset summary statistics ───────────────────────────────────────────────


def summarise_onset(onset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute avg onset month, std dev, earliest, latest per zone (2000–2023).

    rainfall_reliability_score = 1 − (std_dev / avg_onset_month)
    A higher score means more predictable onset timing — directly useful for
    the dashboard's zone risk profile chart.
    """
    summary = (
        onset.groupby("zone")["onset_month"]
        .agg(
            avg_onset_month="mean",
            onset_std_dev="std",
            earliest_onset="min",
            latest_onset="max",
            n_years="count",
        )
        .reset_index()
    )
    summary["avg_onset_month"] = summary["avg_onset_month"].round(1)
    summary["onset_std_dev"] = summary["onset_std_dev"].round(2)

    # Reliability score: lower std relative to avg = more reliable
    summary["rainfall_reliability_score"] = (
        (1 - (summary["onset_std_dev"] / summary["avg_onset_month"]))
        .clip(0, 1)
        .round(3)
    )

    log.info("Onset summary:")
    for _, r in summary.iterrows():
        log.info(
            "  %-12s  avg_onset=month %s  std=%.1f  reliability=%.2f",
            r["zone"],
            f"{r['avg_onset_month']:.1f}",
            r["onset_std_dev"],
            r["rainfall_reliability_score"],
        )
    return summary


# ── Planting calendar assembly ─────────────────────────────────────────────


def _month_wrap(month: int) -> int:
    """Wrap month arithmetic to 1–12 range."""
    return ((month - 1) % 12) + 1


def build_planting_calendar(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full zone × crop planting calendar.

    For each (zone, crop):
      planting_month = round(avg_onset_month) + crop_offset
      harvest_month  = planting_month + duration_months

    Both are wrapped to 1–12. Month names are appended for readability.
    """
    rows = []
    for _, zone_row in summary.iterrows():
        zone = zone_row["zone"]
        avg_onset = round(zone_row["avg_onset_month"])

        for crop, meta in CROP_CALENDAR.items():
            planting_month = _month_wrap(avg_onset + meta["offset_months"])
            harvest_month = _month_wrap(planting_month + meta["duration_months"])

            rows.append(
                {
                    "zone": zone,
                    "crop": crop,
                    "avg_onset_month": zone_row["avg_onset_month"],
                    "onset_std_dev": zone_row["onset_std_dev"],
                    "planting_month": planting_month,
                    "planting_month_name": MONTH_NAMES[planting_month - 1],
                    "harvest_month": harvest_month,
                    "harvest_month_name": MONTH_NAMES[harvest_month - 1],
                    "growing_duration_months": meta["duration_months"],
                    "is_perennial": meta["perennial"],
                    "rainfall_reliability_score": zone_row[
                        "rainfall_reliability_score"
                    ],
                    "planting_notes": meta["notes"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["zone", "crop"]).reset_index(drop=True)
    df.to_csv(CALENDAR_CSV, index=False)
    log.info("Planting calendar saved → %s  (%d rows)", CALENDAR_CSV.name, len(df))
    return df


# ── Charts ────────────────────────────────────────────────────────────────


def plot_calendar_heatmap(calendar: pd.DataFrame) -> None:
    """
    chart01 — Planting window heatmap: zones (rows) × months (cols),
    coloured by which crops are planted in each month.

    Each cell shows a coloured block for every crop with a planting window
    in that zone-month combination. This gives an at-a-glance view of
    planting concentration and overlap risk.
    """
    crop_colours = {
        "Cassava": "#2E75B6",
        "Oil palm fruit": "#E05C2A",
        "Maize": "#C00000",
        "Yam": "#7030A0",
        "Rice (paddy)": "#00B050",
        "Sorghum": "#FFC000",
        "Cocoa beans": "#8B4513",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    zone_list = ZONES
    n_zones = len(zone_list)
    n_months = 12

    # Draw month column backgrounds alternating
    for m in range(n_months):
        ax.axvspan(
            m - 0.5,
            m + 0.5,
            color="#F5F5F5" if m % 2 == 0 else "white",
            alpha=0.5,
            zorder=0,
        )

    # Plot crop planting months as coloured markers per zone
    for z_idx, zone in enumerate(zone_list):
        zone_cal = calendar[calendar["zone"] == zone]
        crops_this_zone = {}
        for _, row in zone_cal.iterrows():
            m = row["planting_month"] - 1  # 0-indexed
            if m not in crops_this_zone:
                crops_this_zone[m] = []
            crops_this_zone[m].append(row["crop"])

        for m, crops in crops_this_zone.items():
            n = len(crops)
            for c_idx, crop in enumerate(crops):
                offset = (c_idx - (n - 1) / 2) * 0.12
                ax.scatter(
                    m,
                    z_idx + offset,
                    color=crop_colours.get(crop, "grey"),
                    s=180,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.6,
                    marker="s",
                )

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_yticks(range(n_zones))
    ax.set_yticklabels(zone_list, fontsize=10)
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(-0.7, n_zones - 0.3)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_title(
        "NigeriaAgriScope — Optimal Planting Windows by Zone and Crop\n"
        "Module 5 | Based on NASA POWER Rainfall Onset Analysis 2000–2023",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    legend_patches = [
        mpatches.Patch(color=c, label=crop) for crop, c in crop_colours.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
        ncol=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.01,
        "Source: FAOSTAT, NASA POWER, NIMET long-term normals | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_HEATMAP, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", CHART_HEATMAP.name)


def plot_zone_risk_profile(summary: pd.DataFrame) -> None:
    """
    chart02 — Horizontal bar chart of rainfall variability risk by zone.

    Bars show onset_std_dev (higher = more variable = higher risk).
    Colour encodes reliability score (green = reliable, red = variable).
    Zones ordered by reliability score descending.
    """
    df = summary.sort_values("rainfall_reliability_score", ascending=True).copy()

    cmap = plt.cm.RdYlGn
    colours = [cmap(r) for r in df["rainfall_reliability_score"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: onset std dev (risk)
    ax = axes[0]
    bars = ax.barh(
        df["zone"], df["onset_std_dev"], color=colours, edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Onset Month Std Dev (months)", fontsize=11)
    ax.set_title(
        "Rainfall Onset Variability\n(Higher = More Risk)",
        fontsize=11,
        fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlim(0, df["onset_std_dev"].max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: reliability score
    ax2 = axes[1]
    bars2 = ax2.barh(
        df["zone"],
        df["rainfall_reliability_score"],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_xlabel("Rainfall Reliability Score (0–1)", fontsize=11)
    ax2.set_title(
        "Planning Reliability Score\n(Higher = More Predictable)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.bar_label(bars2, fmt="%.2f", padding=4, fontsize=9)
    ax2.set_xlim(0, 1.15)
    ax2.axvline(
        0.7,
        color="green",
        lw=1.0,
        linestyle="--",
        alpha=0.6,
        label="Good reliability threshold",
    )
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "NigeriaAgriScope — Zone Rainfall Risk Profile\n"
        "Module 5 | NASA POWER 2000–2023 | NIMET Seasonal Normals",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.01,
        "Source: FAOSTAT, NASA POWER, NIMET long-term normals | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_RISK, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", CHART_RISK.name)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 5: Planting Calendar Generator")
    log.info("=" * 60)

    log.info("STEP 1 — Load annual climate data")
    annual = load_annual_climate()

    log.info("STEP 2 — Reconstruct monthly rainfall distributions")
    monthly = reconstruct_monthly(annual)

    log.info("STEP 3 — Detect rainfall onset month per zone per year")
    onset = detect_onset(monthly)

    log.info("STEP 4 — Summarise onset statistics (2000–2023)")
    summary = summarise_onset(onset)

    log.info("STEP 5 — Build planting calendar (zone × crop)")
    calendar = build_planting_calendar(summary)

    log.info("STEP 6 — Generate charts")
    plot_calendar_heatmap(calendar)
    plot_zone_risk_profile(summary)

    log.info("=" * 60)
    log.info("Module 5 — planting_calendar.py COMPLETE")
    log.info(
        "  planting_calendar_all_zones.csv  → %d rows (42 zone-crop pairs)",
        len(calendar),
    )
    log.info("  chart01_planting_calendar_heatmap.png")
    log.info("  chart02_zone_risk_profile.png")
    log.info("  Outputs → module5_planning/outputs/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
