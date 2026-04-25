"""
NigeriaAgriScope — Module 5: Operations Schedule Generator
===========================================================
Generates a week-by-week agronomic operations schedule for all 42 zone-crop
pairs (7 crops × 6 zones), anchored to planting month outputs from
planting_calendar.py.

Schedule design
---------------
Each crop has a phase-based schedule template defined as week offsets from
the planting date (week 0). The template is anchored to the zone-specific
planting month from planting_calendar_all_zones.csv, converting week offsets
to approximate calendar months for planning purposes.

Week-to-month conversion: week_offset // 4 + planting_month (wrapped to 1–12).
This is a planning approximation — the schedule is not a precise agronomic
prescription but a structural planning tool for the M6 Power BI export and
the M3 Page 5 operations widget.

Phase taxonomy
--------------
  PRE_PLANTING   : land preparation, input procurement, nursery raising
  PLANTING       : sowing, transplanting, staking (yam)
  EARLY_GROWTH   : establishment, gap filling, first weeding
  VEGETATIVE     : active growth, fertilizer top-dress, pest scouting
  REPRODUCTIVE   : flowering/tuber bulking, water-critical period
  PRE_HARVEST    : maturity assessment, withdrawal of irrigation
  HARVEST        : harvesting, threshing, bunch cutting
  POST_HARVEST   : drying, storage preparation, field cleanup

Sources for week schedules
--------------------------
  IITA Crop Production Guide for Smallholder Farmers in Nigeria (2021)
  FAOSTAT Nigeria crop calendars
  Akobundu & Agyakwa (1987) A Handbook of West African Weeds
  NAERLS/ABU Zaria Crop Management Bulletins (2022)

Outputs (→ module5_planning/outputs/)
  operations_schedule_all_zones.csv
  operations_schedule_summary.csv   (pivot: zone × crop → key activity months)

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5_operations_schedule")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CALENDAR_CSV = ROOT / "module5_planning" / "outputs" / "planting_calendar_all_zones.csv"
OUT_DIR = ROOT / "module5_planning" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULE_CSV = OUT_DIR / "operations_schedule_all_zones.csv"
SUMMARY_CSV = OUT_DIR / "operations_schedule_summary.csv"

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


# ── Schedule templates ─────────────────────────────────────────────────────
# Each entry: (phase, activity, start_week_offset, end_week_offset, notes)
# Week 0 = planting week. Negative weeks = pre-planting.
# Weeks are relative to the planting date derived from planting_calendar.py.

SCHEDULE_TEMPLATES: dict[str, list[tuple]] = {
    "Maize": [
        (
            "PRE_PLANTING",
            "Land clearing and ploughing",
            -4,
            -3,
            "Deep tillage 20–25cm; remove crop residues",
        ),
        (
            "PRE_PLANTING",
            "Soil testing and input procurement",
            -3,
            -2,
            "Procure certified seed, NPK 15:15:15, herbicide",
        ),
        (
            "PRE_PLANTING",
            "Harrowing and ridging",
            -2,
            -1,
            "Final seedbed preparation; form ridges or flat bed",
        ),
        (
            "PLANTING",
            "Sowing",
            0,
            0,
            "2–3 seeds/hole at 75×25cm spacing; thin to 1 plant",
        ),
        (
            "EARLY_GROWTH",
            "Pre-emergence herbicide application",
            1,
            1,
            "Apply within 3 days of planting on moist soil",
        ),
        (
            "EARLY_GROWTH",
            "Gap filling",
            2,
            2,
            "Replace failed stands; maintain plant population",
        ),
        (
            "EARLY_GROWTH",
            "First weeding",
            3,
            4,
            "Hand-hoe or post-emergence herbicide at 3 WAP",
        ),
        (
            "VEGETATIVE",
            "First fertilizer application (NPK)",
            4,
            4,
            "Apply NPK 15:15:15 at 4 WAP; side-dress and cover",
        ),
        (
            "VEGETATIVE",
            "Second weeding",
            6,
            7,
            "Before canopy closure; critical for yield protection",
        ),
        (
            "VEGETATIVE",
            "Pest scouting — stem borer",
            5,
            8,
            "Scout weekly; apply insecticide at threshold",
        ),
        (
            "REPRODUCTIVE",
            "Second fertilizer application (Urea)",
            8,
            8,
            "Urea top-dress at tasselling; do not fold into leaves",
        ),
        (
            "REPRODUCTIVE",
            "Pollination monitoring",
            9,
            10,
            "Ensure good silk coverage; no pesticide during silking",
        ),
        (
            "REPRODUCTIVE",
            "Fall armyworm scouting",
            8,
            12,
            "High-risk period; apply biopesticide if threshold met",
        ),
        (
            "PRE_HARVEST",
            "Maturity assessment",
            14,
            15,
            "Check husk colour and grain hardness; moisture <25%",
        ),
        (
            "HARVEST",
            "Harvesting and field drying",
            15,
            16,
            "Snap cobs; sun-dry to <13% moisture for storage",
        ),
        (
            "POST_HARVEST",
            "Shelling and bagging",
            16,
            17,
            "Shell when dry; use hermetic bags for storage",
        ),
        (
            "POST_HARVEST",
            "Field cleanup and compost preparation",
            17,
            18,
            "Incorporate stover; prepare compost pit for off-season",
        ),
    ],
    "Cassava": [
        (
            "PRE_PLANTING",
            "Land clearing and primary tillage",
            -4,
            -3,
            "Remove stumps; plough to 30cm depth",
        ),
        (
            "PRE_PLANTING",
            "Stem cutting procurement and curing",
            -2,
            -1,
            "Select disease-free cuttings 25cm long; cure 3–5 days",
        ),
        (
            "PRE_PLANTING",
            "Mound or ridge formation",
            -1,
            0,
            "Form ridges 30cm high; spacing 1×1m",
        ),
        (
            "PLANTING",
            "Planting stem cuttings",
            0,
            0,
            "Plant at 45° angle, 2/3 buried; 10,000 cuttings/ha",
        ),
        (
            "EARLY_GROWTH",
            "Gap filling",
            2,
            3,
            "Replace failed cuttings before canopy closure",
        ),
        (
            "EARLY_GROWTH",
            "First weeding",
            4,
            6,
            "Critical period: weed-free first 3 months",
        ),
        (
            "VEGETATIVE",
            "First fertilizer application",
            8,
            8,
            "NPK 15:15:15 at 2 months; place 10cm from stem",
        ),
        ("VEGETATIVE", "Second weeding", 10, 12, "Before secondary canopy closure"),
        (
            "VEGETATIVE",
            "Pest scouting — whitefly and CMD",
            8,
            20,
            "Scout fortnightly; rogue CMV-infected plants",
        ),
        (
            "VEGETATIVE",
            "Second fertilizer application",
            16,
            16,
            "Potassium sulphate at 4 months for tuber bulking",
        ),
        (
            "REPRODUCTIVE",
            "Tuber bulking monitoring",
            24,
            36,
            "Monitor canopy colour; yellowing = nutrient stress",
        ),
        (
            "PRE_HARVEST",
            "Maturity assessment",
            36,
            40,
            "Check HCN content if bitter variety; test tuber size",
        ),
        (
            "HARVEST",
            "Harvesting",
            40,
            44,
            "Mechanical or manual uprooting; avoid tuber damage",
        ),
        (
            "POST_HARVEST",
            "Processing or storage",
            44,
            48,
            "Process within 24–48hrs (perishable); or wax-coat",
        ),
        (
            "POST_HARVEST",
            "Field ratoon or new land prep",
            44,
            52,
            "For ratoon crop: cut stems to 50cm; fertilize",
        ),
    ],
    "Yam": [
        (
            "PRE_PLANTING",
            "Mound construction",
            -4,
            -3,
            "Form 1×1m mounds 30–50cm high before rains; labour-intensive",
        ),
        (
            "PRE_PLANTING",
            "Seed yam sett preparation",
            -3,
            -2,
            "Cut setts 150–200g; dust with fungicide/ash; cure 7 days",
        ),
        (
            "PRE_PLANTING",
            "Staking material procurement",
            -2,
            -1,
            "Procure 5,000 stakes/ha; pre-plant before sowing",
        ),
        (
            "PLANTING",
            "Planting seed yam setts",
            0,
            0,
            "Plant on top of mound; 5cm depth; 1 sett/mound",
        ),
        (
            "EARLY_GROWTH",
            "Staking young vines",
            3,
            4,
            "Stake vines at 3–4 WAP to prevent lodging",
        ),
        (
            "EARLY_GROWTH",
            "First weeding",
            3,
            5,
            "Hand weed; avoid disturbing tuber zone",
        ),
        (
            "VEGETATIVE",
            "First fertilizer application",
            6,
            6,
            "NPK 15:15:15 at 6 WAP; side-dress around mound",
        ),
        (
            "VEGETATIVE",
            "Second weeding",
            8,
            10,
            "Critical for tuber bulking; avoid root disturbance",
        ),
        (
            "VEGETATIVE",
            "Second fertilizer application",
            10,
            10,
            "Muriate of Potash for tuber quality",
        ),
        (
            "VEGETATIVE",
            "Pest scouting — yam mosaic virus",
            6,
            20,
            "Scout weekly; rogue infected vines immediately",
        ),
        (
            "REPRODUCTIVE",
            "Tuber monitoring",
            20,
            28,
            "Observe vine senescence; partial harvest possible",
        ),
        (
            "PRE_HARVEST",
            "Vine senescence assessment",
            26,
            28,
            "80% vine dry-down indicates harvest readiness",
        ),
        (
            "HARVEST",
            "Harvesting and destaking",
            28,
            30,
            "Manual dig; avoid cuts; destake before lifting",
        ),
        (
            "POST_HARVEST",
            "Barn storage or yam barn construction",
            30,
            32,
            "Stack in shaded barn; inspect monthly for rot",
        ),
    ],
    "Rice (paddy)": [
        (
            "PRE_PLANTING",
            "Nursery bed preparation",
            -4,
            -4,
            "Prepare raised nursery bed 1×10m per ha field",
        ),
        (
            "PRE_PLANTING",
            "Seed soaking and pregermination",
            -3,
            -3,
            "Soak 24hr; drain 24hr; sow at pin-point germination",
        ),
        (
            "PRE_PLANTING",
            "Nursery seeding",
            -3,
            -2,
            "Broadcast 40g/m² in nursery; flood to 2cm depth",
        ),
        (
            "PRE_PLANTING",
            "Land puddling and bund construction",
            -2,
            -1,
            "Puddle field; form bunds to retain water",
        ),
        (
            "PLANTING",
            "Transplanting",
            0,
            1,
            "Transplant 3-week-old seedlings; 2–3 per hill; 20×20cm",
        ),
        (
            "EARLY_GROWTH",
            "Flood management",
            1,
            3,
            "Maintain 5cm flood depth; drain for weeding",
        ),
        (
            "EARLY_GROWTH",
            "First weeding (drain and hand-weed)",
            2,
            3,
            "Drain field; weed; re-flood within 24hrs",
        ),
        (
            "VEGETATIVE",
            "First fertilizer application (NPK)",
            3,
            3,
            "Apply before re-flooding; broadcast and flood",
        ),
        (
            "VEGETATIVE",
            "Second weeding or herbicide",
            5,
            6,
            "Post-emergence herbicide or drain-and-hoe",
        ),
        (
            "VEGETATIVE",
            "Pest scouting — stem borer and BPH",
            4,
            10,
            "Scout twice weekly; avoid broad-spectrum at tillering",
        ),
        (
            "REPRODUCTIVE",
            "Second fertilizer — Urea at panicle",
            8,
            8,
            "Panicle initiation top-dress; drain 3 days before",
        ),
        (
            "REPRODUCTIVE",
            "Water management at flowering",
            9,
            11,
            "Maintain 5cm flood; critical for pollination",
        ),
        (
            "PRE_HARVEST",
            "Drain field and maturity check",
            13,
            14,
            "Drain 2 weeks before harvest; check grain moisture",
        ),
        (
            "HARVEST",
            "Harvesting and threshing",
            14,
            16,
            "Harvest at <25% moisture; thresh within 24hrs",
        ),
        (
            "POST_HARVEST",
            "Drying and milling preparation",
            16,
            18,
            "Sun-dry to <14%; bag for milling or storage",
        ),
    ],
    "Sorghum": [
        (
            "PRE_PLANTING",
            "Land clearing and tillage",
            -4,
            -3,
            "Minimum tillage acceptable; remove previous stubble",
        ),
        (
            "PRE_PLANTING",
            "Input procurement",
            -2,
            -1,
            "Procure certified seed, NPK fertilizer",
        ),
        (
            "PLANTING",
            "Sowing",
            0,
            0,
            "Drill or spot sow; 75×20cm; thin to 1 plant at 2 WAP",
        ),
        (
            "EARLY_GROWTH",
            "Thinning and gap filling",
            2,
            3,
            "Thin to 1 seedling/hole; fill gaps with reserve seed",
        ),
        ("EARLY_GROWTH", "First weeding", 3, 5, "Critical period; weed before 4 WAP"),
        (
            "VEGETATIVE",
            "Fertilizer application",
            5,
            5,
            "NPK at 5 WAP; side-dress; avoid waterlogging",
        ),
        ("VEGETATIVE", "Second weeding", 7, 8, "Before panicle initiation"),
        (
            "VEGETATIVE",
            "Pest scouting — shoot fly, aphids",
            2,
            10,
            "Scout weekly; apply systemic insecticide at threshold",
        ),
        (
            "REPRODUCTIVE",
            "Bird scare implementation",
            14,
            18,
            "Continuous bird-scaring from grain filling to harvest",
        ),
        (
            "REPRODUCTIVE",
            "Mould monitoring",
            15,
            20,
            "Check for grain mould in high-humidity zones",
        ),
        (
            "PRE_HARVEST",
            "Maturity assessment",
            18,
            20,
            "Panicle colour and grain hardness check",
        ),
        (
            "HARVEST",
            "Harvesting",
            20,
            21,
            "Cut panicles; head by head; avoid shattering",
        ),
        (
            "POST_HARVEST",
            "Threshing and drying",
            21,
            22,
            "Sun-dry to <12% moisture; bag in hermetic storage",
        ),
    ],
    "Oil palm fruit": [
        (
            "PRE_PLANTING",
            "Pre-nursery preparation",
            -8,
            -6,
            "Germinate seeds; transfer to pre-nursery at 3 months",
        ),
        (
            "PRE_PLANTING",
            "Main nursery",
            -6,
            -1,
            "Grow in main nursery 12 months; harden off before planting",
        ),
        (
            "PRE_PLANTING",
            "Land clearing and holing",
            -2,
            -1,
            "Clear undergrowth; dig 60×60×60cm holes; apply rock phos",
        ),
        (
            "PLANTING",
            "Transplanting seedlings",
            0,
            0,
            "Plant 18-month seedlings at 9×9m triangular; mulch basin",
        ),
        (
            "EARLY_GROWTH",
            "Establishment weeding",
            2,
            4,
            "Circle weed monthly for first 2 years",
        ),
        (
            "EARLY_GROWTH",
            "Cover crop establishment",
            2,
            8,
            "Sow Pueraria cover between palms to suppress weeds",
        ),
        (
            "VEGETATIVE",
            "First year fertilizer",
            12,
            12,
            "Muriate of Potash + Urea at 12 months per palm",
        ),
        (
            "VEGETATIVE",
            "Frond management",
            16,
            20,
            "Remove oldest fronds; do not over-prune",
        ),
        ("VEGETATIVE", "Second year fertilizer", 24, 24, "Full NPK per palm at year 2"),
        (
            "REPRODUCTIVE",
            "First bunch monitoring (yr 3+)",
            36,
            52,
            "Count bunches; thin excess in first bearing year",
        ),
        (
            "REPRODUCTIVE",
            "Bunch harvest cycle begins",
            40,
            52,
            "Harvest ripe bunches every 10–14 days",
        ),
        (
            "HARVEST",
            "Ongoing bunch harvesting",
            40,
            52,
            "Use harvesting chisel; harvest at black fruit threshold",
        ),
        (
            "POST_HARVEST",
            "FFB transport and processing",
            40,
            52,
            "Transport within 24hrs to avoid FFA rise",
        ),
    ],
    "Cocoa beans": [
        (
            "PRE_PLANTING",
            "Nursery pod selection and seeding",
            -12,
            -8,
            "Select pods from high-yielding trees; sow in poly bags",
        ),
        (
            "PRE_PLANTING",
            "Shade tree establishment",
            -8,
            -4,
            "Plant plantain or Gliricidia as temporary shade",
        ),
        (
            "PRE_PLANTING",
            "Land preparation and pegging",
            -4,
            -2,
            "Peg holes at 3×3m; apply basal fertilizer",
        ),
        (
            "PLANTING",
            "Transplanting cocoa seedlings",
            0,
            0,
            "Transplant 4–6 month seedlings; plant under shade",
        ),
        (
            "EARLY_GROWTH",
            "Mulching and establishment care",
            1,
            8,
            "Mulch basin; water if dry; control weeds by hand",
        ),
        (
            "VEGETATIVE",
            "Shade management",
            8,
            16,
            "Gradually reduce shade as canopy develops",
        ),
        (
            "VEGETATIVE",
            "Fertilizer application",
            12,
            12,
            "NPK at 12 months; apply in ring 30cm from stem",
        ),
        (
            "VEGETATIVE",
            "Pruning — jorquette formation",
            16,
            20,
            "Shape jorquette; remove chupons and dead wood",
        ),
        (
            "VEGETATIVE",
            "Black pod disease scouting",
            12,
            32,
            "Spray copper fungicide at 2-week intervals in wet season",
        ),
        (
            "REPRODUCTIVE",
            "Flushing and flowering monitoring",
            24,
            32,
            "Ensure good shade; remove mummified pods",
        ),
        (
            "REPRODUCTIVE",
            "Cherelle wilt assessment",
            28,
            36,
            "Normal to lose 80% of chereelles; do not panic",
        ),
        (
            "HARVEST",
            "Pod breaking and bean extraction",
            36,
            44,
            "Harvest yellow/orange pods; extract beans same day",
        ),
        (
            "POST_HARVEST",
            "Fermentation",
            36,
            37,
            "Ferment 5–6 days in boxes; turn daily from day 2",
        ),
        (
            "POST_HARVEST",
            "Drying and grading",
            37,
            38,
            "Sun-dry to 7.5% moisture; grade and bag 62.5kg",
        ),
    ],
}


# ── Data loading ──────────────────────────────────────────────────────────


def load_planting_calendar() -> pd.DataFrame:
    """Load planting_calendar_all_zones.csv from planting_calendar.py output."""
    if not CALENDAR_CSV.exists():
        raise FileNotFoundError(
            f"Planting calendar not found at {CALENDAR_CSV}. "
            "Run module5_planning/planting_calendar.py first."
        )
    df = pd.read_csv(CALENDAR_CSV)
    log.info("Loaded planting calendar: %d zone-crop rows", len(df))
    return df


# ── Schedule generation ────────────────────────────────────────────────────


def _week_to_month(start_week: int, planting_month: int) -> str:
    """Convert a week offset relative to planting to an approximate calendar month name."""
    month_offset = start_week // 4
    month = ((planting_month - 1 + month_offset) % 12) + 1
    return MONTH_NAMES[month - 1]


def generate_schedule(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Generate full operations schedule for all zone-crop pairs.

    For each (zone, crop) in the planting calendar:
      1. Look up the planting_month
      2. Retrieve the schedule template for that crop
      3. Anchor all week offsets to the planting month
      4. Append zone/crop metadata

    Returns a flat DataFrame with one row per schedule activity.
    """
    all_rows = []
    missing_templates = set()

    for _, cal_row in calendar.iterrows():
        zone = cal_row["zone"]
        crop = cal_row["crop"]
        planting_month = int(cal_row["planting_month"])
        harvest_month = int(cal_row["harvest_month"])
        reliability = cal_row.get("rainfall_reliability_score", None)

        template = SCHEDULE_TEMPLATES.get(crop)
        if template is None:
            missing_templates.add(crop)
            continue

        for phase, activity, start_wk, end_wk, notes in template:
            start_month_name = _week_to_month(start_wk, planting_month)
            end_month_name = _week_to_month(end_wk, planting_month)

            # Resolve to month number for sorting
            start_month_num = ((planting_month - 1 + start_wk // 4) % 12) + 1

            all_rows.append(
                {
                    "zone": zone,
                    "crop": crop,
                    "planting_month": planting_month,
                    "planting_month_name": MONTH_NAMES[planting_month - 1],
                    "harvest_month": harvest_month,
                    "harvest_month_name": MONTH_NAMES[harvest_month - 1],
                    "rainfall_reliability_score": reliability,
                    "phase": phase,
                    "activity": activity,
                    "start_week_offset": start_wk,
                    "end_week_offset": end_wk,
                    "approx_start_month": start_month_name,
                    "approx_end_month": end_month_name,
                    "approx_start_month_num": start_month_num,
                    "activity_notes": notes,
                }
            )

    if missing_templates:
        log.warning("No schedule template found for crops: %s", missing_templates)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["zone", "crop", "start_week_offset"]).reset_index(drop=True)

    log.info(
        "Schedule generated: %d activity rows across %d zone-crop pairs",
        len(df),
        df.groupby(["zone", "crop"]).ngroups,
    )
    return df


# ── Summary pivot ──────────────────────────────────────────────────────────


def build_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary: one row per (zone, crop) with key milestone months.

    Columns:
      zone, crop, planting_month_name, first_fertilizer_month,
      first_harvest_month, harvest_month_name, total_activities
    """
    rows = []
    for (zone, crop), grp in schedule.groupby(["zone", "crop"]):
        planting = grp["planting_month_name"].iloc[0]
        harvest = grp["harvest_month_name"].iloc[0]

        fert_rows = grp[grp["activity"].str.contains("fertilizer|Fertilizer", na=False)]
        first_fert = (
            fert_rows["approx_start_month"].iloc[0] if len(fert_rows) > 0 else "N/A"
        )

        harv_rows = grp[grp["phase"] == "HARVEST"]
        first_harv = (
            harv_rows["approx_start_month"].iloc[0] if len(harv_rows) > 0 else harvest
        )

        rows.append(
            {
                "zone": zone,
                "crop": crop,
                "planting_month": planting,
                "first_fertilizer_month": first_fert,
                "first_harvest_month": first_harv,
                "harvest_month": harvest,
                "total_schedule_activities": len(grp),
                "reliability_score": grp["rainfall_reliability_score"].iloc[0],
            }
        )

    df_summary = pd.DataFrame(rows).sort_values(["zone", "crop"]).reset_index(drop=True)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    log.info("Summary saved → %s  (%d rows)", SUMMARY_CSV.name, len(df_summary))
    return df_summary


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 5: Operations Schedule Generator")
    log.info("=" * 60)

    log.info("STEP 1 — Load planting calendar")
    calendar = load_planting_calendar()

    log.info("STEP 2 — Generate week-by-week schedule (all 42 zone-crop pairs)")
    schedule = generate_schedule(calendar)

    log.info("STEP 3 — Save full schedule CSV")
    schedule.to_csv(SCHEDULE_CSV, index=False)
    log.info("Full schedule saved → %s  (%d rows)", SCHEDULE_CSV.name, len(schedule))

    log.info("STEP 4 — Build and save summary pivot")
    summary = build_summary(schedule)

    log.info("=" * 60)
    log.info("Module 5 — operations_schedule.py COMPLETE")
    log.info("  operations_schedule_all_zones.csv → %d activity rows", len(schedule))
    log.info("  operations_schedule_summary.csv   → %d zone-crop pairs", len(summary))
    log.info("  Outputs → module5_planning/outputs/")
    log.info("")
    log.info("  SAMPLE — South West Cocoa beans schedule:")

    sample = schedule[
        (schedule["zone"] == "South West") & (schedule["crop"] == "Cocoa beans")
    ][["phase", "activity", "approx_start_month", "approx_end_month"]].head(5)
    for _, r in sample.iterrows():
        log.info(
            "    [%-16s] %-40s %s → %s",
            r["phase"],
            r["activity"][:40],
            r["approx_start_month"],
            r["approx_end_month"],
        )

    log.info("=" * 60)


if __name__ == "__main__":
    main()
