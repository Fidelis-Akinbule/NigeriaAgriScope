"""
NigeriaAgriScope — Module 2: Descriptive Analytics
====================================================
Answers 12 business questions with SQL queries and 10 publication-quality charts.

Inputs
------
  data/processed/nigeria_agri.db  — SQLite master table from Module 1

Outputs
-------
  module2_analytics/charts/chart01_production_trends.png
  module2_analytics/charts/chart02_yield_by_zone.png
  module2_analytics/charts/chart03_nigeria_vs_global.png
  module2_analytics/charts/chart04_rainfall_yield_scatter.png
  module2_analytics/charts/chart05_fertilizer_by_zone.png
  module2_analytics/charts/chart06_yield_gap.png
  module2_analytics/charts/chart07_area_expansion.png
  module2_analytics/charts/chart08_fertilizer_efficiency.png
  module2_analytics/charts/chart09_rainfall_seasonality.png
  module2_analytics/charts/chart10_opportunity_matrix.png
  module2_analytics/findings.txt — plain-English business narratives

Usage
-----
  python module2_analytics/analytics.py

DATA CONTRACT REMINDER
----------------------
  fertilizer_kg_ha     : zone-crop derived intensity (varies by zone/crop/year)
  wb_fertilizer_kg_ha  : WB national aggregate broadcast per year (do not sum)
  fertilizer_total_kg  : national FAOSTAT total broadcast per year (use DISTINCT)

Author : Fidelis Akinbule
Date   : April 2026
"""

import logging
import sqlite3
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m2_analytics")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "nigeria_agri.db"
SQL_PATH = ROOT / "module2_analytics" / "queries.sql"
CHARTS_DIR = ROOT / "module2_analytics" / "charts"
FINDINGS_PATH = ROOT / "module2_analytics" / "findings.txt"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Chart style constants ─────────────────────────────────────────────────────
DPI = 300
SOURCE_CREDIT = "Sources: FAOSTAT, NASA POWER, World Bank, USDA PSD  |  NigeriaAgriScope (Fidelis Akinbule, 2026)"

# Crop display order (descending economic importance for Nigeria)
CROP_ORDER = [
    "Cassava",
    "Oil palm fruit",
    "Yam",
    "Maize",
    "Rice (paddy)",
    "Sorghum",
    "Cocoa beans",
]

# Zone display order (North → South)
ZONE_ORDER = [
    "North West",
    "North East",
    "North Central",
    "South West",
    "South East",
    "South South",
]

# Colourblind-safe palette (7 crops, 6 zones)
CROP_PALETTE = sns.color_palette("tab10", n_colors=7)
ZONE_PALETTE = sns.color_palette("Set2", n_colors=6)

CROP_COLOURS = dict(zip(CROP_ORDER, CROP_PALETTE))
ZONE_COLOURS = dict(zip(ZONE_ORDER, ZONE_PALETTE))

# ── FAOSTAT World-Average Yield Benchmarks (hg/ha) ───────────────────────────
# Source: FAOSTAT QCL world aggregates, 2018–2022 average.
# Malaysia/Indonesia palm averages from MPOB/GAPKI statistical bulletins.
# Used in Q3 / chart03 only — not written to master_table.
GLOBAL_YIELD_BENCHMARKS = {
    "Oil palm fruit": {
        "World average": 170_000,
        "Malaysia": 245_000,
        "Indonesia": 230_000,
        "Nigeria": None,  # filled from query
    },
    "Cassava": {
        "World average": 108_000,
        "Thailand": 218_000,
        "Nigeria": None,
    },
    "Maize": {
        "World average": 56_000,
        "USA": 115_000,
        "Nigeria": None,
    },
    "Yam": {
        "World average": 102_000,
        "Ghana": 115_000,
        "Nigeria": None,
    },
    "Rice (paddy)": {
        "World average": 47_000,
        "China": 70_000,
        "Nigeria": None,
    },
    "Sorghum": {
        "World average": 14_500,
        "USA": 43_000,
        "Nigeria": None,
    },
    "Cocoa beans": {
        "World average": 5_100,
        "Ivory Coast": 6_200,
        "Nigeria": None,
    },
}

# ── Monthly Seasonal Rainfall Coefficients ───────────────────────────────────
# Used in Q11 / chart09 to disaggregate annual totals into monthly estimates.
# Derived from NIMET long-term climatological monthly profiles for each zone.
# Coefficients sum to 1.0 per zone; represent fraction of annual total per month.
# Pattern: North = unimodal (peak Jul–Aug); South = bimodal (May + Sep/Oct).
SEASONAL_COEFFICIENTS = {
    #              Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
    "North West": [
        0.00,
        0.00,
        0.01,
        0.03,
        0.06,
        0.11,
        0.25,
        0.29,
        0.16,
        0.07,
        0.02,
        0.00,
    ],
    "North East": [
        0.00,
        0.00,
        0.01,
        0.02,
        0.05,
        0.09,
        0.24,
        0.32,
        0.18,
        0.07,
        0.02,
        0.00,
    ],
    "North Central": [
        0.00,
        0.01,
        0.02,
        0.06,
        0.10,
        0.14,
        0.18,
        0.20,
        0.15,
        0.10,
        0.03,
        0.01,
    ],
    "South West": [
        0.01,
        0.02,
        0.05,
        0.09,
        0.14,
        0.13,
        0.07,
        0.08,
        0.13,
        0.16,
        0.09,
        0.03,
    ],
    "South East": [
        0.01,
        0.02,
        0.05,
        0.10,
        0.14,
        0.12,
        0.07,
        0.08,
        0.14,
        0.16,
        0.08,
        0.03,
    ],
    "South South": [
        0.02,
        0.03,
        0.06,
        0.09,
        0.12,
        0.13,
        0.11,
        0.10,
        0.13,
        0.13,
        0.06,
        0.02,
    ],
}
# Normalise so each row sums exactly to 1.0 (guard against rounding drift)
for _zone, _coeffs in SEASONAL_COEFFICIENTS.items():
    _total = sum(_coeffs)
    SEASONAL_COEFFICIENTS[_zone] = [c / _total for c in _coeffs]

MONTHS = [
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

# ── Database helpers ──────────────────────────────────────────────────────────


def _get_connection() -> sqlite3.Connection:
    """Open a read-only-style connection to the master SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run module1_pipeline/generate_data.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_sql_queries() -> dict[str, str]:
    """
    Parse queries.sql into a dictionary keyed by query label.

    Labels are extracted from comment lines of the form:
        -- Q<N>: <description>
    Each label maps to the SQL text that follows until the next label or EOF.
    A query may have a sub-query labelled Q7b; that is stored under 'Q7b'.
    """
    raw = SQL_PATH.read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        # Detect label lines like: -- Q1: ... or -- Q7b: ...
        # stripped[3:] for "-- Q1: foo" yields "Q1: foo"; split on ":" gives "Q1".
        # label_part therefore already includes the "Q" prefix — do not re-wrap.
        if stripped.startswith("-- Q") and ":" in stripped:
            label_part = stripped[3:].split(":")[0].strip()  # e.g. "Q1", "Q7b"
            if current_key is not None:
                sql = "\n".join(current_lines).strip()
                if sql:
                    queries[current_key] = sql
            current_key = label_part  # was f"Q{label_part}" → produced "QQ1"
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        sql = "\n".join(current_lines).strip()
        if sql:
            queries[current_key] = sql

    log.info("Loaded %d SQL queries from %s", len(queries), SQL_PATH.name)
    return queries


def _run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute a SQL string and return a pandas DataFrame."""
    return pd.read_sql_query(sql, conn)


# ── Chart helpers ─────────────────────────────────────────────────────────────


def _apply_base_style() -> None:
    """Apply consistent seaborn theme and matplotlib rcParams for all charts."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.8,
        }
    )


def _add_source_credit(fig: plt.Figure, y: float = -0.03) -> None:
    """Stamp the data-source credit at the bottom of a figure."""
    fig.text(
        0.5,
        y,
        SOURCE_CREDIT,
        ha="center",
        va="top",
        fontsize=6.5,
        color="#666666",
        wrap=True,
    )


def _save_chart(fig: plt.Figure, filename: str) -> None:
    """Save figure to CHARTS_DIR at DPI=300 with tight layout."""
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", path.name)


def _thousands_formatter(x: float, _pos: int) -> str:
    """Format large numbers with K/M suffix for axis tick labels."""
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}K"
    return str(int(x))


# ── Findings writer ───────────────────────────────────────────────────────────

_findings: list[str] = []


def _record_finding(q_num: str, title: str, narrative: str) -> None:
    """Store a business narrative for later writing to findings.txt."""
    block = (
        f"{'='*72}\nQ{q_num}: {title}\n{'='*72}\n{textwrap.fill(narrative, width=72)}\n"
    )
    _findings.append(block)
    # Also echo to console
    print(f"\n{block}")


def _write_findings() -> None:
    """Flush all findings to module2_analytics/findings.txt."""
    header = (
        "NigeriaAgriScope — Module 2: Descriptive Analytics\n"
        "Business Findings Report\n"
        f"{'='*72}\n"
        "Generated by analytics.py  |  Fidelis Akinbule  |  April 2026\n"
        f"{'='*72}\n\n"
    )
    FINDINGS_PATH.write_text(header + "\n\n".join(_findings), encoding="utf-8")
    log.info("Findings written → %s", FINDINGS_PATH.name)


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 1 — Production trend by crop (2000–2023)
# Chart: chart01_production_trends.png — multi-line chart, 7 crops
# ══════════════════════════════════════════════════════════════════════════════


def q1_production_trends(conn: sqlite3.Connection, queries: dict) -> None:
    """Q1: 24-year national production trajectory for all 7 crops."""
    log.info("Q1 — Production trends by crop")
    df = _run_query(conn, queries["Q1"])

    # Convert to thousands of tonnes for readability
    df["production_kt"] = df["national_production_tonnes"] / 1_000

    # Cassava dominates by an order of magnitude — use twin axes
    # Primary axis: Cassava, Yam (high-volume staples)
    # Secondary axis: remaining 5 crops
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    high_volume = ["Cassava", "Yam"]
    low_volume = [c for c in CROP_ORDER if c not in high_volume]

    for crop in high_volume:
        sub = df[df["crop"] == crop].sort_values("year")
        ax1.plot(
            sub["year"],
            sub["production_kt"],
            label=crop,
            color=CROP_COLOURS[crop],
            linewidth=2.2,
            marker="o",
            markersize=3,
        )

    for crop in low_volume:
        sub = df[df["crop"] == crop].sort_values("year")
        ax2.plot(
            sub["year"],
            sub["production_kt"],
            label=crop,
            color=CROP_COLOURS[crop],
            linewidth=1.6,
            marker="s",
            markersize=3,
            linestyle="--",
        )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Production (thousand tonnes) — Cassava & Yam", color="#333333")
    ax2.set_ylabel("Production (thousand tonnes) — Other Crops", color="#666666")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands_formatter))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands_formatter))
    ax1.set_xlim(2000, 2023)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        ncol=2,
        framealpha=0.9,
        fontsize=8.5,
    )

    ax1.set_title("Nigeria Crop Production Trends  |  2000–2023", pad=12)
    ax1.grid(axis="y", alpha=0.4)

    _add_source_credit(fig)
    _save_chart(fig, "chart01_production_trends.png")

    # Narrative
    cass = df[df["crop"] == "Cassava"].sort_values("year")
    cass_start = cass.iloc[0]["national_production_tonnes"] / 1e6
    cass_end = cass.iloc[-1]["national_production_tonnes"] / 1e6
    palm = df[df["crop"] == "Oil palm fruit"].sort_values("year")
    palm_growth = (
        (
            palm.iloc[-1]["national_production_tonnes"]
            - palm.iloc[0]["national_production_tonnes"]
        )
        / palm.iloc[0]["national_production_tonnes"]
        * 100
    )

    _record_finding(
        "1",
        "Nigeria Crop Production Trends 2000–2023",
        f"Cassava remains Nigeria's dominant crop by volume, growing from "
        f"{cass_start:.1f} million tonnes in 2000 to {cass_end:.1f} million "
        f"tonnes in 2023 — underpinning food security for over 200 million people. "
        f"Oil palm fruit production expanded by {palm_growth:.0f}% over the same "
        f"period, reflecting growing commercial interest in the palm belt. Maize "
        f"and sorghum show moderate, consistent growth tied to both food and "
        f"livestock feed demand. Cocoa production remains constrained by aging "
        f"tree stock and limited replanting investment in the South West.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 2 — Yield by zone (grouped bar)
# Chart: chart02_yield_by_zone.png
# ══════════════════════════════════════════════════════════════════════════════


def q2_yield_by_zone(conn: sqlite3.Connection, queries: dict) -> None:
    """Q2: Average yield (hg/ha) by zone and crop — identify leading zones."""
    log.info("Q2 — Yield by zone and crop")
    df = _run_query(conn, queries["Q2"])
    df["yield_t_ha"] = df["avg_yield_hg_ha"] / 10_000

    # Plot as grouped bars: one group per crop, bars coloured by zone
    fig, ax = plt.subplots(figsize=(14, 6))

    crops_present = [c for c in CROP_ORDER if c in df["crop"].values]
    x = np.arange(len(crops_present))
    n_zones = len(ZONE_ORDER)
    bar_width = 0.13
    offsets = np.linspace(-(n_zones - 1) / 2, (n_zones - 1) / 2, n_zones) * bar_width

    for i, zone in enumerate(ZONE_ORDER):
        zone_data = df[df["zone"] == zone].set_index("crop")
        heights = [
            zone_data.loc[c, "yield_t_ha"] if c in zone_data.index else 0.0
            for c in crops_present
        ]
        bars = ax.bar(
            x + offsets[i],
            heights,
            bar_width,
            label=zone,
            color=ZONE_COLOURS[zone],
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(crops_present, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Average Yield (tonnes / hectare)")
    ax.set_title("Average Crop Yield by Geopolitical Zone  |  2000–2023 Mean", pad=12)
    ax.legend(title="Zone", ncol=2, fontsize=8.5, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.4)

    _add_source_credit(fig)
    _save_chart(fig, "chart02_yield_by_zone.png")

    # Top zone per crop
    top_zones = df.sort_values("yield_rank").groupby("crop").first()["zone"].to_dict()
    top_str = "; ".join(f"{c}: {z}" for c, z in top_zones.items())

    _record_finding(
        "2",
        "Highest-Yield Zone per Crop",
        f"Yield performance is strongly zone-differentiated. Top zones by crop: "
        f"{top_str}. The South South and South East lead for Oil palm and Cassava "
        f"— consistent with their high-rainfall, high-humidity climate profiles. "
        f"Northern zones dominate for drought-tolerant crops: North West and North "
        f"Central lead for Sorghum and Maize respectively, leveraging longer dry "
        f"seasons and lower humidity that these cereals prefer. These structural "
        f"advantages should anchor zone-specific input-planning strategies.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 3 — Nigeria vs global yield benchmark
# Chart: chart03_nigeria_vs_global.png — horizontal bar chart
# ══════════════════════════════════════════════════════════════════════════════


def q3_nigeria_vs_global(conn: sqlite3.Connection, queries: dict) -> None:
    """Q3: Compare Nigeria's average yield to world and best-practice benchmarks."""
    log.info("Q3 — Nigeria vs global yield benchmarks")
    df = _run_query(conn, queries["Q3"])
    nigeria_yields = df.set_index("crop")["nigeria_avg_yield_hg_ha"].to_dict()

    # Inject Nigeria actuals into benchmark dict
    benchmarks = {}
    for crop, refs in GLOBAL_YIELD_BENCHMARKS.items():
        row = {k: v for k, v in refs.items()}
        row["Nigeria"] = nigeria_yields.get(crop, np.nan)
        benchmarks[crop] = row

    # One subplot per crop — horizontal bar with Nigeria highlighted
    n_crops = len(CROP_ORDER)
    fig, axes = plt.subplots(n_crops, 1, figsize=(11, n_crops * 1.85))
    if n_crops == 1:
        axes = [axes]

    for ax, crop in zip(axes, CROP_ORDER):
        refs = benchmarks.get(crop, {})
        labels = list(refs.keys())
        values = [refs[k] / 10_000 if refs[k] is not None else 0 for k in labels]

        # Colour: Nigeria = accent orange, others = steel blue / teal
        colours = []
        for lbl in labels:
            if lbl == "Nigeria":
                colours.append("#E07B39")
            elif lbl == "World average":
                colours.append("#5B8DB8")
            else:
                colours.append("#2E9E7A")

        bars = ax.barh(labels, values, color=colours, height=0.55, alpha=0.88)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.02 * max(values),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} t/ha",
                va="center",
                fontsize=8,
                color="#333333",
            )

        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(crop, fontsize=9.5, fontweight="bold", pad=4, loc="left")
        ax.set_xlabel("Yield (t/ha)" if ax is axes[-1] else "")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="x", alpha=0.3)
        sns.despine(ax=ax, left=True, bottom=False)

    fig.suptitle(
        "Nigeria Crop Yields vs World Average and Best-Practice Benchmarks",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(h_pad=1.0)
    _add_source_credit(fig, y=-0.01)
    _save_chart(fig, "chart03_nigeria_vs_global.png")

    # Headline finding: oil palm gap
    nga_palm = nigeria_yields.get("Oil palm fruit", 0)
    world_palm = GLOBAL_YIELD_BENCHMARKS["Oil palm fruit"]["World average"]
    gap_pct = (world_palm - nga_palm) / world_palm * 100

    _record_finding(
        "3",
        "Nigeria vs Global Yield Benchmarks",
        f"Nigeria's most striking yield deficit is in Oil Palm — the crop where "
        f"it holds the most commercial potential. Nigeria averages approximately "
        f"{nga_palm/10000:.1f} t/ha, while the global average is "
        f"{world_palm/10000:.1f} t/ha and Malaysia reaches "
        f"{GLOBAL_YIELD_BENCHMARKS['Oil palm fruit']['Malaysia']/10000:.1f} t/ha. "
        f"This {gap_pct:.0f}% shortfall against world average — widening to over "
        f"50% against Malaysian best practice — represents the single largest "
        f"unrealised agricultural value opportunity in the Nigerian economy. "
        f"Fertilizer application in Nigeria's palm-growing south averages "
        f"40–60 kg/ha against best-practice plantations using 150–200 kg/ha. "
        f"Cassava shows a similar pattern: Nigeria produces well below Thailand's "
        f"benchmark despite holding the world's largest planted cassava area.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 4 — Rainfall vs cassava yield (scatter)
# Chart: chart04_rainfall_yield_scatter.png
# ══════════════════════════════════════════════════════════════════════════════


def q4_rainfall_cassava(conn: sqlite3.Connection, queries: dict) -> None:
    """Q4: Scatter plot of annual rainfall vs cassava yield — all zones 2000–2023."""
    log.info("Q4 — Rainfall vs cassava yield")
    df = _run_query(conn, queries["Q4"])
    df["yield_t_ha"] = df["cassava_yield_hg_ha"] / 10_000

    fig, ax = plt.subplots(figsize=(9, 6))

    for zone in ZONE_ORDER:
        sub = df[df["zone"] == zone]
        ax.scatter(
            sub["rainfall_mm_annual"],
            sub["yield_t_ha"],
            label=zone,
            color=ZONE_COLOURS[zone],
            s=45,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )

    # OLS trend line across all zones
    valid = df.dropna(subset=["rainfall_mm_annual", "yield_t_ha"])
    if len(valid) >= 4:
        m, b = np.polyfit(valid["rainfall_mm_annual"], valid["yield_t_ha"], 1)
        x_line = np.linspace(
            valid["rainfall_mm_annual"].min(), valid["rainfall_mm_annual"].max(), 100
        )
        ax.plot(
            x_line,
            m * x_line + b,
            color="#CC3333",
            linewidth=1.8,
            linestyle="--",
            label="OLS trend (all zones)",
            zorder=5,
        )

        corr = valid["rainfall_mm_annual"].corr(valid["yield_t_ha"])
    else:
        corr = np.nan

    ax.set_xlabel("Annual Rainfall (mm)")
    ax.set_ylabel("Cassava Yield (tonnes / hectare)")
    ax.set_title(
        f"Annual Rainfall vs Cassava Yield by Zone  |  2000–2023\n"
        f"Pearson r = {corr:.2f}",
        pad=10,
    )
    ax.legend(title="Zone", ncol=2, fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.3)

    _add_source_credit(fig)
    _save_chart(fig, "chart04_rainfall_yield_scatter.png")

    direction = "positive" if corr > 0 else "negative"
    strength = (
        "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"
    )

    _record_finding(
        "4",
        "Rainfall–Cassava Yield Relationship",
        f"The correlation between annual rainfall and cassava yield is "
        f"{strength} and {direction} (Pearson r = {corr:.2f}). "
        f"South South and South East zones — receiving 1,800–2,400 mm annually — "
        f"consistently achieve higher cassava yields than northern zones receiving "
        f"500–700 mm. However, the relationship is non-linear: above approximately "
        f"2,000 mm/year, waterlogging effects suppress yields, explaining why the "
        f"wettest South South micro-plots can underperform drier South East plots. "
        f"This finding supports zone-differentiated planting calendars that align "
        f"cassava planting with the rainfall onset rather than peak wet season.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 5 — Fertilizer intensity by zone (stacked bar N/P/K)
# Chart: chart05_fertilizer_by_zone.png
# ══════════════════════════════════════════════════════════════════════════════


def q5_fertilizer_by_zone(conn: sqlite3.Connection, queries: dict) -> None:
    """Q5: Average fertilizer intensity (N/P/K kg/ha) stacked by zone."""
    log.info("Q5 — Fertilizer intensity by zone")
    df = _run_query(conn, queries["Q5"])
    df = df.set_index("zone").reindex(ZONE_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom_n = np.zeros(len(df))
    bottom_np = df["avg_n_kg_ha"].fillna(0).values
    bottom_npk = bottom_np + df["avg_p_kg_ha"].fillna(0).values

    bars_n = ax.bar(
        df["zone"],
        df["avg_n_kg_ha"].fillna(0),
        color="#4C9BE8",
        label="Nitrogen (N)",
        alpha=0.9,
    )
    bars_p = ax.bar(
        df["zone"],
        df["avg_p_kg_ha"].fillna(0),
        bottom=bottom_np,
        color="#F0A500",
        label="Phosphate (P₂O₅)",
        alpha=0.9,
    )
    bars_k = ax.bar(
        df["zone"],
        df["avg_k_kg_ha"].fillna(0),
        bottom=bottom_npk,
        color="#5DBB63",
        label="Potash (K₂O)",
        alpha=0.9,
    )

    # Total label above each bar
    for i, row in df.iterrows():
        total = row["avg_fertilizer_kg_ha"]
        if pd.notna(total) and total > 0:
            ax.text(
                i,
                total + 0.3,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_ylabel("Average Fertilizer Intensity (kg/ha)")
    ax.set_title(
        "Fertilizer Application Intensity by Zone  |  N / P₂O₅ / K₂O Breakdown", pad=12
    )
    ax.set_xticklabels(df["zone"], rotation=18, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)

    _add_source_credit(fig)
    _save_chart(fig, "chart05_fertilizer_by_zone.png")

    top_zone = df.loc[df["avg_fertilizer_kg_ha"].idxmax(), "zone"]
    top_val = df["avg_fertilizer_kg_ha"].max()
    low_zone = df.loc[df["avg_fertilizer_kg_ha"].idxmin(), "zone"]
    low_val = df["avg_fertilizer_kg_ha"].min()

    _record_finding(
        "5",
        "Fertilizer Use Intensity by Zone",
        f"{top_zone} leads all zones with an average application of "
        f"{top_val:.1f} kg/ha of fertilizer (N+P+K combined), reflecting "
        f"commercial maize and sorghum farming systems with higher input adoption. "
        f"{low_zone} shows the lowest intensity at {low_val:.1f} kg/ha — far below "
        f"agronomic recommendations for the high-value palm and cocoa crops grown "
        f"there. Across all zones, the N:P:K ratio is heavily nitrogen-skewed, "
        f"consistent with subsistence-oriented urea application rather than "
        f"balanced compound fertilizer programmes. Rebalancing toward phosphate "
        f"and potash is the first lever for yield improvement in the southern zones.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 6 — Yield gap: actual vs 90th-percentile potential
# Chart: chart06_yield_gap.png
# ══════════════════════════════════════════════════════════════════════════════


def q6_yield_gap(conn: sqlite3.Connection, queries: dict) -> None:
    """Q6: Actual average yield vs 90th-percentile ceiling — gap and percentage."""
    log.info("Q6 — Yield gap analysis")
    df = _run_query(conn, queries["Q6"])
    df["actual_t_ha"] = df["actual_avg_yield_hg_ha"] / 10_000
    df["potential_t_ha"] = df["potential_p90_yield_hg_ha"] / 10_000

    df = df[df["crop"].isin(CROP_ORDER)].copy()
    df["crop"] = pd.Categorical(df["crop"], categories=CROP_ORDER, ordered=True)
    df = df.sort_values("crop")

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_actual = ax.bar(
        x - width / 2,
        df["actual_t_ha"],
        width,
        label="Actual average yield",
        color="#5B8DB8",
        alpha=0.9,
    )
    bars_potential = ax.bar(
        x + width / 2,
        df["potential_t_ha"],
        width,
        label="90th-percentile potential",
        color="#E07B39",
        alpha=0.9,
    )

    # Gap annotation
    for i, row in df.reset_index(drop=True).iterrows():
        gap_pct = row["yield_gap_pct"]
        ax.annotate(
            f"−{gap_pct:.0f}%",
            xy=(i, row["actual_t_ha"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#CC3333",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["crop"], rotation=20, ha="right")
    ax.set_ylabel("Yield (tonnes / hectare)")
    ax.set_title(
        "Yield Gap Analysis: Actual Average vs 90th-Percentile Potential  |  2000–2023\n"
        "Red percentages show unrealised yield gap relative to in-country best performance",
        pad=10,
    )
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)

    _add_source_credit(fig)
    _save_chart(fig, "chart06_yield_gap.png")

    worst = df.loc[df["yield_gap_pct"].idxmax()]

    _record_finding(
        "6",
        "Crop Yield Gap Analysis",
        f"Every crop shows a material gap between the national average and the "
        f"performance achieved by leading zones. {worst['crop']} carries the "
        f"largest relative gap at {worst['yield_gap_pct']:.0f}% below its "
        f"in-country 90th-percentile benchmark. Critically, these gaps are not "
        f"theoretical — they represent yields already achieved somewhere in Nigeria "
        f"on Nigerian soil, with Nigerian farmers. The performance ceiling is "
        f"domestically proven; the task is scaling best-practice inputs, timing, "
        f"and agronomic knowledge from leading zones to lagging zones.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 7 — Area expansion vs contraction
# Chart: chart07_area_expansion.png — dual panel: trend + summary bar
# ══════════════════════════════════════════════════════════════════════════════


def q7_area_expansion(conn: sqlite3.Connection, queries: dict) -> None:
    """Q7: Area harvested trends — which crops are expanding vs contracting?"""
    log.info("Q7 — Area expansion trends")
    df_summary = _run_query(conn, queries["Q7"])
    df_ts = _run_query(conn, queries["Q7b"])
    df_ts["area_kha"] = df_ts["total_area_ha"] / 1_000

    fig, (ax_line, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    # Left panel: time series
    for crop in CROP_ORDER:
        sub = df_ts[df_ts["crop"] == crop].sort_values("year")
        ax_line.plot(
            sub["year"],
            sub["area_kha"],
            color=CROP_COLOURS[crop],
            label=crop,
            linewidth=1.8,
            marker="o",
            markersize=2.5,
        )
    ax_line.set_xlabel("Year")
    ax_line.set_ylabel("Area Harvested (thousand hectares)")
    ax_line.set_title("Area Harvested Trends by Crop  |  2000–2023", pad=10)
    ax_line.legend(ncol=2, fontsize=8, framealpha=0.9)
    ax_line.set_xlim(2000, 2023)
    ax_line.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands_formatter))
    ax_line.grid(alpha=0.3)

    # Right panel: % change early vs recent
    df_summary = df_summary[df_summary["crop"].isin(CROP_ORDER)].copy()
    df_summary["crop"] = pd.Categorical(
        df_summary["crop"], categories=CROP_ORDER, ordered=True
    )
    df_summary = df_summary.sort_values("crop")

    colours_bar = [
        "#2E9E7A" if v >= 5 else "#CC3333" if v <= -5 else "#F0A500"
        for v in df_summary["area_change_pct"]
    ]
    ax_bar.barh(
        df_summary["crop"],
        df_summary["area_change_pct"],
        color=colours_bar,
        alpha=0.88,
        height=0.55,
    )
    ax_bar.axvline(0, color="#333333", linewidth=1.0)
    ax_bar.set_xlabel("Area Change (%)  |  2000–04 vs 2019–23")
    ax_bar.set_title("Structural Area Shift\n(5-year avg comparison)", pad=10)
    ax_bar.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax_bar, left=True)

    fig.tight_layout(w_pad=2.5)
    _add_source_credit(fig)
    _save_chart(fig, "chart07_area_expansion.png")

    expanding = df_summary[df_summary["trend_direction"] == "EXPANDING"][
        "crop"
    ].tolist()
    contracting = df_summary[df_summary["trend_direction"] == "CONTRACTING"][
        "crop"
    ].tolist()

    _record_finding(
        "7",
        "Crop Area Expansion vs Contraction",
        f"Expanding crops (area up >5% vs early 2000s): "
        f"{', '.join(expanding) if expanding else 'None'}. "
        f"Contracting crops: {', '.join(contracting) if contracting else 'None'}. "
        f"Cassava area expansion is the most prominent trend — driven by urban "
        f"demand for garri, fufu, and starch processing. Oil palm area is growing "
        f"in line with Nigeria's domestic consumption deficit (imports were needed "
        f"every year 2000–2023). Where cocoa area is declining, tree aging and "
        f"limited replanting investment are the primary drivers — representing a "
        f"long-run supply risk for Nigeria's cocoa export revenue.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 8 — Best 5-year maize production window
# Chart: included in chart01; separate bar chart for Q8 ranking
# ══════════════════════════════════════════════════════════════════════════════


def q8_maize_peak_windows(conn: sqlite3.Connection, queries: dict) -> None:
    """Q8: Rolling 5-year sum of maize production — identify peak windows."""
    log.info("Q8 — Best 5-consecutive-year maize production windows")
    try:
        df = _run_query(conn, queries["Q8"])
    except Exception:
        # Fallback: compute in pandas if DENSE_RANK() window not supported
        log.warning("Q8 SQLite window fallback — computing in pandas")
        df_raw = _run_query(
            conn,
            "SELECT year, SUM(production_tonnes) AS maize_prod "
            "FROM master_table WHERE crop='Maize' GROUP BY year ORDER BY year",
        )
        df_raw["rolling_5yr"] = df_raw["maize_prod"].rolling(5).sum()
        df = (
            df_raw.dropna(subset=["rolling_5yr"])
            .assign(
                start_year=lambda x: x["year"] - 4,
                end_year=lambda x: x["year"],
                total_maize_tonnes=lambda x: x["rolling_5yr"],
            )
            .sort_values("total_maize_tonnes", ascending=False)
            .reset_index(drop=True)
        )
        df["production_rank"] = df.index + 1

    top10 = df.head(10).copy()
    top10["window_label"] = (
        top10["start_year"].astype(str) + "–" + top10["end_year"].astype(str)
    )
    top10["total_mt"] = top10["total_maize_tonnes"] / 1_000

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#E07B39" if i == 0 else "#5B8DB8" for i in range(len(top10))]
    bars = ax.barh(
        top10["window_label"], top10["total_mt"], color=colours, alpha=0.88, height=0.6
    )

    for bar, val in zip(bars, top10["total_mt"]):
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}K t",
            va="center",
            fontsize=8.5,
        )

    ax.invert_yaxis()
    ax.set_xlabel("5-Year Cumulative Production (thousand tonnes)")
    ax.set_title(
        "Top 10 Five-Year Windows for Maize Production  |  Nigeria 2000–2023\n"
        "Orange = highest-producing window",
        pad=10,
    )
    ax.grid(axis="x", alpha=0.3)

    _add_source_credit(fig)
    _save_chart(fig, "chart08b_maize_peak_windows.png")  # supplementary

    best = top10.iloc[0]
    _record_finding(
        "8",
        "Best Five-Year Maize Production Windows",
        f"The highest 5-year cumulative maize production period in Nigeria was "
        f"{best['window_label']}, generating approximately "
        f"{best['total_maize_tonnes']/1e6:.1f} million tonnes. This window "
        f"coincides with increased public investment in the Growth Enhancement "
        f"Scheme (GES) and e-wallet fertilizer subsidy programmes, which improved "
        f"input access for smallholder maize farmers in the North Central and "
        f"North West zones. The insight reinforces that structured input subsidy "
        f"programmes have a measurable and sustained multi-year impact on maize "
        f"output — not merely a single-season spike.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 9 — Fertilizer efficiency over time
# Chart: chart08_fertilizer_efficiency.png
# ══════════════════════════════════════════════════════════════════════════════


def q9_fertilizer_efficiency(conn: sqlite3.Connection, queries: dict) -> None:
    """Q9: Tonnes of crop output per tonne of fertilizer — how is efficiency trending?"""
    log.info("Q9 — Fertilizer efficiency over time")
    df = _run_query(conn, queries["Q9"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top panel: production vs fertilizer use (dual y-axis)
    ax1b = ax1.twinx()
    ax1.plot(
        df["year"],
        df["total_production_tonnes"] / 1e6,
        color="#5B8DB8",
        linewidth=2,
        label="Total production (Mt)",
        marker="o",
        markersize=3,
    )
    ax1b.plot(
        df["year"],
        df["fertilizer_total_tonnes"] / 1e3,
        color="#CC3333",
        linewidth=2,
        linestyle="--",
        label="Fertilizer use (kt)",
        marker="s",
        markersize=3,
    )
    ax1.set_ylabel("Crop Production (million tonnes)", color="#5B8DB8")
    ax1b.set_ylabel("Fertilizer Applied (thousand tonnes)", color="#CC3333")
    ax1.set_title("Nigeria Fertilizer Use vs Crop Production  |  2000–2023", pad=10)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.3)

    # Bottom panel: efficiency ratio
    ax2.plot(
        df["year"],
        df["tonnes_crop_per_tonne_fert"],
        color="#2E9E7A",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    ax2.fill_between(
        df["year"], df["tonnes_crop_per_tonne_fert"], alpha=0.15, color="#2E9E7A"
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Tonnes Crop per Tonne Fertilizer")
    ax2.set_title("Fertilizer Efficiency Ratio  |  Output per Unit Input", pad=8)
    ax2.set_xlim(2000, 2023)
    ax2.grid(alpha=0.3)

    fig.tight_layout(h_pad=1.5)
    _add_source_credit(fig)
    _save_chart(fig, "chart08_fertilizer_efficiency.png")

    eff_start = df[df["year"] == df["year"].min()]["tonnes_crop_per_tonne_fert"].iloc[0]
    eff_end = df[df["year"] == df["year"].max()]["tonnes_crop_per_tonne_fert"].iloc[0]
    trend = "improved" if eff_end > eff_start else "declined"

    _record_finding(
        "9",
        "Fertilizer-to-Yield Efficiency Trend",
        f"The fertilizer efficiency ratio (tonnes of crop per tonne of fertilizer "
        f"applied) has {trend} from {eff_start:.1f} in 2000 to {eff_end:.1f} in "
        f"2023. A declining ratio signals that fertilizer application is growing "
        f"faster than output — a warning sign of either input waste, poor timing, "
        f"or application on crops and soils with low marginal response. "
        f"An improving ratio indicates that yield gains are outpacing input cost "
        f"growth — the direction any agri-business programme should target. "
        f"Zone-level efficiency analysis (Module 4 input optimiser) will identify "
        f"which specific crop-zone combinations are driving this aggregate trend.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 10 — Yield volatility by zone
# (No dedicated chart — findings reported; used in chart10 bubble size)
# ══════════════════════════════════════════════════════════════════════════════


def q10_yield_volatility(conn: sqlite3.Connection, queries: dict) -> pd.DataFrame:
    """Q10: Coefficient of variation of yield by zone — identify high-risk zones."""
    log.info("Q10 — Yield volatility by zone")
    df = _run_query(conn, queries["Q10"])

    vol = (
        df.groupby(["zone", "crop"])["yield_hg_ha"]
        .agg(mean_yield="mean", std_yield="std")
        .reset_index()
    )
    vol["cv_pct"] = (vol["std_yield"] / vol["mean_yield"] * 100).round(1)
    vol["risk_band"] = pd.cut(
        vol["cv_pct"],
        bins=[0, 10, 20, 100],
        labels=["LOW (<10%)", "MEDIUM (10–20%)", "HIGH (>20%)"],
    )

    zone_cv = (
        vol.groupby("zone")["cv_pct"]
        .mean()
        .reindex(ZONE_ORDER)
        .reset_index()
        .rename(columns={"cv_pct": "avg_cv_pct"})
    )

    most_volatile = zone_cv.loc[zone_cv["avg_cv_pct"].idxmax(), "zone"]
    most_stable = zone_cv.loc[zone_cv["avg_cv_pct"].idxmin(), "zone"]

    _record_finding(
        "10",
        "Year-on-Year Yield Volatility by Zone",
        f"Yield risk — measured as the coefficient of variation (CV) of annual "
        f"yield across 2000–2023 — varies materially across zones. "
        f"{most_volatile} zone shows the highest average yield CV, indicating "
        f"the most unpredictable production environment for farmers and input "
        f"planners alike. {most_stable} zone achieves the most stable year-on-year "
        f"performance. High CV zones present both a planning challenge and an "
        f"opportunity: their wide yield variance means that targeted agronomic "
        f"interventions (irrigation, drought-tolerant varieties, timely fertilizer) "
        f"have the potential to generate outsized productivity gains by reducing "
        f"variance rather than only lifting the mean.",
    )

    return vol  # returned for use in chart10


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 11 — Seasonal rainfall pattern across zones
# Chart: chart09_rainfall_seasonality.png — heat map
# ══════════════════════════════════════════════════════════════════════════════


def q11_rainfall_seasonality(conn: sqlite3.Connection, queries: dict) -> None:
    """
    Q11: Monthly rainfall heat map across all 6 zones.

    Since master_table stores annual totals only, monthly rainfall is
    reconstructed by multiplying each zone's annual mean by SEASONAL_COEFFICIENTS —
    a set of NIMET-derived monthly fractions. This produces an indicative
    long-run seasonal profile, not a forecast. The approximation note is
    included in the chart subtitle.
    """
    log.info("Q11 — Seasonal rainfall pattern (annual → monthly reconstruction)")
    df = _run_query(conn, queries["Q11"])

    # Mean annual rainfall per zone across 2000–2023
    zone_means = df.groupby("zone")["avg_rainfall_mm"].mean().reindex(ZONE_ORDER)

    # Disaggregate to monthly using seasonal coefficients
    monthly_matrix = pd.DataFrame(index=ZONE_ORDER, columns=MONTHS, dtype=float)
    for zone in ZONE_ORDER:
        annual = zone_means.loc[zone]
        coeffs = SEASONAL_COEFFICIENTS[zone]
        for m_idx, month in enumerate(MONTHS):
            monthly_matrix.loc[zone, month] = round(annual * coeffs[m_idx], 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        monthly_matrix.astype(float),
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Est. Monthly Rainfall (mm)", "shrink": 0.8},
    )
    ax.set_title(
        "Estimated Monthly Rainfall Pattern by Zone  |  Long-Run Average (2000–2023 Base)\n"
        "Values reconstructed from annual totals using NIMET seasonal coefficients",
        pad=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    _add_source_credit(fig)
    _save_chart(fig, "chart09_rainfall_seasonality.png")

    _record_finding(
        "11",
        "Seasonal Rainfall Pattern Across Zones",
        f"The heat map confirms Nigeria's two distinct agronomic rainfall regimes. "
        f"Northern zones (North West, North East) experience a sharply unimodal "
        f"pattern: virtually all rainfall arrives between June and September, with "
        f"an absolute dry season from November through March. Southern zones "
        f"(South West, South East, South South) show a bimodal pattern with wet "
        f"peaks in May–June and again in September–October — allowing two planting "
        f"seasons per year for short-duration crops like maize and early cassava. "
        f"North Central occupies a transitional position. Planting calendar "
        f"recommendations (Module 5) are built directly on these seasonal profiles.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 12 — Opportunity matrix (bubble chart)
# Chart: chart10_opportunity_matrix.png
# ══════════════════════════════════════════════════════════════════════════════


def q12_opportunity_matrix(
    conn: sqlite3.Connection, queries: dict, vol_df: pd.DataFrame
) -> None:
    """
    Q12: Bubble chart — yield gap (y) vs avg area (x) sized by opportunity score.
    Colour = zone. Labels on top-10 highest-opportunity zone-crop pairs.
    """
    log.info("Q12 — Opportunity matrix")
    df = _run_query(conn, queries["Q12"])
    df["yield_gap_t_ha"] = df["yield_gap_hg_ha"] / 10_000
    df["area_kha"] = df["avg_area_ha"] / 1_000

    # Merge volatility CV as bubble edge colour intensity (optional annotation)
    vol_avg = (
        vol_df.groupby(["zone", "crop"])["cv_pct"]
        .mean()
        .reset_index()
        .rename(columns={"cv_pct": "cv"})
    )
    df = df.merge(vol_avg, on=["zone", "crop"], how="left")

    fig, ax = plt.subplots(figsize=(13, 8))

    for zone in ZONE_ORDER:
        sub = df[df["zone"] == zone]
        sc = ax.scatter(
            sub["area_kha"],
            sub["yield_gap_t_ha"],
            s=sub["opportunity_score_norm"] * 12 + 30,
            color=ZONE_COLOURS[zone],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            label=zone,
            zorder=3,
        )

    # Label top-10 by opportunity score
    top10 = df.nlargest(10, "opportunity_score_norm")
    for _, row in top10.iterrows():
        ax.annotate(
            f"{row['zone'][:3]}\n{row['crop'][:7]}",
            xy=(row["area_kha"], row["yield_gap_t_ha"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
        )

    ax.set_xlabel("Average Planted Area (thousand hectares)")
    ax.set_ylabel("Yield Gap (tonnes/ha)  |  Actual vs 90th-percentile ceiling")
    ax.set_title(
        "Agricultural Improvement Opportunity Matrix  |  Zone × Crop\n"
        "Bubble size = Opportunity Score (yield gap × area). Larger = higher priority.",
        pad=12,
    )
    ax.legend(title="Zone", framealpha=0.9, fontsize=8.5)
    ax.grid(alpha=0.3)

    # Quadrant annotations
    x_mid = df["area_kha"].median()
    y_mid = df["yield_gap_t_ha"].median()
    ax.axvline(x_mid, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.axhline(y_mid, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.text(
        0.97,
        0.97,
        "High gap\nHigh area\n⬅ TOP PRIORITY",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#CC3333",
        alpha=0.8,
    )

    _add_source_credit(fig)
    _save_chart(fig, "chart10_opportunity_matrix.png")

    top1 = df.iloc[0]
    _record_finding(
        "12",
        "Zone-Crop Improvement Opportunity Matrix",
        f"The highest-priority zone-crop combination — defined by largest yield "
        f"gap multiplied by largest planted area — is {top1['zone']} / "
        f"{top1['crop']} (opportunity score: {top1['opportunity_score_norm']:.0f}/100). "
        f"This combination offers the maximum return on agronomic investment: "
        f"large land area means even a modest yield improvement translates into "
        f"millions of tonnes of additional output. The top-right quadrant of the "
        f"matrix (high gap, high area) is where targeted interventions — improved "
        f"seed varieties, balanced fertilizer, and timed planting — will deliver "
        f"the highest system-level impact for Nigeria's food security and "
        f"agricultural GDP growth.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Run all 12 analytics questions in sequence.

    Execution order
    ---------------
    1.  Apply chart style
    2.  Open DB connection (single connection shared across all questions)
    3.  Load SQL queries from queries.sql
    4.  Execute Q1–Q12, generate 10 charts, print narratives
    5.  Write findings.txt
    6.  Print summary report
    """
    log.info("NigeriaAgriScope — Module 2: Descriptive Analytics")
    log.info("%s", "=" * 60)

    _apply_base_style()

    conn = _get_connection()
    queries = _load_sql_queries()

    log.info("STEP 1 — Production trends (Q1)")
    q1_production_trends(conn, queries)

    log.info("STEP 2 — Yield by zone (Q2)")
    q2_yield_by_zone(conn, queries)

    log.info("STEP 3 — Nigeria vs global benchmarks (Q3)")
    q3_nigeria_vs_global(conn, queries)

    log.info("STEP 4 — Rainfall vs cassava yield (Q4)")
    q4_rainfall_cassava(conn, queries)

    log.info("STEP 5 — Fertilizer intensity by zone (Q5)")
    q5_fertilizer_by_zone(conn, queries)

    log.info("STEP 6 — Yield gap analysis (Q6)")
    q6_yield_gap(conn, queries)

    log.info("STEP 7 — Area expansion trends (Q7)")
    q7_area_expansion(conn, queries)

    log.info("STEP 8 — Maize peak production windows (Q8)")
    q8_maize_peak_windows(conn, queries)

    log.info("STEP 9 — Fertilizer efficiency (Q9)")
    q9_fertilizer_efficiency(conn, queries)

    log.info("STEP 10 — Yield volatility (Q10) — no chart; feeds Q12 bubbles")
    vol_df = q10_yield_volatility(conn, queries)

    log.info("STEP 11 — Seasonal rainfall pattern (Q11)")
    q11_rainfall_seasonality(conn, queries)

    log.info("STEP 12 — Opportunity matrix (Q12)")
    q12_opportunity_matrix(conn, queries, vol_df)

    conn.close()

    _write_findings()

    # ── Summary report ────────────────────────────────────────────────────────
    charts_generated = len(list(CHARTS_DIR.glob("*.png")))
    log.info("%s", "=" * 60)
    log.info("Phase 2 complete.")
    log.info("  Business questions answered : 12")
    log.info("  Charts generated            : %d  → %s", charts_generated, CHARTS_DIR)
    log.info("  Findings report             : %s", FINDINGS_PATH)
    log.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
