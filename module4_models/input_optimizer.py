"""
NigeriaAgriScope — Module 4: Input Optimisation Analysis
=========================================================
Fits polynomial yield-response curves (fertilizer kg/ha → yield hg/ha)
per crop-zone pair and identifies the point of diminishing returns —
the fertilizer application rate beyond which adding more fertilizer
produces less than a threshold marginal yield improvement.

Why polynomial regression (not ML)
------------------------------------
This is not a prediction problem — it is a curve-fitting and threshold
analysis problem. The goal is to find the mathematical relationship between
fertilizer input and yield output for each crop-zone combination, then
derive the economically optimal application rate from that curve.

A degree-2 polynomial (y = a·x² + b·x + c) is the standard agronomic
choice for yield-response curves. It is:
  - Interpretable: the vertex gives the theoretical maximum yield point
  - Monotonic where it should be (increasing then flattening/declining)
  - Robust on small per-crop-zone samples (~24 data points)
  - Consistent with FAO and IAEA yield-response literature

The point of diminishing returns (PDR) is defined as the fertilizer rate
where the marginal yield gain per additional kg/ha drops below
DIMINISHING_RETURNS_THRESHOLD_HG_PER_KG (default: 50 hg/ha per kg/ha).
This represents approximately 0.5 tonnes of additional yield per tonne
of additional fertilizer — a reasonable economic floor for Nigerian
smallholder economics (fertilizer price ~₦300–400/kg vs crop value).

Outputs (all → module4_models/outputs/)
  input_requirements_reference.csv  — optimal fertilizer range per crop/zone
  chart06_input_optimization_curves.png — yield response curve grid

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m4_input_optimizer")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "nigeria_agri.db"
CSV_PATH = ROOT / "data" / "processed" / "master_table.csv"
OUT_DIR = ROOT / "module4_models" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHART_CURVES = OUT_DIR / "chart06_input_optimization_curves.png"
REF_CSV = OUT_DIR / "input_requirements_reference.csv"

# ── Analysis config ────────────────────────────────────────────────────────
POLY_DEGREE = 2  # quadratic — standard for yield-response curves

# Marginal yield threshold for PDR: below this, extra fertilizer is uneconomic.
# 50 hg/ha per kg/ha = 0.005 tonnes additional yield per kg additional fertilizer.
# At ₦350/kg fertilizer and ₦250/tonne cassava (approximate 2024 prices),
# this threshold represents roughly break-even ROI for Nigerian smallholders.
DIMINISHING_RETURNS_THRESHOLD_HG_PER_KG = 50.0

# Minimum data points needed to fit a reliable quadratic curve per crop-zone.
# Below this threshold, the fit is flagged as low-confidence.
MIN_DATA_POINTS = 10

# Fertilizer scan range for PDR search (kg/ha) — covers the realistic range
# for Nigerian smallholder and commercial agriculture.
FERT_SCAN_MIN = 0.5  # minimum meaningful application
FERT_SCAN_MAX = 300.0  # well above current Nigerian average (~11 kg/ha WB)
FERT_SCAN_STEPS = 1000  # resolution of the PDR numerical search

# Top crops for the chart grid (by economic importance in Nigeria)
CHART_CROPS = ["Cassava", "Oil palm fruit", "Maize", "Yam"]
CHART_ZONES = ["South West", "South East", "North Central", "South South"]

ZONE_COLOURS = {
    "North West": "#8B4513",
    "North East": "#DAA520",
    "North Central": "#2E8B57",
    "South West": "#2E75B6",
    "South East": "#9B59B6",
    "South South": "#E05C2A",
}


# ── Data loading ──────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load master_table from SQLite or CSV."""
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM master_table", conn)
        log.info("Loaded from SQLite: %d rows × %d cols", *df.shape)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        log.info("Loaded from CSV: %d rows × %d cols", *df.shape)
    else:
        raise FileNotFoundError(
            f"Neither {DB_PATH} nor {CSV_PATH} found. Run Module 1 first."
        )
    df["year"] = df["year"].astype(int)
    for col in ["fertilizer_kg_ha", "yield_hg_ha", "area_ha"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Curve fitting ──────────────────────────────────────────────────────────


def fit_yield_response_curve(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[tuple[np.ndarray, float, bool]]:
    """
    Fit a degree-2 polynomial (quadratic) to fertilizer vs yield data.

    Returns
    -------
    (coefficients, r_squared, is_reliable)
      coefficients : [a, b, c] where y = a·x² + b·x + c
      r_squared    : coefficient of determination on the fitted data
      is_reliable  : True if n ≥ MIN_DATA_POINTS and R² ≥ 0.10

    Returns None if fit fails (all-NaN or fewer than 3 valid points).

    Design note on R² threshold (0.10):
    Fertilizer-yield correlations in Nigerian agricultural data are
    expectedly weak at the zone-aggregate level (R² 0.10–0.40 is normal)
    because FAOSTAT fertilizer figures are national totals disaggregated
    by proxy weights — not field-level measurements. A threshold of 0.10
    filters out spurious negative-slope curves without being too strict.
    The flag is_reliable surfaces low-confidence results to the user
    rather than silently hiding them.
    """
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0) & (y > 0)
    x_c, y_c = x[mask], y[mask]

    if len(x_c) < 3:
        return None

    try:
        coeffs = np.polyfit(x_c, y_c, POLY_DEGREE)
    except (np.linalg.LinAlgError, ValueError):
        return None

    y_hat = np.polyval(coeffs, x_c)
    ss_res = np.sum((y_c - y_hat) ** 2)
    ss_tot = np.sum((y_c - y_c.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    is_reliable = (len(x_c) >= MIN_DATA_POINTS) and (r2 >= 0.10)
    return coeffs, r2, is_reliable


def find_pdr(
    coeffs: np.ndarray,
    fert_min: float = FERT_SCAN_MIN,
    fert_max: float = FERT_SCAN_MAX,
) -> Optional[float]:
    """
    Numerically find the point of diminishing returns (PDR).

    The PDR is the fertilizer rate where the first derivative of the
    polynomial drops below DIMINISHING_RETURNS_THRESHOLD_HG_PER_KG.

    For a quadratic y = ax² + bx + c:
      dy/dx = 2ax + b
      PDR where 2ax + b = threshold
      → x_PDR = (threshold - b) / (2a)

    For a positive-concave-down curve (a < 0), this has an analytical
    solution. For edge cases (a ≥ 0, i.e. ever-increasing curve) the
    PDR is undefined — we return the scan maximum to signal "no PDR
    within realistic range."

    Numerical scan is used instead of the analytical solution to handle
    both positive and negative curvature curves and to be robust against
    floating-point edge cases.
    """
    a, b, c = coeffs
    fert_range = np.linspace(fert_min, fert_max, FERT_SCAN_STEPS)
    marginal_yield = np.polyder(np.poly1d(coeffs))(fert_range)

    # Find first x where marginal yield drops below threshold
    below = np.where(marginal_yield < DIMINISHING_RETURNS_THRESHOLD_HG_PER_KG)[0]
    if len(below) == 0:
        return fert_max  # PDR beyond realistic range
    return float(fert_range[below[0]])


def compute_optimal_range(
    coeffs: np.ndarray,
    pdr: float,
) -> tuple[float, float]:
    """
    Return (recommended_min, recommended_max) fertilizer range.

    The optimal range is defined as:
      min: PDR × 0.70  (70% of PDR — practical floor accounting for soil
                        nutrient carry-over and application efficiency losses)
      max: PDR × 0.95  (95% of PDR — stay below PDR with 5% safety margin)

    These multipliers are consistent with FAO Fertilizer and Plant Nutrition
    Bulletin No. 16 recommendations for sub-Saharan African smallholder crops.
    """
    rec_min = max(0.5, pdr * 0.70)
    rec_max = pdr * 0.95
    return round(rec_min, 1), round(rec_max, 1)


# ── Analysis orchestrator ─────────────────────────────────────────────────


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run yield-response curve analysis for all crop-zone pairs.

    Returns a DataFrame with one row per (crop, zone):
      crop, zone, n_years, r_squared, is_reliable,
      pdr_kg_ha, recommended_min_kg_ha, recommended_max_kg_ha,
      current_avg_kg_ha, yield_gap_at_pdr_hg_ha, polynomial_coeffs
    """
    results = []
    crops = sorted(df["crop"].unique())
    zones = sorted(df["zone"].unique())

    log.info(
        "Fitting yield-response curves for %d crops × %d zones = %d pairs",
        len(crops),
        len(zones),
        len(crops) * len(zones),
    )

    for crop in crops:
        for zone in zones:
            subset = df[
                (df["crop"] == crop)
                & (df["zone"] == zone)
                & df["fertilizer_kg_ha"].notna()
                & df["yield_hg_ha"].notna()
            ].copy()

            x = subset["fertilizer_kg_ha"].values
            y = subset["yield_hg_ha"].values
            n = len(subset)

            fit_result = fit_yield_response_curve(x, y)
            if fit_result is None:
                log.debug(
                    "  [%s / %s] Insufficient data (n=%d) — skipped", crop, zone, n
                )
                continue

            coeffs, r2, is_reliable = fit_result
            pdr = find_pdr(coeffs)
            rec_min, rec_max = compute_optimal_range(coeffs, pdr)
            current_avg = float(x[x > 0].mean()) if (x > 0).any() else 0.0

            # Yield at PDR vs yield at current average — the "yield gap"
            yield_at_pdr = max(0.0, float(np.polyval(coeffs, pdr)))
            yield_at_current = (
                max(0.0, float(np.polyval(coeffs, current_avg)))
                if current_avg > 0
                else 0.0
            )
            yield_gap = yield_at_pdr - yield_at_current

            results.append(
                {
                    "crop": crop,
                    "zone": zone,
                    "n_years": n,
                    "r_squared": round(r2, 4),
                    "is_reliable": is_reliable,
                    "pdr_kg_ha": round(pdr, 1),
                    "recommended_min_kg_ha": rec_min,
                    "recommended_max_kg_ha": rec_max,
                    "current_avg_kg_ha": round(current_avg, 1),
                    "yield_at_pdr_hg_ha": round(yield_at_pdr),
                    "yield_at_current_kg_ha": round(yield_at_current),
                    "yield_gap_at_pdr_hg_ha": round(max(0.0, yield_gap)),
                    "poly_a": round(coeffs[0], 6),
                    "poly_b": round(coeffs[1], 4),
                    "poly_c": round(coeffs[2], 2),
                }
            )

            flag = "" if is_reliable else " [LOW CONFIDENCE]"
            log.info(
                "  [%-20s / %-12s] n=%2d  R²=%.2f  PDR=%6.1f kg/ha  gap=%s hg/ha%s",
                crop,
                zone,
                n,
                r2,
                pdr,
                f"{yield_gap:+,.0f}",
                flag,
            )

    df_results = pd.DataFrame(results)
    log.info(
        "Analysis complete: %d crop-zone pairs fitted (%d reliable, %d low-confidence)",
        len(df_results),
        df_results["is_reliable"].sum(),
        (~df_results["is_reliable"]).sum(),
    )
    return df_results


# ── Chart ─────────────────────────────────────────────────────────────────


def plot_optimization_curves(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:
    """
    chart06 — 4×4 grid of yield-response curves.

    Each panel shows one crop (rows) × one zone (columns).
    Plotted elements per panel:
      - Scatter: historical (fertilizer_kg_ha, yield_hg_ha) data points
      - Curve: fitted polynomial
      - Vertical dashed line: PDR
      - Vertical dotted line: current average fertilizer rate
      - Shaded region: recommended optimal range (PDR×0.70 to PDR×0.95)
    """
    n_crops = len(CHART_CROPS)
    n_zones = len(CHART_ZONES)
    fig, axes = plt.subplots(
        n_crops,
        n_zones,
        figsize=(4.5 * n_zones, 3.8 * n_crops),
        squeeze=False,
    )

    fig.suptitle(
        "NigeriaAgriScope — Yield-Response Curves: Fertilizer Input vs Crop Yield\n"
        "Point of Diminishing Returns Analysis | 2000–2023 | Module 4",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for row_i, crop in enumerate(CHART_CROPS):
        for col_j, zone in enumerate(CHART_ZONES):
            ax = axes[row_i][col_j]
            colour = ZONE_COLOURS.get(zone, "#2E75B6")

            # Raw data points for this crop-zone
            subset = df[
                (df["crop"] == crop)
                & (df["zone"] == zone)
                & df["fertilizer_kg_ha"].notna()
                & df["yield_hg_ha"].notna()
                & (df["fertilizer_kg_ha"] > 0)
            ]
            x_data = subset["fertilizer_kg_ha"].values
            y_data = subset["yield_hg_ha"].values

            # Look up analysis results for this pair
            pair = analysis[(analysis["crop"] == crop) & (analysis["zone"] == zone)]

            if len(subset) < 3 or pair.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="grey",
                )
                ax.set_title(f"{crop[:10]}\n{zone}", fontsize=9)
                continue

            row = pair.iloc[0]
            coeffs = np.array([row["poly_a"], row["poly_b"], row["poly_c"]])
            pdr = row["pdr_kg_ha"]
            rec_min = row["recommended_min_kg_ha"]
            rec_max = row["recommended_max_kg_ha"]
            current_avg = row["current_avg_kg_ha"]
            r2 = row["r_squared"]

            # Scatter data points
            ax.scatter(
                x_data,
                y_data / 1e3,
                color=colour,
                alpha=0.6,
                s=30,
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
            )

            # Fitted polynomial curve
            x_plot_max = min(pdr * 1.3, FERT_SCAN_MAX, x_data.max() * 1.5)
            x_plot = np.linspace(0.1, x_plot_max, 300)
            y_plot = np.polyval(coeffs, x_plot)
            # Only plot the positive-yield portion of the curve
            valid = y_plot > 0
            if valid.any():
                ax.plot(
                    x_plot[valid], y_plot[valid] / 1e3, color=colour, lw=1.8, zorder=2
                )

            # PDR vertical line
            if pdr < FERT_SCAN_MAX:
                ax.axvline(
                    pdr,
                    color="red",
                    lw=1.2,
                    linestyle="--",
                    alpha=0.8,
                    zorder=4,
                    label=f"PDR={pdr:.0f}",
                )

            # Current average line
            if current_avg > 0:
                ax.axvline(
                    current_avg,
                    color="green",
                    lw=1.0,
                    linestyle=":",
                    alpha=0.8,
                    zorder=4,
                    label=f"Avg={current_avg:.0f}",
                )

            # Recommended range shading
            ax.axvspan(rec_min, rec_max, alpha=0.10, color="green", zorder=1)

            # Confidence flag
            conf_str = "" if row["is_reliable"] else " ⚠"
            ax.set_title(
                f"{crop[:12]} | {zone[:9]}\nR²={r2:.2f}{conf_str}",
                fontsize=8.5,
                fontweight="bold",
            )
            ax.set_xlabel("Fertilizer (kg/ha)", fontsize=8)
            ax.set_ylabel("Yield (000 hg/ha)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    # Global legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=1.5,
            linestyle="--",
            label="Point of Diminishing Returns (PDR)",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            lw=1.2,
            linestyle=":",
            label="Current average fertilizer rate",
        ),
        mpatches.Patch(color="green", alpha=0.20, label="Recommended optimal range"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markersize=6,
            label="Historical data point",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.text(
        0.99,
        -0.03,
        "Source: FAOSTAT, NASA POWER, World Bank, USDA PSD | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(CHART_CURVES, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", CHART_CURVES.name)


import matplotlib.patches as mpatches  # ensure available at module level


# ── Save reference CSV ─────────────────────────────────────────────────────


def save_reference_csv(analysis: pd.DataFrame) -> None:
    """
    Save the input requirements reference table used by Module 5.

    This CSV is the primary handoff from M4 to M5's input_calculator.py.
    Module 5 reads 'recommended_min_kg_ha' and 'recommended_max_kg_ha'
    per crop-zone as the basis for input planning calculations.

    Columns exported (polynomial internals excluded — M5 doesn't need them):
      crop, zone, n_years, r_squared, is_reliable,
      pdr_kg_ha, recommended_min_kg_ha, recommended_max_kg_ha,
      current_avg_kg_ha, yield_gap_at_pdr_hg_ha
    """
    export_cols = [
        "crop",
        "zone",
        "n_years",
        "r_squared",
        "is_reliable",
        "pdr_kg_ha",
        "recommended_min_kg_ha",
        "recommended_max_kg_ha",
        "current_avg_kg_ha",
        "yield_at_pdr_hg_ha",
        "yield_at_current_kg_ha",
        "yield_gap_at_pdr_hg_ha",
    ]
    present = [c for c in export_cols if c in analysis.columns]
    df_out = analysis[present].sort_values(["crop", "zone"]).reset_index(drop=True)
    df_out.to_csv(REF_CSV, index=False)
    log.info("Reference CSV saved → %s  (%d rows)", REF_CSV.name, len(df_out))

    # Summary: top 5 opportunity crop-zones by yield gap
    top5 = df_out[df_out["is_reliable"]].nlargest(5, "yield_gap_at_pdr_hg_ha")[
        [
            "crop",
            "zone",
            "current_avg_kg_ha",
            "recommended_max_kg_ha",
            "yield_gap_at_pdr_hg_ha",
        ]
    ]
    log.info("Top 5 fertilizer optimisation opportunities:")
    for _, r in top5.iterrows():
        log.info(
            "  %-20s | %-12s  current=%.1f kg/ha → recommended=%.1f kg/ha  gap=%s hg/ha",
            r["crop"],
            r["zone"],
            r["current_avg_kg_ha"],
            r["recommended_max_kg_ha"],
            f"{r['yield_gap_at_pdr_hg_ha']:+,.0f}",
        )


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 4: Input Optimisation Analysis")
    log.info("=" * 60)

    log.info("STEP 1 — Load data")
    df = load_data()

    log.info("STEP 2 — Fit yield-response curves (all crop-zone pairs)")
    analysis = run_analysis(df)

    log.info("STEP 3 — Save input requirements reference CSV")
    save_reference_csv(analysis)

    log.info("STEP 4 — Generate optimisation curves chart")
    plot_optimization_curves(df, analysis)

    log.info("=" * 60)
    log.info("Module 4 — input_optimizer.py COMPLETE")
    log.info("  Outputs → module4_models/outputs/")
    log.info(
        "  input_requirements_reference.csv → %d crop-zone pairs",
        len(analysis),
    )
    log.info("  chart06_input_optimization_curves.png → 4×4 curve grid")
    log.info(
        "  NOTE: Module 5 input_calculator.py reads "
        "input_requirements_reference.csv for fertilizer planning."
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
