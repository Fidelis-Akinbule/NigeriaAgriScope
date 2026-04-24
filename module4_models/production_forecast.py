"""
NigeriaAgriScope — Module 4: Production Volume Forecast
========================================================
Uses Meta's Prophet library to generate 3-year national production forecasts
(2024–2026) for Oil palm fruit, Cassava, and Maize.

Why Prophet for this task
-------------------------
Prophet is designed for business time series with the following properties —
all of which apply here:
  - Moderate length (24 annual data points, 2000–2023)
  - Possible trend changepoints (Nigeria's cassava boom ~2005, COVID dip ~2020)
  - Yearly seasonality not meaningful at annual grain (disabled deliberately)
  - Irregular outliers that should not dictate the long-run trend

Prophet's additive model (trend + seasonality + error) produces interpretable
confidence intervals, which the dashboard renders as shaded forecast bands.
This matches the "Trend-Based Production Outlook" chart already stubbed on
Page 5 of the Streamlit dashboard.

Why national-level only (not zone-level)
-----------------------------------------
Zone-level Prophet would fit 7 crops × 6 zones = 42 separate models on
series of only 24 points each. Short per-zone series amplify noise and produce
unreliable confidence intervals. National totals are smoother, more consistent
with published references, and are what Page 5 currently visualises.

Train/forecast window
---------------------
  Historical : 2000–2023  (24 annual data points)
  Forecast   : 2024, 2025, 2026

Outputs (all → module4_models/outputs/)
  production_forecast_oil_palm_fruit.csv
  production_forecast_cassava.csv
  production_forecast_maize.csv
  chart04_palm_oil_forecast.png
  chart05_cassava_forecast.png
  — chart06 (input optimisation curves) is produced by input_optimizer.py

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
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m4_production_forecast")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "nigeria_agri.db"
CSV_PATH = ROOT / "data" / "processed" / "master_table.csv"
OUT_DIR = ROOT / "module4_models" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Forecast config ────────────────────────────────────────────────────────
FORECAST_CROPS = ["Oil palm fruit", "Cassava", "Maize"]
FORECAST_HORIZON_YEARS = 3  # 2024, 2025, 2026
HISTORICAL_START = 2000
HISTORICAL_END = 2023

# Prophet model settings — tuned for short annual agricultural series.
# changepoint_prior_scale=0.15 allows moderate flexibility in trend direction
# changes (e.g., the 2020 COVID disruption) without overfitting to single-year
# outliers. yearly_seasonality=False: annual data has no within-year seasonality
# by definition. weekly_seasonality and daily_seasonality are always False for
# annual data.
PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.15,
    "seasonality_prior_scale": 0.1,
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "interval_width": 0.80,  # 80% confidence interval
    "uncertainty_samples": 1000,
}

# Chart colour map — consistent with Page 5 Streamlit line chart colours
CROP_COLOURS = {
    "Oil palm fruit": "#E05C2A",  # orange-red (matches M3 palette)
    "Cassava": "#2E75B6",  # blue
    "Maize": "#C00000",  # dark red
}


# ── Data loading ──────────────────────────────────────────────────────────


def load_national_production() -> pd.DataFrame:
    """
    Load master_table and aggregate to national annual production per crop.

    Aggregation: sum zone-level production_tonnes across all 6 zones.
    This reconstructs the FAOSTAT national total (conservation was validated
    in Module 1 validate_master() with ZONE_CONSERVATION_TOL=2%).
    """
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM master_table", conn)
        log.info("Loaded from SQLite: %d rows", len(df))
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        log.info("Loaded from CSV: %d rows", len(df))
    else:
        raise FileNotFoundError(
            f"Neither {DB_PATH} nor {CSV_PATH} found. Run Module 1 first."
        )

    df["year"] = df["year"].astype(int)
    df["production_tonnes"] = pd.to_numeric(df["production_tonnes"], errors="coerce")

    national = (
        df[df["crop"].isin(FORECAST_CROPS)]
        .groupby(["crop", "year"], as_index=False)["production_tonnes"]
        .sum()
    )
    log.info(
        "National production assembled: %d crop-year rows for %s",
        len(national),
        FORECAST_CROPS,
    )
    return national


# ── Prophet series preparation ────────────────────────────────────────────


def prepare_prophet_df(national: pd.DataFrame, crop: str) -> pd.DataFrame:
    """
    Convert national production series for one crop into Prophet's required
    format: columns 'ds' (datetime) and 'y' (value).

    Prophet requires datetime 'ds'. Annual data is mapped to Jan 1 of each
    year — the exact date within the year is irrelevant for annual-grain
    forecasting; Prophet's trend component operates at the year level.

    Log-transform of 'y':
    Agricultural production series are right-skewed (large outlier years pull
    the trend up). Log-transforming 'y' before fitting and inverse-transforming
    forecasts produces more symmetric residuals and tighter confidence intervals.
    The inverse transform is applied before saving outputs.
    """
    crop_df = national[national["crop"] == crop].copy()
    crop_df = crop_df[
        (crop_df["year"] >= HISTORICAL_START) & (crop_df["year"] <= HISTORICAL_END)
    ].sort_values("year")

    prophet_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(crop_df["year"].astype(str) + "-01-01"),
            "y": np.log1p(crop_df["production_tonnes"].values),  # log1p for stability
            "y_raw": crop_df["production_tonnes"].values,  # kept for diagnostics
        }
    )
    log.info(
        "  [%s] %d data points, production range: %s – %s tonnes",
        crop,
        len(prophet_df),
        f"{crop_df['production_tonnes'].min():,.0f}",
        f"{crop_df['production_tonnes'].max():,.0f}",
    )
    return prophet_df


# ── Prophet fitting ────────────────────────────────────────────────────────


def fit_prophet(prophet_df: pd.DataFrame, crop: str) -> tuple[Prophet, pd.DataFrame]:
    """
    Fit a Prophet model and generate 3-year forecasts.

    Cross-validation note: with only 24 annual data points, formal Prophet
    cross-validation (prophet.diagnostics.cross_validation) requires at least
    3 data points in the initial training window and 1 in each validation
    horizon. We run it with initial='14 years', period='2 years', horizon='3 years'
    giving 4 cutpoints. Results are logged for transparency but not used to
    tune the model (sample too small for reliable hyperparameter selection).
    """
    model = Prophet(**PROPHET_CONFIG)
    # Fit on log-transformed 'y' — 'y_raw' column is ignored by Prophet
    fit_df = prophet_df[["ds", "y"]].copy()
    model.fit(fit_df)

    # Generate future dates: 3 years beyond the last historical data point
    last_year = prophet_df["ds"].dt.year.max()
    future_dates = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [f"{last_year + i}-01-01" for i in range(1, FORECAST_HORIZON_YEARS + 1)]
            )
        }
    )
    future = pd.concat([fit_df[["ds"]], future_dates], ignore_index=True)
    forecast = model.predict(future)

    # Inverse log-transform all prediction columns
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[f"{col}_tonnes"] = np.expm1(forecast[col]).clip(lower=0)

    # Run cross-validation for logging (suppressed on failure — small dataset)
    try:
        df_cv = cross_validation(
            model,
            initial="5114 days",  # ~14 years
            period="730 days",  # ~2 years
            horizon="1095 days",  # ~3 years
            disable_tqdm=True,
        )
        pm = performance_metrics(df_cv)
        log.info(
            "  [%s] CV MAPE: %.1f%%  RMSE: %s tonnes",
            crop,
            pm["mape"].mean() * 100,
            f"{pm['rmse'].mean():,.0f}",
        )
    except Exception as exc:
        log.warning("  [%s] CV skipped: %s", crop, exc)

    log.info(
        "  [%s] Forecast %d–%d: %s tonnes",
        crop,
        last_year + 1,
        last_year + FORECAST_HORIZON_YEARS,
        " / ".join(
            f"{int(v):,}"
            for v in forecast[forecast["ds"].dt.year > last_year]["yhat_tonnes"]
        ),
    )
    return model, forecast


# ── Save forecast CSVs ────────────────────────────────────────────────────


def save_forecast_csv(
    forecast: pd.DataFrame,
    prophet_df: pd.DataFrame,
    crop: str,
) -> Path:
    """
    Save a clean forecast CSV consumable by Page 5 of the dashboard.

    Columns:
      year, type (actual/forecast), production_tonnes,
      lower_80ci_tonnes, upper_80ci_tonnes
    """
    last_hist_year = prophet_df["ds"].dt.year.max()

    rows = []

    # Historical actuals (inverse-transformed back to raw tonnes for reference)
    for _, row in prophet_df.iterrows():
        rows.append(
            {
                "year": row["ds"].year,
                "type": "actual",
                "production_tonnes": round(np.expm1(row["y"])),
                "lower_80ci_tonnes": None,
                "upper_80ci_tonnes": None,
            }
        )

    # Forecast rows only
    fc_rows = forecast[forecast["ds"].dt.year > last_hist_year]
    for _, row in fc_rows.iterrows():
        rows.append(
            {
                "year": row["ds"].year,
                "type": "forecast",
                "production_tonnes": round(row["yhat_tonnes"]),
                "lower_80ci_tonnes": round(row["yhat_lower_tonnes"]),
                "upper_80ci_tonnes": round(row["yhat_upper_tonnes"]),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    safe_name = crop.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = OUT_DIR / f"production_forecast_{safe_name}.csv"
    df_out.to_csv(out_path, index=False)
    log.info("  CSV saved → %s", out_path.name)
    return out_path


# ── Charts ────────────────────────────────────────────────────────────────


def plot_forecast(
    forecast: pd.DataFrame,
    prophet_df: pd.DataFrame,
    crop: str,
    chart_path: Path,
) -> None:
    """
    chart04 / chart05 — Prophet forecast with 80% confidence band.

    Layout mirrors the Page 5 Streamlit chart: actual line (solid),
    forecast line (dashed), shaded confidence band. Vertical dashed line
    marks the train/forecast boundary.
    """
    colour = CROP_COLOURS.get(crop, "#2E75B6")
    last_hist_year = prophet_df["ds"].dt.year.max()

    hist_years = prophet_df["ds"].dt.year.values
    hist_vals = np.expm1(prophet_df["y"].values)

    fc_all = forecast.copy()
    fc_all["year_num"] = fc_all["ds"].dt.year

    fc_hist = fc_all[fc_all["year_num"] <= last_hist_year]
    fc_future = fc_all[fc_all["year_num"] > last_hist_year]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Historical actuals
    ax.plot(
        hist_years,
        hist_vals / 1e6,
        color=colour,
        lw=2.0,
        label="Actual production",
        zorder=3,
    )
    ax.scatter(hist_years, hist_vals / 1e6, color=colour, s=28, zorder=4)

    # Prophet fitted values over historical window
    ax.plot(
        fc_hist["year_num"],
        fc_hist["yhat_tonnes"] / 1e6,
        color=colour,
        lw=1.2,
        linestyle=":",
        alpha=0.6,
        label="Prophet fitted (historical)",
        zorder=2,
    )

    # Forecast line
    # Connect last actual to first forecast for visual continuity
    connect_x = [last_hist_year, fc_future["year_num"].iloc[0]]
    connect_y = [hist_vals[-1] / 1e6, fc_future["yhat_tonnes"].iloc[0] / 1e6]
    ax.plot(connect_x, connect_y, color=colour, lw=1.5, linestyle="--", alpha=0.5)

    ax.plot(
        fc_future["year_num"],
        fc_future["yhat_tonnes"] / 1e6,
        color=colour,
        lw=2.0,
        linestyle="--",
        label=f"Forecast 2024–2026",
        zorder=3,
    )
    ax.scatter(
        fc_future["year_num"],
        fc_future["yhat_tonnes"] / 1e6,
        color=colour,
        s=60,
        marker="D",
        zorder=5,
    )

    # 80% confidence band (forecast years only)
    ax.fill_between(
        fc_future["year_num"],
        fc_future["yhat_lower_tonnes"] / 1e6,
        fc_future["yhat_upper_tonnes"] / 1e6,
        color=colour,
        alpha=0.15,
        label="80% confidence interval",
    )

    # Train/forecast boundary line
    ax.axvline(last_hist_year + 0.5, color="grey", lw=1.0, linestyle="--", alpha=0.7)
    ax.text(
        last_hist_year + 0.6,
        ax.get_ylim()[1] * 0.95,
        "← Historical  |  Forecast →",
        fontsize=9,
        color="grey",
        va="top",
    )

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Production (million tonnes)", fontsize=11)
    ax.set_title(
        f"NigeriaAgriScope — {crop} National Production Forecast\n"
        f"Prophet Model | 2000–2023 Historical | 2024–2026 Forecast | 80% CI",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.text(
        0.99,
        0.01,
        "Source: FAOSTAT, NASA POWER, World Bank, USDA PSD | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )

    # Annotate forecast values
    for _, row in fc_future.iterrows():
        ax.annotate(
            f"{row['yhat_tonnes'] / 1e6:.1f}M",
            xy=(row["year_num"], row["yhat_tonnes"] / 1e6),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colour,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", chart_path.name)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 4: Production Forecast (Prophet)")
    log.info("=" * 60)

    log.info("STEP 1 — Load and aggregate national production data")
    national = load_national_production()

    chart_paths = {
        "Oil palm fruit": OUT_DIR / "chart04_palm_oil_forecast.png",
        "Cassava": OUT_DIR / "chart05_cassava_forecast.png",
    }

    all_forecasts: dict[str, pd.DataFrame] = {}

    for crop in FORECAST_CROPS:
        log.info("─" * 50)
        log.info("Processing: %s", crop)

        log.info("  STEP 2a — Prepare Prophet DataFrame")
        prophet_df = prepare_prophet_df(national, crop)

        log.info("  STEP 2b — Fit Prophet model and generate forecast")
        model, forecast = fit_prophet(prophet_df, crop)

        log.info("  STEP 2c — Save forecast CSV")
        save_forecast_csv(forecast, prophet_df, crop)

        log.info("  STEP 2d — Generate forecast chart")
        if crop in chart_paths:
            plot_forecast(forecast, prophet_df, crop, chart_paths[crop])
        else:
            # Maize: save chart even though not in spec's named chart list
            maize_path = OUT_DIR / "chart05b_maize_forecast.png"
            plot_forecast(forecast, prophet_df, crop, maize_path)

        all_forecasts[crop] = forecast

    log.info("=" * 60)
    log.info("Module 4 — production_forecast.py COMPLETE")
    log.info("  CSVs and charts saved to: module4_models/outputs/")
    log.info("  Crops forecasted: %s", ", ".join(FORECAST_CROPS))
    log.info("  Forecast years: 2024, 2025, 2026")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
