"""
pages/page5_forecast.py
========================
Forecast & Planning — forward-looking yield forecasts and planting calendar.

Module 4 (ML models) and Module 5 (planning engine) do NOT yet exist.
This page renders gracefully with stub content when those files are absent.

Layout:
  ROW 1: Conditional M4 availability banner
  ROW 2: Trend-based production outlook (numpy.polyfit, always shown)
  ROW 3: Planting calendar (M5 CSV if available, else hardcoded stub)
  ROW 4: Input requirements per hectare (stub values)
  ROW 5: Operational timeline Gantt (static — Plotly horizontal bars)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import CROP_ORDER
from utils.metrics import compute_national_production

# ── M4 / M5 file presence checks ─────────────────────────────────────────────
_MODEL_PATH = Path("module4_models/outputs/yield_model.pkl")
_CALENDAR_PATH = Path("module5_planning/outputs/planting_calendar_all_zones.csv")
M4_AVAILABLE = _MODEL_PATH.exists()
M5_AVAILABLE = _CALENDAR_PATH.exists()

# ── Forecast crops ────────────────────────────────────────────────────────────
FORECAST_CROPS = ["Cassava", "Oil palm fruit", "Maize"]

CROP_COLOURS = {
    "Cassava": "#1F77B4",
    "Oil palm fruit": "#FF7F0E",
    "Maize": "#D62728",
}

# ── Hardcoded planting calendar stub (Part 7, Unit 8) ────────────────────────
PLANTING_STUB = [
    {
        "Zone": "North West",
        "Crop": "Maize",
        "Plant": "May",
        "Harvest": "Aug",
        "Risk": "Low",
    },
    {
        "Zone": "North West",
        "Crop": "Sorghum",
        "Plant": "Jun",
        "Harvest": "Oct",
        "Risk": "Low",
    },
    {
        "Zone": "North East",
        "Crop": "Sorghum",
        "Plant": "Jun",
        "Harvest": "Oct",
        "Risk": "Medium",
    },
    {
        "Zone": "North Central",
        "Crop": "Yam",
        "Plant": "Feb",
        "Harvest": "Oct",
        "Risk": "Low",
    },
    {
        "Zone": "North Central",
        "Crop": "Maize",
        "Plant": "Apr",
        "Harvest": "Jul",
        "Risk": "Low",
    },
    {
        "Zone": "South West",
        "Crop": "Cassava",
        "Plant": "Apr",
        "Harvest": "Mar+",
        "Risk": "Low",
    },
    {
        "Zone": "South West",
        "Crop": "Cocoa beans",
        "Plant": "Sep",
        "Harvest": "Nov",
        "Risk": "Medium",
    },
    {
        "Zone": "South East",
        "Crop": "Oil palm fruit",
        "Plant": "Mar",
        "Harvest": "Dec",
        "Risk": "Low",
    },
    {
        "Zone": "South East",
        "Crop": "Cassava",
        "Plant": "Apr",
        "Harvest": "Mar+",
        "Risk": "Low",
    },
    {
        "Zone": "South South",
        "Crop": "Oil palm fruit",
        "Plant": "Mar",
        "Harvest": "Dec",
        "Risk": "Low",
    },
    {
        "Zone": "South South",
        "Crop": "Cassava",
        "Plant": "May",
        "Harvest": "Apr+",
        "Risk": "Low",
    },
]

# ── Input requirements stub (Part 7, Unit 8 — ROW 4) ─────────────────────────
INPUT_STUB = [
    {
        "Crop": "Cassava",
        "Fert. (kg/ha)": 80,
        "Seed units/ha": 10_000,
        "Labour days/ha": 45,
        "Risk": "Low",
    },
    {
        "Crop": "Oil palm fruit",
        "Fert. (kg/ha)": 150,
        "Seed units/ha": 136,
        "Labour days/ha": 60,
        "Risk": "Low",
    },
    {
        "Crop": "Yam",
        "Fert. (kg/ha)": 90,
        "Seed units/ha": 12_000,
        "Labour days/ha": 75,
        "Risk": "Low",
    },
    {
        "Crop": "Maize",
        "Fert. (kg/ha)": 120,
        "Seed units/ha": 25_000,
        "Labour days/ha": 35,
        "Risk": "Low",
    },
    {
        "Crop": "Rice (paddy)",
        "Fert. (kg/ha)": 110,
        "Seed units/ha": 80,
        "Labour days/ha": 50,
        "Risk": "Medium",
    },
    {
        "Crop": "Sorghum",
        "Fert. (kg/ha)": 60,
        "Seed units/ha": 15_000,
        "Labour days/ha": 30,
        "Risk": "Medium",
    },
    {
        "Crop": "Cocoa beans",
        "Fert. (kg/ha)": 100,
        "Seed units/ha": 1_100,
        "Labour days/ha": 90,
        "Risk": "Medium",
    },
]

# ── Gantt phase data ──────────────────────────────────────────────────────────
# month offsets (1=Jan). start, duration in months.
_GANTT_PHASES = [
    # crop,              phase,             start, dur, colour
    ("Cassava", "Land Prep", 3, 1, "#AEC6E8"),
    ("Cassava", "Planting", 4, 1, "#1F77B4"),
    ("Cassava", "Growing", 5, 9, "#7FB3D3"),
    ("Cassava", "Harvest", 14, 2, "#0D3D6B"),
    ("Oil palm fruit", "Land Prep", 2, 1, "#FDBE85"),
    ("Oil palm fruit", "Planting", 3, 1, "#FF7F0E"),
    ("Oil palm fruit", "Growing", 4, 8, "#FFC07A"),
    ("Oil palm fruit", "Harvest", 12, 1, "#C45E00"),
    ("Maize", "Land Prep", 3, 1, "#F5A5A8"),
    ("Maize", "Planting", 4, 1, "#D62728"),
    ("Maize", "Growing", 5, 3, "#F08A8B"),
    ("Maize", "Harvest", 8, 1, "#8B0000"),
]

_SOURCE_NOTE = (
    "Sources: FAOSTAT · NASA POWER · World Bank · USDA PSD  |  NigeriaAgriScope"
)


def _hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    """Convert a 6-digit hex colour string to an rgba() string accepted by Plotly."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        font=dict(family="Inter, Arial, sans-serif", size=12),
        title_font=dict(size=14, color="#1a1a1a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1
        ),
        margin=dict(t=60, b=65, l=50, r=30),
        annotations=[
            dict(
                text=_SOURCE_NOTE,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.18,
                font=dict(size=9, color="#888888"),
                showarrow=False,
            )
        ],
    )
    return fig


def render(df: pd.DataFrame) -> None:
    st.title("📅 Forecast & Planning")

    # ── ROW 1: M4 availability banner ─────────────────────────────────────────
    if not M4_AVAILABLE:
        st.warning(
            "⚙️ **ML yield forecasts will appear here after Module 4** "
            "(`module4_models/yield_model.py`) has been trained. "
            "Run that script to generate `module4_models/outputs/yield_model.pkl`."
        )
    else:
        st.success("✅ Module 4 model detected — ML forecasts active.")

    # ── ROW 2: Trend-based production outlook ─────────────────────────────────
    st.subheader("Trend-Based Production Outlook  |  Linear Extrapolation 2024–2026")
    st.caption(
        "Simple linear trend extrapolation from 2000–2023 national production data "
        "(numpy.polyfit, degree=1). Shaded band = ±1 std dev of historical residuals. "
        "XGBoost yield model forecasts will replace this when Module 4 is trained."
    )

    nat = compute_national_production(df)
    fig_forecast = go.Figure()

    for crop in FORECAST_CROPS:
        c_data = nat[nat["crop"] == crop].sort_values("year")
        years = c_data["year"].values
        prod = c_data["national_production_tonnes"].values

        # Fit linear trend
        m, b = np.polyfit(years, prod, 1)
        residuals = prod - (m * years + b)
        std_res = residuals.std()

        # Historical line
        fig_forecast.add_trace(
            go.Scatter(
                x=years,
                y=prod,
                mode="lines",
                name=f"{crop} (actual)",
                line=dict(color=CROP_COLOURS[crop], width=2),
                hovertemplate=f"{crop} actual<br>%{{x}}: %{{y:,.0f}} t",
            )
        )

        # Forecast 2024–2026
        f_years = np.array([2024, 2025, 2026])
        f_pred = m * f_years + b
        f_upper = f_pred + std_res
        f_lower = f_pred - std_res

        # Confidence band (filled area)
        fig_forecast.add_trace(
            go.Scatter(
                x=np.concatenate([f_years, f_years[::-1]]),
                y=np.concatenate([f_upper, f_lower[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(CROP_COLOURS[crop], alpha=0.14),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{crop} CI",
            )
        )

        # Forecast line
        fig_forecast.add_trace(
            go.Scatter(
                x=f_years,
                y=f_pred,
                mode="lines+markers",
                name=f"{crop} (forecast)",
                line=dict(color=CROP_COLOURS[crop], dash="dash", width=1.5),
                marker=dict(symbol="circle-open", size=7),
                hovertemplate=f"{crop} forecast<br>%{{x}}: %{{y:,.0f}} t",
            )
        )

    fig_forecast.add_vline(
        x=2023.5,
        line_dash="dot",
        line_color="#888",
        opacity=0.6,
        annotation_text="Forecast →",
        annotation_position="top right",
    )
    _apply_layout(
        fig_forecast, "National Production Forecast — Cassava · Oil Palm · Maize"
    )
    fig_forecast.update_yaxes(title_text="Production (tonnes)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── ROW 3: Planting calendar ──────────────────────────────────────────────
    st.subheader("Optimal Planting Windows")
    if not M5_AVAILABLE:
        st.info(
            "📅 Full optimised planting windows will appear here after Module 5 "
            "(`module5_planning/planting_calendar.py`) has been run. "
            "Showing agronomic literature defaults below."
        )
        cal_df = pd.DataFrame(PLANTING_STUB)

        def _risk_style(val):
            return {
                "High": "color: #CC3333",
                "Medium": "color: #F0A500",
                "Low": "color: #2E9E7A",
            }.get(val, "")

        st.dataframe(
            cal_df.style.map(_risk_style, subset=["Risk"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        cal_df = pd.read_csv(_CALENDAR_PATH)
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

    # ── ROW 4: Input requirements ─────────────────────────────────────────────
    st.subheader("Estimated Input Requirements per Hectare")
    st.caption("Agronomic standard values — full calibration pending Module 5.")
    st.dataframe(
        pd.DataFrame(INPUT_STUB).style.format(
            {
                "Fert. (kg/ha)": "{:.0f}",
                "Seed units/ha": "{:,}",
                "Labour days/ha": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── ROW 5: Operational timeline Gantt ────────────────────────────────────
    st.subheader("Operational Timeline  |  Cassava · Oil Palm · Maize")
    st.caption(
        "Exact dates generated by Module 5 operations_schedule.py — illustrative schedule shown below."
    )

    # Build Gantt using Plotly horizontal bars (go.Bar with base parameter)
    # px.timeline requires datetime objects — using numeric months is simpler
    # and avoids year-wrapping issues for crops with >12-month cycles.
    month_labels = [
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
        "Jan+",
        "Feb+",
        "Mar+",
        "Apr+",
    ]

    fig_gantt = go.Figure()
    gantt_crops = list(dict.fromkeys(p[0] for p in _GANTT_PHASES))  # preserve order

    for crop, phase, start, dur, colour in _GANTT_PHASES:
        y_pos = gantt_crops.index(crop)
        fig_gantt.add_trace(
            go.Bar(
                x=[dur],
                base=[start - 1],
                y=[crop],
                orientation="h",
                name=phase,
                marker_color=colour,
                text=phase,
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"{crop} — {phase}<br>Months {start}–{start+dur-1}",
                showlegend=False,
            )
        )

    fig_gantt.update_layout(
        template="plotly_white",
        title="Indicative Crop Cycle Calendar  |  Month Offsets from Jan",
        barmode="overlay",
        font=dict(family="Inter, Arial, sans-serif", size=11),
        title_font=dict(size=14, color="#1a1a1a"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(16)),
            ticktext=month_labels[:16],
            title="Month",
        ),
        yaxis=dict(title="Crop", autorange="reversed"),
        margin=dict(t=60, b=50, l=120, r=30),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        annotations=[
            dict(
                text=_SOURCE_NOTE + " · Agronomic literature",
                xref="paper",
                yref="paper",
                x=0,
                y=-0.12,
                font=dict(size=9, color="#888888"),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig_gantt, use_container_width=True)
