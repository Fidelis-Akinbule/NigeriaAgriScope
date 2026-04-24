"""
pages/page3_inputs.py
======================
Input Efficiency — fertilizer efficiency and input-output relationships by zone.

Layout:
  ROW 1: 3 metric cards
  ROW 2: N/P/K stacked bar by zone
  ROW 3: Scatter (fert_kg_ha × yield_t_ha) + efficiency ratio trend line
  ROW 4: ROI recommendation table

⚠ DATA CONTRACT:
  fertilizer_kg_ha      = zone-crop intensity   ← USE THIS for input-efficiency
  wb_fertilizer_kg_ha   = WB national aggregate ← do NOT use for zone-level analysis
  fertilizer_total_kg   = broadcast per year    ← NEVER sum across rows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import CROP_ORDER, ZONE_ORDER
from utils.metrics import (
    RECOMMENDED_FERT_KG_HA,
    compute_fertilizer_efficiency,
    compute_roi_table,
)

ZONE_COLOURS = {
    "North West": "#66C2A5",
    "North East": "#FC8D62",
    "North Central": "#8DA0CB",
    "South West": "#E78AC3",
    "South East": "#A6D854",
    "South South": "#FFD92F",
}
_SOURCE_NOTE = (
    "Sources: FAOSTAT · NASA POWER · World Bank · USDA PSD  |  NigeriaAgriScope"
)
POSITIVE_COLOUR = "#2E9E7A"
NEGATIVE_COLOUR = "#CC3333"
NEUTRAL_COLOUR = "#F0A500"


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
    st.title("🧪 Input Efficiency")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    zone_options = ["All Zones"] + ZONE_ORDER
    crop_options = ["All Crops"] + CROP_ORDER
    selected_zone = st.sidebar.selectbox("Select Zone", zone_options)
    selected_crop = st.sidebar.selectbox("Select Crop", crop_options)

    filtered = df.copy()
    if selected_zone != "All Zones":
        filtered = filtered[filtered["zone"] == selected_zone]
    if selected_crop != "All Crops":
        filtered = filtered[filtered["crop"] == selected_crop]

    latest_year = int(df["year"].max())
    recent_5 = filtered[filtered["year"] >= latest_year - 4]

    # ── ROW 1: Metric cards ───────────────────────────────────────────────────
    # Card 1: national avg fertilizer intensity (zone-crop level) — latest 5-yr
    avg_fert_intensity = recent_5["fertilizer_kg_ha"].mean()

    # Card 2: gap vs 150 kg/ha palm oil best-practice benchmark
    oil_palm_rec = RECOMMENDED_FERT_KG_HA.get("Oil palm fruit", 150.0)
    palm_fert = recent_5[recent_5["crop"] == "Oil palm fruit"][
        "fertilizer_kg_ha"
    ].mean()
    palm_gap = oil_palm_rec - palm_fert if not np.isnan(palm_fert) else None

    # Card 3: efficiency trend direction
    eff_df = compute_fertilizer_efficiency(df)
    eff_last = eff_df.sort_values("year").tail(5)["fert_efficiency_ratio"]
    m_eff, _ = np.polyfit(range(len(eff_last)), eff_last, 1)
    trend_arrow = "↑ Improving" if m_eff > 0 else "↓ Declining"
    trend_colour = POSITIVE_COLOUR if m_eff > 0 else NEGATIVE_COLOUR

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Avg Fertilizer Intensity",
        f"{avg_fert_intensity:.1f} kg/ha",
        "Latest 5-year avg (zone-crop level)",
    )
    c2.metric(
        "Oil Palm vs Best Practice",
        f"{palm_fert:.1f} kg/ha applied" if palm_gap is not None else "N/A",
        f"{palm_gap:+.0f} kg/ha vs 150 kg/ha rec." if palm_gap is not None else "",
        delta_color="inverse",
    )
    c3.metric("Efficiency Trend", trend_arrow, "Tonnes crop per tonne fertilizer")

    st.divider()

    # ── ROW 2: N/P/K stacked bar by zone ─────────────────────────────────────
    # Use _zone columns (zone-crop allocated) — NOT national broadcast columns
    npk_zone = (
        filtered.groupby("zone")[
            [
                "fertilizer_n_kg_zone",
                "fertilizer_p_kg_zone",
                "fertilizer_k_kg_zone",
                "area_ha",
            ]
        ]
        .sum()
        .reset_index()
    )
    # Convert to kg/ha intensity
    for col, label in [
        ("fertilizer_n_kg_zone", "N"),
        ("fertilizer_p_kg_zone", "P₂O₅"),
        ("fertilizer_k_kg_zone", "K₂O"),
    ]:
        npk_zone[label] = npk_zone[col] / npk_zone["area_ha"]

    npk_long = npk_zone.melt(
        id_vars="zone",
        value_vars=["N", "P₂O₅", "K₂O"],
        var_name="Nutrient",
        value_name="kg/ha",
    )
    fig_npk = px.bar(
        npk_long,
        x="zone",
        y="kg/ha",
        color="Nutrient",
        color_discrete_map={"N": "#4C9BE8", "P₂O₅": "#F0A500", "K₂O": "#5DBB63"},
        barmode="stack",
        labels={"zone": "Zone", "kg/ha": "Fertilizer Intensity (kg/ha)"},
        category_orders={"zone": ZONE_ORDER},
        title="Fertilizer Intensity by Zone — N / P₂O₅ / K₂O",
    )
    fig_npk.update_traces(hovertemplate="%{fullData.name}: %{y:.1f} kg/ha")
    _apply_layout(fig_npk)
    st.plotly_chart(fig_npk, use_container_width=True)

    # ── ROW 3: Scatter + efficiency ratio ─────────────────────────────────────
    col_left, col_right = st.columns([6, 4])

    with col_left:
        scatter_df = filtered.dropna(subset=["fertilizer_kg_ha", "yield_t_ha"])
        fig_scatter = px.scatter(
            scatter_df,
            x="fertilizer_kg_ha",
            y="yield_t_ha",
            color="zone",
            size="area_ha",
            color_discrete_map=ZONE_COLOURS,
            hover_data={
                "zone": True,
                "crop": True,
                "year": True,
                "yield_t_ha": ":.2f",
                "fertilizer_kg_ha": ":.1f",
                "area_ha": False,
            },
            labels={
                "fertilizer_kg_ha": "Fertilizer Intensity (kg/ha)",
                "yield_t_ha": "Yield (t/ha)",
                "zone": "Zone",
            },
            title="Fertilizer Intensity vs Yield — Zone-Crop-Year",
        )
        # OLS trend line via numpy.polyfit — no statsmodels dependency
        x_vals = scatter_df["fertilizer_kg_ha"].values
        y_vals = scatter_df["yield_t_ha"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=m * x_line + b,
                mode="lines",
                name=f"OLS trend (slope={m:.3f})",
                line=dict(color="#333333", dash="dash", width=1.5),
            )
        )
        _apply_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        fig_eff = px.line(
            eff_df,
            x="year",
            y="fert_efficiency_ratio",
            labels={"fert_efficiency_ratio": "t Crop / t Fertilizer", "year": "Year"},
            title="Fertilizer Efficiency Ratio  |  2000–2023",
        )
        fig_eff.update_traces(
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Year %{x}: %{y:.1f} t/t",
        )
        _apply_layout(fig_eff)
        st.plotly_chart(fig_eff, use_container_width=True)

    # ── ROW 4: ROI recommendation table ──────────────────────────────────────
    st.subheader("Input ROI Recommendations")
    st.caption(
        "Recommended values based on agronomic literature. "
        "Yield uplift estimates assume linear response — actual response varies by soil type. "
        "Full calibration pending Module 5 input calculator."
    )
    roi_df = compute_roi_table(df)
    if selected_zone != "All Zones":
        roi_df = roi_df[roi_df["zone"] == selected_zone]
    if selected_crop != "All Crops":
        roi_df = roi_df[roi_df["crop"] == selected_crop]

    def _colour_priority(val):
        return {
            "High": f"color: {NEGATIVE_COLOUR}; font-weight: bold",
            "Medium": f"color: {NEUTRAL_COLOUR}; font-weight: bold",
            "Low": f"color: {POSITIVE_COLOUR}",
        }.get(str(val), "")

    styled_roi = (
        roi_df.rename(
            columns={
                "zone": "Zone",
                "crop": "Crop",
                "current_fert_kg_ha": "Current (kg/ha)",
                "recommended_fert_kg_ha": "Recommended (kg/ha)",
                "estimated_yield_uplift_t_ha": "Est. Uplift (t/ha)",
                "priority_tier": "Priority",
            }
        )
        .style.format(
            {
                "Current (kg/ha)": "{:.1f}",
                "Recommended (kg/ha)": "{:.0f}",
                "Est. Uplift (t/ha)": "{:.2f}",
            }
        )
        .map(_colour_priority, subset=["Priority"])
    )
    st.dataframe(styled_roi, use_container_width=True, hide_index=True)
