"""
pages/page4_climate.py
=======================
Climate Intelligence — rainfall trends, drought risk, yield-climate correlations.

Layout:
  ROW 1: 3 metric cards
  ROW 2: Rainfall multi-line with per-zone 85% drought threshold
  ROW 3: Pearson correlation heatmap + temperature area chart
  ROW 4: Drought risk bar (count of events by zone)
  ROW 5: Drought event log table

Technical note: numpy.polyfit only — statsmodels NOT used (Streamlit Cloud
install issues). All trend lines built from np.polyfit.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Climate & Forecasts", page_icon="🌦")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import ZONE_ORDER

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
_CLIMATE_COLS = [
    "rainfall_mm_annual",
    "temp_avg_celsius",
    "humidity_pct",
    "solar_radiation",
    "fertilizer_kg_ha",
]
_CLIMATE_LABELS = {
    "rainfall_mm_annual": "Rainfall",
    "temp_avg_celsius": "Temperature",
    "humidity_pct": "Humidity",
    "solar_radiation": "Solar Rad.",
    "fertilizer_kg_ha": "Fertilizer",
}


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
    st.title("🌦️ Climate Intelligence")

    # ── Sidebar filter ────────────────────────────────────────────────────────
    selected_zones = st.sidebar.multiselect(
        "Select Zones", ZONE_ORDER, default=ZONE_ORDER
    )
    if not selected_zones:
        st.warning("Select at least one zone.")
        return

    zone_df = df[df["zone"].isin(selected_zones)]
    latest_year = int(df["year"].max())

    # Distinct zone-year rows (rainfall is broadcast across crops)
    climate_zy = df.drop_duplicates(subset=["zone", "year"])

    # ── ROW 1: Metric cards ───────────────────────────────────────────────────
    recent_5_zy = climate_zy[climate_zy["year"] >= latest_year - 4]
    zone_rain_5 = recent_5_zy.groupby("zone")["rainfall_mm_annual"].mean()
    wettest = zone_rain_5.idxmax()
    driest = zone_rain_5.idxmin()

    # Drought events 2019–2023 across ALL zones
    drought_recent = (
        (
            df[(df["year"] >= latest_year - 4) & (df["drought_flag"] == 1)]
            .drop_duplicates(subset=["zone", "year"])
            .shape[0]
        )
        if "drought_flag" in df.columns
        else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Wettest Zone (5-yr avg)", wettest, f"{zone_rain_5[wettest]:.0f} mm/yr")
    c2.metric("Driest Zone (5-yr avg)", driest, f"{zone_rain_5[driest]:.0f} mm/yr")
    c3.metric(
        f"Drought Events {latest_year-4}–{latest_year}",
        str(drought_recent),
        "Zone-years below 85% of historical mean",
    )

    st.divider()

    # ── ROW 2: Rainfall multi-line with per-zone drought threshold ────────────
    rain_zy = climate_zy[climate_zy["zone"].isin(selected_zones)].sort_values("year")

    # Compute per-zone 85% threshold
    zone_thresholds = (
        climate_zy.groupby("zone")["rainfall_mm_annual"]
        .mean()
        .mul(0.85)
        .rename("threshold_mm")
    )

    fig_rain = go.Figure()
    for zone in selected_zones:
        z_data = rain_zy[rain_zy["zone"] == zone]
        fig_rain.add_trace(
            go.Scatter(
                x=z_data["year"],
                y=z_data["rainfall_mm_annual"],
                mode="lines",
                name=zone,
                line=dict(color=ZONE_COLOURS.get(zone, "#888888")),
                hovertemplate=f"{zone}<br>%{{x}}: %{{y:.0f}} mm",
            )
        )
        threshold = zone_thresholds.get(zone)
        if threshold is not None:
            fig_rain.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=ZONE_COLOURS.get(zone, "#888888"),
                opacity=0.4,
                annotation_text=f"{zone[:2]} 85%",
                annotation_position="right",
                annotation_font_size=9,
            )
    _apply_layout(
        fig_rain, "Annual Rainfall by Zone  |  2000–2023  (dashed = drought threshold)"
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    # ── ROW 3: Pearson correlation heatmap + temperature area chart ───────────
    col_left, col_right = st.columns(2)

    with col_left:
        # Compute Pearson r between yield_hg_ha and each climate/input column,
        # grouped by crop. Use zone-year grain to avoid over-inflating r from
        # cross-zone repetition of broadcast climate values.
        corr_records = []
        for crop_name, crop_grp in df.groupby("crop"):
            for col in _CLIMATE_COLS:
                if col in crop_grp.columns:
                    valid = crop_grp[["yield_hg_ha", col]].dropna()
                    if len(valid) > 5:
                        r = valid["yield_hg_ha"].corr(valid[col])
                        corr_records.append(
                            {"Crop": crop_name, "Driver": _CLIMATE_LABELS[col], "r": r}
                        )

        corr_df = pd.DataFrame(corr_records)
        if not corr_df.empty:
            corr_pivot = corr_df.pivot(index="Crop", columns="Driver", values="r")
            fig_heat = px.imshow(
                corr_pivot,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                title="Climate & Input Drivers of Yield — Pearson r",
                labels={"color": "Pearson r"},
            )
            fig_heat.update_layout(
                template="plotly_white",
                font=dict(family="Inter, Arial, sans-serif", size=11),
                title_font=dict(size=13, color="#1a1a1a"),
                margin=dict(t=60, b=40, l=120, r=30),
                coloraxis_colorbar=dict(title="r"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Insufficient data for correlation matrix.")

    with col_right:
        temp_zy = climate_zy[climate_zy["zone"].isin(selected_zones)].sort_values(
            "year"
        )
        fig_temp = go.Figure()
        for zone in selected_zones:
            z_temp = temp_zy[temp_zy["zone"] == zone]
            hex_colour = ZONE_COLOURS.get(zone, "#888888")
            fig_temp.add_trace(
                go.Scatter(
                    x=z_temp["year"],
                    y=z_temp["temp_avg_celsius"],
                    mode="lines",
                    name=zone,
                    fill="tozeroy",
                    line=dict(color=hex_colour),
                    fillcolor=_hex_to_rgba(hex_colour, alpha=0.18),
                    hovertemplate=f"{zone}<br>%{{x}}: %{{y:.1f}} °C",
                )
            )
        _apply_layout(fig_temp, "Mean Annual Temperature by Zone  |  2000–2023")
        fig_temp.update_yaxes(title_text="Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

    # ── ROW 4: Drought risk events bar ────────────────────────────────────────
    if "drought_flag" in df.columns:
        drought_counts = (
            df[df["drought_flag"] == 1]
            .drop_duplicates(subset=["zone", "year"])
            .groupby("zone")
            .size()
            .reindex(ZONE_ORDER, fill_value=0)
            .reset_index()
            .rename(columns={0: "drought_events", "index": "zone"})
        )
        # Rename columns properly after reset_index
        drought_counts.columns = ["zone", "drought_events"]

        def _risk_colour(n):
            if n > 5:
                return "#CC3333"
            if n >= 3:
                return "#F0A500"
            return "#2E9E7A"

        colours = [_risk_colour(n) for n in drought_counts["drought_events"]]
        fig_drought = go.Figure(
            go.Bar(
                x=drought_counts["zone"],
                y=drought_counts["drought_events"],
                marker_color=colours,
                text=drought_counts["drought_events"],
                textposition="outside",
                hovertemplate="%{x}: %{y} drought years",
            )
        )
        _apply_layout(
            fig_drought,
            "Drought Risk Events by Zone  |  2000–2023  (years below 85% of mean)",
        )
        fig_drought.update_layout(showlegend=False, yaxis_title="Drought Event Count")
        st.plotly_chart(fig_drought, use_container_width=True)

    # ── ROW 5: Drought event log ──────────────────────────────────────────────
    st.subheader("Drought Event Log")
    if "drought_flag" in df.columns and "zone_mean_rainfall" in df.columns:
        # One row per zone-year drought event (not per zone-crop-year)
        event_log = (
            df[df["drought_flag"] == 1]
            .drop_duplicates(subset=["zone", "year"])[
                ["zone", "year", "rainfall_mm_annual", "zone_mean_rainfall"]
            ]
            .copy()
        )
        event_log["pct_of_normal"] = (
            event_log["rainfall_mm_annual"] / event_log["zone_mean_rainfall"] * 100
        ).round(1)
        # Attach crops grown in each zone from the spec
        zone_crops = (
            df.groupby("zone")["crop"]
            .apply(lambda x: ", ".join(sorted(x.unique())))
            .rename("affected_crops")
        )
        event_log = event_log.join(zone_crops, on="zone")
        event_log = event_log[event_log["zone"].isin(selected_zones)]
        st.dataframe(
            event_log.rename(
                columns={
                    "zone": "Zone",
                    "year": "Year",
                    "rainfall_mm_annual": "Actual Rainfall (mm)",
                    "zone_mean_rainfall": "Zone Mean (mm)",
                    "pct_of_normal": "% of Normal",
                    "affected_crops": "Crops in Zone",
                }
            )
            .sort_values(["Zone", "Year"])
            .style.format(
                {
                    "Actual Rainfall (mm)": "{:.0f}",
                    "Zone Mean (mm)": "{:.0f}",
                    "% of Normal": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Drought flags not yet computed. Ensure enrich() was called in app.py.")
