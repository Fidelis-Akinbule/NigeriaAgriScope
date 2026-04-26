"""
pages/page1_command_centre.py
==============================
National Command Centre — executive overview.
High-signal, low-noise first view for any user.

Layout:
  Headline insight banner
  ROW 1: 4 KPI metric cards
  ROW 2: Production trend (line) + latest-year zone production (stacked bar)
  ROW 3: 5-year yield trend for top 3 crops by volume
  ROW 4: Colour-coded zone-crop performance table
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="National Overview", page_icon="🏠")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import CROP_ORDER, ZONE_ORDER
from utils.metrics import (
    GLOBAL_BENCHMARKS_HG_HA,
    compute_national_production,
)

# ── Visual design constants (Part 8) ─────────────────────────────────────────
ZONE_COLOURS = {
    "North West": "#66C2A5",
    "North East": "#FC8D62",
    "North Central": "#8DA0CB",
    "South West": "#E78AC3",
    "South East": "#A6D854",
    "South South": "#FFD92F",
}
CROP_COLOURS = {
    "Cassava": "#1F77B4",
    "Oil palm fruit": "#FF7F0E",
    "Yam": "#2CA02C",
    "Maize": "#D62728",
    "Rice (paddy)": "#9467BD",
    "Sorghum": "#8C564B",
    "Cocoa beans": "#E377C2",
}
POSITIVE_COLOUR = "#2E9E7A"
NEGATIVE_COLOUR = "#CC3333"
NEUTRAL_COLOUR = "#F0A500"

_SOURCE_NOTE = (
    "Sources: FAOSTAT · NASA POWER · World Bank · USDA PSD  |  NigeriaAgriScope"
)

TOP_VOLUME_CROPS = ["Cassava", "Oil palm fruit", "Yam"]


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply standard chart layout (Part 8)."""
    fig.update_layout(
        template="plotly_white",
        title=title,
        font=dict(family="Inter, Arial, sans-serif", size=12),
        title_font=dict(size=14, color="#1a1a1a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1
        ),
        margin=dict(t=60, b=60, l=50, r=30),
        annotations=[
            dict(
                text=_SOURCE_NOTE,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.15,
                font=dict(size=9, color="#888888"),
                showarrow=False,
            )
        ],
    )
    return fig


def _fmt_tonnes(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M t"
    if v >= 1_000:
        return f"{v/1_000:.0f}K t"
    return f"{v:.0f} t"


def render(df: pd.DataFrame) -> None:
    st.title("🏠 National Command Centre")

    latest_year = int(df["year"].max())

    # ── Headline insight banner ───────────────────────────────────────────────
    st.info(
        "📌 **KEY FINDING:** Nigeria's oil palm yield averages ~10.5 t/ha against a world "
        "average of 17 t/ha and Malaysian best practice of 24.5 t/ha — a 38% gap "
        "representing billions in unrealised agricultural value. "
        "See **Crop Performance** for full benchmark analysis."
    )

    # ── ROW 1: KPI Metric Cards ───────────────────────────────────────────────
    latest_df = df[df["year"] == latest_year]

    # Card 1: total national production latest year
    total_prod = latest_df["production_tonnes"].sum()

    # Card 2: oil palm yield gap vs world average
    op_avg_yield = latest_df[latest_df["crop"] == "Oil palm fruit"][
        "yield_hg_ha"
    ].mean()
    world_op = GLOBAL_BENCHMARKS_HG_HA["Oil palm fruit"]["World average"]
    op_gap_pct = (op_avg_yield - world_op) / world_op * 100  # negative = below

    # Card 3: top-performing zone current year (avg yield_t_ha across all crops)
    zone_yield = (
        latest_df.groupby("zone")["yield_t_ha"].mean().sort_values(ascending=False)
    )
    top_zone = zone_yield.index[0]
    top_zone_yield = zone_yield.iloc[0]

    # Card 4: zone-crop combos in decline YoY
    # yoy_yield_change_pct was added by enrich() in app.py
    decline_count = (
        int(df[df["year"] == latest_year]["yoy_yield_change_pct"].lt(0).sum())
        if "yoy_yield_change_pct" in df.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌾 Total Production", _fmt_tonnes(total_prod), f"{latest_year}")
    c2.metric(
        "🌴 Oil Palm vs World Avg",
        f"{op_avg_yield/10_000:.1f} t/ha",
        f"{op_gap_pct:+.1f}% vs world avg",
        delta_color="inverse",
    )
    c3.metric("🏆 Top Zone", top_zone, f"{top_zone_yield:.2f} t/ha avg")
    c4.metric(
        "📉 Combos in Decline",
        str(decline_count),
        delta_color="inverse",
        help="Zone-crop pairs with negative YoY yield change in latest year",
    )

    st.divider()

    # ── ROW 2: Production trend + latest-year zone stacked bar ───────────────
    col_left, col_right = st.columns([6, 4])

    with col_left:
        nat = compute_national_production(df)
        fig_trend = px.line(
            nat,
            x="year",
            y="national_production_tonnes",
            color="crop",
            color_discrete_map=CROP_COLOURS,
            labels={
                "national_production_tonnes": "Production (t)",
                "year": "Year",
                "crop": "Crop",
            },
            category_orders={"crop": CROP_ORDER},
        )
        fig_trend.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:,.0f} t")
        _apply_layout(fig_trend, "National Production by Crop  |  2000–2023")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        zone_prod = (
            latest_df.groupby(["zone", "crop"])["production_tonnes"].sum().reset_index()
        )
        fig_zone = px.bar(
            zone_prod,
            x="production_tonnes",
            y="zone",
            color="crop",
            orientation="h",
            color_discrete_map=CROP_COLOURS,
            labels={
                "production_tonnes": "Production (t)",
                "zone": "Zone",
                "crop": "Crop",
            },
            category_orders={"zone": ZONE_ORDER, "crop": CROP_ORDER},
        )
        fig_zone.update_traces(hovertemplate="%{fullData.name}<br>%{y}: %{x:,.0f} t")
        _apply_layout(fig_zone, f"Production by Zone  |  {latest_year}")
        st.plotly_chart(fig_zone, use_container_width=True)

    # ── ROW 3: 5-year yield trend for top 3 volume crops ─────────────────────
    five_yr_start = latest_year - 4
    trend_df = df[(df["crop"].isin(TOP_VOLUME_CROPS)) & (df["year"] >= five_yr_start)]
    # Average across zones per crop-year for cleaner grouped bar
    trend_agg = trend_df.groupby(["crop", "year"])["yield_t_ha"].mean().reset_index()
    fig_5yr = px.bar(
        trend_agg,
        x="year",
        y="yield_t_ha",
        color="crop",
        barmode="group",
        color_discrete_map=CROP_COLOURS,
        labels={"yield_t_ha": "Avg Yield (t/ha)", "year": "Year", "crop": "Crop"},
        category_orders={"crop": TOP_VOLUME_CROPS},
    )
    fig_5yr.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f} t/ha")
    _apply_layout(
        fig_5yr,
        f"5-Year Yield Trend — Cassava · Oil Palm · Yam  |  {five_yr_start}–{latest_year}",
    )
    st.plotly_chart(fig_5yr, use_container_width=True)

    # ── ROW 4: Performance table with conditional formatting ─────────────────
    st.subheader("Zone–Crop Performance Table")
    perf = (
        latest_df[
            ["zone", "crop", "yield_t_ha", "yoy_yield_change_pct", "drought_flag"]
        ]
        .copy()
        .sort_values("yoy_yield_change_pct", ascending=True)  # worst first
    )
    perf["drought_flag"] = perf["drought_flag"].map({1: "🔴", 0: "✅"})
    perf = perf.rename(
        columns={
            "zone": "Zone",
            "crop": "Crop",
            "yield_t_ha": "Yield (t/ha)",
            "yoy_yield_change_pct": "YoY Change (%)",
            "drought_flag": "Drought",
        }
    )

    def _colour_yoy(val):
        if pd.isna(val):
            return ""
        return f"color: {POSITIVE_COLOUR}" if val > 0 else f"color: {NEGATIVE_COLOUR}"

    styled = perf.style.format(
        {"Yield (t/ha)": "{:.2f}", "YoY Change (%)": "{:+.1f}"}
    ).applymap(_colour_yoy, subset=["YoY Change (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
