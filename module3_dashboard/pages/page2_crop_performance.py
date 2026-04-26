"""
pages/page2_crop_performance.py
================================
Crop Performance — deep-dive into any crop across all zones.
Benchmark gaps. Top / bottom zone performers.

Layout:
  ROW 1: 3 metric cards
  ROW 2: Yield by zone over time (multi-line)
  ROW 3: Benchmark bar (Nigeria vs world) + Area harvested by zone (line)
  ROW 4: Top 3 / Bottom 3 zone tables
  ROW 5: Benchmark methodology expander
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Yield Analysis", page_icon="📈")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import CROP_ORDER, ZONE_ORDER
from utils.metrics import GLOBAL_BENCHMARKS_HG_HA, compute_zone_ranking

ZONE_COLOURS = {
    "North West": "#66C2A5",
    "North East": "#FC8D62",
    "North Central": "#8DA0CB",
    "South West": "#E78AC3",
    "South East": "#A6D854",
    "South South": "#FFD92F",
}
POSITIVE_COLOUR = "#2E9E7A"
NEGATIVE_COLOUR = "#CC3333"
_SOURCE_NOTE = (
    "Sources: FAOSTAT · NASA POWER · World Bank · USDA PSD  |  NigeriaAgriScope"
)


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
    st.title("🌾 Crop Performance")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    selected_crop = st.sidebar.selectbox("Select Crop", CROP_ORDER, index=0)
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider("Year Range", min_yr, max_yr, (min_yr, max_yr))

    crop_df = df[
        (df["crop"] == selected_crop)
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    ]

    # ── ROW 1: Metric cards ───────────────────────────────────────────────────
    avg_yield_t = crop_df["yield_t_ha"].mean()
    world_hg = GLOBAL_BENCHMARKS_HG_HA.get(selected_crop, {}).get("World average")
    gap_vs_world = (
        ((avg_yield_t * 10_000 - world_hg) / world_hg * 100) if world_hg else None
    )
    ranked = compute_zone_ranking(df, selected_crop, year_range)
    top_zone = ranked.iloc[0]["zone"] if not ranked.empty else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Average Yield", f"{avg_yield_t:.2f} t/ha", f"{year_range[0]}–{year_range[1]}"
    )
    c2.metric(
        "Gap vs World Average",
        f"{gap_vs_world:+.1f}%" if gap_vs_world is not None else "N/A",
        delta_color="inverse" if gap_vs_world and gap_vs_world < 0 else "normal",
    )
    c3.metric("Leading Zone", top_zone)

    st.divider()

    # ── ROW 2: Yield by zone over time ────────────────────────────────────────
    fig_lines = px.line(
        crop_df,
        x="year",
        y="yield_t_ha",
        color="zone",
        color_discrete_map=ZONE_COLOURS,
        labels={"yield_t_ha": "Yield (t/ha)", "year": "Year", "zone": "Zone"},
        category_orders={"zone": ZONE_ORDER},
        title=f"{selected_crop} — Yield by Zone  |  {year_range[0]}–{year_range[1]}",
    )
    fig_lines.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f} t/ha")
    _apply_layout(fig_lines)
    st.plotly_chart(fig_lines, use_container_width=True)

    # ── ROW 3: Benchmark bar + area chart ─────────────────────────────────────
    col_left, col_right = st.columns([55, 45])

    with col_left:
        benchmarks = GLOBAL_BENCHMARKS_HG_HA.get(selected_crop, {})
        nigeria_avg_hg = crop_df["yield_hg_ha"].mean()
        bench_data = {"Nigeria (avg)": nigeria_avg_hg}
        bench_data.update(benchmarks)
        bench_df = pd.DataFrame(
            [{"Label": k, "Yield (t/ha)": v / 10_000} for k, v in bench_data.items()]
        )
        colour_map = {
            lbl: (
                "#D62728"
                if "Nigeria" in lbl
                else "#5B8DB8" if "World" in lbl else "#2E9E7A"
            )
            for lbl in bench_df["Label"]
        }
        fig_bench = px.bar(
            bench_df,
            x="Yield (t/ha)",
            y="Label",
            orientation="h",
            color="Label",
            color_discrete_map=colour_map,
            title=f"{selected_crop} — Nigeria vs Global Benchmarks",
        )
        fig_bench.update_traces(hovertemplate="%{y}: %{x:.2f} t/ha", showlegend=False)
        fig_bench.update_layout(yaxis={"categoryorder": "total ascending"})
        _apply_layout(fig_bench)
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_right:
        area_df = crop_df.groupby(["year", "zone"])["area_ha"].sum().reset_index()
        fig_area = px.line(
            area_df,
            x="year",
            y="area_ha",
            color="zone",
            color_discrete_map=ZONE_COLOURS,
            labels={"area_ha": "Area (ha)", "year": "Year", "zone": "Zone"},
            category_orders={"zone": ZONE_ORDER},
            title=f"{selected_crop} — Area Harvested by Zone",
        )
        fig_area.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:,.0f} ha")
        _apply_layout(fig_area)
        st.plotly_chart(fig_area, use_container_width=True)

    # ── ROW 4: Top 3 / Bottom 3 tables ───────────────────────────────────────
    col_t, col_b = st.columns(2)
    nat_avg = crop_df["yield_t_ha"].mean()

    with col_t:
        st.markdown("**🏆 Top 3 Zones**")
        top3 = ranked.head(3).copy()
        top3["% vs National Avg"] = (
            (top3["avg_yield_t_ha"] - nat_avg) / nat_avg * 100
        ).map("{:+.1f}%".format)
        st.dataframe(
            top3[["zone", "avg_yield_t_ha", "rank", "% vs National Avg"]]
            .rename(
                columns={
                    "zone": "Zone",
                    "avg_yield_t_ha": "Avg Yield (t/ha)",
                    "rank": "Rank",
                }
            )
            .style.format({"Avg Yield (t/ha)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.markdown("**⚠️ Bottom 3 Zones**")
        bot3 = ranked.tail(3).copy()
        bot3["% vs National Avg"] = (
            (bot3["avg_yield_t_ha"] - nat_avg) / nat_avg * 100
        ).map("{:+.1f}%".format)
        st.dataframe(
            bot3[["zone", "avg_yield_t_ha", "rank", "% vs National Avg"]]
            .rename(
                columns={
                    "zone": "Zone",
                    "avg_yield_t_ha": "Avg Yield (t/ha)",
                    "rank": "Rank",
                }
            )
            .style.format({"Avg Yield (t/ha)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── ROW 5: Methodology expander ───────────────────────────────────────────
    with st.expander("📊 Benchmark Methodology"):
        st.markdown(
            """
**Potential yield ceiling — 90th percentile**

The yield gap displayed on this page is computed against the **90th percentile**
of historical yield_hg_ha for each crop across all six zones and all 24 years
(2000–2023). This choice is deliberate:

- Using the **maximum** yield introduces outlier sensitivity — a single
  exceptional season in one zone inflates the gap for all other zones.
- The **90th percentile** represents a *domestically achievable ceiling* —
  it has been reached by at least one zone-year combination in the dataset,
  making it a credible near-term policy target.

**Global benchmarks**

World-average yields are sourced from FAOSTAT national statistics, averaged
over the 2018–2022 period to smooth inter-annual variability. Comparator
country benchmarks (Malaysia for Oil Palm, Thailand for Cassava, USA for Maize
and Sorghum, etc.) represent best-practice peers selected on the basis of
production scale and agro-climatic comparability where possible.

**Data sources:** FAOSTAT QCL (area, production, yield) · NASA POWER (climate)
· World Bank WDI (macro) · USDA PSD (palm oil, cassava supply/demand)
            """
        )
