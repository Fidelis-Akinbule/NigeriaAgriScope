"""
module3_dashboard/app.py
========================
NigeriaAgriScope — entry point.

Design decisions:
  - load_data() + enrich() called ONCE here; enriched df passed to every
    page's render(df). Avoids redundant cache misses on page switches.
  - st.sidebar.radio used for navigation (not st.navigation) — broadest
    version compatibility across Streamlit 1.33–1.35 (Part 7, Unit 1).
  - Page modules imported at top level; each exposes a render(df) function.
"""

import streamlit as st

from utils.data_loader import load_data
from utils.metrics import enrich

import pages.page1_command_centre as p1
import pages.page2_crop_performance as p2
import pages.page3_inputs as p3
import pages.page4_climate as p4
import pages.page5_forecast as p5

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NigeriaAgriScope | Agricultural Intelligence",
    layout="wide",
    page_icon="🌱",
)

# ── Load and enrich data once ─────────────────────────────────────────────────
# enrich() adds: yield_t_ha, yield_gap cols, yoy_change, drought_flag,
# opportunity_score_norm — all pages receive a fully prepared DataFrame.
df = enrich(load_data())

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 NigeriaAgriScope")
    st.caption("Agricultural Performance Intelligence System")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏠 National Command Centre",
            "🌾 Crop Performance",
            "🧪 Input Efficiency",
            "🌦️ Climate Intelligence",
            "📅 Forecast & Planning",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "**Data sources:** FAOSTAT · NASA POWER · World Bank · USDA PSD\n\n"
        "**Author:** Fidelis Akinbule\n\n"
        "Data coverage: 2000–2023"
    )

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🏠 National Command Centre":
    p1.render(df)
elif page == "🌾 Crop Performance":
    p2.render(df)
elif page == "🧪 Input Efficiency":
    p3.render(df)
elif page == "🌦️ Climate Intelligence":
    p4.render(df)
elif page == "📅 Forecast & Planning":
    p5.render(df)
