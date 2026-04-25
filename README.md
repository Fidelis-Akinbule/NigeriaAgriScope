# NigeriaAgriScope 🌾

**[Live Dashboard →](https://nigeriaagriscope.streamlit.app)** &nbsp;|&nbsp; Nigeria's first open-source agricultural intelligence system — yield forecasting, input optimisation, and planting calendars for 7 crops across 6 geopolitical zones, built on 24 years of FAOSTAT, NASA, and World Bank data.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%3D0.991-189AB4?style=flat)
![Prophet](https://img.shields.io/badge/Prophet-2024--2026-4285F4?style=flat)
![Power BI](https://img.shields.io/badge/Power%20BI-Ready-F2C811?style=flat&logo=powerbi&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)

---

## The Problem

Nigeria is the world's largest cassava producer and Africa's largest food market — yet it produces only 1.2 tonnes of maize per hectare against a global average of 5.8 tonnes. That gap is not a soil problem. It is an information problem. Smallholder farmers lack access to zone-specific data on when to plant, how much fertilizer to apply, and what yield to expect. The result: ₦2.3 trillion in annual yield losses from suboptimal input timing and application rates.

## The Solution

NigeriaAgriScope closes the information gap by integrating four international data sources into a unified agricultural intelligence platform:

- **Yield forecasting** — an XGBoost model trained on 24 years of zone-level data predicts next-year crop yield with R² = 0.991, enabling evidence-based input investment decisions
- **Input optimisation** — polynomial yield-response curves identify the point of diminishing returns for fertilizer application across all 42 crop-zone combinations, translating to ₦-per-hectare cost savings
- **Planting calendars** — NASA POWER rainfall onset analysis generates zone-specific planting windows, reducing late-planting yield penalties
- **Production forecasting** — Prophet time-series models project national cassava, oil palm, and maize output to 2026, informing supply chain planning

## Live Demo

**Dashboard:** [https://nigeriaagriscope.streamlit.app](https://nigeriaagriscope.streamlit.app)

Five dashboard pages:
1. **National Overview** — production trends, zone comparison, crop ranking
2. **Yield Analysis** — historical yield by zone, fertilizer efficiency, climate correlation
3. **Input Planner** — fertilizer recommendations, cost breakdown, diminishing returns curves
4. **Forecasts** — XGBoost yield predictions, Prophet production outlook 2024–2026
5. **Planning Calendar** — zone-specific planting windows, operations schedule, risk profile

---

## Key Findings

> Data-driven insights from 24 years (2000–2023) of FAOSTAT, NASA POWER, World Bank, and USDA PSD records across 6 Nigerian geopolitical zones and 7 crops.

1. **South South leads oil palm yield but is underutilising its fertilizer potential.** The zone applies an average of 9.1 kg/ha of fertilizer against a computed point of diminishing returns of 95 kg/ha — a 10× gap representing an estimated +153 hg/ha yield uplift at optimal application rates.

2. **North Central yam faces the largest absolute yield gap.** Optimal fertilizer application (172 kg/ha) could unlock +30,538 hg/ha above current yield levels — the highest single crop-zone improvement opportunity in the dataset.

3. **Rainfall onset is perfectly predictable in the South East and North West.** Both zones recorded a reliability score of 1.00 across 24 years, meaning planting calendar adherence alone (March for South East, June for North West) is sufficient to eliminate rainfall-timing risk.

4. **The XGBoost model captures 99.1% of yield variance** (R² = 0.991, MAE = 2,900 hg/ha on the 2019–2023 test set), with lagged yield and rolling 3-year rainfall as the dominant predictors — confirming that yield momentum and multi-year climate trends, not single-season inputs, drive Nigerian agricultural output.

---

## Architecture

```
NigeriaAgriScope/
│
├── module1_pipeline/
│   └── generate_data.py          # FAOSTAT + NASA POWER + WB + USDA → SQLite
│
├── module2_analytics/
│   └── analytics.py              # 12 SQL queries + 10 exploratory charts
│
├── module3_dashboard/
│   ├── app.py                    # Streamlit entry point
│   └── pages/
│       ├── page1_overview.py
│       ├── page2_yield.py
│       ├── page3_inputs.py
│       ├── page4_forecast.py
│       └── page5_planning.py
│
├── module4_models/
│   ├── yield_model.py            # XGBoost: R²=0.991, MAE=2,900 hg/ha
│   ├── production_forecast.py    # Prophet: 3-crop national forecast 2024–2026
│   ├── input_optimizer.py        # Polynomial PDR curves, 42 crop-zone pairs
│   └── yield_model.pkl           # Bundled model + encoders + feature medians
│
├── module5_planning/
│   ├── planting_calendar.py      # NASA POWER onset analysis → planting windows
│   ├── input_calculator.py       # Seed + labour + agrochem + fertilizer cost
│   └── operations_schedule.py    # Week-by-week agronomic schedule templates
│
├── module6_powerbi/
│   └── export_powerbi.py         # 8-sheet star-schema Excel data mart
│
├── data/
│   ├── raw/                      # Source API downloads (gitignored)
│   └── processed/
│       ├── nigeria_agri.db       # SQLite master table (gitignored)
│       └── master_table.csv      # Cloud-ready CSV snapshot
│
└── requirements.txt
```

---

## Data Sources

| Source | Dataset | Coverage | URL |
|---|---|---|---|
| FAOSTAT QCL | Crop production: yield, area, output | 7 crops, 2000–2023 | [fao.org/faostat](https://www.fao.org/faostat) |
| NASA POWER | Monthly climate: rainfall, temperature, humidity | 6 zone centroids | [power.larc.nasa.gov](https://power.larc.nasa.gov) |
| World Bank WDI | Agricultural GDP share, fertilizer consumption, rural population | Nigeria, 2000–2023 | [data.worldbank.org](https://data.worldbank.org) |
| USDA PSD | Palm oil & cassava supply/demand balance | Nigeria, 2000–2023 | [apps.fas.usda.gov/psdonline](https://apps.fas.usda.gov/psdonline) |

All data is fetched programmatically with API fallback to local CSV when APIs are unavailable.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Prophet-4285F4?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## JD Requirements Mapping

| Requirement | Where demonstrated | Evidence |
|---|---|---|
| Data pipeline design | Module 1 | 5-source parallel API pipeline with fallback logic |
| SQL proficiency | Module 2 | 12 analytical SQL queries on SQLite |
| Machine learning | Module 4 | XGBoost with TimeSeriesSplit CV, SHAP explainability |
| Time-series forecasting | Module 4 | Prophet with log-transform, 80% CI, CV metrics |
| Dashboard development | Module 3 | 5-page Streamlit app, live on Streamlit Cloud |
| Business intelligence | Module 6 | 8-table Power BI star-schema data mart |
| Agricultural domain knowledge | Module 5 | IITA/NAERLS agronomic standards, NIMET onset analysis |
| Python (advanced) | All modules | 14 production scripts, 2,000+ lines |
| Data storytelling | Module 7 | Key findings with specific numbers, LinkedIn post |
| Nigerian market context | All modules | Zone-level analysis, NBS price benchmarks, CBN FX rate |

---

## How to Run Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/Fidelis-Akinbule/NigeriaAgriScope.git
cd NigeriaAgriScope
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Build the data pipeline (fetches from APIs — takes ~3 minutes)
python module1_pipeline/generate_data.py

# 3. Train models
python module4_models/yield_model.py
python module4_models/production_forecast.py
python module4_models/input_optimizer.py

# 4. Generate planning outputs
python module5_planning/planting_calendar.py
python module5_planning/input_calculator.py
python module5_planning/operations_schedule.py

# 5. Launch dashboard
streamlit run module3_dashboard/app.py
```

**Requirements:** Python 3.11+, internet connection for initial API fetch.
If APIs are unavailable, all modules fall back to validated local estimates automatically.

---

## Author

**Fidelis Akinbule** — Data Scientist | Lagos, Nigeria

[GitHub](https://github.com/Fidelis-Akinbule) · [LinkedIn](https://linkedin.com/in/fidelis-akinbule)

> *"NigeriaAgriScope is not a practice project — it is a functioning agricultural intelligence system built on real data that tells a real story about Nigerian food security."*