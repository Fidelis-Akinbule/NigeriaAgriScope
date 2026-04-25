# NigeriaAgriScope — Power BI Import Guide

**File:** `NigeriaAgriScope_PowerBI_DataMart.xlsx`
**Author:** Fidelis Akinbule | April 2026

---

## Step 1 — Import the workbook into Power BI Desktop

1. Open **Power BI Desktop**
2. Click **Home → Get Data → Excel Workbook**
3. Select `NigeriaAgriScope_PowerBI_DataMart.xlsx`
4. In the Navigator panel, tick **all 8 data sheets** (exclude `_COVER`):
   - `dim_zone_crop`
   - `fact_yield_history`
   - `fact_yield_forecast`
   - `fact_production_forecast`
   - `fact_input_requirements`
   - `fact_planting_calendar`
   - `fact_operations_summary`
   - `fact_operations_detail`
5. Click **Transform Data** → in Power Query, verify column types are correct
   (year = Whole Number, costs = Decimal, text = Text)
6. Click **Close & Apply**

---

## Step 2 — Build the star-schema relationships

Go to **Model view** (left sidebar icon). Create these relationships:

| From table (many side) | Column | To table (one side) | Column |
|---|---|---|---|
| fact_yield_history | zone_crop_key | dim_zone_crop | zone_crop_key |
| fact_yield_forecast | zone_crop_key | dim_zone_crop | zone_crop_key |
| fact_input_requirements | zone_crop_key | dim_zone_crop | zone_crop_key |
| fact_planting_calendar | zone_crop_key | dim_zone_crop | zone_crop_key |
| fact_operations_summary | zone_crop_key | dim_zone_crop | zone_crop_key |
| fact_operations_detail | zone_crop_key | dim_zone_crop | zone_crop_key |

`fact_production_forecast` is national-level (no zone) — connect on `crop` only:

| From table | Column | To table | Column |
|---|---|---|---|
| fact_production_forecast | crop | dim_zone_crop | crop |

Set all relationship cardinality to **Many-to-One (*)→(1)**.

---

## Step 3 — Suggested DAX measures

Create these in a dedicated `_Measures` table:

```dax
-- Yield gap as percentage of PDR yield
Yield Gap % =
DIVIDE(
    SUM(fact_input_requirements[yield_gap_at_pdr_hg_ha]),
    SUM(fact_yield_history[yield_hg_ha]),
    0
)

-- Fertilizer efficiency ratio (yield per kg fertilizer)
Fertilizer Efficiency =
DIVIDE(
    AVERAGE(fact_yield_history[yield_hg_ha]),
    AVERAGE(fact_yield_history[fertilizer_kg_ha]),
    0
)

-- Total input cost (Naira, millions)
Total Input Cost (₦M) =
DIVIDE(SUM(fact_input_requirements[total_input_cost_naira_per_ha]), 1000000)

-- Model accuracy (test set MAPE)
Model MAPE % =
AVERAGEX(
    fact_yield_forecast,
    fact_yield_forecast[abs_pct_error]
)

-- Forecast vs actual production growth
Production Growth % =
VAR actual_2023 =
    CALCULATE(SUM(fact_production_forecast[production_tonnes]),
              fact_production_forecast[year] = 2023,
              fact_production_forecast[type] = "actual")
VAR forecast_2026 =
    CALCULATE(SUM(fact_production_forecast[production_tonnes]),
              fact_production_forecast[year] = 2026,
              fact_production_forecast[type] = "forecast")
RETURN DIVIDE(forecast_2026 - actual_2023, actual_2023, 0)
```

---

## Step 4 — Suggested report pages

### Page 1 — National Overview
- Card: Total crops tracked, zones covered, years of data
- Line chart: `fact_yield_history` yield_hg_ha by year (all crops)
- Map visual: zone → production_tonnes (use Nigeria shapefile or built-in map)
- Slicer: crop (from dim_zone_crop)

### Page 2 — Yield Intelligence
- Scatter: actual vs predicted yield (fact_yield_forecast)
- Bar: SHAP feature importance (static image from chart01 — import as image)
- Line: yield trend 2000–2023 by zone (fact_yield_history filtered by crop slicer)

### Page 3 — Production Forecast
- Line + shaded band: production_tonnes + CI bounds (fact_production_forecast)
- Slicer: crop (Cassava / Oil palm fruit / Maize)
- Card: 2026 forecast vs 2023 actual growth %

### Page 4 — Input Planning
- Stacked bar: cost breakdown per crop (fact_input_requirements)
- Table: zone × crop with recommended fertilizer range + PDR
- Gauge: current_avg_kg_ha vs recommended_max_kg_ha (per selected zone-crop)

### Page 5 — Planting Calendar
- Matrix: zone (rows) × month (cols), coloured by planting month
- Bar: rainfall reliability score by zone
- Table: planting + harvest months with growing duration

### Page 6 — Operations Schedule (Drill-through)
- Table: fact_operations_detail filtered by selected zone-crop
- Set drill-through from Page 5 on zone_crop_key → this page
- Timeline visual: phase × week_offset coloured by phase category

---

## Data sources

| Source | Coverage | API |
|---|---|---|
| FAOSTAT QCL | Crop production 2000–2023 | fenixservices.fao.org |
| FAOSTAT RFN | Fertilizer by nutrient | fenixservices.fao.org |
| NASA POWER | Monthly climate 6 zones | power.larc.nasa.gov |
| World Bank WDI | Agricultural macro-indicators | api.worldbank.org |
| USDA PSD | Palm oil & cassava supply/demand | apps.fas.usda.gov |

**Model details:** XGBoost R²=0.991, MAE=2,900 hg/ha | Prophet 80% CI | PDR threshold=50 hg/ha per kg/ha
**Price benchmarks:** NBS Q3 2024 | ₦820/kg NPK 15:15:15 | ₦750/kg Urea | Exchange rate ₦1,580/USD
