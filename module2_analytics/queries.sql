-- =============================================================================
-- NigeriaAgriScope — Module 2: Descriptive Analytics
-- SQL Queries File
-- =============================================================================
-- Source table : master_table (data/processed/nigeria_agri.db)
-- Grain        : one row per (zone, crop, year)
-- Zones        : 6 geopolitical zones
-- Crops        : 7 (Oil palm fruit, Cassava, Maize, Yam, Rice (paddy),
--                    Sorghum, Cocoa beans)
-- Years        : 2000–2023 (24 years)
-- Author       : Fidelis Akinbule  |  April 2026
-- =============================================================================
--
-- DATA CONTRACT REMINDERS
-- -----------------------
--   fertilizer_kg_ha     : zone-crop derived intensity (varies by zone/crop/year)
--   wb_fertilizer_kg_ha  : WB national aggregate (broadcast identically per year)
--   fertilizer_total_kg  : national FAOSTAT total kg (broadcast; use DISTINCT year
--                          to avoid n×42 overcounting across zone-crop rows)
-- =============================================================================


-- ----------------------------------------------------------------------------
-- Q1: Production trend by crop and year
-- What is the 24-year production trajectory for each of the 7 crops?
-- Aggregates zone-level rows to national totals.
-- ----------------------------------------------------------------------------
-- Q1: Production trend by crop and year
SELECT
    crop,
    year,
    SUM(production_tonnes) AS national_production_tonnes
FROM master_table
GROUP BY crop, year
ORDER BY crop, year;


-- ----------------------------------------------------------------------------
-- Q2: Highest-yield zone per crop (ranked)
-- Which zone achieves the best average yield (hg/ha) for each crop?
-- Uses average across all years to smooth single-year outliers.
-- ----------------------------------------------------------------------------
-- Q2: Average yield by zone and crop (ranked descending within each crop)
SELECT
    crop,
    zone,
    ROUND(AVG(yield_hg_ha), 0)                                AS avg_yield_hg_ha,
    ROUND(AVG(yield_hg_ha) / 10000.0, 2)                      AS avg_yield_t_ha,
    RANK() OVER (PARTITION BY crop ORDER BY AVG(yield_hg_ha) DESC) AS yield_rank
FROM master_table
WHERE yield_hg_ha IS NOT NULL
GROUP BY crop, zone
ORDER BY crop, yield_rank;


-- ----------------------------------------------------------------------------
-- Q3: Nigeria national average yield per crop (for benchmark comparison)
-- The Python layer joins this to hardcoded global benchmarks.
-- Using all-year average to represent the long-run production profile.
-- ----------------------------------------------------------------------------
-- Q3: Nigeria average yield per crop
SELECT
    crop,
    ROUND(AVG(yield_hg_ha), 0)        AS nigeria_avg_yield_hg_ha,
    ROUND(MIN(yield_hg_ha), 0)        AS nigeria_min_yield_hg_ha,
    ROUND(MAX(yield_hg_ha), 0)        AS nigeria_max_yield_hg_ha,
    COUNT(DISTINCT year)              AS years_of_data
FROM master_table
WHERE yield_hg_ha IS NOT NULL
GROUP BY crop
ORDER BY nigeria_avg_yield_hg_ha DESC;


-- ----------------------------------------------------------------------------
-- Q4: Rainfall vs cassava yield (scatter input)
-- What is the relationship between annual rainfall and cassava yield
-- across all zones and years?
-- ----------------------------------------------------------------------------
-- Q4: Annual rainfall and cassava yield by zone and year
SELECT
    zone,
    year,
    ROUND(AVG(rainfall_mm_annual), 1)  AS rainfall_mm_annual,
    ROUND(AVG(yield_hg_ha), 0)         AS cassava_yield_hg_ha
FROM master_table
WHERE crop = 'Cassava'
  AND rainfall_mm_annual IS NOT NULL
  AND yield_hg_ha IS NOT NULL
GROUP BY zone, year
ORDER BY zone, year;


-- ----------------------------------------------------------------------------
-- Q5: Fertilizer application intensity by zone
-- Which zones apply the most fertilizer per hectare?
-- fertilizer_kg_ha is the zone-crop derived intensity figure (DATA CONTRACT).
-- Average across crops gives the zone-level picture.
-- Split into N/P/K components for the stacked bar chart.
-- NOTE: _zone columns are zone-allocated fractions of the national total.
--       Dividing by area_ha converts to intensity (kg/ha).
-- ----------------------------------------------------------------------------
-- Q5: Average fertilizer intensity by zone (N, P, K breakdown)
SELECT
    zone,
    ROUND(AVG(fertilizer_kg_ha), 2)   AS avg_fertilizer_kg_ha,
    ROUND(
        AVG(CASE WHEN area_ha > 0
                 THEN CAST(fertilizer_n_kg_zone AS REAL) / area_ha
                 ELSE NULL END), 2)   AS avg_n_kg_ha,
    ROUND(
        AVG(CASE WHEN area_ha > 0
                 THEN CAST(fertilizer_p_kg_zone AS REAL) / area_ha
                 ELSE NULL END), 2)   AS avg_p_kg_ha,
    ROUND(
        AVG(CASE WHEN area_ha > 0
                 THEN CAST(fertilizer_k_kg_zone AS REAL) / area_ha
                 ELSE NULL END), 2)   AS avg_k_kg_ha
FROM master_table
WHERE area_ha > 0
GROUP BY zone
ORDER BY avg_fertilizer_kg_ha DESC;


-- ----------------------------------------------------------------------------
-- Q6: Yield gap — actual vs potential — per crop
-- "Potential yield" is defined as the 90th-percentile yield achieved within
-- the dataset for each crop (a practical, data-driven ceiling that excludes
-- extreme outliers while representing realistic best performance).
-- Gap = (potential - actual) / potential * 100 expressed as a percentage.
-- ----------------------------------------------------------------------------
-- Q6: Yield gap per crop (actual average vs 90th-percentile ceiling)
WITH percentiles AS (
    SELECT
        crop,
        -- SQLite has no built-in PERCENTILE_CONT; approximate with sorted window.
        -- Uses a subquery approach that is compatible with SQLite 3.25+.
        AVG(yield_hg_ha) AS avg_yield
    FROM master_table
    WHERE yield_hg_ha IS NOT NULL
    GROUP BY crop
),
potential AS (
    SELECT
        crop,
        -- 90th percentile approximated as: order rows by yield, take row at
        -- ceiling(0.9 * count) — equivalent to PERCENTILE_DISC(0.9).
        yield_hg_ha AS p90_yield
    FROM (
        SELECT
            crop,
            yield_hg_ha,
            ROW_NUMBER() OVER (PARTITION BY crop ORDER BY yield_hg_ha)   AS rn,
            COUNT(*)     OVER (PARTITION BY crop)                         AS total
        FROM master_table
        WHERE yield_hg_ha IS NOT NULL
    ) ranked
    WHERE rn = CAST(CEIL(0.9 * total) AS INTEGER)
)
SELECT
    p.crop,
    ROUND(a.avg_yield, 0)                                     AS actual_avg_yield_hg_ha,
    ROUND(p.p90_yield, 0)                                     AS potential_p90_yield_hg_ha,
    ROUND(p.p90_yield - a.avg_yield, 0)                       AS yield_gap_hg_ha,
    ROUND((p.p90_yield - a.avg_yield) / p.p90_yield * 100, 1) AS yield_gap_pct
FROM potential p
JOIN percentiles a ON p.crop = a.crop
ORDER BY yield_gap_pct DESC;


-- ----------------------------------------------------------------------------
-- Q7: Area expansion vs contraction by crop over time
-- Which crops have been growing in planted area and which are shrinking?
-- Compares first-5-year average to last-5-year average as a structural trend.
-- ----------------------------------------------------------------------------
-- Q7: Area harvested trend — early period vs recent period comparison
WITH early AS (
    SELECT crop, ROUND(AVG(area_ha), 0) AS early_area_ha
    FROM master_table
    WHERE year BETWEEN 2000 AND 2004
    GROUP BY crop
),
recent AS (
    SELECT crop, ROUND(AVG(area_ha), 0) AS recent_area_ha
    FROM master_table
    WHERE year BETWEEN 2019 AND 2023
    GROUP BY crop
),
annual AS (
    SELECT crop, year, SUM(area_ha) AS total_area_ha
    FROM master_table
    GROUP BY crop, year
)
SELECT
    e.crop,
    e.early_area_ha,
    r.recent_area_ha,
    ROUND((CAST(r.recent_area_ha AS REAL) - e.early_area_ha)
          / e.early_area_ha * 100, 1)   AS area_change_pct,
    CASE
        WHEN r.recent_area_ha > e.early_area_ha * 1.05 THEN 'EXPANDING'
        WHEN r.recent_area_ha < e.early_area_ha * 0.95 THEN 'CONTRACTING'
        ELSE 'STABLE'
    END                                  AS trend_direction
FROM early e
JOIN recent r ON e.crop = r.crop
ORDER BY area_change_pct DESC;


-- Q7b: Full annual area time series (for line chart in analytics.py)
SELECT
    crop,
    year,
    SUM(area_ha) AS total_area_ha
FROM master_table
GROUP BY crop, year
ORDER BY crop, year;


-- ----------------------------------------------------------------------------
-- Q8: Five consecutive years of highest maize production (rolling window)
-- Which 5-year window achieved the greatest cumulative maize output?
-- Uses a self-join approach for SQLite 3.24 compatibility (window functions
-- for range frames are limited; the analytics.py layer falls back to pandas
-- rolling() if the SQLite version pre-dates 3.25).
-- ----------------------------------------------------------------------------
-- Q8: 5-year rolling sum of national maize production
WITH maize_national AS (
    SELECT year, SUM(production_tonnes) AS maize_prod
    FROM master_table
    WHERE crop = 'Maize'
    GROUP BY year
),
rolling AS (
    SELECT
        m.year                                  AS end_year,
        m.year - 4                              AS start_year,
        SUM(h.maize_prod)                       AS rolling_5yr_total
    FROM maize_national m
    JOIN maize_national h
      ON h.year BETWEEN m.year - 4 AND m.year
    GROUP BY m.year
    HAVING COUNT(*) = 5           -- only complete 5-year windows
)
SELECT
    start_year,
    end_year,
    ROUND(rolling_5yr_total, 0)  AS total_maize_tonnes,
    DENSE_RANK() OVER (ORDER BY rolling_5yr_total DESC) AS production_rank
FROM rolling
ORDER BY rolling_5yr_total DESC;


-- ----------------------------------------------------------------------------
-- Q9: Fertilizer-to-yield efficiency over time
-- How has the ratio of production per kilogram of fertilizer changed?
-- National-level: SUM of all production / national fertilizer total.
-- fertilizer_total_kg is broadcast across all zone-crop rows per year;
-- use DISTINCT to extract one value per year before dividing.
-- ----------------------------------------------------------------------------
-- Q9: National fertilizer efficiency ratio by year
WITH fert_by_year AS (
    -- One row per year — extract the national total without multiplying
    SELECT DISTINCT year, fertilizer_total_kg
    FROM master_table
    WHERE fertilizer_total_kg IS NOT NULL
      AND fertilizer_total_kg > 0
),
prod_by_year AS (
    SELECT year, SUM(production_tonnes) AS total_production_tonnes
    FROM master_table
    GROUP BY year
)
SELECT
    p.year,
    ROUND(p.total_production_tonnes, 0)                               AS total_production_tonnes,
    ROUND(f.fertilizer_total_kg / 1000.0, 1)                         AS fertilizer_total_tonnes,
    ROUND(p.total_production_tonnes /
          (f.fertilizer_total_kg / 1000.0), 2)                       AS tonnes_crop_per_tonne_fert,
    ROUND(f.fertilizer_total_kg /
          NULLIF(p.total_production_tonnes, 0) * 1000.0, 4)          AS kg_fert_per_tonne_crop
FROM prod_by_year p
JOIN fert_by_year f ON p.year = f.year
ORDER BY p.year;


-- ----------------------------------------------------------------------------
-- Q10: Year-on-year yield volatility by zone (coefficient of variation)
-- Which zones show the most unpredictable yield outcomes — a risk signal
-- for planting decisions and input procurement.
-- CV = (STDEV / MEAN) * 100.  SQLite lacks STDEV; Python handles computation.
-- This query delivers the raw year-zone-crop yield series for pandas processing.
-- ----------------------------------------------------------------------------
-- Q10: Annual yield by zone and crop (for CV computation in Python)
SELECT
    zone,
    crop,
    year,
    yield_hg_ha
FROM master_table
WHERE yield_hg_ha IS NOT NULL
ORDER BY zone, crop, year;


-- ----------------------------------------------------------------------------
-- Q11: Seasonal rainfall pattern across zones
-- The master table stores annual aggregates only. Monthly disaggregation is
-- performed in analytics.py using zone-specific seasonal coefficients derived
-- from NIMET long-term monthly profiles. This query supplies the annual
-- totals that anchor the disaggregation.
-- ----------------------------------------------------------------------------
-- Q11: Annual rainfall by zone (anchor for seasonal reconstruction)
SELECT
    zone,
    year,
    ROUND(AVG(rainfall_mm_annual), 1) AS avg_rainfall_mm
FROM master_table
WHERE rainfall_mm_annual IS NOT NULL
GROUP BY zone, year
ORDER BY zone, year;


-- ----------------------------------------------------------------------------
-- Q12: Zone-crop combinations with highest improvement potential
-- Defined as: large planted area × large yield gap = high-value opportunity.
-- "Opportunity Score" = (potential_yield - actual_yield) * avg_area_ha
-- Normalised to a 0–100 scale for chart readability.
-- ----------------------------------------------------------------------------
-- Q12: Opportunity matrix — yield gap vs planted area by zone-crop
WITH stats AS (
    SELECT
        zone,
        crop,
        ROUND(AVG(yield_hg_ha), 0)         AS avg_yield_hg_ha,
        ROUND(AVG(area_ha), 0)             AS avg_area_ha,
        ROUND(MAX(yield_hg_ha), 0)         AS max_yield_hg_ha
    FROM master_table
    WHERE yield_hg_ha IS NOT NULL
      AND area_ha IS NOT NULL
    GROUP BY zone, crop
),
scored AS (
    SELECT
        zone,
        crop,
        avg_yield_hg_ha,
        avg_area_ha,
        max_yield_hg_ha,
        ROUND(max_yield_hg_ha - avg_yield_hg_ha, 0)         AS yield_gap_hg_ha,
        ROUND((max_yield_hg_ha - avg_yield_hg_ha)
              * avg_area_ha / 1e9, 4)                        AS opportunity_score_raw
    FROM stats
)
SELECT
    zone,
    crop,
    avg_yield_hg_ha,
    avg_area_ha,
    max_yield_hg_ha,
    yield_gap_hg_ha,
    opportunity_score_raw,
    -- Normalise to 0–100 for bubble chart size encoding
    ROUND(
        opportunity_score_raw
        / MAX(opportunity_score_raw) OVER () * 100.0
    , 1)                                                     AS opportunity_score_norm
FROM scored
WHERE avg_area_ha > 0
  AND yield_gap_hg_ha > 0
ORDER BY opportunity_score_raw DESC;