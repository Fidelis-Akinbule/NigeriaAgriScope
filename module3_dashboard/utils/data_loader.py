"""
utils/data_loader.py
====================
Single data-access layer for NigeriaAgriScope dashboard.

Design decisions:
  - Environment detection via DB file presence (not env vars) — more reliable
    across Streamlit Cloud versions (see Part 9 of build spec).
  - @st.cache_data(ttl=3600) prevents re-reads on every Streamlit rerun.
  - All filtering, aggregation, and KPI computation is done in metrics.py or
    inline in page files — this module is read-only, zero-transform.
  - Convenience accessors (get_years, get_zones, get_crops) derive from the
    loaded DataFrame so they are always consistent with the actual data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path constants ────────────────────────────────────────────────────────────
_DB_PATH = Path("data/processed/nigeria_agri.db")
_CSV_PATH = Path("data/processed/master_table.csv")

# ── Canonical ordering (mirrors Part 4 of build spec) ────────────────────────
CROP_ORDER = [
    "Cassava",
    "Oil palm fruit",
    "Yam",
    "Maize",
    "Rice (paddy)",
    "Sorghum",
    "Cocoa beans",
]

ZONE_ORDER = [
    "North West",
    "North East",
    "North Central",
    "South West",
    "South East",
    "South South",
]

# ── Environment detection ─────────────────────────────────────────────────────


def _use_csv() -> bool:
    """
    Return True when the SQLite DB is absent (i.e. Streamlit Cloud deployment).
    The DB is gitignored; the CSV is force-added — presence of the DB is the
    reliable local/cloud discriminator (see Part 9 of build spec).
    """
    return not _DB_PATH.exists()


# ── Primary loader ────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Load the master_table and return a clean, typed DataFrame.

    Local:  reads from SQLite (nigeria_agri.db → master_table)
    Cloud:  reads from master_table.csv

    Schema contract — dtypes enforced after loading:
      year          → int64
      zone, crop    → str (object)
      All numeric   → float64 (pandas default from both sources)

    Returns
    -------
    pd.DataFrame
        Full master_table (~1,008 rows × 33 columns).
        No filtering or aggregation applied.
    """
    if _use_csv():
        df = pd.read_csv(_CSV_PATH)
    else:
        with sqlite3.connect(_DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM master_table", conn)

    # ── Dtype enforcement ────────────────────────────────────────────────────
    # year must be int so year-range slider comparisons work correctly.
    df["year"] = df["year"].astype(int)

    # zone and crop must be str — guards against accidental categorical dtype
    # from CSV inference, which would break string comparisons in filter widgets.
    df["zone"] = df["zone"].astype(str)
    df["crop"] = df["crop"].astype(str)

    return df


# ── Convenience accessors ─────────────────────────────────────────────────────
# These are thin wrappers — they do not cache independently because load_data()
# is already cached. Calling them on each page rerun is negligible overhead.


def get_years(df: pd.DataFrame) -> list[int]:
    """Return sorted list of all unique years in the dataset."""
    return sorted(df["year"].unique().tolist())


def get_zones(df: pd.DataFrame) -> list[str]:
    """
    Return zones in canonical ZONE_ORDER, filtered to those present in df.
    Preserves order even if a future data refresh drops a zone.
    """
    present = set(df["zone"].unique())
    return [z for z in ZONE_ORDER if z in present]


def get_crops(df: pd.DataFrame) -> list[str]:
    """
    Return crops in canonical CROP_ORDER, filtered to those present in df.
    """
    present = set(df["crop"].unique())
    return [c for c in CROP_ORDER if c in present]
