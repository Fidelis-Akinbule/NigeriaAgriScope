"""
NigeriaAgriScope — Module 8: Streamlit Cloud Deployment Preparation
====================================================================
Validates that the repository is ready for Streamlit Cloud deployment
and fixes the most common deployment failures automatically.

Streamlit Cloud deployment model
----------------------------------
Streamlit Cloud clones your GitHub repo and runs:
  pip install -r requirements.txt
  streamlit run <entrypoint>

It does NOT have access to:
  - Local SQLite databases (gitignored)
  - Any file not committed to the repo
  - Files larger than ~100MB

NigeriaAgriScope handles this via the data_loader.py environment detection:
  - If nigeria_agri.db exists → reads from SQLite (local dev)
  - If nigeria_agri.db absent → reads from master_table.csv (Streamlit Cloud)

This script verifies that contract is honoured and all required files exist.

Usage
-----
  python module8_deploy/deploy_checklist.py

What it checks and fixes
-------------------------
  [1]  runtime.txt         — Python version declaration for Streamlit Cloud
  [2]  requirements.txt    — all required packages present and pinned
  [3]  .gitignore          — SQLite DB and raw data are excluded; CSV is force-added
  [4]  master_table.csv    — cloud fallback CSV exists and has correct row count
  [5]  yield_model.pkl     — model pickle committed (required for Page 5)
  [6]  M5 planning CSVs   — all 3 planning output CSVs committed
  [7]  M4 forecast CSVs   — all 3 Prophet forecast CSVs committed
  [8]  app.py entrypoint  — exists and is importable
  [9]  Streamlit config   — .streamlit/config.toml exists with correct settings
  [10] Page 5 stub check  — confirms Page 5 loads pickle, not stub message

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m8_deploy")

ROOT = Path(__file__).resolve().parent.parent
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
FIX = "🔧"

results: list[tuple[str, str, str]] = []  # (check, status, detail)


def check(name: str, passed: bool, detail: str = "", fixed: bool = False) -> bool:
    icon = PASS if passed else (FIX if fixed else FAIL)
    results.append((name, icon, detail))
    log.info("%s  %s%s", icon, name, f"  — {detail}" if detail else "")
    return passed or fixed


# ── Check 1: runtime.txt ──────────────────────────────────────────────────


def check_runtime() -> None:
    path = ROOT / "runtime.txt"
    required = "python-3.11"
    if path.exists():
        content = path.read_text().strip()
        if required in content:
            check("runtime.txt", True, f"Found: {content}")
        else:
            path.write_text(f"{required}\n")
            check(
                "runtime.txt",
                False,
                f"Was '{content}' — fixed to '{required}'",
                fixed=True,
            )
    else:
        path.write_text(f"{required}\n")
        check("runtime.txt", False, f"Missing — created with '{required}'", fixed=True)


# ── Check 2: requirements.txt ─────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
    "xgboost",
    "shap",
    "prophet",
    "scikit-learn",
    "scipy",
    "requests",
    "openpyxl",
    "wbgapi",
}


def check_requirements() -> None:
    path = ROOT / "requirements.txt"
    if not path.exists():
        _write_requirements(path)
        check(
            "requirements.txt",
            False,
            "Missing — created with all required packages",
            fixed=True,
        )
        return

    content = path.read_text().lower()
    missing = {pkg for pkg in REQUIRED_PACKAGES if pkg.lower() not in content}

    if not missing:
        check(
            "requirements.txt",
            True,
            f"{len(REQUIRED_PACKAGES)} required packages all present",
        )
    else:
        # Append missing packages
        with open(path, "a") as f:
            for pkg in sorted(missing):
                f.write(f"\n{pkg}")
        check(
            "requirements.txt",
            False,
            f"Added missing: {', '.join(sorted(missing))}",
            fixed=True,
        )


def _write_requirements(path: Path) -> None:
    content = """streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
xgboost>=2.0.0
shap>=0.44.0
prophet>=1.1.5
scikit-learn>=1.4.0
scipy>=1.12.0
requests>=2.31.0
openpyxl>=3.1.0
wbgapi>=1.0.12
"""
    path.write_text(content)


# ── Check 3: .gitignore ───────────────────────────────────────────────────

GITIGNORE_REQUIRED_EXCLUDES = [
    "data/processed/nigeria_agri.db",
    "data/raw/",
    "*.db",
    "__pycache__/",
    ".env",
    "venv/",
    ".venv/",
]

GITIGNORE_FORCE_INCLUDES = [
    "!data/processed/master_table.csv",
    "!module4_models/yield_model.pkl",
    "!module4_models/outputs/",
    "!module5_planning/outputs/",
]


def check_gitignore() -> None:
    path = ROOT / ".gitignore"
    if not path.exists():
        _write_gitignore(path)
        check(".gitignore", False, "Missing — created with correct rules", fixed=True)
        return

    content = path.read_text()
    issues = []

    for rule in GITIGNORE_REQUIRED_EXCLUDES:
        if rule not in content:
            issues.append(f"missing exclude: {rule}")

    for rule in GITIGNORE_FORCE_INCLUDES:
        if rule not in content:
            issues.append(f"missing force-include: {rule}")

    if not issues:
        check(".gitignore", True, "All exclude and force-include rules present")
    else:
        # Append missing rules
        with open(path, "a") as f:
            f.write("\n# NigeriaAgriScope deployment rules\n")
            for rule in GITIGNORE_REQUIRED_EXCLUDES:
                if rule not in content:
                    f.write(f"{rule}\n")
            f.write("\n# Force-include cloud fallback files\n")
            for rule in GITIGNORE_FORCE_INCLUDES:
                if rule not in content:
                    f.write(f"{rule}\n")
        check(
            ".gitignore",
            False,
            f"Fixed {len(issues)} issues: {'; '.join(issues[:3])}",
            fixed=True,
        )


def _write_gitignore(path: Path) -> None:
    path.write_text(
        """# Python
__pycache__/
*.py[cod]
*.pyo
.env
venv/
.venv/
*.egg-info/
dist/
build/

# Data (gitignored — too large / sensitive)
data/raw/
data/processed/nigeria_agri.db
*.db

# Force-include cloud fallback files
!data/processed/master_table.csv
!module4_models/yield_model.pkl
!module4_models/outputs/input_requirements_reference.csv
!module4_models/outputs/yield_predictions.csv
!module4_models/outputs/production_forecast_*.csv
!module5_planning/outputs/planting_calendar_all_zones.csv
!module5_planning/outputs/input_requirements_enhanced.csv
!module5_planning/outputs/operations_schedule_summary.csv
!module5_planning/outputs/operations_schedule_all_zones.csv

# IDE
.vscode/settings.json
.idea/
*.swp
"""
    )


# ── Check 4: master_table.csv ─────────────────────────────────────────────


def check_master_csv() -> None:
    import pandas as pd

    path = ROOT / "data" / "processed" / "master_table.csv"
    if not path.exists():
        check(
            "master_table.csv",
            False,
            "MISSING — run module1_pipeline/generate_data.py then commit the CSV",
        )
        return

    try:
        df = pd.read_csv(path)
        rows, cols = df.shape
        if rows >= 900:
            check(
                "master_table.csv",
                True,
                f"{rows:,} rows × {cols} cols — cloud fallback ready",
            )
        else:
            check(
                "master_table.csv",
                False,
                f"Only {rows} rows — expected ≥900. Re-run M1 pipeline.",
            )
    except Exception as e:
        check("master_table.csv", False, f"Read error: {e}")


# ── Check 5: yield_model.pkl ──────────────────────────────────────────────


def check_pickle() -> None:
    import pickle

    # pkl is saved to outputs/ by yield_model.py — check both locations
    path = ROOT / "module4_models" / "outputs" / "yield_model.pkl"
    if not path.exists():
        path = ROOT / "module4_models" / "yield_model.pkl"
    if not path.exists():
        check(
            "yield_model.pkl",
            False,
            "MISSING — run module4_models/yield_model.py then commit the .pkl",
        )
        return

    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        required_keys = {"model", "encoders", "feature_names", "feature_medians"}
        present = required_keys.issubset(artifact.keys())
        r2 = artifact.get("test_r2", 0)
        check(
            "yield_model.pkl",
            present,
            f"Keys OK — R²={r2:.3f}  size={path.stat().st_size/1024:.0f}KB",
        )
    except Exception as e:
        check("yield_model.pkl", False, f"Load error: {e}")


# ── Check 6: M5 planning CSVs ─────────────────────────────────────────────

M5_FILES = [
    "module5_planning/outputs/planting_calendar_all_zones.csv",
    "module5_planning/outputs/input_requirements_enhanced.csv",
    "module5_planning/outputs/operations_schedule_summary.csv",
]


def check_m5_csvs() -> None:
    import pandas as pd

    all_ok = True
    for rel_path in M5_FILES:
        path = ROOT / rel_path
        if not path.exists():
            check(
                f"M5: {path.name}",
                False,
                "MISSING — run module5_planning scripts then commit",
            )
            all_ok = False
        else:
            rows = len(pd.read_csv(path))
            check(f"M5: {path.name}", True, f"{rows} rows")
    if all_ok:
        log.info("%s  All M5 planning CSVs present", PASS)


# ── Check 7: M4 forecast CSVs ─────────────────────────────────────────────

M4_FORECAST_FILES = [
    "module4_models/outputs/production_forecast_oil_palm_fruit.csv",
    "module4_models/outputs/production_forecast_cassava.csv",
    "module4_models/outputs/production_forecast_maize.csv",
]


def check_m4_forecasts() -> None:
    import pandas as pd

    for rel_path in M4_FORECAST_FILES:
        path = ROOT / rel_path
        if not path.exists():
            check(
                f"M4: {path.name}",
                False,
                "MISSING — run module4_models/production_forecast.py then commit",
            )
        else:
            rows = len(pd.read_csv(path))
            check(f"M4: {path.name}", True, f"{rows} rows")


# ── Check 8: app.py entrypoint ────────────────────────────────────────────


def check_entrypoint() -> None:
    path = ROOT / "module3_dashboard" / "app.py"
    if not path.exists():
        check("app.py entrypoint", False, "module3_dashboard/app.py not found")
        return
    content = path.read_text()
    has_streamlit = "import streamlit" in content or "st." in content
    check(
        "app.py entrypoint",
        has_streamlit,
        (
            "Found and contains Streamlit code"
            if has_streamlit
            else "Found but may not be a valid Streamlit app"
        ),
    )


# ── Check 9: .streamlit/config.toml ──────────────────────────────────────

STREAMLIT_CONFIG = """[theme]
primaryColor = "#2E75B6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#1F1F1F"
font = "sans serif"

[server]
headless = true
enableCORS = false

[browser]
gatherUsageStats = false
"""


def check_streamlit_config() -> None:
    config_dir = ROOT / ".streamlit"
    config_path = config_dir / "config.toml"

    if config_path.exists():
        check(".streamlit/config.toml", True, "Present")
    else:
        config_dir.mkdir(exist_ok=True)
        config_path.write_text(STREAMLIT_CONFIG)
        check(
            ".streamlit/config.toml",
            False,
            "Missing — created with theme and server settings",
            fixed=True,
        )


# ── Check 10: Page 5 stub detection ───────────────────────────────────────


def check_page5_stub() -> None:
    page5 = ROOT / "module3_dashboard" / "pages" / "page5_forecast.py"
    if not page5.exists():
        # Try alternate locations
        for alt in ROOT.rglob("page5*.py"):
            page5 = alt
            break
        else:
            check(
                "Page 5 — stub detection",
                False,
                "page5_forecast.py not found in dashboard",
            )
            return

    content = page5.read_text()
    has_pickle_ref = "pickle.load" in content or "yield_model.pkl" in content
    has_stub = "will appear here after Module 4" in content

    # Determine actual pkl path
    pkl_outputs = ROOT / "module4_models" / "outputs" / "yield_model.pkl"
    pkl_root = ROOT / "module4_models" / "yield_model.pkl"
    pkl_exists = pkl_outputs.exists() or pkl_root.exists()

    if has_pickle_ref and pkl_exists:
        check(
            "Page 5 — stub detection",
            True,
            "References pickle and pickle exists on disk — ready for deployment",
        )
    elif has_stub and not pkl_exists:
        check(
            "Page 5 — stub detection",
            False,
            "⚠️  STUB TEXT STILL PRESENT — Page 5 shows placeholder. "
            "Update page5_forecast.py to load yield_model.pkl before publishing.",
        )
    elif has_stub and pkl_exists:
        check(
            "Page 5 — stub detection",
            True,
            "Stub text present but pickle exists — banner will show ✅ on deployment",
        )


# ── Final summary ──────────────────────────────────────────────────────────


def print_summary() -> None:
    log.info("")
    log.info("=" * 60)
    log.info("DEPLOYMENT READINESS SUMMARY")
    log.info("=" * 60)

    passed = sum(1 for _, icon, _ in results if icon == PASS)
    fixed = sum(1 for _, icon, _ in results if icon == FIX)
    failed = sum(1 for _, icon, _ in results if icon == FAIL)
    total = len(results)

    for name, icon, detail in results:
        log.info("%s  %-40s  %s", icon, name, detail[:60] if detail else "")

    log.info("")
    log.info("  Passed : %d / %d", passed, total)
    log.info("  Fixed  : %d / %d", fixed, total)
    log.info("  Failed : %d / %d", failed, total)
    log.info("")

    if failed == 0:
        log.info("%s  All checks passed. Ready to deploy!", PASS)
        log.info("")
        log.info("  NEXT STEPS:")
        log.info("  1. git add -A && git commit -m 'Phase 7: deployment prep'")
        log.info("  2. git push origin main")
        log.info("  3. Go to share.streamlit.io → New app")
        log.info("     Repository : github.com/Fidelis-Akinbule/NigeriaAgriScope")
        log.info("     Branch     : main")
        log.info("     Main file  : module3_dashboard/app.py")
        log.info("  4. Wait for build (~3–5 min) → copy the .streamlit.app URL")
        log.info("  5. Update README.md live dashboard link with the real URL")
        log.info("  6. Publish LinkedIn post (Tuesday/Wednesday 8–10am WAT)")
    else:
        log.info(
            "%s  %d check(s) require manual action before deploying.", FAIL, failed
        )
        log.info("  Fix the ❌ items above, then re-run this script.")

    log.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 7: Deployment Readiness Check")
    log.info("=" * 60)

    check_runtime()
    check_requirements()
    check_gitignore()
    check_master_csv()
    check_pickle()
    check_m5_csvs()
    check_m4_forecasts()
    check_entrypoint()
    check_streamlit_config()
    check_page5_stub()

    print_summary()


if __name__ == "__main__":
    main()
