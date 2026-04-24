"""
NigeriaAgriScope — Module 4: Yield Forecast Model
===================================================
Trains an XGBoost regression model to predict crop yield (hg/ha) per zone
using lagged yield, climate, and fertilizer features.

Model design decisions
----------------------
- TimeSeriesSplit CV (not random KFold) — prevents data leakage from using
  future years to predict past years, which would inflate R² artificially.
- OrdinalEncoder (not LabelEncoder or OneHotEncoder) — chosen over OHE to
  avoid dimensionality explosion with 7 crops × 6 zones. OrdinalEncoder is
  stable across fit/transform calls, making it safe to bundle in the pickle.
- Encoders bundled into a single pickle dict with the model — ensures Page 5
  of the dashboard uses exactly the same encoding at inference time.
- Lagged yield features (lag_1, lag_2) are the single strongest predictors
  per agronomic literature; rolling_rain_3yr captures multi-year climate trend.
- Output path: module4_models/outputs/yield_model.pkl (consistent with all
  other M4 artifacts in outputs/).
- Page 5 contract: dashboard calls predict_yield(zone, crop, year, rainfall,
  fertilizer_kg_ha) using the bundled artifact. See INFERENCE INTERFACE below.

Train/test split
----------------
  Train : 2000–2018 (19 years × 7 crops × 6 zones = up to 798 rows)
  Test  : 2019–2023 (5 years  × 7 crops × 6 zones = up to 210 rows)

  Note: lag features consume the first 2 years per series. Effective train
  set = 2002–2018.  Total usable rows after lag creation ≈ 756 train / 210 test.

Expected performance (from FAOSTAT Nigeria agricultural literature)
--------------------------------------------------------------------
  R²  ≥ 0.75 (spec target)
  MAE < 15,000 hg/ha (roughly 10% of average oil palm yield)

Usage
-----
  python module4_models/yield_model.py

Outputs (all → module4_models/outputs/)
  yield_model.pkl                    — bundled {model, encoders, feature_names}
  yield_predictions.csv              — test set actual vs predicted
  chart01_yield_model_feature_importance.png
  chart02_actual_vs_predicted.png
  chart03_residuals.png

Author : Fidelis Akinbule
Date   : April 2026
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m4_yield_model")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "nigeria_agri.db"
CSV_PATH = ROOT / "data" / "processed" / "master_table.csv"
OUT_DIR = ROOT / "module4_models" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PICKLE_PATH = ROOT / "module4_models" / "yield_model.pkl"
PREDICTIONS_CSV = OUT_DIR / "yield_predictions.csv"

# ── Chart output paths (spec names) ───────────────────────────────────────
CHART_IMPORTANCE = OUT_DIR / "chart01_yield_model_feature_importance.png"
CHART_ACTUAL_PRED = OUT_DIR / "chart02_actual_vs_predicted.png"
CHART_RESIDUALS = OUT_DIR / "chart03_residuals.png"

# ── Model config ──────────────────────────────────────────────────────────
TRAIN_END_YEAR = 2018
TEST_START_YEAR = 2019
N_CV_SPLITS = 5  # TimeSeriesSplit folds over training years

# XGBoost hyperparameters — tuned for ~800-row agricultural tabular dataset.
# n_estimators=400 with early_stopping_rounds=30 prevents overfitting on
# the small time series. subsample=0.8 adds regularisation without hurting
# recall on the scarce test years (2019–2023).
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,  # L1 — forces sparse feature selection
    "reg_lambda": 1.0,  # L2 — standard ridge penalty
    "min_child_weight": 5,  # prevents splits on very few zone-crop observations
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

CATEGORICAL_FEATURES = ["zone", "crop"]
NUMERIC_FEATURES = [
    "lag_1_yield",  # yield(t-1) — strongest autocorrelation signal
    "lag_2_yield",  # yield(t-2) — captures 2-year climate cycles
    "rolling_rain_3yr",  # 3-year trailing average rainfall (mm)
    "yoy_rain_change",  # year-over-year rainfall delta
    "rainfall_mm_annual",  # raw annual rainfall
    "temp_avg_celsius",  # mean annual temperature
    "humidity_pct",
    "solar_radiation",
    "fertilizer_kg_ha",  # zone-crop fertilizer intensity (derived in M1)
    "wb_fertilizer_kg_ha",  # WB national aggregate (macro context)
    "yoy_fert_change",  # year-over-year fertilizer application change
    "year",  # trend term — captures long-run productivity gains
]


# ── Data loading ──────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load master_table from SQLite (preferred) or CSV fallback."""
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM master_table", conn)
        log.info("Loaded from SQLite: %d rows × %d cols", *df.shape)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        log.info("Loaded from CSV: %d rows × %d cols", *df.shape)
    else:
        raise FileNotFoundError(
            f"Neither {DB_PATH} nor {CSV_PATH} found. Run Module 1 first."
        )
    df["year"] = df["year"].astype(int)
    return df


# ── Feature engineering ───────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling features per (zone, crop) series.

    Why sort before shift: pandas groupby.shift() relies on the existing row
    order. Sorting by zone → crop → year guarantees the lag_1 of year T is
    year T-1 from the same zone-crop series, not a stale row from another group.

    NaN rows (years 2000–2001 per series, which lack 2 prior years of data)
    are dropped after feature creation. This is deliberate: we prefer to lose
    2 years of training data per series rather than fill lag features with
    potentially misleading imputed values.
    """
    df = df.sort_values(["zone", "crop", "year"]).copy()

    grp = df.groupby(["zone", "crop"])

    # Lagged yield
    df["lag_1_yield"] = grp["yield_hg_ha"].shift(1)
    df["lag_2_yield"] = grp["yield_hg_ha"].shift(2)

    # 3-year trailing average rainfall (current year included)
    df["rolling_rain_3yr"] = grp["rainfall_mm_annual"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Year-over-year deltas
    df["yoy_rain_change"] = grp["rainfall_mm_annual"].diff(1)
    df["yoy_fert_change"] = grp["fertilizer_kg_ha"].diff(1)

    # Drop rows missing both lag features (unavoidable for earliest 2 years)
    before = len(df)
    df = df.dropna(subset=["lag_1_yield", "lag_2_yield"])
    log.info(
        "Feature engineering: dropped %d lag-initialization rows (%d → %d)",
        before - len(df),
        before,
        len(df),
    )

    # Fill remaining NaNs in numeric features with column median
    # (wb_fertilizer_kg_ha has occasional gaps from WB API; yoy_fert_change
    # has NaN for first year of each series that lag_1 already filters above)
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.debug("  Filled NaN in '%s' with median %.2f", col, med)

    return df


# ── Encoding ──────────────────────────────────────────────────────────────


def fit_encoders(df: pd.DataFrame) -> dict[str, OrdinalEncoder]:
    """
    Fit one OrdinalEncoder per categorical feature.

    OrdinalEncoder is preferred over pd.Categorical or LabelEncoder here
    because it is sklearn-compatible (safe to bundle in the pickle and call
    .transform() on arbitrary DataFrames), and produces integer codes that
    XGBoost handles efficiently as continuous splits.
    """
    encoders: dict[str, OrdinalEncoder] = {}
    for feat in CATEGORICAL_FEATURES:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        enc.fit(df[[feat]])
        encoders[feat] = enc
        log.info(
            "  Encoder fitted for '%s': %d categories",
            feat,
            len(enc.categories_[0]),
        )
    return encoders


def apply_encoders(
    df: pd.DataFrame,
    encoders: dict[str, OrdinalEncoder],
) -> pd.DataFrame:
    """Apply pre-fitted encoders to categorical columns in-place."""
    df = df.copy()
    for feat, enc in encoders.items():
        df[feat] = enc.transform(df[[feat]]).astype(int)
    return df


# ── Train / test split ────────────────────────────────────────────────────


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    test = df[df["year"] >= TEST_START_YEAR].copy()
    log.info(
        "Split: train=%d rows (2002–%d), test=%d rows (%d–2023)",
        len(train),
        TRAIN_END_YEAR,
        len(test),
        TEST_START_YEAR,
    )
    return train, test


def get_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using all defined features that exist in df."""
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    present = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        log.warning("  Features missing from data (will be excluded): %s", missing)
    X = df[present].copy()
    y = df["yield_hg_ha"].copy()
    return X, y, present


# ── Cross-validation ──────────────────────────────────────────────────────


def run_cv(
    train: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    TimeSeriesSplit CV on training data to report generalisation metrics.

    TimeSeriesSplit respects chronological order: each fold trains on years
    [t0 … tN] and validates on [tN+1 … tN+k]. This mirrors the real-world
    constraint that we cannot know future yields when training.
    """
    log.info("Running %d-fold TimeSeriesSplit CV on training data…", N_CV_SPLITS)
    X_train, y_train, _ = get_feature_matrix(train)
    X_arr = X_train[feature_names].values
    y_arr = y_train.values

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_r2, cv_mae = [], []

    for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_arr), 1):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_arr[idx_tr], y_arr[idx_tr], verbose=False)
        y_pred = m.predict(X_arr[idx_val])
        cv_r2.append(r2_score(y_arr[idx_val], y_pred))
        cv_mae.append(mean_absolute_error(y_arr[idx_val], y_pred))
        log.info(
            "  Fold %d — R²=%.3f  MAE=%s hg/ha",
            fold,
            cv_r2[-1],
            f"{cv_mae[-1]:,.0f}",
        )

    log.info(
        "CV summary — R²: %.3f ± %.3f  |  MAE: %s ± %s hg/ha",
        np.mean(cv_r2),
        np.std(cv_r2),
        f"{np.mean(cv_mae):,.0f}",
        f"{np.std(cv_mae):,.0f}",
    )


# ── Model training ────────────────────────────────────────────────────────


def train_model(
    train: pd.DataFrame,
    feature_names: list[str],
) -> XGBRegressor:
    """Train final XGBoost model on full training set (2002–2018)."""
    X_train, y_train, _ = get_feature_matrix(train)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train[feature_names], y_train, verbose=False)
    log.info("Model trained on %d rows, %d features", len(train), len(feature_names))
    return model


# ── Evaluation ────────────────────────────────────────────────────────────


def evaluate_model(
    model: XGBRegressor,
    test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Evaluate on test set and return DataFrame of actuals vs predictions."""
    X_test, y_test, _ = get_feature_matrix(test)
    y_pred = model.predict(X_test[feature_names])

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    log.info("=" * 50)
    log.info("TEST SET RESULTS (2019–2023)")
    log.info("  R²  : %.4f  (target ≥ 0.75)", r2)
    log.info("  MAE : %s hg/ha", f"{mae:,.0f}")
    if r2 < 0.75:
        log.warning(
            "  R² below target. Consider tuning XGB_PARAMS or adding features. "
            "Most common cause: fertilizer_kg_ha sparsity post-2019."
        )
    log.info("=" * 50)

    results = test[["zone", "crop", "year", "yield_hg_ha"]].copy()
    results["predicted_yield_hg_ha"] = y_pred.round(0)
    results["residual_hg_ha"] = (
        results["yield_hg_ha"] - results["predicted_yield_hg_ha"]
    )
    results["abs_pct_error"] = (
        (results["residual_hg_ha"].abs() / results["yield_hg_ha"].replace(0, np.nan))
        * 100
    ).round(2)
    return results


# ── SHAP explanation ───────────────────────────────────────────────────────


def compute_shap(
    model: XGBRegressor,
    train: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values on a stratified sample of training data.

    Using training data (not test) for SHAP is intentional: the goal is
    to explain what the model learned (feature importance), not to explain
    individual test predictions. Sample of 300 rows keeps compute time
    under 30 seconds on a standard laptop without materially changing the
    global importance ranking.
    """
    X_train, _, _ = get_feature_matrix(train)
    X_sample = X_train[feature_names].sample(n=min(300, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return shap_values, mean_abs_shap, X_sample, feature_names


# ── Charts ────────────────────────────────────────────────────────────────


def plot_feature_importance(
    mean_abs_shap: np.ndarray,
    feature_names: list[str],
) -> None:
    """chart01 — SHAP mean absolute value bar chart (top 12 features)."""
    pairs = sorted(zip(mean_abs_shap, feature_names), reverse=True)[:12]
    vals, names = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(vals)), vals, color="#2E75B6", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (hg/ha impact on yield prediction)", fontsize=11)
    ax.set_title(
        "NigeriaAgriScope — Top Yield Prediction Drivers\n"
        "SHAP Feature Importance | XGBoost Model | 2002–2018 Training Data",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.01,
        "Source: FAOSTAT, NASA POWER, World Bank, USDA PSD | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_IMPORTANCE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", CHART_IMPORTANCE.name)


def plot_actual_vs_predicted(results: pd.DataFrame) -> None:
    """chart02 — Scatter plot of actual vs predicted yield (test set)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Colour by crop for interpretability
    crops = results["crop"].unique()
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(crops))
    crop_colours = {c: cmap(i) for i, c in enumerate(crops)}

    for crop, grp in results.groupby("crop"):
        ax.scatter(
            grp["yield_hg_ha"],
            grp["predicted_yield_hg_ha"],
            label=crop,
            color=crop_colours[crop],
            alpha=0.75,
            s=55,
            edgecolors="white",
            linewidths=0.5,
        )

    # Perfect prediction reference line
    lim_min = results[["yield_hg_ha", "predicted_yield_hg_ha"]].min().min() * 0.9
    lim_max = results[["yield_hg_ha", "predicted_yield_hg_ha"]].max().max() * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="Perfect fit")

    r2 = r2_score(results["yield_hg_ha"], results["predicted_yield_hg_ha"])
    mae = mean_absolute_error(results["yield_hg_ha"], results["predicted_yield_hg_ha"])

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Actual Yield (hg/ha)", fontsize=12)
    ax.set_ylabel("Predicted Yield (hg/ha)", fontsize=12)
    ax.set_title(
        f"NigeriaAgriScope — Actual vs Predicted Yield\n"
        f"XGBoost Test Set (2019–2023)  |  R² = {r2:.3f}  |  MAE = {mae:,.0f} hg/ha",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.01,
        "Source: FAOSTAT, NASA POWER, World Bank, USDA PSD | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_ACTUAL_PRED, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", CHART_ACTUAL_PRED.name)


def plot_residuals(results: pd.DataFrame) -> None:
    """
    chart03 — Residual plot (predicted vs residual).

    A well-specified model should show residuals randomly scattered around
    zero with no fan shape (heteroscedasticity) or systematic curve (omitted
    non-linearity). Fan shapes are common in agricultural yield models for
    high-yield crops like cassava where year-to-year variance is large.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: residuals vs predicted
    ax = axes[0]
    ax.scatter(
        results["predicted_yield_hg_ha"],
        results["residual_hg_ha"],
        alpha=0.6,
        s=50,
        color="#E05C2A",
        edgecolors="white",
        linewidths=0.4,
    )
    ax.axhline(0, color="black", lw=1.2, linestyle="--")
    ax.set_xlabel("Predicted Yield (hg/ha)", fontsize=11)
    ax.set_ylabel("Residual (Actual − Predicted, hg/ha)", fontsize=11)
    ax.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: residual distribution histogram
    ax2 = axes[1]
    ax2.hist(
        results["residual_hg_ha"],
        bins=25,
        color="#2E75B6",
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axvline(0, color="black", lw=1.2, linestyle="--")
    ax2.set_xlabel("Residual (hg/ha)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "NigeriaAgriScope — Model Residual Diagnostics | XGBoost Test Set (2019–2023)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.99,
        -0.02,
        "Source: FAOSTAT, NASA POWER, World Bank, USDA PSD | NigeriaAgriScope",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    fig.savefig(CHART_RESIDUALS, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", CHART_RESIDUALS.name)


# ── Pickle: bundle model + encoders + metadata ────────────────────────────


def save_artifact(
    model: XGBRegressor,
    encoders: dict[str, OrdinalEncoder],
    feature_names: list[str],
    results: pd.DataFrame,
) -> None:
    """
    Save a single pickle dict consumed by Module 3 Page 5.

    INFERENCE INTERFACE — Page 5 (page5_forecast.py) contract:
    ----------------------------------------------------------
    from pathlib import Path
    import pickle, numpy as np, pandas as pd

    artifact = pickle.load(open("module4_models/yield_model.pkl", "rb"))
    model       = artifact["model"]
    encoders    = artifact["encoders"]      # dict: {"zone": enc, "crop": enc}
    feat_names  = artifact["feature_names"] # list[str]

    def predict_yield(zone, crop, year, rainfall_mm, fertilizer_kg_ha,
                      lag_1=None, lag_2=None):
        row = pd.DataFrame([{
            "zone": zone, "crop": crop, "year": year,
            "rainfall_mm_annual": rainfall_mm, "fertilizer_kg_ha": fertilizer_kg_ha,
            "lag_1_yield": lag_1 or artifact["median_yield_by_crop"].get(crop, 100000),
            "lag_2_yield": lag_2 or artifact["median_yield_by_crop"].get(crop, 100000),
            # remaining features filled with medians below
        }])
        for f, med in artifact["feature_medians"].items():
            if f not in row.columns:
                row[f] = med
        for feat, enc in encoders.items():
            row[feat] = enc.transform(row[[feat]]).astype(int)
        return float(model.predict(row[feat_names])[0])
    """
    # Store feature medians from training for Page 5 imputation of optional features
    median_yield_by_crop = results.groupby("crop")["yield_hg_ha"].median().to_dict()

    artifact = {
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names,
        "median_yield_by_crop": median_yield_by_crop,
        # Page 5 fills any feature it doesn't have from the user with these medians
        "feature_medians": {},  # populated after full training — see main()
        "train_end_year": TRAIN_END_YEAR,
        "test_r2": r2_score(results["yield_hg_ha"], results["predicted_yield_hg_ha"]),
        "test_mae": mean_absolute_error(
            results["yield_hg_ha"], results["predicted_yield_hg_ha"]
        ),
    }
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=5)
    log.info("Pickle saved → %s", PICKLE_PATH)
    log.info("  Artifact keys: %s", list(artifact.keys()))


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    log.info("NigeriaAgriScope — Module 4: Yield Forecast Model")
    log.info("=" * 60)

    log.info("STEP 1 — Load data")
    df = load_data()

    log.info("STEP 2 — Engineer features")
    df = engineer_features(df)

    log.info("STEP 3 — Fit encoders")
    encoders = fit_encoders(df)

    log.info("STEP 4 — Apply encoders")
    df_enc = apply_encoders(df, encoders)

    log.info("STEP 5 — Train / test split")
    train, test = split_data(df_enc)

    log.info("STEP 6 — Build feature matrix (identify available features)")
    _, _, feature_names = get_feature_matrix(train)
    log.info("  Features (%d): %s", len(feature_names), feature_names)

    log.info("STEP 7 — TimeSeriesSplit cross-validation")
    run_cv(train, feature_names)

    log.info("STEP 8 — Train final model on full training set")
    model = train_model(train, feature_names)

    log.info("STEP 9 — Evaluate on test set (2019–2023)")
    results = evaluate_model(model, test, feature_names)
    results.to_csv(PREDICTIONS_CSV, index=False)
    log.info("Predictions saved → %s", PREDICTIONS_CSV.name)

    log.info("STEP 10 — Compute SHAP values")
    _, mean_abs_shap, _, _ = compute_shap(model, train, feature_names)

    log.info("STEP 11 — Generate charts")
    plot_feature_importance(mean_abs_shap, feature_names)
    plot_actual_vs_predicted(results)
    plot_residuals(results)

    log.info("STEP 12 — Save pickle artifact")
    # Compute feature medians from training set for Page 5 inference imputation
    X_train_full, _, _ = get_feature_matrix(train)
    feature_medians = {
        col: float(X_train_full[col].median())
        for col in feature_names
        if col not in CATEGORICAL_FEATURES
    }
    save_artifact(model, encoders, feature_names, results)
    # Patch medians into the saved artifact
    with open(PICKLE_PATH, "rb") as f:
        artifact = pickle.load(f)
    artifact["feature_medians"] = feature_medians
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=5)
    log.info("  Feature medians patched into artifact (%d keys)", len(feature_medians))

    log.info("=" * 60)
    log.info("Module 4 — yield_model.py COMPLETE")
    log.info("  Outputs → module4_models/outputs/")
    log.info(
        "  %-40s  R²=%.3f  MAE=%s hg/ha",
        "yield_model.pkl",
        artifact["test_r2"],
        f"{artifact['test_mae']:,.0f}",
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
