"""
ml_training/train-s5.py
S5 Quality Shop — Specialized ML Training Pipeline

Source  : ml_training/data/processed_s5.csv (173 rows, age 0-7)
Output  : model_registry/variant_s5/
           └─ vehicle_price_catboost.cbm
           └─ vehicle_price_lightgbm.txt
           └─ ensemble_bundle.pkl
           └─ model_metadata.json

Training notes:
- Small dataset (173 rows) → high regularization to prevent overfitting
- NO segment sub-models (too few rows per segment)
- CatBoost + LightGBM ensemble (no XGBoost — too few rows)
- 70/30 train/val split to preserve test quality
- Registers as 'variant_s5' in model_registry/registry.json
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
from ml_training import registry_helper

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATASET      = Path(__file__).resolve().parent / "data" / "processed_s5.csv"
SCRIPT_NAME  = "train-s5"
VARIANT_ID   = "variant_4"
ARTIFACT_DIR = ROOT / "model_registry" / VARIANT_ID
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = 42
TARGET        = "selling_price"
LOG_TARGET    = "log_selling_price"

# S5 features — aligned to main model features for runtime compatibility
CAT_FEATURES: list[str] = [
    "brand", "model", "variant",
    "city", "rto",
    "fuel_type", "transmission", "seller_type", "color",
]
NUMERIC_FEATURES: list[str] = [
    "vehicle_age", "odometer_reading", "km_per_year", "owner_count", "certified",
]
FEATURES: list[str] = CAT_FEATURES + NUMERIC_FEATURES

# Small-dataset tuned hyperparams — reduced depth, moderate regularization
CB_PARAMS: dict[str, Any] = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 5,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE,
    "l2_leaf_reg": 12,
    "min_data_in_leaf": 3,
    "early_stopping_rounds": 200,
    "verbose": 0,
}
LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 24,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "min_child_samples": 3,
    "lambda_l2": 10.0,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}
LGB_ROUNDS = 3000


# ── LOGGER ─────────────────────────────────────────────────────────────────────
def _setup_logger(artifact_dir: Path) -> logging.Logger:
    log_path = artifact_dir / "training_s5.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"s5.{SCRIPT_NAME}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


log = _setup_logger(ARTIFACT_DIR)


def _set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_set_seeds()


# ── METRICS ────────────────────────────────────────────────────────────────────
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


# ── DATA LOADING ────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    log.info(f"Loading: {DATASET}")
    df = pd.read_csv(DATASET)
    log.info(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # Ensure all required features exist
    for col in FEATURES:
        if col not in df.columns:
            if col in CAT_FEATURES:
                df[col] = "unknown"
            else:
                df[col] = 0.0

    # Fill nulls
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use log target if available; otherwise compute it
    if LOG_TARGET in df.columns:
        y_log = df[LOG_TARGET].values
    else:
        y_log = np.log1p(df[TARGET].values)

    y = df[TARGET].values
    X = df[FEATURES].copy()

    log.info(f"  Price range: ₹{y.min():,.0f} – ₹{y.max():,.0f}, Median: ₹{np.median(y):,.0f}")
    return X, y, y_log, df


# ── BUILD CAT LEVELS MAP (for inference) ───────────────────────────────────────
def build_cat_levels(X: pd.DataFrame) -> dict:
    return {col: sorted(X[col].unique().tolist()) for col in CAT_FEATURES if col in X.columns}


# ── TRAIN CATBOOST ─────────────────────────────────────────────────────────────
def train_catboost(X_tr, y_tr_log, X_val, y_val_log) -> CatBoostRegressor:
    log.info("  Training CatBoost...")
    cat_idx = [X_tr.columns.tolist().index(c) for c in CAT_FEATURES if c in X_tr.columns]
    pool_tr  = Pool(X_tr, y_tr_log,  cat_features=cat_idx)
    pool_val = Pool(X_val, y_val_log, cat_features=cat_idx)
    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(pool_tr, eval_set=pool_val, verbose=200)
    return model


# ── TRAIN LIGHTGBM ─────────────────────────────────────────────────────────────
def train_lightgbm(X_tr, y_tr_log, X_val, y_val_log) -> lgb.Booster:
    log.info("  Training LightGBM...")
    cat_cols = [c for c in CAT_FEATURES if c in X_tr.columns]
    X_tr_lgb  = X_tr.copy()
    X_val_lgb = X_val.copy()
    for col in cat_cols:
        X_tr_lgb[col]  = X_tr_lgb[col].astype("category")
        X_val_lgb[col] = X_val_lgb[col].astype("category")

    ds_tr  = lgb.Dataset(X_tr_lgb,  label=y_tr_log,  categorical_feature=cat_cols, free_raw_data=False)
    ds_val = lgb.Dataset(X_val_lgb, label=y_val_log,  categorical_feature=cat_cols, free_raw_data=False, reference=ds_tr)

    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(400)]
    model = lgb.train(LGB_PARAMS, ds_tr, LGB_ROUNDS, valid_sets=[ds_val], callbacks=callbacks)
    return model


# ── ENSEMBLE WEIGHT OPTIMIZATION ───────────────────────────────────────────────
def optimize_weights(preds_dict: dict, y_true_log: np.ndarray) -> dict:
    keys = list(preds_dict.keys())
    mat  = np.column_stack([preds_dict[k] for k in keys])

    def loss(w):
        w = np.abs(w) / np.sum(np.abs(w))
        pred = mat @ w
        return float(np.sqrt(np.mean((pred - y_true_log) ** 2)))

    w0 = np.ones(len(keys)) / len(keys)
    res = minimize(loss, w0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6})
    w   = np.abs(res.x) / np.sum(np.abs(res.x))
    return {k: round(float(v), 4) for k, v in zip(keys, w)}


# ── PREDICT HELPERS ────────────────────────────────────────────────────────────
def cb_predict(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    Xc = X.copy()
    for col in CAT_FEATURES:
        if col in Xc.columns:
            Xc[col] = Xc[col].astype(str)
    return model.predict(Xc[model.feature_names_])


def lgb_predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    Xl = X.copy()
    for col in CAT_FEATURES:
        if col in Xl.columns:
            Xl[col] = Xl[col].astype("category")
    return model.predict(Xl)


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    DIV = "=" * 72
    log.info(DIV)
    log.info(f"  {SCRIPT_NAME}  ·  {VARIANT_ID}  ·  S5 Quality Shop Model")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(DIV)

    # ── Load data
    # pyrefly: ignore [bad-unpacking]
    X, y, y_log, df_full = load_data()
    cat_levels = build_cat_levels(X)

    # ── Train/val split (80/20 — small dataset needs most rows for training)
    X_tr, X_val, y_tr, y_val, y_tr_log, y_val_log = train_test_split(
        X, y, y_log, test_size=0.20, random_state=RANDOM_STATE,
    )
    log.info(f"\n  Train: {len(X_tr)} rows | Val: {len(X_val)} rows")

    # ── Train models
    log.info("\n[1/3] Training base models...")
    cb_model  = train_catboost(X_tr, y_tr_log, X_val, y_val_log)
    lgb_model = train_lightgbm(X_tr, y_tr_log, X_val, y_val_log)

    # ── Val predictions
    cb_pred_log  = cb_predict(cb_model, X_val)
    lgb_pred_log = lgb_predict(lgb_model, X_val)

    # ── Optimize ensemble weights
    log.info("\n[2/3] Optimizing ensemble weights...")
    weights = optimize_weights({"catboost": cb_pred_log, "lightgbm": lgb_pred_log}, y_val_log)
    log.info(f"  Weights: {weights}")

    # ── Ensemble predictions (on val set, in price space)
    ens_log  = weights["catboost"] * cb_predict(cb_model, X_val) + weights["lightgbm"] * lgb_predict(lgb_model, X_val)
    y_pred   = np.expm1(ens_log)

    # ── Metrics
    val_metrics = calc_metrics(y_val, y_pred)
    log.info(f"\n  Validation Metrics:")
    log.info(f"    MAPE : {val_metrics['mape']:.2f}%")
    log.info(f"    MAE  : ₹{val_metrics['mae']:,.0f}")
    log.info(f"    RMSE : ₹{val_metrics['rmse']:,.0f}")
    log.info(f"    R²   : {val_metrics['r2']:.4f}")

    # ── Save artifacts
    log.info("\n[3/3] Saving model artifacts...")

    cb_path  = ARTIFACT_DIR / "vehicle_price_catboost.cbm"
    lgb_path = ARTIFACT_DIR / "vehicle_price_lightgbm.txt"
    cb_model.save_model(str(cb_path))
    lgb_model.save_model(str(lgb_path))
    log.info(f"  ✓ CatBoost: {cb_path}")
    log.info(f"  ✓ LightGBM: {lgb_path}")

    bundle = {
        "catboost":   cb_model,
        "lightgbm":   lgb_model,
        "weights":    weights,
        "cat_levels": cat_levels,
        "cat_features": CAT_FEATURES,
        "features":   FEATURES,
    }
    bundle_path = ARTIFACT_DIR / "ensemble_bundle.pkl"
    joblib.dump(bundle, bundle_path)
    log.info(f"  ✓ Bundle: {bundle_path}")

    metadata = {
        "variant_id":    VARIANT_ID,
        "script":        SCRIPT_NAME,
        "dataset":       "processed_s5.csv",
        "trained_at":    datetime.now().isoformat(),
        "features":      FEATURES,
        "categorical_features": CAT_FEATURES,
        "model_type":    "CatBoost + LightGBM (S5 Small-Data Ensemble)",
        "weights":       weights,
        "metrics":       val_metrics,
        "train_rows":    int(len(X_tr)),
        "val_rows":      int(len(X_val)),
        "s5_max_age":    7,
        "notes": (
            "Specialized model for quality car shops selling 0-7 year old cars. "
            "High regularization for small dataset (173 rows). "
            "Only activated when vehicle_age <= 7 and the model/brand is known in the S5 dataset. "
            "Falls back to variant_1 (+8% premium) when vehicle is not in S5 catalog."
        ),
    }
    meta_path = ARTIFACT_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata: {meta_path}")

    # ── Register in registry.json (never auto-promotes to default — S5 is specialist only)
    registry_path = ROOT / "model_registry" / "registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"default": "variant_1", "variants": {}}

    registry["variants"][VARIANT_ID] = {
        "dataset":      "processed_s5.csv",
        "trained_at":   datetime.now().isoformat(),
        "metrics":      val_metrics,
        "artifact_path": f"model_registry/{VARIANT_ID}",
        "status":       "s5_quality",
        "s5_max_age":   7,
        "description":  "S5 Quality Shop — specialized model for 0-7 year premium vehicles (variant_4)",
        "activation_condition": "vehicle_age <= 7 AND model known in S5 catalog",
    }
    # Preserve existing default — S5 never becomes the global default
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"  Registered in registry.json as '{VARIANT_ID}' (status=s5_quality, default unchanged)")

    elapsed = time.time() - t0
    log.info(f"\n{DIV}")
    log.info(f"  S5 training complete in {elapsed:.1f}s")
    log.info(f"  MAPE: {val_metrics['mape']:.2f}% | R²: {val_metrics['r2']:.4f}")
    log.info(f"  Saved to: {ARTIFACT_DIR}")
    log.info(DIV)


if __name__ == "__main__":
    main()
