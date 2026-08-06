from __future__ import annotations
import importlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
from ml_training import registry_helper
# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATASET      = Path(__file__).resolve().parent / "data" / "processed_overall.csv"
IS_OFFICIAL  = False
SCRIPT_NAME  = "train-1"
VARIANT_ID   = registry_helper.next_variant_id()
ARTIFACT_DIR = registry_helper.get_variant_dir(VARIANT_ID)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE     = 42
TARGET           = "selling_price"
MIN_SEGMENT_ROWS = 200
SEGMENTS: dict[str, tuple[int, int]] = {
    "0_6_lakh":     (0,           600_000),
    "6_12_lakh":    (600_000,   1_200_000),
    "12_plus_lakh": (1_200_000, 20_000_000),
}
CAT_FEATURES: list[str] = [
    "brand", "model", "variant",
    "locality", "rto",
    "fuel_type", "transmission", "seller_type", "color",
]
NUMERIC_FEATURES: list[str] = [
    "vehicle_age", "odometer_reading", "km_per_year", "owner_count",
    "certified", "pincode",
]
FEATURES: list[str] = CAT_FEATURES + NUMERIC_FEATURES
CB_PARAMS: dict[str, Any] = {
    "iterations": 3000, "learning_rate": 0.03, "depth": 6,
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE, "l2_leaf_reg": 10,
    "min_data_in_leaf": 25, "early_stopping_rounds": 150,
}
LGB_PARAMS: dict[str, Any] = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.03, "num_leaves": 48,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_child_samples": 25,
    "lambda_l2": 5.0, "verbosity": -1, "seed": RANDOM_STATE,
}
LGB_ROUNDS = 3000
XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.03, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 25, "lambda": 5.0, "seed": RANDOM_STATE,
}
XGB_ROUNDS = 3000
SEG_CB_PARAMS: dict[str, Any] = {
    "iterations": 2000, "learning_rate": 0.03, "depth": 7,
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE, "l2_leaf_reg": 5,
    "min_data_in_leaf": 10, "early_stopping_rounds": 100,
}
# ── LOGGER ─────────────────────────────────────────────────────────────────────
# Console → INFO-only, concise.
def _setup_logger(artifact_dir: Path) -> logging.Logger:
    log_path = artifact_dir / "training.log"
    fmt_file    = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fmt_console = logging.Formatter("%(message)s")
    file_handler    = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt_file)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt_console)
    logger = logging.getLogger(f"autopricer.{SCRIPT_NAME}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
log = _setup_logger(ARTIFACT_DIR)
# ── RANDOM SEEDS ───────────────────────────────────────────────────────────────
def _set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.debug(f"Random seed set to {seed}")
_set_seeds()
# ── WEIGHTS & BIASES ───────────────────────────────────────────────────────────
_wandb_mod = None
try:
    _wandb_mod = importlib.import_module("wandb")
except Exception:
    _wandb_mod = None
_WANDB_ENABLED = False
_wandb_run     = None
def _init_wandb(config: dict) -> None:
    global _WANDB_ENABLED, _wandb_run
    if _wandb_mod is None:
        log.debug("W&B package not installed — skipping experiment tracking")
        return
    try:
        _wandb_run = _wandb_mod.init(
            project="autopricer-ml",
            name=f"{SCRIPT_NAME}-v{VARIANT_ID}-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=config, tags=[SCRIPT_NAME, "variant"], reinit=True,
        )
        _WANDB_ENABLED = True
        log.debug("W&B run initialised")
    except Exception as exc:
        log.debug(f"W&B unavailable: {exc}")
def _wandb_log(data: dict) -> None:
    if _WANDB_ENABLED and _wandb_run:
        try:
            _wandb_run.log(data)
        except Exception:
            pass
def _wandb_log_artifact(path: Path, artifact_type: str = "dataset") -> None:
    if _WANDB_ENABLED and _wandb_run and _wandb_mod:
        try:
            art = _wandb_mod.Artifact(path.name, type=artifact_type)
            art.add_file(str(path))
            _wandb_run.log_artifact(art)
        except Exception:
            pass
def _finish_wandb() -> None:
    if _WANDB_ENABLED and _wandb_run:
        try:
            _wandb_run.finish()
        except Exception:
            pass
# ── METRICS ───────────────────────────────────────────────────────────────────
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred)
    mae  = mean_absolute_error(y_true_price, y_pred_price)
    rmse = math.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2   = r2_score(np.log1p(y_true_price), np.log1p(y_pred_price))
    mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4), "MAPE": round(mape, 2)}
# ── DATASET ────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    df = pd.read_csv(DATASET, low_memory=False)
    log.debug(f"Loaded {len(df):,} rows from {DATASET.name}")
    return df
def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(50_000, 20_000_000)]
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = [c for c in ["vehicle_age", "odometer_reading", "km_per_year", "owner_count"]
                if c in df.columns]
    df = df.dropna(subset=required)
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    log.debug(f"Clean: {before:,} → {len(df):,} rows (dropped {before - len(df):,})")
    return df
# ── SPLIT & FRAMES ─────────────────────────────────────────────────────────────
def split_dataset(df: pd.DataFrame):
    X = df[[f for f in FEATURES if f in df.columns]]
    y = np.log1p(df[TARGET])
    return train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, shuffle=True)
def build_category_levels(df: pd.DataFrame) -> dict:
    levels = {}
    for col in CAT_FEATURES:
        if col not in df.columns:
            levels[col] = ["unknown"]
            continue
        vals = df[col].astype(str).fillna("unknown").unique().tolist()
        if "unknown" not in vals:
            vals.append("unknown")
        levels[col] = sorted(vals)
    return levels
def prepare_frames(df: pd.DataFrame, category_levels: dict, encoders: dict | None = None):
    available = [f for f in FEATURES if f in df.columns]
    frame = df[available].copy()
    for f in FEATURES:
        if f not in frame.columns:
            frame[f] = "unknown" if f in CAT_FEATURES else 0
    for col in CAT_FEATURES:
        known = set(category_levels.get(col, ["unknown"]))
        frame[col] = frame[col].astype(str).apply(lambda x: x if x in known else "unknown")
    for col in NUMERIC_FEATURES:
        if col in frame.columns:
            med = frame[col].median()
            frame[col] = frame[col].fillna(0 if pd.isna(med) else med)
    cb_frame  = frame.copy()
    lgb_frame = frame.copy()
    active_encoders: dict = {}
    for col in CAT_FEATURES:
        if encoders is None:
            enc = LabelEncoder()
            enc.fit(category_levels.get(col, ["unknown"]))
            active_encoders[col] = enc
        else:
            active_encoders[col] = encoders[col]
        lgb_frame[col] = active_encoders[col].transform(lgb_frame[col])
    xgb_frame = lgb_frame.copy()
    return cb_frame, lgb_frame, xgb_frame, active_encoders
def prepare_training_frames(X_train: pd.DataFrame, X_val: pd.DataFrame) -> dict:
    cat_levels = build_category_levels(X_train)
    cb_tr, lgb_tr, xgb_tr, encoders = prepare_frames(X_train, cat_levels)
    cb_v,  lgb_v,  xgb_v,  _        = prepare_frames(X_val,   cat_levels, encoders)
    return {
        "category_levels": cat_levels, "encoders": encoders,
        "catboost": {"train": cb_tr, "val": cb_v},
        "lightgbm": {"train": lgb_tr, "val": lgb_v},
        "xgboost":  {"train": xgb_tr, "val": xgb_v},
    }
# ── MODEL TRAINING ─────────────────────────────────────────────────────────────
def train_catboost(X_tr, y_tr, X_v, y_v) -> CatBoostRegressor:
    log.debug(f"CatBoost params: {CB_PARAMS}")
    cat_cols = [c for c in CAT_FEATURES if c in X_tr.columns]
    model = CatBoostRegressor(**CB_PARAMS, verbose=200)
    model.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
              eval_set=Pool(X_v, y_v, cat_features=cat_cols), use_best_model=True)
    return model
def train_lightgbm(X_tr, y_tr, X_v, y_v) -> lgb.Booster:
    log.debug(f"LightGBM params: {LGB_PARAMS}")
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr),
                      valid_sets=[lgb.Dataset(X_v, label=y_v)], num_boost_round=LGB_ROUNDS,
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    return model
def train_xgboost(X_tr, y_tr, X_v, y_v) -> xgb.Booster:
    log.debug(f"XGBoost params: {XGB_PARAMS}")
    model = xgb.train(XGB_PARAMS, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=XGB_ROUNDS,
                      evals=[(xgb.DMatrix(X_v, label=y_v), "val")],
                      early_stopping_rounds=150, verbose_eval=200)
    return model
# ── EVALUATION ─────────────────────────────────────────────────────────────────
def predict(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    if model_name == "CatBoost": return model.predict(X)
    if model_name == "LightGBM": return model.predict(X)
    if model_name == "XGBoost":  return model.predict(xgb.DMatrix(X))
    raise ValueError(f"Unknown model: {model_name}")
def evaluate_model(model, model_name: str, X, y) -> tuple[dict, np.ndarray]:
    preds  = predict(model, model_name, X)
    scores = calculate_metrics(y, preds)
    log.debug(f"{model_name} MAE=₹{scores['MAE']:,.0f} MAPE={scores['MAPE']:.2f}% R2={scores['R2']:.4f}")
    return scores, preds
def optimise_weights(cb_p, lgb_p, xgb_p, y_true) -> np.ndarray:
    def neg_r2(w):
        w = np.array(w) / np.sum(w)
        ens = w[0]*cb_p + w[1]*lgb_p + w[2]*xgb_p
        return -r2_score(np.log1p(np.expm1(y_true)), np.log1p(np.expm1(ens)))
    res = minimize(neg_r2, x0=[1/3, 1/3, 1/3], method="SLSQP",
                   bounds=[(0, 1)]*3, constraints={"type": "eq", "fun": lambda w: sum(w)-1})
    return res.x / res.x.sum()
def evaluate_ensemble(w, cb, lgb_p, xgb_p, y) -> dict:
    ens = w[0]*cb + w[1]*lgb_p + w[2]*xgb_p
    return calculate_metrics(y, ens)
# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────────
def _save_feature_importance(importance: dict[str, float], model_name: str) -> None:
    sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    csv_path   = ARTIFACT_DIR / f"feature_importance_{model_name.lower()}.csv"
    pd.DataFrame(sorted_imp.items(), columns=["feature", "importance"]).to_csv(csv_path, index=False)
    log.debug(f"Saved feature importance: {csv_path.name}")
    _wandb_log_artifact(csv_path, artifact_type="importance")
def generate_feature_importances(cb_model, lgb_model, xgb_model, features: list[str]) -> dict:
    importances: dict = {}
    try:
        cb_imp = dict(zip(features, cb_model.get_feature_importance()))
        _save_feature_importance(cb_imp, "catboost")
        importances["catboost"] = dict(sorted(cb_imp.items(), key=lambda x: x[1], reverse=True)[:10])
    except Exception as e:
        log.warning(f"CatBoost feature importance failed: {e}")
    try:
        lgb_imp = dict(zip(lgb_model.feature_name(), lgb_model.feature_importance("gain").tolist()))
        _save_feature_importance(lgb_imp, "lightgbm")
        importances["lightgbm"] = dict(sorted(lgb_imp.items(), key=lambda x: x[1], reverse=True)[:10])
    except Exception as e:
        log.warning(f"LightGBM feature importance failed: {e}")
    try:
        xgb_imp = xgb_model.get_score(importance_type="gain")
        _save_feature_importance(xgb_imp, "xgboost")
        importances["xgboost"] = dict(sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10])
    except Exception as e:
        log.warning(f"XGBoost feature importance failed: {e}")
    return importances
# ── ARTIFACTS ──────────────────────────────────────────────────────────────────
def save_artifacts(cat_model, lgb_model, xgb_model, weights, cat_levels, encoders,
                   rows_used: int, val_metrics: dict, seg_active: list[str]) -> None:
    cat_model.save_model(str(ARTIFACT_DIR / "vehicle_price_catboost.cbm"))
    lgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_lightgbm.txt"))
    xgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_xgboost.json"))
    bundle = {
        "weights": {"catboost": float(weights[0]), "lightgbm": float(weights[1]), "xgboost": float(weights[2])},
        "category_levels": cat_levels,
        "encoders": {col: enc.classes_.tolist() for col, enc in encoders.items()},
        "features": FEATURES, "cat_features": CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES, "segments": SEGMENTS,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "ensemble_bundle.pkl")
    model_metadata = {
        "variant_id":       VARIANT_ID,
        "script":           SCRIPT_NAME,
        "dataset_name":     DATASET.name,
        "rows_used":        rows_used,
        "features":         FEATURES,
        "cat_features":     CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "val_metrics":      val_metrics,
        "ensemble_weights": {
            "catboost": float(weights[0]),
            "lightgbm": float(weights[1]),
            "xgboost":  float(weights[2]),
        },
        "active_segment_models": seg_active,
        "hyperparameters": {
            "catboost": CB_PARAMS,
            "lightgbm": {**LGB_PARAMS, "num_boost_round": LGB_ROUNDS},
            "xgboost":  {**XGB_PARAMS, "num_boost_round": XGB_ROUNDS},
        },
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    log.debug("Artifacts saved")
# ── SEGMENTED TRAINING ─────────────────────────────────────────────────────────
def train_segment_model(seg_name, seg_df, global_model, global_cat_levels) -> tuple:
    log.debug(f"Segment: {seg_name} ({len(seg_df):,} rows)")
    if len(seg_df) < MIN_SEGMENT_ROWS:
        log.debug(f"Skip {seg_name} — only {len(seg_df)} rows")
        return None, None, None
    X = seg_df[[f for f in FEATURES if f in seg_df.columns]]
    y = np.log1p(seg_df[TARGET])
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    seg_levels     = build_category_levels(X_tr)
    cb_tr, _, _, _ = prepare_frames(X_tr, seg_levels)
    cb_v,  _, _, _ = prepare_frames(X_v,  seg_levels)
    cat_cols = [c for c in CAT_FEATURES if c in cb_tr.columns]
    seg_model = CatBoostRegressor(**SEG_CB_PARAMS, verbose=0)
    seg_model.fit(Pool(cb_tr, y_tr, cat_features=cat_cols),
                  eval_set=Pool(cb_v, y_v, cat_features=cat_cols), use_best_model=True)
    seg_scores    = calculate_metrics(y_v, seg_model.predict(cb_v))
    cb_v_g, _, _, _ = prepare_frames(X_v, global_cat_levels)
    global_scores   = calculate_metrics(y_v, global_model.predict(cb_v_g))
    if seg_scores["MAPE"] < global_scores["MAPE"]:
        log.debug(f"Segment {seg_name} ACTIVE (MAPE {seg_scores['MAPE']:.2f}% < {global_scores['MAPE']:.2f}%)")
        return seg_model, seg_levels, seg_scores
    log.debug(f"Segment {seg_name} inactive (global MAPE {global_scores['MAPE']:.2f}% ≤ {seg_scores['MAPE']:.2f}%)")
    return None, None, global_scores
def train_segmented_models(df, global_model, global_cat_levels) -> dict:
    results = {}
    for seg_name, (pmin, pmax) in SEGMENTS.items():
        mask   = df[TARGET].between(pmin, pmax)
        seg_df = df[mask].copy()
        m, lv, sc = train_segment_model(seg_name, seg_df, global_model, global_cat_levels)
        results[seg_name] = {
            "model": m, "cat_levels": lv, "scores": sc,
            "active": m is not None, "price_range": (pmin, pmax), "row_count": int(mask.sum()),
        }
    return results
def save_segment_artifacts(segment_results) -> tuple[dict, list[str]]:
    routing: dict   = {}
    active: list[str] = []
    for seg_name, r in segment_results.items():
        if r["active"]:
            r["model"].save_model(str(ARTIFACT_DIR / f"segment_{seg_name}.cbm"))
            joblib.dump(r["cat_levels"], ARTIFACT_DIR / f"segment_{seg_name}_levels.pkl")
            active.append(seg_name)
            routing[seg_name] = {
                "active":      True,
                "model_file":  f"segment_{seg_name}.cbm",
                "levels_file": f"segment_{seg_name}_levels.pkl",
                "price_range": r["price_range"],
                "row_count":   r["row_count"],
                "mape":        r["scores"]["MAPE"] if r["scores"] else None,
                "r2":          r["scores"]["R2"]   if r["scores"] else None,
            }
        else:
            routing[seg_name] = {
                "active":      False,
                "fallback":    "global",
                "price_range": r["price_range"],
                "row_count":   r["row_count"],
                "mape":        r["scores"]["MAPE"] if r["scores"] else None,
            }
    with open(ARTIFACT_DIR / "routing_table.json", "w") as f:
        json.dump(routing, f, indent=2)
    return routing, active
# ── CONSOLE SUMMARY ────────────────────────────────────────────────────────────
def _print_summary(df_train, df_val, cat_sc, lgb_sc, xgb_sc, val_scores,
                   weights, comparison, active_segs, top_features, total_sec) -> None:
    W = 72
    line = "─" * W
    header = f"  AutoPricer  ·  {SCRIPT_NAME}  ·  Variant {VARIANT_ID}"
    print(f"\n{'━' * W}")
    print(header)
    print(f"{'━' * W}")
    # Dataset
    print(f"\n  Dataset  : {DATASET.name}  ({len(df_train) + len(df_val):,} rows  →  train {len(df_train):,} / val {len(df_val):,})")
    print(f"  Features : {len(FEATURES)} total  ({len(CAT_FEATURES)} categorical, {len(NUMERIC_FEATURES)} numeric)\n")
    # Model metrics table
    print(f"  {line}")
    print(f"  {'Model':<14}  {'MAE (₹)':>12}  {'RMSE (₹)':>12}  {'MAPE (%)':>9}  {'R²':>7}")
    print(f"  {line}")
    for row in [
        ("CatBoost",  cat_sc),
        ("LightGBM",  lgb_sc),
        ("XGBoost",   xgb_sc),
        ("Ensemble ★", val_scores),
    ]:
        name, sc = row
        marker = "★" if "★" in name else " "
        print(f"  {name:<14}  {sc['MAE']:>12,.0f}  {sc['RMSE']:>12,.0f}  {sc['MAPE']:>8.2f}%  {sc['R2']:>7.4f}")
    print(f"  {line}")
    # Ensemble weights
    print(f"\n  Ensemble Weights")
    print(f"    CatBoost  {weights[0]*100:5.1f}%")
    print(f"    LightGBM  {weights[1]*100:5.1f}%")
    print(f"    XGBoost   {weights[2]*100:5.1f}%")
    # Segments
    print(f"\n  Segment Models  →  {'  '.join(active_segs) if active_segs else 'None (global only)'}")
    # Top features (CatBoost by default)
    cb_top = top_features.get("catboost", {})
    if cb_top:
        top5 = list(cb_top.items())[:5]
        max_score = max(v for _, v in top5)
        print(f"\n  Top Features (CatBoost importance)")
        for feat, score in top5:
            bar_len = int(score / max_score * 20)
            print(f"    {feat:<25}  {'█' * bar_len}  {score:,.0f}")
    # Artifacts
    artifacts = [
        "vehicle_price_catboost.cbm",
        "vehicle_price_lightgbm.txt",
        "vehicle_price_xgboost.json",
        "ensemble_bundle.pkl",
        "routing_table.json",
        "model_metadata.json",
        "model_comparison.csv",
        "feature_importance_catboost.csv",
        "feature_importance_lightgbm.csv",
        "feature_importance_xgboost.csv",
    ]
    print(f"\n  Artifacts  →  {ARTIFACT_DIR}")
    for a in artifacts:
        p = ARTIFACT_DIR / a
        exists = "✓" if p.exists() else "·"
        print(f"    {exists}  {a}")
    print(f"\n  Total training time  :  {total_sec:.1f}s")
    print(f"{'━' * W}\n")
# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────
def train_all_models() -> dict:
    t0 = time.perf_counter()
    log.info(f"AutoPricer ML Training [{SCRIPT_NAME}]  Variant {VARIANT_ID}")
    print(f"\nLoading {DATASET.name} ...")
    df = load_dataset()
    df = clean_training_data(df)
    X_train, X_val, y_train, y_val = split_dataset(df)
    frames     = prepare_training_frames(X_train, X_val)
    cat_levels = frames["category_levels"]
    _init_wandb({
        "script": SCRIPT_NAME, "dataset": DATASET.name, "variant_id": VARIANT_ID,
        "rows": len(df), "features": len(FEATURES), "random_seed": RANDOM_STATE,
        "catboost": CB_PARAMS,
        "lightgbm": {**LGB_PARAMS, "num_boost_round": LGB_ROUNDS},
        "xgboost":  {**XGB_PARAMS, "num_boost_round": XGB_ROUNDS},
    })
    # Train
    print("Training CatBoost ...")
    try:
        cat_model = train_catboost(frames["catboost"]["train"], y_train, frames["catboost"]["val"], y_val)
    except Exception:
        log.error("CatBoost training failed\n" + traceback.format_exc())
        raise
    print("Training LightGBM ...")
    try:
        lgb_model = train_lightgbm(frames["lightgbm"]["train"], y_train, frames["lightgbm"]["val"], y_val)
    except Exception:
        log.error("LightGBM training failed\n" + traceback.format_exc())
        raise
    print("Training XGBoost ...")
    try:
        xgb_model = train_xgboost(frames["xgboost"]["train"], y_train, frames["xgboost"]["val"], y_val)
    except Exception:
        log.error("XGBoost training failed\n" + traceback.format_exc())
        raise
    # Evaluate
    print("Evaluating ...")
    cat_sc, cat_p = evaluate_model(cat_model, "CatBoost", frames["catboost"]["val"], y_val)
    lgb_sc, lgb_p = evaluate_model(lgb_model, "LightGBM", frames["lightgbm"]["val"], y_val)
    xgb_sc, xgb_p = evaluate_model(xgb_model, "XGBoost",  frames["xgboost"]["val"],  y_val)
    weights    = optimise_weights(cat_p, lgb_p, xgb_p, y_val)
    val_scores = evaluate_ensemble(weights, cat_p, lgb_p, xgb_p, y_val)
    _wandb_log({
        "catboost_mae": cat_sc["MAE"],  "catboost_mape": cat_sc["MAPE"],  "catboost_r2": cat_sc["R2"],
        "lightgbm_mae": lgb_sc["MAE"],  "lightgbm_mape": lgb_sc["MAPE"],  "lightgbm_r2": lgb_sc["R2"],
        "xgboost_mae":  xgb_sc["MAE"],  "xgboost_mape":  xgb_sc["MAPE"],  "xgboost_r2":  xgb_sc["R2"],
        "ensemble_mae": val_scores["MAE"], "ensemble_mape": val_scores["MAPE"], "ensemble_r2": val_scores["R2"],
        "weights_catboost": float(weights[0]),
        "weights_lightgbm": float(weights[1]),
        "weights_xgboost":  float(weights[2]),
    })
    # Comparison CSV
    comparison = pd.DataFrame([
        {"Model": "CatBoost",  **cat_sc},
        {"Model": "LightGBM",  **lgb_sc},
        {"Model": "XGBoost",   **xgb_sc},
        {"Model": "Ensemble",  **val_scores},
    ]).sort_values("R2", ascending=False)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    log.debug("Saved model_comparison.csv")
    _wandb_log_artifact(ARTIFACT_DIR / "model_comparison.csv", artifact_type="report")
    # Feature importance (CSV only)
    print("Generating feature importances ...")
    top_features = generate_feature_importances(cat_model, lgb_model, xgb_model, FEATURES)
    # Segmented models
    print("Training segment models ...")
    seg_results   = train_segmented_models(df, cat_model, cat_levels)
    routing_table, active_segs = save_segment_artifacts(seg_results)
    val_metrics_full = {
        "CatBoost": cat_sc, "LightGBM": lgb_sc, "XGBoost": xgb_sc, "Ensemble": val_scores
    }
    save_artifacts(cat_model, lgb_model, xgb_model, weights, cat_levels,
                   frames["encoders"], len(df), val_metrics_full, active_segs)
    _wandb_log_artifact(ARTIFACT_DIR / "model_metadata.json", artifact_type="metadata")
    # Dataset catalog
    try:
        _cdf = pd.read_csv(DATASET, usecols=["brand", "model", "variant"])
        _cdf = _cdf.dropna(subset=["brand", "model"])
        for c in ["brand", "model", "variant"]:
            _cdf[c] = _cdf[c].astype(str).str.strip().str.lower()
        _STRIP_TOKENS = set([
            "petrol", "diesel", "crdi", "cng", "lpg", "electric", "ev", "vtvt", "tdci", "mpi", "dci", "ddis",
            "tsi", "tdi", "gdi", "tgdi", "cdti", "idtec", "ivtec", "k10", "k12", "k15", "boostjet", "smart", "hybrid",
            "at", "mt", "cvt", "dct", "amt", "ivt", "dsg", "automatic", "manual", "str", "shvs",
            "dsl", "ptl", "bs6", "bs4", "bsiv", "bs3", "unknown", "nan", "null", "none", "car", "model", "variant",
            "5sp", "6sp", "5-speed", "6-speed", "7-speed", "8-speed", "5mt", "6mt", "6at", "5at", "speed",
            "drive", "2wd", "4wd", "awd", "4x2", "4x4", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
        ])
        _ENGINE_PAT = re.compile(r"\b\d+\.\d+l?\b|\b\d{3,4}cc?\b|\b\d+\.\d+\b")
        def _norm_var(v_raw: str, m_name: str = "") -> str:
            if not v_raw or not isinstance(v_raw, str):
                return ""
            t = v_raw.lower().strip()
            if t in ("unknown", "nan", "null", "none", "-", "", "base model"):
                return ""
            if m_name:
                for w in m_name.lower().split():
                    if len(w) > 2:
                        t = t.replace(w, "")
            t = _ENGINE_PAT.sub("", t)
            t = re.sub(r"[\(\)\[\]\/\-\,\_\.\+]", " ", t)
            toks = [tk for tk in t.split() if tk not in _STRIP_TOKENS and not tk.isdigit() and len(tk) > 0]
            if not toks:
                return ""
            res = " ".join(toks).upper()
            res = re.sub(r"\bSX\s+O\b", "SX (O)", res)
            res = re.sub(r"\bS\s+O\b", "S (O)", res)
            res = re.sub(r"\bZX\s+O\b", "ZX (O)", res)
            res = re.sub(r"\bZXI\s+PLUS\b", "ZXI+", res)
            res = re.sub(r"\bVXI\s+PLUS\b", "VXI+", res)
            res = re.sub(r"\bLXI\s+PLUS\b", "LXI+", res)
            res = re.sub(r"\bXZ\s+PLUS\b", "XZ+", res)
            res = re.sub(r"\bXT\s+PLUS\b", "XT+", res)
            return res
        catalog: dict = {}
        for brand, bdf in _cdf.groupby("brand"):
            catalog[brand] = {}
            for model_, mdf in bdf.groupby("model"):
                norm_vars = set()
                for v in mdf["variant"].dropna():
                    nv = _norm_var(v, model_)
                    if nv:
                        norm_vars.add(nv)
                catalog[brand][model_] = sorted(list(norm_vars), key=str.casefold)
        with open(ARTIFACT_DIR / "dataset_catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        log.debug(f"Saved dataset_catalog.json ({len(catalog)} brands)")
    except Exception as exc:
        log.warning(f"dataset_catalog.json failed: {exc}")
    # Registry
    registry_helper.register_variant(
        variant_id=VARIANT_ID, artifact_dir=ARTIFACT_DIR,
        dataset_name=DATASET.name,
        metrics={"mae": val_scores["MAE"], "rmse": val_scores["RMSE"],
                 "r2": val_scores["R2"],   "mape": val_scores["MAPE"]},
    )
    _finish_wandb()
    total_sec = time.perf_counter() - t0
    log.info(f"COMPLETE — Ensemble MAPE {val_scores['MAPE']:.2f}%  R2 {val_scores['R2']:.4f}  ({total_sec:.1f}s)")
    _print_summary(X_train, X_val, cat_sc, lgb_sc, xgb_sc, val_scores,
                   weights, comparison, active_segs, top_features, total_sec)
    return {"comparison": comparison, "segments": seg_results}
if __name__ == "__main__":
    try:
        train_all_models()
    except Exception:
        log.critical("Training pipeline crashed", exc_info=True)
        sys.exit(1)
