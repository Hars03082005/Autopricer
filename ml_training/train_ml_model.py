"""
PricerPoint ML Training Pipeline
Version 8.0

Models:
    • CatBoost
    • LightGBM
    • XGBoost

Changes from v7.0:
    • Segments switched from budget/economy/mid/premium to explicit
      price bands: 0-6 lakh, 6-12 lakh, 12+ lakh (matches business tiers)
    • Removed duplicated pipeline execution (was training everything twice)
    • encoders now saved as plain class lists (classes_.tolist()), not
      raw LabelEncoder objects — avoids "cannot pickle 'module' object"
      and makes ensemble_bundle.pkl portable across sklearn versions
"""

# pyrefly: ignore [invalid-syntax]
from __future__ import annotations
import json
import math
import warnings

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

from scipy.optimize import minimize

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# CONFIGURATION

ROOT        = Path(__file__).resolve().parents[1]
DATASET     = Path(__file__).resolve().parent / "data" / "processed_with owner filled.csv"
ARTIFACT_DIR = ROOT / "model_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
DIV = "=" * 80

# FEATURES

TARGET = "selling_price"

CAT_FEATURES = [
    "brand",
    "model",
    "variant",
    "color",
    "segment_class",
    "fuel_type",
    "transmission",
    "seller_type",
]

NUMERIC_FEATURES = [
    "vehicle_age",
    "odometer_reading",
    "km_per_year",
    "owner_count",
    "inspected",
    "brand_tier",
    "age_km_interaction",
    "ownership_trust_score",
    "vehicle_health_score",
    "is_high_mileage",
]

FEATURES = CAT_FEATURES + NUMERIC_FEATURES

# PRICE SEGMENTS  (business-driven lakh bands)

SEGMENTS = {
    "0_6_lakh":    (0,          600_000),
    "6_12_lakh":   (600_000,  1_200_000),
    "12_plus_lakh": (1_200_000, 20_000_000),
}

MIN_SEGMENT_ROWS = 300

# METRICS

def calculate_metrics(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(np.log1p(y_true), np.log1p(y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
        "MAPE": round(mape, 2),
    }

# CATEGORY LEVELS

def build_category_levels(df):
    levels = {}
    for col in CAT_FEATURES:
        values = df[col].astype(str).fillna("unknown").unique().tolist()
        if "unknown" not in values:
            values.append("unknown")
        levels[col] = sorted(values)
    return levels

# DATA PREPARATION

def prepare_frames(df, category_levels, encoders=None):
    """
    Prepares data splits.
    If encoders are provided, uses them to transform categoricals (Validation/Test/Inference).
    If None, fits encoders (Training phase only).
    """
    frame = df[FEATURES].copy()

    for col in CAT_FEATURES:
        known = set(category_levels[col])
        frame[col] = (
            frame[col].astype(str)
            .apply(lambda x: x if x in known else "unknown")
        )

    for col in NUMERIC_FEATURES:
        median = frame[col].median()
        frame[col] = frame[col].fillna(0 if np.isnan(median) else median)

    cb_frame = frame.copy()

    lgb_frame = frame.copy()
    active_encoders = {}

    for col in CAT_FEATURES:
        if encoders is None:
            enc = LabelEncoder()
            enc.fit(category_levels[col])
            active_encoders[col] = enc
        else:
            active_encoders[col] = encoders[col]

        lgb_frame[col] = active_encoders[col].transform(lgb_frame[col])

    xgb_frame = lgb_frame.copy()

    return cb_frame, lgb_frame, xgb_frame, active_encoders


def prepare_training_frames(X_train, X_val, X_test):
    print("\nPreparing model inputs with synchronized encoders …")

    category_levels = build_category_levels(X_train)

    cb_train, lgb_train, xgb_train, fitted_encoders = prepare_frames(X_train, category_levels, encoders=None)

    cb_val, lgb_val, xgb_val, _ = prepare_frames(X_val, category_levels, encoders=fitted_encoders)
    cb_test, lgb_test, xgb_test, _ = prepare_frames(X_test, category_levels, encoders=fitted_encoders)

    return {
        "category_levels": category_levels,
        "encoders": fitted_encoders,
        "catboost":  {"train": cb_train,  "val": cb_val,  "test": cb_test},
        "lightgbm":  {"train": lgb_train, "val": lgb_val, "test": lgb_test},
        "xgboost":   {"train": xgb_train, "val": xgb_val, "test": xgb_test},
    }

# LOAD / VALIDATE / CLEAN

def load_dataset():
    print(DIV)
    print("Loading Dataset")
    print(DIV)

    df = pd.read_csv(DATASET)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  • {col}")
    return df


def validate_dataset(df):
    print(f"\n{DIV}")
    print("VALIDATING DATASET")
    print(DIV)

    required = FEATURES + [TARGET]
    missing  = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns:\n{missing}")

    print("✓ All required columns present")

    derived = ["brand_tier", "age_km_interaction", "ownership_trust_score",
               "vehicle_health_score", "is_high_mileage"]
    for col in derived:
        if col not in df.columns:
            print(f"  ⚠  Derived feature '{col}' missing — run clean_data.py first")

    return df


def clean_training_data(df):
    print(f"\n{DIV}")
    print("CLEANING DATA")
    print(DIV)

    before = len(df)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(50_000, 20_000_000)]

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vehicle_age", "odometer_reading", "km_per_year", "owner_count"])
    df["inspected"] = df["inspected"].fillna(0).astype(int)

    df["brand_tier"]             = df["brand_tier"].fillna(1)
    df["age_km_interaction"]     = df["age_km_interaction"].fillna(0)
    df["ownership_trust_score"]  = df["ownership_trust_score"].fillna(75)
    df["vehicle_health_score"]   = df["vehicle_health_score"].fillna(50)
    df["is_high_mileage"]        = df["is_high_mileage"].fillna(0)

    for col in CAT_FEATURES:
        df[col] = (
            df[col].fillna("unknown").astype(str)
            .str.strip().str.lower()
        )

    print(f"Removed {before - len(df):,} invalid rows")
    print(f"Remaining rows : {len(df):,}")
    return df

# TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)

def split_dataset(df):
    print(f"\n{DIV}")
    print("TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)")
    print(DIV)

    X = df[FEATURES]
    y = np.log1p(df[TARGET])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test       : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# GLOBAL MODEL TRAINERS

def train_catboost(X_train, y_train, X_val, y_val):
    print("\nTraining CatBoost …")

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5,
        min_data_in_leaf=20,
        early_stopping_rounds=100,
        verbose=100,
    )
    model.fit(
        Pool(X_train, y_train, cat_features=CAT_FEATURES),
        eval_set=Pool(X_val, y_val, cat_features=CAT_FEATURES),
        use_best_model=True,
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM …")

    model = lgb.train(
        {
            "objective":        "regression",
            "metric":           "rmse",
            "learning_rate":    0.03,
            "num_leaves":       64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "min_child_samples": 20,
            "verbosity":        -1,
            "seed":             RANDOM_STATE,
        },
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost …")

    model = xgb.train(
        {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "learning_rate":    0.03,
            "max_depth":        8,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "seed":             RANDOM_STATE,
        },
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=3000,
        evals=[(xgb.DMatrix(X_val, label=y_val), "Validation")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    return model

# PREDICT / EVALUATE

def predict(model, model_name, X):
    if model_name == "CatBoost":
        return model.predict(X)
    if model_name == "LightGBM":
        return model.predict(X)
    if model_name == "XGBoost":
        return model.predict(xgb.DMatrix(X))
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, model_name, X, y, split_label="Validation"):
    preds  = predict(model, model_name, X)
    scores = calculate_metrics(y, preds)

    print(f"\n{'='*60}")
    print(f"{model_name}  [{split_label}]")
    print("="*60)
    print(f"MAE  : ₹{scores['MAE']:,.0f}")
    print(f"RMSE : ₹{scores['RMSE']:,.0f}")
    print(f"MAPE : {scores['MAPE']:.2f}%")
    print(f"R²   : {scores['R2']:.4f}")

    return scores, preds

# ENSEMBLE WEIGHT OPTIMISATION

def optimise_ensemble_weights(cb_preds, lgb_preds, xgb_preds, y_true):
    print(f"\n{DIV}")
    print("ENSEMBLE WEIGHT OPTIMISATION")
    print(DIV)

    def neg_r2(weights):
        w = np.array(weights)
        w = w / w.sum()
        ensemble = w[0]*cb_preds + w[1]*lgb_preds + w[2]*xgb_preds
        y_t = np.expm1(y_true)
        y_p = np.expm1(ensemble)
        return -r2_score(np.log1p(y_t), np.log1p(y_p))

    result = minimize(
        neg_r2,
        x0=[1/3, 1/3, 1/3],
        method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
    )

    weights = result.x / result.x.sum()
    print(f"CatBoost  weight : {weights[0]:.4f}  ({weights[0]*100:.1f}%)")
    print(f"LightGBM  weight : {weights[1]:.4f}  ({weights[1]*100:.1f}%)")
    print(f"XGBoost   weight : {weights[2]:.4f}  ({weights[2]*100:.1f}%)")

    return weights


def evaluate_ensemble(weights, cb_preds, lgb_preds, xgb_preds, y_true, split_label="Validation"):
    ensemble = weights[0]*cb_preds + weights[1]*lgb_preds + weights[2]*xgb_preds
    scores   = calculate_metrics(y_true, ensemble)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE  [{split_label}]")
    print("="*60)
    print(f"MAE  : ₹{scores['MAE']:,.0f}")
    print(f"RMSE : ₹{scores['RMSE']:,.0f}")
    print(f"MAPE : {scores['MAPE']:.2f}%")
    print(f"R²   : {scores['R2']:.4f}")

    return scores

# SAVE GLOBAL ARTIFACTS

def save_artifacts(cat_model, lgb_model, xgb_model, weights, category_levels, encoders, metadata):
    print(f"\n{DIV}")
    print("SAVING ARTIFACTS")
    print(DIV)

    cat_model.save_model(str(ARTIFACT_DIR / "vehicle_price_catboost.cbm"))
    lgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_lightgbm.txt"))
    xgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_xgboost.json"))
    print("  ✓ CatBoost   → vehicle_price_catboost.cbm")
    print("  ✓ LightGBM   → vehicle_price_lightgbm.txt")
    print("  ✓ XGBoost    → vehicle_price_xgboost.json")

    # Store encoder classes as plain lists — NOT LabelEncoder objects.
    # This is what was causing "cannot pickle 'module' object".
    encoders_serializable = {
        col: enc.classes_.tolist() for col, enc in encoders.items()
    }

    ensemble_bundle = {
        "weights": {
            "catboost":  float(weights[0]),
            "lightgbm":  float(weights[1]),
            "xgboost":   float(weights[2]),
        },
        "category_levels": category_levels,
        "encoders":         encoders_serializable,
        "features":         FEATURES,
        "cat_features":     CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "segments":         SEGMENTS,   # so dashboard.py knows the band edges
    }
    joblib.dump(ensemble_bundle, ARTIFACT_DIR / "ensemble_bundle.pkl")
    print("  ✓ Ensemble   → ensemble_bundle.pkl  (weights + category_levels)")

    with open(ARTIFACT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("  ✓ Report     → training_report.json")

# SEGMENT (PRICE-BAND) MODEL TRAINING

def train_segment_model(segment_name, seg_df, global_model, global_cat_levels):
    """
    Train a CatBoost model on one price band.
    Compare its test MAPE against the global model on the same rows.
    Return the segment model only if it beats global — else return None.
    """

    print(f"\n{'─'*60}")
    print(f"  SEGMENT: {segment_name.upper()}  ({len(seg_df):,} rows)")
    print(f"{'─'*60}")

    if len(seg_df) < MIN_SEGMENT_ROWS:
        print(f"  ⚠  Only {len(seg_df)} rows — below {MIN_SEGMENT_ROWS} minimum")
        print(f"  ✗  Skipping — global model will be used for {segment_name}")
        return None, None, None

    X = seg_df[FEATURES]
    y = np.log1p(seg_df[TARGET])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    seg_cat_levels = build_category_levels(X_train)

    cb_train, _, _, _ = prepare_frames(X_train, seg_cat_levels)
    cb_val,   _, _, _ = prepare_frames(X_val,   seg_cat_levels)
    cb_test,  _, _, _ = prepare_frames(X_test,  seg_cat_levels)

    seg_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5,
        min_data_in_leaf=10,
        early_stopping_rounds=100,
        verbose=200,
    )
    seg_model.fit(
        Pool(cb_train, y_train, cat_features=CAT_FEATURES),
        eval_set=Pool(cb_val, y_val, cat_features=CAT_FEATURES),
        use_best_model=True,
    )

    seg_preds  = seg_model.predict(cb_test)
    seg_scores = calculate_metrics(y_test, seg_preds)

    cb_test_global, _, _, _ = prepare_frames(X_test, global_cat_levels)
    global_preds  = global_model.predict(cb_test_global)
    global_scores = calculate_metrics(y_test, global_preds)

    print(f"\n  Segment model  MAPE: {seg_scores['MAPE']:.2f}%   R²: {seg_scores['R2']:.4f}")
    print(f"  Global  model  MAPE: {global_scores['MAPE']:.2f}%   R²: {global_scores['R2']:.4f}")

    if seg_scores["MAPE"] < global_scores["MAPE"]:
        improvement = global_scores["MAPE"] - seg_scores["MAPE"]
        print(f"  ✅ Segment model WINS by {improvement:.2f}% MAPE — activating")
        return seg_model, seg_cat_levels, seg_scores
    else:
        print(f"  ✗  Global model is better — {segment_name} will use global fallback")
        return None, None, global_scores


def train_segmented_models(df, global_model, global_cat_levels):

    print(f"\n{DIV}")
    print("SEGMENTED TRAINING  (price-band models)")
    print(DIV)
    print("Training one CatBoost model per price band: 0-6L / 6-12L / 12L+.")
    print("Segment model only activated if MAPE < global model on same rows.\n")

    segment_results = {}

    for seg_name, (price_min, price_max) in SEGMENTS.items():

        mask   = df[TARGET].between(price_min, price_max)
        seg_df = df[mask].copy()

        seg_model, seg_levels, seg_scores = train_segment_model(
            seg_name, seg_df, global_model, global_cat_levels
        )

        segment_results[seg_name] = {
            "model":       seg_model,
            "cat_levels":  seg_levels,
            "scores":      seg_scores,
            "active":      seg_model is not None,
            "price_range": (price_min, price_max),
            "row_count":   int(mask.sum()),
        }

    return segment_results


def save_segment_artifacts(segment_results):

    print(f"\n{DIV}")
    print("SAVING SEGMENT ARTIFACTS")
    print(DIV)

    routing_table = {}

    for seg_name, result in segment_results.items():

        if result["active"]:
            model_path = ARTIFACT_DIR / f"segment_{seg_name}.cbm"
            result["model"].save_model(str(model_path))

            levels_path = ARTIFACT_DIR / f"segment_{seg_name}_levels.pkl"
            joblib.dump(result["cat_levels"], levels_path)

            routing_table[seg_name] = {
                "active":      True,
                "model_file":  f"segment_{seg_name}.cbm",
                "levels_file": f"segment_{seg_name}_levels.pkl",
                "price_range": result["price_range"],
                "row_count":   result["row_count"],
                "mape":        result["scores"]["MAPE"] if result["scores"] else None,
                "r2":          result["scores"]["R2"]   if result["scores"] else None,
            }
            print(f"  ✓ {seg_name:<14} → segment_{seg_name}.cbm  (MAPE {result['scores']['MAPE']:.2f}%)")

        else:
            routing_table[seg_name] = {
                "active":      False,
                "model_file":  None,
                "levels_file": None,
                "price_range": result["price_range"],
                "row_count":   result["row_count"],
                "fallback":    "global",
                "mape":        result["scores"]["MAPE"] if result["scores"] else None,
            }
            print(f"  ✗ {seg_name:<14} → using global fallback  "
                  f"(global MAPE on segment: {result['scores']['MAPE'] if result['scores'] else 'N/A'}%)")

    with open(ARTIFACT_DIR / "routing_table.json", "w", encoding="utf-8") as f:
        json.dump(routing_table, f, indent=4)

    print(f"\n  ✓ Routing table → routing_table.json")
    return routing_table


def print_segment_summary(segment_results):

    print(f"\n{DIV}")
    print("SEGMENT SUMMARY")
    print(DIV)

    rows = []
    for seg_name, result in segment_results.items():
        rows.append({
            "Segment":    seg_name,
            "Rows":       result["row_count"],
            "Active":     "✅ YES" if result["active"] else "✗ global",
            "MAPE":       f"{result['scores']['MAPE']:.2f}%" if result["scores"] else "N/A",
            "R²":         f"{result['scores']['R2']:.4f}"   if result["scores"] else "N/A",
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

# FULL PIPELINE — global + price-band segments  (single entry point)

def train_all_models():

    # Global model
    df = load_dataset()
    df = validate_dataset(df)
    df = clean_training_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    frames     = prepare_training_frames(X_train, X_val, X_test)
    cat_levels = frames["category_levels"]

    cat_model = train_catboost(
        frames["catboost"]["train"], y_train,
        frames["catboost"]["val"],   y_val,
    )
    lgb_model = train_lightgbm(
        frames["lightgbm"]["train"], y_train,
        frames["lightgbm"]["val"],   y_val,
    )
    xgb_model = train_xgboost(
        frames["xgboost"]["train"], y_train,
        frames["xgboost"]["val"],   y_val,
    )

    cat_val_scores, cat_val_preds = evaluate_model(cat_model, "CatBoost", frames["catboost"]["val"], y_val)
    lgb_val_scores, lgb_val_preds = evaluate_model(lgb_model, "LightGBM", frames["lightgbm"]["val"], y_val)
    xgb_val_scores, xgb_val_preds = evaluate_model(xgb_model, "XGBoost",  frames["xgboost"]["val"],  y_val)

    weights    = optimise_ensemble_weights(cat_val_preds, lgb_val_preds, xgb_val_preds, y_val)
    val_scores = evaluate_ensemble(weights, cat_val_preds, lgb_val_preds, xgb_val_preds, y_val, "Validation")

    print(f"\n{DIV}")
    print("FINAL TEST SET METRICS  (unbiased)")
    print(DIV)

    cat_test_preds = predict(cat_model, "CatBoost", frames["catboost"]["test"])
    lgb_test_preds = predict(lgb_model, "LightGBM", frames["lightgbm"]["test"])
    xgb_test_preds = predict(xgb_model, "XGBoost",  frames["xgboost"]["test"])

    cat_test_scores, _ = evaluate_model(cat_model, "CatBoost", frames["catboost"]["test"], y_test, "Test")
    lgb_test_scores, _ = evaluate_model(lgb_model, "LightGBM", frames["lightgbm"]["test"], y_test, "Test")
    xgb_test_scores, _ = evaluate_model(xgb_model, "XGBoost",  frames["xgboost"]["test"],  y_test, "Test")

    test_scores = evaluate_ensemble(weights, cat_test_preds, lgb_test_preds, xgb_test_preds, y_test, "Test")

    comparison = pd.DataFrame([
        {"Model": "CatBoost", **cat_test_scores},
        {"Model": "LightGBM", **lgb_test_scores},
        {"Model": "XGBoost",  **xgb_test_scores},
        {"Model": "Ensemble", **test_scores},
    ]).sort_values("R2", ascending=False)

    print(f"\n{DIV}")
    print("MODEL COMPARISON  [Test Set]")
    print(DIV)
    print(comparison.to_string(index=False))

    metadata = {
        "training_time":    datetime.now().isoformat(),
        "dataset":          str(DATASET),
        "rows_used":        len(df),
        "features":         FEATURES,
        "cat_features":     CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "segment_definitions": SEGMENTS,
        "ensemble_weights": {
            "catboost": float(weights[0]),
            "lightgbm": float(weights[1]),
            "xgboost":  float(weights[2]),
        },
        "validation_metrics": {
            "CatBoost": cat_val_scores,
            "LightGBM": lgb_val_scores,
            "XGBoost":  xgb_val_scores,
            "Ensemble": val_scores,
        },
        "test_metrics": {
            "CatBoost": cat_test_scores,
            "LightGBM": lgb_test_scores,
            "XGBoost":  xgb_test_scores,
            "Ensemble": test_scores,
        },
    }

    save_artifacts(cat_model, lgb_model, xgb_model, weights, cat_levels, frames["encoders"], metadata)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    # Price-band segmented training
    segment_results = train_segmented_models(df, cat_model, cat_levels)
    routing_table    = save_segment_artifacts(segment_results)
    print_segment_summary(segment_results)

    metadata["segments"] = {
        name: {
            "active":      r["active"],
            "row_count":   r["row_count"],
            "price_range": r["price_range"],
            "mape":        r["scores"]["MAPE"] if r["scores"] else None,
        }
        for name, r in segment_results.items()
    }
    with open(ARTIFACT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n{DIV}")
    print("TRAINING COMPLETE")
    print(DIV)
    print(f"  Global MAPE  : {test_scores['MAPE']:.2f}%")
    print(f"  Global R²    : {test_scores['R2']:.4f}")
    active = [n for n, r in segment_results.items() if r["active"]]
    print(f"  Active segs  : {active if active else 'none — all using global'}")
    print(f"  Artifacts    : {ARTIFACT_DIR}")

    return {"comparison": comparison, "metadata": metadata, "segments": segment_results}


if __name__ == "__main__":
    train_all_models()

    # Generate dataset_catalog.json (brand → model → variants)
    print("\nGenerating dataset_catalog.json …")
    try:
        _df_cat = pd.read_csv(DATASET, usecols=["brand", "model", "variant"])
        _df_cat = _df_cat.dropna(subset=["brand", "model"])
        _df_cat["brand"]   = _df_cat["brand"].astype(str).str.strip().str.lower()
        _df_cat["model"]   = _df_cat["model"].astype(str).str.strip().str.lower()
        _df_cat["variant"] = _df_cat["variant"].astype(str).str.strip().str.lower()
        _catalog: dict = {}
        for _brand, _bdf in _df_cat.groupby("brand"):
            _catalog[_brand] = {}
            for _model, _mdf in _bdf.groupby("model"):
                _variants = sorted(_mdf["variant"].dropna().unique().tolist())
                _catalog[_brand][_model] = _variants
        _cat_path = ARTIFACT_DIR / "dataset_catalog.json"
        with open(_cat_path, "w", encoding="utf-8") as _f:
            json.dump(_catalog, _f, indent=2)
        _n_brands = len(_catalog)
        _n_models = sum(len(m) for m in _catalog.values())
        print(f"  Saved {_n_brands} brands / {_n_models} models → {_cat_path}")
    except Exception as _e:
        print(f"  WARNING: catalog generation failed: {_e}")