"""
PriceRef ML Training Pipeline
Version 6.0

Models:
    • CatBoost
    • LightGBM
    • XGBoost

Dataset:
    processed_cell7_dataset.csv

Author:
    PriceRef
"""

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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    Path(__file__).resolve().parent
    / "data"
    / "processed_cell7_dataset.csv"
)

ARTIFACT_DIR = ROOT / "model_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# =============================================================================
# FEATURES
# =============================================================================

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

]

FEATURES = CAT_FEATURES + NUMERIC_FEATURES

# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        np.log1p(y_true),
        np.log1p(y_pred),
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            /
            (y_true + 1e-8)
        )
    ) * 100

    return {

        "MAE": round(mae,2),

        "RMSE": round(rmse,2),

        "R2": round(r2,4),

        "MAPE": round(mape,2),

    }

# =============================================================================
# CATEGORY LEVELS
# =============================================================================

def build_category_levels(df):

    levels = {}

    for col in CAT_FEATURES:

        values = (
            df[col]
            .astype(str)
            .fillna("unknown")
            .unique()
            .tolist()
        )

        if "unknown" not in values:
            values.append("unknown")

        levels[col] = sorted(values)

    return levels

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_frames(df, category_levels):

    frame = df[FEATURES].copy()

    # Handle categorical columns

    for col in CAT_FEATURES:

        known = set(category_levels[col])

        frame[col] = (
            frame[col]
            .astype(str)
            .apply(
                lambda x:
                x if x in known
                else "unknown"
            )
        )

    # Handle numeric columns

    for col in NUMERIC_FEATURES:

        median = frame[col].median()

        if np.isnan(median):
            median = 0

        frame[col] = frame[col].fillna(median)

    # CatBoost Frame

    cb_frame = frame.copy()

    # LightGBM Frame

    lgb_frame = frame.copy()

    for col in CAT_FEATURES:

        encoder = LabelEncoder()

        lgb_frame[col] = encoder.fit_transform(
            lgb_frame[col].astype(str)
        )

    # XGBoost uses same frame

    xgb_frame = lgb_frame.copy()

    return cb_frame, lgb_frame, xgb_frame# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset():

    print("=" * 80)
    print("Loading Processed Dataset")
    print("=" * 80)

    df = pd.read_csv(DATASET)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns")

    for col in df.columns:
        print(f"  • {col}")

    return df


# =============================================================================
# VALIDATE DATASET
# =============================================================================

def validate_dataset(df):

    print("\n" + "=" * 80)
    print("VALIDATING DATASET")
    print("=" * 80)

    required_columns = FEATURES + [TARGET]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    print("✓ Required columns verified")

    print("\nMissing Values")

    for col in required_columns:

        print(
            f"{col:<25}"
            f"{df[col].isna().sum():>8}"
        )

    return df


# =============================================================================
# CLEAN TRAINING DATA
# =============================================================================

def clean_training_data(df):

    print("\n" + "=" * 80)
    print("CLEANING DATA")
    print("=" * 80)

    before = len(df)

    # ---------------------------------------------------
    # Target
    # ---------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df.dropna(subset=[TARGET])

    df = df[
        df[TARGET].between(
            50_000,
            20_000_000,
        )
    ]

    # ---------------------------------------------------
    # Numeric Features
    # ---------------------------------------------------

    for col in NUMERIC_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "vehicle_age",
            "odometer_reading",
            "km_per_year",
            "owner_count",
        ]
    )

    df["inspected"] = (
        df["inspected"]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------
    # Categorical Features
    # ---------------------------------------------------

    for col in CAT_FEATURES:

        df[col] = (

            df[col]

            .fillna("unknown")

            .astype(str)

            .str.strip()

            .str.lower()

        )

    print(f"Removed {before-len(df):,} invalid rows")

    print(f"Remaining Rows : {len(df):,}")

    return df


# =============================================================================
# TRAIN / VALIDATION SPLIT
# =============================================================================

def split_dataset(df):

    print("\n" + "=" * 80)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 80)

    X = df[FEATURES]

    y = np.log1p(df[TARGET])

    X_train, X_valid, y_train, y_valid = train_test_split(

        X,

        y,

        test_size=0.30,

        random_state=RANDOM_STATE,

        shuffle=True,

    )

    print(f"Training Samples   : {len(X_train):,}")

    print(f"Validation Samples : {len(X_valid):,}")

    return (

        X_train,

        X_valid,

        y_train,

        y_valid,

    )


# =============================================================================
# PREPARE MODEL INPUTS
# =============================================================================

def prepare_training_frames(

    X_train,

    X_valid,

):

    print("\nPreparing Model Inputs...")

    category_levels = build_category_levels(
        X_train
    )

    cb_train, lgb_train, xgb_train = prepare_frames(

        X_train,

        category_levels,

    )

    cb_valid, lgb_valid, xgb_valid = prepare_frames(

        X_valid,

        category_levels,

    )

    return {

        "category_levels": category_levels,

        "catboost": {

            "train": cb_train,

            "valid": cb_valid,

        },

        "lightgbm": {

            "train": lgb_train,

            "valid": lgb_valid,

        },

        "xgboost": {

            "train": xgb_train,

            "valid": xgb_valid,

        },

    }# =============================================================================
# CATBOOST
# =============================================================================

def train_catboost(

    X_train,

    y_train,

    X_valid,

    y_valid,

):

    print("\nTraining CatBoost...")

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

    train_pool = Pool(

        X_train,

        y_train,

        cat_features=CAT_FEATURES,

    )

    valid_pool = Pool(

        X_valid,

        y_valid,

        cat_features=CAT_FEATURES,

    )

    model.fit(

        train_pool,

        eval_set=valid_pool,

        use_best_model=True,

    )

    return model


# =============================================================================
# LIGHTGBM
# =============================================================================

def train_lightgbm(

    X_train,

    y_train,

    X_valid,

    y_valid,

):

    print("\nTraining LightGBM...")

    train_set = lgb.Dataset(

        X_train,

        label=y_train,

    )

    valid_set = lgb.Dataset(

        X_valid,

        label=y_valid,

    )

    params = {

        "objective": "regression",

        "metric": "rmse",

        "learning_rate": 0.03,

        "num_leaves": 64,

        "feature_fraction": 0.8,

        "bagging_fraction": 0.8,

        "bagging_freq": 5,

        "min_child_samples": 20,

        "verbosity": -1,

        "seed": RANDOM_STATE,

    }

    model = lgb.train(

        params,

        train_set,

        valid_sets=[valid_set],

        num_boost_round=3000,

        callbacks=[

            lgb.early_stopping(100),

            lgb.log_evaluation(100),

        ],

    )

    return model


# =============================================================================
# XGBOOST
# =============================================================================

def train_xgboost(

    X_train,

    y_train,

    X_valid,

    y_valid,

):

    print("\nTraining XGBoost...")

    train_matrix = xgb.DMatrix(

        X_train,

        label=y_train,

    )

    valid_matrix = xgb.DMatrix(

        X_valid,

        label=y_valid,

    )

    params = {

        "objective": "reg:squarederror",

        "eval_metric": "rmse",

        "learning_rate": 0.03,

        "max_depth": 8,

        "subsample": 0.8,

        "colsample_bytree": 0.8,

        "seed": RANDOM_STATE,

    }

    model = xgb.train(

        params,

        train_matrix,

        num_boost_round=3000,

        evals=[

            (valid_matrix, "Validation")

        ],

        early_stopping_rounds=100,

        verbose_eval=100,

    )

    return model


# =============================================================================
# PREDICTION
# =============================================================================

def predict(

    model,

    model_name,

    X,

):

    if model_name == "CatBoost":

        return model.predict(X)

    if model_name == "LightGBM":

        return model.predict(X)

    if model_name == "XGBoost":

        return model.predict(

            xgb.DMatrix(X)

        )

    raise ValueError("Unknown model")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(

    model,

    model_name,

    X_valid,

    y_valid,

):

    predictions = predict(

        model,

        model_name,

        X_valid,

    )

    scores = calculate_metrics(

        y_valid,

        predictions,

    )

    print("\n" + "="*60)

    print(model_name)

    print("="*60)

    print(f"MAE  : {scores['MAE']:,.2f}")

    print(f"RMSE : {scores['RMSE']:,.2f}")

    print(f"MAPE : {scores['MAPE']:.2f}%")

    print(f"R²   : {scores['R2']:.4f}")

    return scores, predictions# =============================================================================
# TRAIN ALL MODELS
# =============================================================================

def train_all_models():

    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------

    df = load_dataset()

    df = validate_dataset(df)

    df = clean_training_data(df)

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = split_dataset(df)

    # ---------------------------------------------------------
    # Prepare Frames
    # ---------------------------------------------------------

    frames = prepare_training_frames(

        X_train,

        X_valid,

    )

    # =========================================================
    # CATBOOST
    # =========================================================

    cat_model = train_catboost(

        frames["catboost"]["train"],

        y_train,

        frames["catboost"]["valid"],

        y_valid,

    )

    cat_metrics, cat_predictions = evaluate_model(

        cat_model,

        "CatBoost",

        frames["catboost"]["valid"],

        y_valid,

    )

    # =========================================================
    # LIGHTGBM
    # =========================================================

    lgb_model = train_lightgbm(

        frames["lightgbm"]["train"],

        y_train,

        frames["lightgbm"]["valid"],

        y_valid,

    )

    lgb_metrics, lgb_predictions = evaluate_model(

        lgb_model,

        "LightGBM",

        frames["lightgbm"]["valid"],

        y_valid,

    )

    # =========================================================
    # XGBOOST
    # =========================================================

    xgb_model = train_xgboost(

        frames["xgboost"]["train"],

        y_train,

        frames["xgboost"]["valid"],

        y_valid,

    )

    xgb_metrics, xgb_predictions = evaluate_model(

        xgb_model,

        "XGBoost",

        frames["xgboost"]["valid"],

        y_valid,

    )

    # =========================================================
    # MODEL COMPARISON
    # =========================================================

    comparison = pd.DataFrame({

        "Model": [

            "CatBoost",

            "LightGBM",

            "XGBoost",

        ],

        "MAE": [

            cat_metrics["MAE"],

            lgb_metrics["MAE"],

            xgb_metrics["MAE"],

        ],

        "RMSE": [

            cat_metrics["RMSE"],

            lgb_metrics["RMSE"],

            xgb_metrics["RMSE"],

        ],

        "MAPE": [

            cat_metrics["MAPE"],

            lgb_metrics["MAPE"],

            xgb_metrics["MAPE"],

        ],

        "R2": [

            cat_metrics["R2"],

            lgb_metrics["R2"],

            xgb_metrics["R2"],

        ]

    })

    comparison = comparison.sort_values(

        by="R2",

        ascending=False,

    )

    print("\n")

    print("="*80)

    print("MODEL COMPARISON")

    print("="*80)

    print(comparison)

    # =========================================================
    # BEST MODEL
    # =========================================================

    best_model_name = comparison.iloc[0]["Model"]

    print("\n")

    print("="*80)

    print(f"BEST MODEL : {best_model_name}")

    print("="*80)

    if best_model_name == "CatBoost":

        best_model = cat_model

        extension = "cbm"

    elif best_model_name == "LightGBM":

        best_model = lgb_model

        extension = "txt"

    else:

        best_model = xgb_model

        extension = "json"

    # =========================================================
    # SAVE MODEL
    # =========================================================

    print("\nSaving Best Model...")

    if best_model_name == "CatBoost":

        best_model.save_model(

            ARTIFACT_DIR /

            "best_model.cbm"

        )

    elif best_model_name == "LightGBM":

        best_model.save_model(

            str(

                ARTIFACT_DIR /

                "best_model.txt"

            )

        )

    else:

        best_model.save_model(

            ARTIFACT_DIR /

            "best_model.json"

        )

    # Save metadata

    metadata = {

        "training_time":

            datetime.now().isoformat(),

        "best_model":

            best_model_name,

        "features":

            FEATURES,

        "categorical_features":

            CAT_FEATURES,

        "numeric_features":

            NUMERIC_FEATURES,

        "metrics": {

            "CatBoost": cat_metrics,

            "LightGBM": lgb_metrics,

            "XGBoost": xgb_metrics,

        }

    }

    with open(

        ARTIFACT_DIR /

        "training_report.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4,

        )

    comparison.to_csv(

        ARTIFACT_DIR /

        "model_comparison.csv",

        index=False,

    )

    print("\n")

    print("="*80)

    print("TRAINING COMPLETED")

    print("="*80)

    print(f"Best Model : {best_model_name}")

    print(f"Artifacts  : {ARTIFACT_DIR}")

    return {

        "comparison": comparison,

        "best_model": best_model_name,

        "metadata": metadata,

    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    train_all_models()