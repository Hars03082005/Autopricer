"""
ml_training/train_unified.py

Unified CatBoost model trained ONLY on the 19 features available at inference.
Fixes:
  - Feature mismatch: model no longer expects locality/rto/seller_type etc.
  - Luxury underprediction: 4x sample weight for cars >20L, 2x for >10L
  - Brand age/km penalty features for correct luxury-old-car pricing
"""
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ml_training import registry_helper

ROOT         = Path(__file__).resolve().parents[1]
DATASET      = Path(__file__).resolve().parent / "data" / "processed_widown-1.csv"
VARIANT_ID   = registry_helper.next_variant_id()
ARTIFACT_DIR = registry_helper.get_variant_dir(VARIANT_ID)
print(f"Unified train -> Variant: {VARIANT_ID}  dir: {ARTIFACT_DIR}")

RANDOM_STATE = 42
CURRENT_YEAR = 2026
TARGET       = "selling_price"

BRAND_TIER_MAP = {
    "datsun": 0,
    "maruti suzuki": 1, "renault": 1, "tata": 1, "chevrolet": 1,
    "fiat": 1, "bajaj": 1,
    "hyundai": 2, "honda": 2, "kia": 2, "ford": 2,
    "volkswagen": 2, "skoda": 2, "nissan": 2, "mitsubishi": 2,
    "mahindra": 2, "citroen": 2,
    "toyota": 3, "mg": 3, "jeep": 3,
    "bmw": 4, "mercedes-benz": 4, "audi": 4, "volvo": 4,
    "mini": 4, "lexus": 4, "jaguar": 4, "land rover": 4,
}
LUXURY_BRANDS = {"bmw", "mercedes-benz", "audi", "volvo", "mini", "lexus",
                 "jaguar", "land rover", "porsche", "maserati", "ferrari",
                 "rolls-royce", "bentley", "lamborghini"}

# Only features available at inference time
CAT_FEATURES = [
    "brand", "model", "variant", "city",
    "segment_class", "fuel_type", "transmission", "color",
]
NUMERIC_FEATURES = [
    "vehicle_age", "odometer_reading", "km_per_year", "owner_count",
    "brand_tier", "age_km_interaction", "is_high_mileage",
    "ownership_trust_score", "vehicle_health_score",
    "brand_age_penalty", "brand_km_penalty",
    "inspected", "high_mileage", "luxury_brand",
]
FEATURES = CAT_FEATURES + NUMERIC_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    brand = df["brand"].str.strip().str.lower()

    df["vehicle_age"]  = (CURRENT_YEAR - df["year"]).clip(lower=0)
    df["km_per_year"]  = (df["odometer_reading"] / df["vehicle_age"].clip(lower=1)).clip(upper=100_000)
    df["brand_tier"]   = brand.map(BRAND_TIER_MAP).fillna(1).astype(float)

    df["ownership_trust_score"] = (
        (1 / df["owner_count"].clip(lower=1)) * 0.5
        + (1 - (df["vehicle_age"] / 35).clip(upper=1)) * 0.3
        + (1 - (df["odometer_reading"] / 600_000).clip(upper=1)) * 0.2
    )
    df["vehicle_health_score"] = (
        (1 - (df["odometer_reading"] / 600_000).clip(upper=1)) * 0.5
        + (1 - (df["vehicle_age"] / 35).clip(upper=1)) * 0.3
        + (1 / df["owner_count"].clip(lower=1)) * 0.2
    )

    df["age_km_interaction"] = df["vehicle_age"] * df["odometer_reading"]
    df["is_high_mileage"]    = (df["km_per_year"] > 15_000).astype(float)
    df["high_mileage"]       = (df["odometer_reading"] > 93_143).astype(float)
    df["luxury_brand"]       = brand.isin(LUXURY_BRANDS).astype(float)

    # New penalty features for luxury car age/km pricing
    df["brand_age_penalty"]  = df["brand_tier"] * df["vehicle_age"]
    df["brand_km_penalty"]   = df["brand_tier"] * (df["odometer_reading"] / 10_000)

    df["inspected"] = df.get("inspected", pd.Series(0, index=df.index)).fillna(0).astype(float)

    # Derive segment_class from brand_tier
    tier = df["brand_tier"]
    df["segment_class"] = np.where(tier >= 4, "luxury",
                          np.where(tier >= 3, "premium", "economy"))

    # Color
    if "color" not in df.columns:
        df["color"] = "unknown"
    df["color"] = df["color"].fillna("unknown").str.strip().str.lower()

    # Clean categoricals
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    return df


def compute_sample_weights(y_log: np.ndarray, brands: pd.Series) -> np.ndarray:
    prices = np.expm1(y_log)
    weights = np.ones(len(prices))
    weights = np.where(prices > 20_00_000, 4.0, weights)   # >₹20L: 4×
    weights = np.where((prices > 10_00_000) & (prices <= 20_00_000), 2.0, weights)  # ₹10-20L: 2×
    weights = np.where(brands.str.lower().isin(LUXURY_BRANDS) & (weights < 2.0), 2.0, weights)
    return weights


def evaluate(model, X, y_log, label):
    pred_log = model.predict(X)
    pred     = np.expm1(pred_log)
    actual   = np.expm1(y_log)
    mae      = mean_absolute_error(actual, pred)
    rmse     = np.sqrt(mean_squared_error(actual, pred))
    r2       = r2_score(actual, pred)
    mape     = float(np.mean(np.abs((actual - pred) / actual.clip(lower=1))) * 100)
    print(f"  [{label}] MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


def train():
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    df = pd.read_csv(DATASET)
    print(f"  Rows: {len(df):,}  Cols: {len(df.columns)}")

    # Drop rows missing target or key features
    df = df.dropna(subset=["selling_price", "brand", "model", "year", "odometer_reading"])
    df = df[df["selling_price"] > 0]
    df["owner_count"] = pd.to_numeric(df.get("owner_count", 1), errors="coerce").fillna(1).clip(lower=1)

    df = engineer_features(df)

    # Log-transform target
    df["log_price"] = np.log1p(df["selling_price"])

    # Drop extreme outliers (outside 1.5×IQR on log scale)
    q1, q3 = df["log_price"].quantile([0.01, 0.99])
    df = df[(df["log_price"] >= q1) & (df["log_price"] <= q3)]
    print(f"  After outlier removal: {len(df):,} rows")

    # Keep only inference-available features
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING — missing features (will skip): {missing}")

    X = df[available]
    y = df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)

    cat_available = [c for c in CAT_FEATURES if c in available]
    weights_train = compute_sample_weights(y_train.values, X_train["brand"])

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Features: {len(available)}  Cat: {len(cat_available)}")

    print("\n{'='*70}")
    print("TRAINING CatBoost (unified)")
    train_pool = Pool(X_train, y_train, cat_features=cat_available, weight=weights_train)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_available)

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=150,
        verbose=200,
        task_type="CPU",
    )
    model.fit(train_pool, eval_set=val_pool)

    print("\n{'='*70}")
    print("METRICS")
    train_sc = evaluate(model, X_train, y_train.values, "train")
    val_sc   = evaluate(model, X_val,   y_val.values,   "val  ")
    test_sc  = evaluate(model, X_test,  y_test.values,  "test ")

    # Save model
    model.save_model(str(ARTIFACT_DIR / "vehicle_price_catboost.cbm"))
    print(f"\n  Saved CatBoost -> {ARTIFACT_DIR / 'vehicle_price_catboost.cbm'}")

    # Build category levels for inference normalisation
    cat_levels: dict = {}
    for col in cat_available:
        levels = X_train[col].astype(str).unique().tolist()
        if "unknown" not in levels:
            levels.append("unknown")
        cat_levels[col] = sorted(levels)

    # model_metadata.json in the format EnsemblePredictor expects
    model_meta = {
        "model_name":              "CatBoostRegressor (Unified-19F)",
        "trained_at":              datetime.now().isoformat(),
        "current_year_used_for_age": CURRENT_YEAR,
        "features":                available,
        "categorical_features":    cat_available,
        "numeric_features":        NUMERIC_FEATURES,
        "ensemble": {
            "enabled": False,
            "weights": {"catboost": 1.0, "lightgbm": 0.0, "xgboost": 0.0},
            "category_levels": cat_levels,
        },
        "global_metrics": {
            "train":      train_sc,
            "validation": val_sc,
            "test":       test_sc,
        },
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    # Dataset catalog for frontend brand/model dropdowns
    try:
        _df = pd.read_csv(DATASET, usecols=["brand", "model", "variant"])
        _df = _df.dropna(subset=["brand", "model"])
        for c in ["brand", "model", "variant"]:
            _df[c] = _df[c].astype(str).str.strip().str.lower()
        catalog: dict = {}
        for brand, bdf in _df.groupby("brand"):
            catalog[brand] = {}
            for mdl, mdf in bdf.groupby("model"):
                catalog[brand][mdl] = sorted(mdf["variant"].dropna().unique().tolist())
        with open(ARTIFACT_DIR / "dataset_catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        print(f"  Saved dataset_catalog.json ({len(catalog)} brands)")
    except Exception as e:
        print(f"  WARNING: catalog generation failed: {e}")

    # Register in model_registry
    registry_helper.register_variant(
        variant_id   = VARIANT_ID,
        artifact_dir = ARTIFACT_DIR,
        dataset_name = DATASET.name,
        metrics      = test_sc,
    )
    registry_helper.copy_to_model_artifacts(ARTIFACT_DIR)

    print(f"\n{'='*70}")
    print(f"DONE — Variant: {VARIANT_ID}")
    print(f"  Test MAPE : {test_sc['mape']:.2f}%")
    print(f"  Test R²   : {test_sc['r2']:.4f}")
    print(f"  Artifacts : {ARTIFACT_DIR}")
    return model_meta


if __name__ == "__main__":
    train()
