"""Train the PricerPoint ML valuation model (v5.0).

Dataset : ml_training/data/cleaned_used_car_dataset.csv  (213,820 rows, 26 cols)
Pipeline :
  - Column rename + feature engineering from the 2026 cleaned dataset
  - Segment mapping  →  economy / premium / luxury
  - 70 / 15 / 15 train / validation / test split
  - CatBoost + LightGBM + XGBoost base learners
  - SLSQP-optimised ensemble weights (maximise validation R²)
  - Global model  +  3 segment models (economy / premium / luxury)
  - Saves all artifacts to model_artifacts/
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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

ROOT         = Path(__file__).resolve().parents[1]
DATA_CSV     = Path(__file__).resolve().parent / "data" / "cleaned_used_car_dataset.csv"
ARTIFACT_DIR = ROOT / "model_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

CURRENT_YEAR = 2026
RANDOM_STATE = 42

# ── Column rename map (dataset → canonical) ────────────────────────────────────
RENAME_MAP = {
    "MAKE":           "brand",
    "MODEL":          "model",
    "TRIM":           "variant",
    "CITY":           "city",
    "RTO":            "rto_state",
    "COLOR":          "color",
    "SEGMENT":        "segment",
    "FUEL":           "fuel_type",
    "TRANS":          "transmission",
    "PRICE":          "selling_price",
    "LIST_PRICE":     "list_price",
    "CERTIFIED":      "inspected_raw",
    "YEAR":           "year",
    "ODOMETER":       "odometer_reading",
    "OWNER":          "owner_raw",
    "Vehicle_Age":    "vehicle_age",
    "Annual_Mileage": "km_per_year",
    "High_Mileage":   "high_mileage",
    "Luxury_Brand":   "luxury_brand",
    "HAS_LIST_PRICE": "has_list_price",
}

# ── Segment mapping: dataset SEGMENT → economy / premium / luxury ──────────────
SEGMENT_MAP: Dict[str, str] = {
    "mass market": "economy",
    "budget":      "economy",
    "unknown":     "economy",
    "assured":     "economy",
    "standard":    "premium",
    "luxury":      "luxury",
    "luxe":        "luxury",
}
SEGMENT_CLASSES = ["economy", "premium", "luxury"]

# ── Feature sets ───────────────────────────────────────────────────────────────
CAT_FEATURES = [
    "brand", "model", "variant", "city", "rto_state",
    "color", "segment_class", "fuel_type", "transmission",
]
NUMERIC_FEATURES = [
    "vehicle_age", "odometer_reading", "km_per_year",
    "owner_count", "ownership_trust_score", "vehicle_health_score",
    "inspected", "high_mileage", "luxury_brand", "has_list_price",
]
FEATURES = CAT_FEATURES + NUMERIC_FEATURES

# ── Condition multipliers (post-prediction calibration) ───────────────────────
CONDITION_MULTIPLIERS = {
    "excellent": 1.035,
    "good":      1.000,
    "average":   0.940,
    "poor":      0.860,
}

# ── Brand → segment routing for inference (when SEGMENT is unknown) ────────────
BRAND_SEGMENT_MAP: Dict[str, str] = {
    # Economy
    "maruti": "economy", "maruti suzuki": "economy", "datsun": "economy",
    "bajaj": "economy", "chevrolet": "economy", "fiat": "economy",
    "opel": "economy", "premier": "economy", "force": "economy",
    "ashok leyland": "economy", "ambassador": "economy",
    "hindustan motors": "economy",
    # Economy-Mid
    "hyundai": "economy", "honda": "economy", "tata": "economy",
    "renault": "economy", "nissan": "economy", "ford": "economy",
    "mahindra": "economy", "mitsubishi": "economy", "isuzu": "economy",
    "citroen": "economy", "dc": "economy",
    # Premium
    "volkswagen": "premium", "skoda": "premium", "toyota": "premium",
    "mg": "premium", "jeep": "premium", "kia": "premium",
    "mini": "premium", "volvo": "premium", "lexus": "premium",
    # Luxury
    "bmw": "luxury", "mercedes-benz": "luxury", "audi": "luxury",
    "jaguar": "luxury", "land rover": "luxury", "porsche": "luxury",
    "maserati": "luxury", "aston martin": "luxury", "bentley": "luxury",
    "rolls-royce": "luxury", "ferrari": "luxury", "lamborghini": "luxury",
    "hummer": "luxury",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_e = np.expm1(y_true)
    y_pred_e = np.expm1(y_pred)
    mae  = mean_absolute_error(y_true_e, y_pred_e)
    rmse = math.sqrt(mean_squared_error(y_true_e, y_pred_e))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true_e - y_pred_e) / (y_true_e + 1e-8))) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "r2": round(r2, 4), "mape": round(mape, 2)}


def blend_predictions(weights: dict, preds: dict) -> np.ndarray:
    out = np.zeros(len(next(iter(preds.values()))))
    for name, w in weights.items():
        out += w * np.asarray(preds[name])
    return out


def optimize_ensemble_weights(y_val: np.ndarray, val_preds: dict) -> dict:
    names = list(val_preds.keys())
    n = len(names)

    def neg_r2(w):
        blended = sum(w[i] * val_preds[names[i]] for i in range(n))
        return -r2_score(y_val, blended)

    result = minimize(
        neg_r2,
        x0=[1.0 / n] * n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
    )
    w = result.x
    return {names[i]: round(float(w[i]), 4) for i in range(n)}


def build_category_levels(df: pd.DataFrame) -> dict:
    levels = {}
    for col in CAT_FEATURES:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique().tolist()
            if "unknown" not in vals:
                vals.append("unknown")
            levels[col] = sorted(vals)
    return levels


def prepare_frames(
    df: pd.DataFrame, cat_levels: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (cb_frame, lgb_frame, xgb_frame) from a dataframe slice."""
    frame = df[FEATURES].copy()

    for col in CAT_FEATURES:
        if col in frame.columns:
            known = set(cat_levels.get(col, []))
            frame[col] = frame[col].astype(str).apply(
                lambda v: v if v in known else "unknown"
            )

    for col in NUMERIC_FEATURES:
        if col in frame.columns:
            med = frame[col].median()
            if np.isnan(med):
                med = 0.0
            frame[col] = frame[col].fillna(med)

    cb = frame.copy()

    lgb_frame = frame.copy()
    for col in CAT_FEATURES:
        if col in lgb_frame.columns:
            le = LabelEncoder()
            lgb_frame[col] = le.fit_transform(lgb_frame[col].astype(str))

    xgb_frame = lgb_frame.copy()
    return cb, lgb_frame, xgb_frame


# ── Model trainers ─────────────────────────────────────────────────────────────

def train_catboost(X_tr, y_tr, X_vl, y_vl, cat_cols: List[str]) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=3,
        min_data_in_leaf=15,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100,
        verbose=False,
        cat_features=[c for c in cat_cols if c in X_tr.columns],
    )
    pool_tr = Pool(X_tr, y_tr, cat_features=[c for c in cat_cols if c in X_tr.columns])
    pool_vl = Pool(X_vl, y_vl, cat_features=[c for c in cat_cols if c in X_vl.columns])
    model.fit(pool_tr, eval_set=pool_vl, use_best_model=True)
    return model


def train_lightgbm(X_tr, y_tr, X_vl, y_vl) -> lgb.Booster:
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval   = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "learning_rate":     0.04,
        "num_leaves":        127,
        "max_depth":         -1,
        "min_child_samples": 15,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         0.1,
        "verbose":           -1,
        "random_state":      RANDOM_STATE,
    }
    cb = lgb.early_stopping(100, verbose=False)
    lg  = lgb.log_evaluation(-1)
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[cb, lg])
    return model


def train_xgboost(X_tr, y_tr, X_vl, y_vl) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_vl, label=y_vl)
    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "learning_rate":    0.04,
        "max_depth":        8,
        "min_child_weight": 15,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "lambda":           1.0,
        "alpha":            0.1,
        "seed":             RANDOM_STATE,
        "verbosity":        0,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return model


def predict(name: str, model, frame: pd.DataFrame) -> np.ndarray:
    if name == "catboost":
        return np.asarray(model.predict(frame))
    if name == "lightgbm":
        return np.asarray(model.predict(frame))
    if name == "xgboost":
        return np.asarray(model.predict(xgb.DMatrix(frame)))
    raise ValueError(name)


# ── Load & prepare ─────────────────────────────────────────────────────────────

def load_and_prepare() -> pd.DataFrame:
    print(f"Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Rename columns to canonical names
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # ── Target ──────────────────────────────────────────────────────────────────
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df.dropna(subset=["selling_price"], inplace=True)
    df = df[df["selling_price"].between(50_000, 20_000_000)]

    # ── Owner count ─────────────────────────────────────────────────────────────
    df["owner_count"] = pd.to_numeric(df["owner_raw"], errors="coerce").fillna(1).clip(1, 6).astype(int)

    # ── Inspected (CERTIFIED → 0/1) ─────────────────────────────────────────────
    def parse_inspected(v):
        if pd.isna(v):
            return 0
        s = str(v).strip().lower()
        return 1 if s in {"true", "1", "yes", "certified", "inspected"} else 0

    df["inspected"] = df["inspected_raw"].apply(parse_inspected)

    # ── RTO state: extract prefix (KA-19 → ka) ──────────────────────────────────
    df["rto_state"] = df["rto_state"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).str.lower().fillna("unknown")

    # ── Normalise categorical columns to lowercase ───────────────────────────────
    for col in ["brand", "model", "variant", "city", "color", "fuel_type", "transmission", "segment"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().fillna("unknown")

    # ── Segment class (economy / premium / luxury) ───────────────────────────────
    df["segment_class"] = df["segment"].map(SEGMENT_MAP).fillna("economy")

    # ── Engineered scores (mirror backend formulas) ──────────────────────────────
    age  = df["vehicle_age"].clip(0, 35)
    km   = df["odometer_reading"].clip(0, 600_000)
    own  = df["owner_count"]

    df["ownership_trust_score"] = (
        (1 / own) * 0.5
        + (1 - (age / 35).clip(0, 1)) * 0.3
        + (1 - (km / 600_000).clip(0, 1)) * 0.2
    ).round(4)

    df["vehicle_health_score"] = (
        (1 - (km / 600_000).clip(0, 1)) * 0.5
        + (1 - (age / 35).clip(0, 1)) * 0.3
        + (1 / own) * 0.2
    ).round(4)

    # ── Clip / guard numeric features ────────────────────────────────────────────
    df["vehicle_age"]      = df["vehicle_age"].clip(0, 35)
    df["odometer_reading"] = df["odometer_reading"].clip(0, 600_000)
    df["km_per_year"]      = df["km_per_year"].clip(0, 100_000)

    # ── Drop unusable rows ────────────────────────────────────────────────────────
    df.dropna(subset=NUMERIC_FEATURES[:4], inplace=True)   # core numerics must exist

    print(f"  Clean rows for training: {len(df):,}")
    return df


# ── Segment model trainer ──────────────────────────────────────────────────────

def train_segment(seg_df: pd.DataFrame, seg_name: str) -> dict | None:
    MIN_ROWS = 500
    if len(seg_df) < MIN_ROWS:
        print(f"  Skipping {seg_name} — only {len(seg_df)} rows (< {MIN_ROWS})")
        return None

    y = np.log1p(seg_df["selling_price"].values)
    X = seg_df[FEATURES]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_vl, X_te, y_vl, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    cat_levels = build_category_levels(seg_df.loc[X_tr.index])
    cb_tr, lgb_tr, xgb_tr = prepare_frames(seg_df.loc[X_tr.index], cat_levels)
    cb_vl, lgb_vl, xgb_vl = prepare_frames(seg_df.loc[X_vl.index], cat_levels)
    cb_te, lgb_te, xgb_te = prepare_frames(seg_df.loc[X_te.index], cat_levels)

    print(f"  CatBoost …")
    s_cb  = train_catboost(cb_tr, y_tr, cb_vl, y_vl, CAT_FEATURES)
    print(f"  LightGBM …")
    s_lgb = train_lightgbm(lgb_tr, y_tr, lgb_vl, y_vl)
    print(f"  XGBoost …")
    s_xgb = train_xgboost(xgb_tr, y_tr, xgb_vl, y_vl)

    s_models = {"catboost": s_cb, "lightgbm": s_lgb, "xgboost": s_xgb}
    s_frames  = {
        "catboost": {"train": cb_tr, "val": cb_vl, "test": cb_te},
        "lightgbm": {"train": lgb_tr, "val": lgb_vl, "test": lgb_te},
        "xgboost":  {"train": xgb_tr, "val": xgb_vl, "test": xgb_te},
    }

    val_preds = {n: predict(n, m, s_frames[n]["val"]) for n, m in s_models.items()}
    s_weights = optimize_ensemble_weights(y_vl, val_preds)

    test_blend = blend_predictions(s_weights, {n: predict(n, m, s_frames[n]["test"]) for n, m in s_models.items()})
    seg_m = metrics(y_te, test_blend)
    print(f"  {seg_name.upper()} → R²: {seg_m['r2']:.4f}  MAE: ₹{seg_m['mae']:,.0f}  MAPE: {seg_m['mape']:.2f}%")

    return {
        "catboost":        s_cb,
        "lightgbm":        s_lgb,
        "xgboost":         s_xgb,
        "weights":         s_weights,
        "features":        FEATURES,
        "cat_features":    CAT_FEATURES,
        "segment_class":   seg_name,
        "category_levels": cat_levels,
        "rows":            len(seg_df),
        "test_metrics":    seg_m,
    }


# ── Main training pipeline ─────────────────────────────────────────────────────

def train() -> None:
    df = load_and_prepare()

    y = np.log1p(df["selling_price"].values)
    X = df[FEATURES]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_vl, X_te, y_vl, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    print(f"\n{'='*60}")
    print("GLOBAL MODEL")
    print(f"{'='*60}")

    cat_levels = build_category_levels(df.loc[X_tr.index])

    cb_tr, lgb_tr, xgb_tr = prepare_frames(df.loc[X_tr.index], cat_levels)
    cb_vl, lgb_vl, xgb_vl = prepare_frames(df.loc[X_vl.index], cat_levels)
    cb_te, lgb_te, xgb_te = prepare_frames(df.loc[X_te.index], cat_levels)

    print("CatBoost …")
    cat_model = train_catboost(cb_tr, y_tr, cb_vl, y_vl, CAT_FEATURES)
    print("LightGBM …")
    lgb_model = train_lightgbm(lgb_tr, y_tr, lgb_vl, y_vl)
    print("XGBoost …")
    xgb_model = train_xgboost(xgb_tr, y_tr, xgb_vl, y_vl)

    g_models = {"catboost": cat_model, "lightgbm": lgb_model, "xgboost": xgb_model}
    g_frames  = {
        "catboost": {"train": cb_tr, "val": cb_vl, "test": cb_te},
        "lightgbm": {"train": lgb_tr, "val": lgb_vl, "test": lgb_te},
        "xgboost":  {"train": xgb_tr, "val": xgb_vl, "test": xgb_te},
    }

    individual_results = {}
    val_preds = {}
    for name, model in g_models.items():
        tr_m = metrics(y_tr, predict(name, model, g_frames[name]["train"]))
        vl_m = metrics(y_vl, predict(name, model, g_frames[name]["val"]))
        te_m = metrics(y_te, predict(name, model, g_frames[name]["test"]))
        individual_results[name] = {
            "train_metrics": tr_m, "validation_metrics": vl_m, "test_metrics": te_m
        }
        val_preds[name] = predict(name, model, g_frames[name]["val"])
        print(f"  {name:12s} → Val R²: {vl_m['r2']:.4f}  Test MAPE: {te_m['mape']:.2f}%")

    weights = optimize_ensemble_weights(y_vl, val_preds)
    print(f"\n  Ensemble weights: {weights}")

    test_blend = blend_predictions(weights, {n: predict(n, m, g_frames[n]["test"]) for n, m in g_models.items()})
    train_blend = blend_predictions(weights, {n: predict(n, m, g_frames[n]["train"]) for n, m in g_models.items()})
    val_blend   = blend_predictions(weights, {n: predict(n, m, g_frames[n]["val"])   for n, m in g_models.items()})

    g_train_m = metrics(y_tr, train_blend)
    g_val_m   = metrics(y_vl, val_blend)
    g_test_m  = metrics(y_te, test_blend)

    overfit_gap    = round(g_train_m["r2"] - g_test_m["r2"], 4)
    overfit_status = "healthy_generalization" if overfit_gap < 0.02 else "mild_overfit" if overfit_gap < 0.05 else "overfit"

    print(f"\n  Global Train → R²: {g_train_m['r2']:.4f}  MAE: ₹{g_train_m['mae']:,.0f}")
    print(f"  Global Val   → R²: {g_val_m['r2']:.4f}  MAE: ₹{g_val_m['mae']:,.0f}")
    print(f"  Global Test  → R²: {g_test_m['r2']:.4f}  MAE: ₹{g_test_m['mae']:,.0f}  MAPE: {g_test_m['mape']:.2f}%")
    print(f"  Overfit gap: {overfit_gap} ({overfit_status})")

    # Save global artifact
    global_artifact = {
        "catboost":        cat_model,
        "lightgbm":        lgb_model,
        "xgboost":         xgb_model,
        "weights":         weights,
        "features":        FEATURES,
        "cat_features":    CAT_FEATURES,
        "segment_class":   "global",
        "brand_segment_map": BRAND_SEGMENT_MAP,
        "category_levels": cat_levels,
        "test_metrics":    g_test_m,
    }
    joblib.dump(global_artifact, ARTIFACT_DIR / "ensemble_global.pkl")

    # Also save individual model files for backward-compat
    cat_model.save_model(str(ARTIFACT_DIR / "vehicle_price_catboost.cbm"))
    lgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_lightgbm.txt"))
    xgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_xgboost.json"))

    # ── Segment models ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SEGMENT MODELS (economy / premium / luxury)")
    print(f"{'='*60}")

    df["segment_class"] = df["segment"].map(SEGMENT_MAP).fillna("economy")
    segment_metrics: dict = {}

    for seg in SEGMENT_CLASSES:
        print(f"\n--- {seg.upper()} ---")
        seg_df  = df[df["segment_class"] == seg].copy()
        artifact = train_segment(seg_df, seg)
        if artifact is None:
            continue
        joblib.dump(artifact, ARTIFACT_DIR / f"ensemble_{seg}.pkl")
        print(f"  Saved: ensemble_{seg}.pkl")
        segment_metrics[seg] = {
            "rows":      artifact["rows"],
            "test_r2":   artifact["test_metrics"]["r2"],
            "test_mae":  artifact["test_metrics"]["mae"],
            "test_mape": artifact["test_metrics"]["mape"],
        }

    # ── Metadata ───────────────────────────────────────────────────────────────
    feat_imp = {}
    try:
        imp = cat_model.get_feature_importance()
        feat_imp = {FEATURES[i]: round(float(imp[i]), 4) for i in range(len(FEATURES))}
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass

    metadata = {
        "model_name":    "CatBoost+LightGBM+XGBoost Segment Ensemble",
        "version":       "5.0",
        "trained_at":    datetime.utcnow().isoformat(),
        "target":        "selling_price",
        "target_transform": "log1p during training, expm1 during prediction",
        "prediction_unit": "INR",
        "dataset":       str(DATA_CSV.name),
        "features":      FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_features":     NUMERIC_FEATURES,
        "split_strategy": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "segment_map":   SEGMENT_MAP,
        "segment_classes": SEGMENT_CLASSES,
        "ensemble": {
            "enabled":  True,
            "strategy": "SLSQP-optimised weighted average on validation R²",
            "weights":  weights,
        },
        "global_metrics": {
            "train":      g_train_m,
            "validation": g_val_m,
            "test":       g_test_m,
        },
        "overfitting_check": {
            "train_r2":         g_train_m["r2"],
            "validation_r2":    g_val_m["r2"],
            "test_r2":          g_test_m["r2"],
            "train_val_r2_gap": overfit_gap,
            "status":           overfit_status,
        },
        "individual_base_models": individual_results,
        "segmented_models":       segment_metrics,
        "feature_importance":     feat_imp,
        "condition_multipliers":  CONDITION_MULTIPLIERS,
        "brand_segment_map":      BRAND_SEGMENT_MAP,
    }

    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(ARTIFACT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Sample CSV
    df.sample(min(500, len(df)), random_state=RANDOM_STATE)[[
        "brand", "model", "vehicle_age", "odometer_reading",
        "fuel_type", "transmission", "segment_class", "selling_price",
    ]].to_csv(ARTIFACT_DIR / "cleaned_training_sample.csv", index=False)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY  (v5.0)")
    print(f"{'='*60}")
    print(f"Global model  → R²: {g_test_m['r2']:.4f}  MAE: ₹{g_test_m['mae']:,.0f}  MAPE: {g_test_m['mape']:.2f}%")
    for seg, m in segment_metrics.items():
        print(f"{seg.upper():<10} → R²: {m['test_r2']:.4f}  MAE: ₹{m['test_mae']:>12,.0f}  "
              f"MAPE: {m['test_mape']:.2f}%  ({m['rows']:,} rows)")
    print(f"{'='*60}")
    print(f"\nAll artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    train()
