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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
from ml_training import registry_helper

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT         = Path(__file__).resolve().parents[1]
DATASET      = Path(__file__).resolve().parent / "data" / "processed_widown1-6.csv"
VARIANT_ID   = registry_helper.next_variant_id()
ARTIFACT_DIR = registry_helper.get_variant_dir(VARIANT_ID)
print(f"Training run -> Variant ID: {VARIANT_ID} ({ARTIFACT_DIR})")

RANDOM_STATE = 42
DIV = "=" * 80

TARGET = "selling_price"

CAT_FEATURES = [
    "brand", "model", "variant", "city", "locality",
    "rto", "segment_class", "fuel_type", "transmission", "seller_type",
]

NUMERIC_FEATURES = [
    "vehicle_age", "odometer_reading", "km_per_year", "owner_count",
    "brand_tier", "age_km_interaction", "ownership_trust_score",
    "vehicle_health_score", "is_high_mileage", "locality_tier",
    "usage_category_num", "locality_density_norm", "popularity_score_log",
]

FEATURES = CAT_FEATURES + NUMERIC_FEATURES

SEGMENTS = {
    "0_6_lakh":     (0,          600_000),
    "6_12_lakh":    (600_000,  1_200_000),
    "12_plus_lakh": (1_200_000, 20_000_000),
}

MIN_SEGMENT_ROWS = 200


def calculate_metrics(y_true, y_pred) -> dict:
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred)
    mae  = mean_absolute_error(y_true_price, y_pred_price)
    rmse = math.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2   = r2_score(np.log1p(y_true_price), np.log1p(y_pred_price))
    mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100
    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,4), "MAPE": round(mape,2)}


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


def prepare_frames(df, category_levels, encoders=None):
    available_features = [f for f in FEATURES if f in df.columns]
    frame = df[available_features].copy()

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

    active_encoders = {}
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


def prepare_training_frames(X_train, X_val):
    print("\nPreparing model inputs ...")
    cat_levels = build_category_levels(X_train)
    cb_train, lgb_train, xgb_train, encoders = prepare_frames(X_train, cat_levels)
    cb_val,   lgb_val,   xgb_val,   _        = prepare_frames(X_val,   cat_levels, encoders)
    return {
        "category_levels": cat_levels,
        "encoders":        encoders,
        "catboost":  {"train": cb_train,  "val": cb_val},
        "lightgbm":  {"train": lgb_train, "val": lgb_val},
        "xgboost":   {"train": xgb_train, "val": xgb_val},
    }


def load_dataset() -> pd.DataFrame:
    print(DIV); print("LOADING DATASET"); print(DIV)
    df = pd.read_csv(DATASET)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}"); print("VALIDATING DATASET"); print(DIV)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    present = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    print(f"Features present : {len(present)} / {len(FEATURES)}")
    if missing:
        print("Features missing (will use defaults):")
        for f in missing:
            print(f"  - {f}")
    return df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}"); print("CLEANING TRAINING DATA"); print(DIV)
    before = len(df)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(50_000, 20_000_000)]

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vehicle_age", "odometer_reading", "km_per_year", "owner_count"])

    fill_defaults = {
        "brand_tier": 1, "age_km_interaction": 0, "ownership_trust_score": 75,
        "vehicle_health_score": 50, "is_high_mileage": 0, "locality_tier": 2,
        "usage_category_num": 0, "locality_density_norm": 0.5,
        "popularity_score_log": 0,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    print(f"Removed  : {before - len(df):,} rows")
    print(f"Remaining: {len(df):,} rows")
    return df


def split_dataset(df):
    print(f"\n{DIV}"); print("TRAIN / VAL SPLIT  (70 / 30)"); print(DIV)
    X = df[[f for f in FEATURES if f in df.columns]]
    y = np.log1p(df[TARGET])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    return X_train, X_val, y_train, y_val


def train_catboost(X_train, y_train, X_val, y_val):
    print("\nTraining CatBoost ...")
    cat_cols = [c for c in CAT_FEATURES if c in X_train.columns]
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=8,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=RANDOM_STATE, l2_leaf_reg=5,
        min_data_in_leaf=15, early_stopping_rounds=100, verbose=200,
    )
    model.fit(
        Pool(X_train, y_train, cat_features=cat_cols),
        eval_set=Pool(X_val, y_val, cat_features=cat_cols),
        use_best_model=True,
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM ...")
    model = lgb.train(
        {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.03, "num_leaves": 64,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "min_child_samples": 15,
            "verbosity": -1, "seed": RANDOM_STATE,
        },
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost ...")
    model = xgb.train(
        {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "learning_rate": 0.03, "max_depth": 8,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": RANDOM_STATE,
        },
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=3000,
        evals=[(xgb.DMatrix(X_val, label=y_val), "val")],
        early_stopping_rounds=100, verbose_eval=200,
    )
    return model


def predict(model, model_name, X):
    if model_name == "CatBoost": return model.predict(X)
    if model_name == "LightGBM": return model.predict(X)
    if model_name == "XGBoost":  return model.predict(xgb.DMatrix(X))
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, model_name, X, y, label="Val"):
    preds  = predict(model, model_name, X)
    scores = calculate_metrics(y, preds)
    print(f"\n{'='*50}")
    print(f"{model_name}  [{label}]")
    print(f"  MAE  : Rs.{scores['MAE']:,.0f}")
    print(f"  RMSE : Rs.{scores['RMSE']:,.0f}")
    print(f"  MAPE : {scores['MAPE']:.2f}%")
    print(f"  R2   : {scores['R2']:.4f}")
    return scores, preds


def optimise_weights(cb_preds, lgb_preds, xgb_preds, y_true):
    print(f"\n{DIV}"); print("ENSEMBLE WEIGHT OPTIMISATION"); print(DIV)

    def neg_r2(w):
        w = np.array(w) / np.sum(w)
        ens = w[0]*cb_preds + w[1]*lgb_preds + w[2]*xgb_preds
        return -r2_score(np.log1p(np.expm1(y_true)), np.log1p(np.expm1(ens)))

    res = minimize(
        neg_r2, x0=[1/3, 1/3, 1/3], method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1}
    )
    w = res.x / res.x.sum()
    print(f"  CatBoost : {w[0]*100:.1f}%")
    print(f"  LightGBM : {w[1]*100:.1f}%")
    print(f"  XGBoost  : {w[2]*100:.1f}%")
    return w


def evaluate_ensemble(w, cb, lgb_p, xgb_p, y, label="Val"):
    ens    = w[0]*cb + w[1]*lgb_p + w[2]*xgb_p
    scores = calculate_metrics(y, ens)
    print(f"\n{'='*50}")
    print(f"ENSEMBLE  [{label}]")
    print(f"  MAE  : Rs.{scores['MAE']:,.0f}")
    print(f"  RMSE : Rs.{scores['RMSE']:,.0f}")
    print(f"  MAPE : {scores['MAPE']:.2f}%")
    print(f"  R2   : {scores['R2']:.4f}")
    return scores


def save_artifacts(cat_model, lgb_model, xgb_model, weights, cat_levels, encoders, metadata):
    print(f"\n{DIV}"); print("SAVING ARTIFACTS"); print(DIV)

    cat_model.save_model(str(ARTIFACT_DIR / "vehicle_price_catboost.cbm"))
    lgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_lightgbm.txt"))
    xgb_model.save_model(str(ARTIFACT_DIR / "vehicle_price_xgboost.json"))

    bundle = {
        "weights": {
            "catboost": float(weights[0]),
            "lightgbm": float(weights[1]),
            "xgboost":  float(weights[2]),
        },
        "category_levels":  cat_levels,
        "encoders":         {col: enc.classes_.tolist() for col, enc in encoders.items()},
        "features":         FEATURES,
        "cat_features":     CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "segments":         SEGMENTS,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "ensemble_bundle.pkl")

        model_meta = {
        "model_name": "CatBoostRegressor",
        "trained_at": metadata.get("training_time"),
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metrics": metadata.get("val_metrics", {}).get("Ensemble", {}),
        "ensemble": {
            "enabled": True,
            "weights": {
                "catboost": float(weights[0]),
                "lightgbm": float(weights[1]),
                "xgboost":  float(weights[2]),
            },
            "category_levels": cat_levels,
        }
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    with open(ARTIFACT_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("  Saved: vehicle_price_catboost.cbm")
    print("  Saved: vehicle_price_lightgbm.txt")
    print("  Saved: vehicle_price_xgboost.json")
    print("  Saved: ensemble_bundle.pkl")
    print("  Saved: model_metadata.json")
    print("  Saved: training_report.json")


def train_segment_model(seg_name, seg_df, global_model, global_cat_levels):
    print(f"\n{'─'*60}")
    print(f"  SEGMENT: {seg_name.upper()}  ({len(seg_df):,} rows)")

    if len(seg_df) < MIN_SEGMENT_ROWS:
        print(f"  SKIP — only {len(seg_df)} rows (min {MIN_SEGMENT_ROWS})")
        return None, None, None

    X = seg_df[[f for f in FEATURES if f in seg_df.columns]]
    y = np.log1p(seg_df[TARGET])

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )

    seg_levels     = build_category_levels(X_tr)
    cb_tr, _, _, _ = prepare_frames(X_tr, seg_levels)
    cb_v,  _, _, _ = prepare_frames(X_v,  seg_levels)

    cat_cols = [c for c in CAT_FEATURES if c in cb_tr.columns]

    seg_model = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=RANDOM_STATE, l2_leaf_reg=5,
        min_data_in_leaf=10, early_stopping_rounds=100, verbose=200,
    )
    seg_model.fit(
        Pool(cb_tr, y_tr, cat_features=cat_cols),
        eval_set=Pool(cb_v, y_v, cat_features=cat_cols),
        use_best_model=True,
    )

    seg_preds    = seg_model.predict(cb_v)
    seg_scores   = calculate_metrics(y_v, seg_preds)

    cb_v_g, _, _, _ = prepare_frames(X_v, global_cat_levels)
    global_preds    = global_model.predict(cb_v_g)
    global_scores   = calculate_metrics(y_v, global_preds)

    print(f"  Segment MAPE : {seg_scores['MAPE']:.2f}%")
    print(f"  Global  MAPE : {global_scores['MAPE']:.2f}%")

    if seg_scores["MAPE"] < global_scores["MAPE"]:
        print(f"  ACTIVE — segment model wins")
        return seg_model, seg_levels, seg_scores
    else:
        print(f"  SKIP   — global model better")
        return None, None, global_scores


def train_segmented_models(df, global_model, global_cat_levels):
    print(f"\n{DIV}"); print("SEGMENTED TRAINING"); print(DIV)
    results = {}
    for seg_name, (pmin, pmax) in SEGMENTS.items():
        mask   = df[TARGET].between(pmin, pmax)
        seg_df = df[mask].copy()
        m, lv, sc = train_segment_model(seg_name, seg_df, global_model, global_cat_levels)
        results[seg_name] = {
            "model": m, "cat_levels": lv, "scores": sc,
            "active": m is not None,
            "price_range": (pmin, pmax), "row_count": int(mask.sum()),
        }
    return results


def save_segment_artifacts(segment_results):
    print(f"\n{DIV}"); print("SAVING SEGMENT ARTIFACTS"); print(DIV)
    routing = {}
    for seg_name, r in segment_results.items():
        if r["active"]:
            r["model"].save_model(str(ARTIFACT_DIR / f"segment_{seg_name}.cbm"))
            joblib.dump(r["cat_levels"], ARTIFACT_DIR / f"segment_{seg_name}_levels.pkl")
            routing[seg_name] = {
                "active": True,
                "model_file":  f"segment_{seg_name}.cbm",
                "levels_file": f"segment_{seg_name}_levels.pkl",
                "price_range": r["price_range"],
                "row_count":   r["row_count"],
                "mape": r["scores"]["MAPE"] if r["scores"] else None,
                "r2":   r["scores"]["R2"]   if r["scores"] else None,
            }
            print(f"  SAVED: {seg_name}  MAPE {r['scores']['MAPE']:.2f}%")
        else:
            routing[seg_name] = {
                "active": False, "fallback": "global",
                "price_range": r["price_range"],
                "row_count":   r["row_count"],
                "mape": r["scores"]["MAPE"] if r["scores"] else None,
            }
            print(f"  SKIP : {seg_name}  using global fallback")

    with open(ARTIFACT_DIR / "routing_table.json", "w") as f:
        json.dump(routing, f, indent=4)
    print("  Saved: routing_table.json")
    return routing


def train_all_models():
    df = load_dataset()
    df = validate_dataset(df)
    df = clean_training_data(df)

    X_train, X_val, y_train, y_val = split_dataset(df)
    frames     = prepare_training_frames(X_train, X_val)
    cat_levels = frames["category_levels"]

    cat_model = train_catboost(frames["catboost"]["train"], y_train, frames["catboost"]["val"], y_val)
    lgb_model = train_lightgbm(frames["lightgbm"]["train"], y_train, frames["lightgbm"]["val"], y_val)
    xgb_model = train_xgboost(frames["xgboost"]["train"],  y_train, frames["xgboost"]["val"],  y_val)

    cat_v_sc, cat_v_p = evaluate_model(cat_model, "CatBoost", frames["catboost"]["val"], y_val)
    lgb_v_sc, lgb_v_p = evaluate_model(lgb_model, "LightGBM", frames["lightgbm"]["val"], y_val)
    xgb_v_sc, xgb_v_p = evaluate_model(xgb_model, "XGBoost",  frames["xgboost"]["val"],  y_val)

    weights    = optimise_weights(cat_v_p, lgb_v_p, xgb_v_p, y_val)
    val_scores = evaluate_ensemble(weights, cat_v_p, lgb_v_p, xgb_v_p, y_val, "Val")

    comparison = pd.DataFrame([
        {"Model": "CatBoost", **cat_v_sc},
        {"Model": "LightGBM", **lgb_v_sc},
        {"Model": "XGBoost",  **xgb_v_sc},
        {"Model": "Ensemble", **val_scores},
    ]).sort_values("R2", ascending=False)

    print(f"\n{DIV}"); print("MODEL COMPARISON [Validation Set]"); print(DIV)
    print(comparison.to_string(index=False))

    metadata = {
        "training_time":    datetime.now().isoformat(),
        "dataset":          str(DATASET),
        "rows_used":        len(df),
        "features":         FEATURES,
        "cat_features":     CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "segment_definitions": {k: list(v) for k, v in SEGMENTS.items()},
        "ensemble_weights": {
            "catboost": float(weights[0]),
            "lightgbm": float(weights[1]),
            "xgboost":  float(weights[2]),
        },
        "val_metrics": {
            "CatBoost": cat_v_sc, "LightGBM": lgb_v_sc,
            "XGBoost":  xgb_v_sc, "Ensemble": val_scores,
        },
    }

    save_artifacts(cat_model, lgb_model, xgb_model,
                   weights, cat_levels, frames["encoders"], metadata)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    seg_results   = train_segmented_models(df, cat_model, cat_levels)
    routing_table = save_segment_artifacts(seg_results)

    print(f"\n{DIV}"); print("SEGMENT SUMMARY"); print(DIV)
    rows = []
    for name, r in seg_results.items():
        rows.append({
            "Segment": name,
            "Rows":    r["row_count"],
            "Active":  "YES" if r["active"] else "global",
            "MAPE":    f"{r['scores']['MAPE']:.2f}%" if r["scores"] else "N/A",
            "R2":      f"{r['scores']['R2']:.4f}"    if r["scores"] else "N/A",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nGenerating dataset_catalog.json ...")
    try:
        _df = pd.read_csv(DATASET, usecols=["brand", "model", "variant"])
        _df = _df.dropna(subset=["brand", "model"])
        for c in ["brand", "model", "variant"]:
            _df[c] = _df[c].astype(str).str.strip().str.lower()
        catalog: dict = {}
        for brand, bdf in _df.groupby("brand"):
            catalog[brand] = {}
            for model, mdf in bdf.groupby("model"):
                catalog[brand][model] = sorted(mdf["variant"].dropna().unique().tolist())
        cat_path = ARTIFACT_DIR / "dataset_catalog.json"
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        print(f"  Saved {len(catalog)} brands → {cat_path}")
    except Exception as e:
        print(f"  WARNING: catalog failed: {e}")

    print(f"\n{DIV}"); print("TRAINING COMPLETE"); print(DIV)
    print(f"  Global MAPE : {val_scores['MAPE']:.2f}%")
    print(f"  Global R2   : {val_scores['R2']:.4f}")
    print(f"  Artifacts   : {ARTIFACT_DIR}")
    reg_metrics = {
        "mae":  val_scores.get("MAE", 0) if isinstance(val_scores, dict) else 0,
        "rmse": val_scores.get("RMSE", 0) if isinstance(val_scores, dict) else 0,
        "r2":   val_scores.get("R2", 0) if isinstance(val_scores, dict) else 0,
        "mape": val_scores.get("MAPE", 0) if isinstance(val_scores, dict) else 0,
    }
    registry_helper.register_variant(
        variant_id=VARIANT_ID,
        artifact_dir=ARTIFACT_DIR,
        dataset_name=DATASET.name,
        metrics=reg_metrics,
    )
    registry_helper.copy_to_model_artifacts(ARTIFACT_DIR)

    return {"comparison": comparison, "metadata": metadata, "segments": seg_results}


if __name__ == "__main__":
    train_all_models()