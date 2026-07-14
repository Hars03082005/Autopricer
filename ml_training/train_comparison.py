"""
train_comparison.py
===================
Train the PriceRef ML ensemble on both processed datasets separately
and produce a side-by-side comparison of all metrics.

Inputs  (already preprocessed by preprocess_flat_csv.py):
  ml_training/data/processed_cell7_dataset.csv
  ml_training/data/processed_owner_assumed_dataset.csv

Outputs:
  model_artifacts_cell7/          <- artifacts from cell7 training
  model_artifacts_owner_assumed/  <- artifacts from owner-assumed training
  model_artifacts_cell7/comparison_report.json
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "cell7": {
        "csv":      DATA_DIR / "processed_cell7_dataset.csv",
        "art_dir":  ROOT / "model_artifacts_cell7",
        "label":    "Cell7 Dataset (original owners)",
    },
    "owner_assumed": {
        "csv":      DATA_DIR / "processed_owner_assumed_dataset.csv",
        "art_dir":  ROOT / "model_artifacts_owner_assumed",
        "label":    "Owner-Assumed Dataset (filled owners)",
    },
}

RANDOM_STATE    = 42
SEGMENT_CLASSES = ["economy", "premium", "luxury"]
DIV = "=" * 72

# ── Feature sets (canonical — matches preprocessed column names) ──────────────
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


def blend(weights: dict, preds: dict) -> np.ndarray:
    out = np.zeros(len(next(iter(preds.values()))))
    for name, w in weights.items():
        out += w * np.asarray(preds[name])
    return out


def optimise_weights(y_val: np.ndarray, val_preds: dict) -> dict:
    names = list(val_preds.keys())
    n = len(names)
    def neg_r2(w):
        blended = sum(w[i] * val_preds[names[i]] for i in range(n))
        return -r2_score(y_val, blended)
    res = minimize(neg_r2, x0=[1.0/n]*n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints={"type":"eq","fun":lambda w: sum(w)-1})
    return {names[i]: round(float(res.x[i]), 4) for i in range(n)}


def build_category_levels(df: pd.DataFrame) -> dict:
    levels = {}
    for col in CAT_FEATURES:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique().tolist()
            if "unknown" not in vals:
                vals.append("unknown")
            levels[col] = sorted(vals)
    return levels


def prepare_frames(df: pd.DataFrame, cat_levels: dict):
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
            frame[col] = frame[col].fillna(0 if np.isnan(med) else med)

    cb = frame.copy()

    lgb_f = frame.copy()
    for col in CAT_FEATURES:
        if col in lgb_f.columns:
            le = LabelEncoder()
            lgb_f[col] = le.fit_transform(lgb_f[col].astype(str))

    xgb_f = lgb_f.copy()
    return cb, lgb_f, xgb_f


# ── Model trainers ─────────────────────────────────────────────────────────────
def train_catboost(X_tr, y_tr, X_vl, y_vl) -> CatBoostRegressor:
    cat_cols = [c for c in CAT_FEATURES if c in X_tr.columns]
    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.04, depth=8,
        l2_leaf_reg=3, min_data_in_leaf=15,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=RANDOM_STATE, early_stopping_rounds=100,
        verbose=False, cat_features=cat_cols,
    )
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_cols),
        eval_set=Pool(X_vl, y_vl, cat_features=cat_cols),
        use_best_model=True,
    )
    return model


def train_lightgbm(X_tr, y_tr, X_vl, y_vl) -> lgb.Booster:
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.04, "num_leaves": 127,
        "min_child_samples": 15, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 0.1,
        "verbose": -1, "random_state": RANDOM_STATE,
    }
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval   = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=2000,
                     valid_sets=[dval],
                     callbacks=[lgb.early_stopping(100, verbose=False),
                                lgb.log_evaluation(-1)])


def train_xgboost(X_tr, y_tr, X_vl, y_vl) -> xgb.Booster:
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.04, "max_depth": 8,
        "min_child_weight": 15, "subsample": 0.8,
        "colsample_bytree": 0.8, "lambda": 1.0, "alpha": 0.1,
        "seed": RANDOM_STATE, "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_vl, label=y_vl)
    return xgb.train(params, dtrain, num_boost_round=2000,
                     evals=[(dval, "val")],
                     early_stopping_rounds=100, verbose_eval=False)


def _predict(name, model, frame) -> np.ndarray:
    if name == "catboost":  return np.asarray(model.predict(frame))
    if name == "lightgbm":  return np.asarray(model.predict(frame))
    if name == "xgboost":   return np.asarray(model.predict(xgb.DMatrix(frame)))
    raise ValueError(name)


# ── Data loader ────────────────────────────────────────────────────────────────
def load_processed(csv_path: Path, label: str) -> pd.DataFrame:
    print(f"\n  Loading {label} from {csv_path.name} …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}  columns: {df.shape[1]}")

    # Validate target
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df.dropna(subset=["selling_price"], inplace=True)
    df = df[df["selling_price"].between(50_000, 20_000_000)]

    # Clip numerics
    df["vehicle_age"]      = pd.to_numeric(df["vehicle_age"], errors="coerce").clip(0, 35).fillna(5)
    df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce").clip(0, 600_000).fillna(50_000)
    df["km_per_year"]      = pd.to_numeric(df["km_per_year"], errors="coerce").clip(0, 100_000).fillna(10_000)

    # Fill remaining numerics
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill categoricals
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
        else:
            df[col] = "unknown"

    # Ensure segment_class column exists
    if "segment_class" not in df.columns:
        df["segment_class"] = "economy"

    df.dropna(subset=NUMERIC_FEATURES[:4], inplace=True)
    print(f"  Clean rows for training: {len(df):,}")
    print(f"  Segment dist: {df['segment_class'].value_counts().to_dict()}")
    return df


# ── Segment trainer ────────────────────────────────────────────────────────────
def train_segment(seg_df: pd.DataFrame, seg_name: str) -> dict | None:
    MIN_ROWS = 300
    if len(seg_df) < MIN_ROWS:
        print(f"  [SKIP] {seg_name} — only {len(seg_df)} rows (< {MIN_ROWS})")
        return None

    y = np.log1p(seg_df["selling_price"].values)
    X = seg_df[FEATURES]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_vl, X_te, y_vl, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    cat_levels = build_category_levels(seg_df.loc[X_tr.index])
    cb_tr, lgb_tr, xgb_tr = prepare_frames(seg_df.loc[X_tr.index], cat_levels)
    cb_vl, lgb_vl, xgb_vl = prepare_frames(seg_df.loc[X_vl.index], cat_levels)
    cb_te, lgb_te, xgb_te = prepare_frames(seg_df.loc[X_te.index], cat_levels)

    print(f"    CatBoost …", end=" ", flush=True)
    s_cb  = train_catboost(cb_tr, y_tr, cb_vl, y_vl)
    print(f"LightGBM …", end=" ", flush=True)
    s_lgb = train_lightgbm(lgb_tr, y_tr, lgb_vl, y_vl)
    print(f"XGBoost …", end=" ", flush=True)
    s_xgb = train_xgboost(xgb_tr, y_tr, xgb_vl, y_vl)
    print("done")

    s_models = {"catboost": s_cb, "lightgbm": s_lgb, "xgboost": s_xgb}
    s_frames  = {
        "catboost": {"val": cb_vl, "test": cb_te},
        "lightgbm": {"val": lgb_vl, "test": lgb_te},
        "xgboost":  {"val": xgb_vl, "test": xgb_te},
    }

    val_preds  = {n: _predict(n, m, s_frames[n]["val"])  for n, m in s_models.items()}
    s_weights  = optimise_weights(y_vl, val_preds)
    test_blend = blend(s_weights, {n: _predict(n, m, s_frames[n]["test"]) for n, m in s_models.items()})
    seg_m = metrics(y_te, test_blend)
    print(f"    {seg_name.upper():<10} R²:{seg_m['r2']:.4f}  MAE:₹{seg_m['mae']:>10,.0f}  MAPE:{seg_m['mape']:.2f}%  rows:{len(seg_df):,}")

    return {
        "catboost": s_cb, "lightgbm": s_lgb, "xgboost": s_xgb,
        "weights": s_weights, "features": FEATURES,
        "cat_features": CAT_FEATURES, "segment_class": seg_name,
        "category_levels": cat_levels, "rows": len(seg_df),
        "test_metrics": seg_m,
    }


# ── Main training run ──────────────────────────────────────────────────────────
def run_training(key: str, cfg: dict) -> dict:
    label   = cfg["label"]
    csv     = cfg["csv"]
    art_dir = cfg["art_dir"]
    art_dir.mkdir(exist_ok=True)

    print(f"\n{DIV}")
    print(f"TRAINING: {label}")
    print(DIV)

    t_start = datetime.now()
    df = load_processed(csv, label)

    y = np.log1p(df["selling_price"].values)
    X = df[FEATURES]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_vl, X_te, y_vl, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    cat_levels = build_category_levels(df.loc[X_tr.index])
    cb_tr, lgb_tr, xgb_tr = prepare_frames(df.loc[X_tr.index], cat_levels)
    cb_vl, lgb_vl, xgb_vl = prepare_frames(df.loc[X_vl.index], cat_levels)
    cb_te, lgb_te, xgb_te = prepare_frames(df.loc[X_te.index], cat_levels)

    print(f"\n  [GLOBAL MODEL]")
    print(f"    Train: {len(X_tr):,}  Val: {len(X_vl):,}  Test: {len(X_te):,}")

    print(f"    CatBoost …", end=" ", flush=True)
    cat_model = train_catboost(cb_tr, y_tr, cb_vl, y_vl)
    print(f"LightGBM …", end=" ", flush=True)
    lgb_model = train_lightgbm(lgb_tr, y_tr, lgb_vl, y_vl)
    print(f"XGBoost …", end=" ", flush=True)
    xgb_model = train_xgboost(xgb_tr, y_tr, xgb_vl, y_vl)
    print("done")

    g_models = {"catboost": cat_model, "lightgbm": lgb_model, "xgboost": xgb_model}
    g_frames  = {
        "catboost": {"train": cb_tr, "val": cb_vl, "test": cb_te},
        "lightgbm": {"train": lgb_tr, "val": lgb_vl, "test": lgb_te},
        "xgboost":  {"train": xgb_tr, "val": xgb_vl, "test": xgb_te},
    }

    # Individual base model metrics
    ind_results = {}
    val_preds   = {}
    for name, model in g_models.items():
        tr_m = metrics(y_tr, _predict(name, model, g_frames[name]["train"]))
        vl_m = metrics(y_vl, _predict(name, model, g_frames[name]["val"]))
        te_m = metrics(y_te, _predict(name, model, g_frames[name]["test"]))
        ind_results[name] = {"train": tr_m, "val": vl_m, "test": te_m}
        val_preds[name]   = _predict(name, model, g_frames[name]["val"])
        print(f"    {name:<12} Val R²:{vl_m['r2']:.4f}  Test MAPE:{te_m['mape']:.2f}%")

    weights    = optimise_weights(y_vl, val_preds)
    print(f"\n    Ensemble weights: {weights}")

    test_blend  = blend(weights, {n: _predict(n, m, g_frames[n]["test"])  for n, m in g_models.items()})
    train_blend = blend(weights, {n: _predict(n, m, g_frames[n]["train"]) for n, m in g_models.items()})
    val_blend   = blend(weights, {n: _predict(n, m, g_frames[n]["val"])   for n, m in g_models.items()})

    g_tr_m = metrics(y_tr, train_blend)
    g_vl_m = metrics(y_vl, val_blend)
    g_te_m = metrics(y_te, test_blend)

    overfit_gap = round(g_tr_m["r2"] - g_te_m["r2"], 4)
    overfit_status = (
        "healthy_generalization" if overfit_gap < 0.02 else
        "mild_overfit" if overfit_gap < 0.05 else "overfit"
    )

    print(f"\n    Global Train → R²:{g_tr_m['r2']:.4f}  MAE:₹{g_tr_m['mae']:,.0f}")
    print(f"    Global Val   → R²:{g_vl_m['r2']:.4f}  MAE:₹{g_vl_m['mae']:,.0f}")
    print(f"    Global Test  → R²:{g_te_m['r2']:.4f}  MAE:₹{g_te_m['mae']:,.0f}  MAPE:{g_te_m['mape']:.2f}%")
    print(f"    Overfit gap  : {overfit_gap} ({overfit_status})")

    # Save global artifacts
    global_art = {
        "catboost": cat_model, "lightgbm": lgb_model, "xgboost": xgb_model,
        "weights": weights, "features": FEATURES, "cat_features": CAT_FEATURES,
        "segment_class": "global", "category_levels": cat_levels,
        "test_metrics": g_te_m,
    }
    joblib.dump(global_art, art_dir / "ensemble_global.pkl")
    cat_model.save_model(str(art_dir / "vehicle_price_catboost.cbm"))
    lgb_model.save_model(str(art_dir / "vehicle_price_lightgbm.txt"))
    xgb_model.save_model(str(art_dir / "vehicle_price_xgboost.json"))

    # ── Segment models ─────────────────────────────────────────────────────────
    print(f"\n  [SEGMENT MODELS]")
    df["segment_class"] = df["segment_class"].fillna("economy")
    seg_metrics: dict = {}
    for seg in SEGMENT_CLASSES:
        print(f"\n  -- {seg.upper()} --")
        seg_df   = df[df["segment_class"] == seg].copy()
        artifact = train_segment(seg_df, seg)
        if artifact is None:
            continue
        joblib.dump(artifact, art_dir / f"ensemble_{seg}.pkl")
        seg_metrics[seg] = {
            "rows": artifact["rows"],
            "test_r2":   artifact["test_metrics"]["r2"],
            "test_mae":  artifact["test_metrics"]["mae"],
            "test_mape": artifact["test_metrics"]["mape"],
            "weights":   artifact["weights"],
        }

    # Feature importance
    feat_imp = {}
    try:
        imp = cat_model.get_feature_importance()
        feat_imp = {FEATURES[i]: round(float(imp[i]), 4) for i in range(len(FEATURES))}
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass

    elapsed = (datetime.now() - t_start).total_seconds()

    # Full metadata
    metadata = {
        "dataset":           key,
        "dataset_label":     label,
        "csv_file":          str(csv.name),
        "trained_at":        datetime.utcnow().isoformat(),
        "model_name":        "CatBoost+LightGBM+XGBoost Segment Ensemble",
        "version":           "5.1",
        "total_rows":        len(df),
        "training_seconds":  round(elapsed, 1),
        "features":          FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_features":     NUMERIC_FEATURES,
        "split":             {"train": 0.70, "val": 0.15, "test": 0.15},
        "ensemble_weights":  weights,
        "global_metrics":    {"train": g_tr_m, "val": g_vl_m, "test": g_te_m},
        "overfitting":       {"gap": overfit_gap, "status": overfit_status},
        "segment_metrics":   seg_metrics,
        "individual_models": ind_results,
        "feature_importance": feat_imp,
    }

    with open(art_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved all artifacts to: {art_dir}")
    print(f"  Training time: {elapsed:.0f}s")
    return metadata


# ── Comparison printer ─────────────────────────────────────────────────────────
def print_comparison(results: dict[str, dict]) -> None:
    keys   = list(results.keys())
    labels = [results[k]["dataset_label"] for k in keys]
    r      = {k: results[k] for k in keys}

    def row(label, vals):
        print(f"  {label:<35} {vals[0]:>22}   {vals[1]:>22}")

    print(f"\n\n{'='*72}")
    print("  COMPARISON REPORT")
    print(f"{'='*72}")
    print(f"  {'Metric':<35} {labels[0]:>22}   {labels[1]:>22}")
    print(f"  {'-'*35} {'-'*22}   {'-'*22}")

    # Dataset info
    row("Total Training Rows",
        [f"{r[k]['total_rows']:,}" for k in keys])
    row("Training Time (sec)",
        [f"{r[k]['training_seconds']:.0f}s" for k in keys])

    # Global metrics
    print(f"\n  GLOBAL ENSEMBLE (test split)")
    for metric, name in [("r2","R² Score"), ("mae","MAE (₹)"), ("rmse","RMSE (₹)"), ("mape","MAPE (%)")]:
        vals = []
        for k in keys:
            v = r[k]["global_metrics"]["test"][metric]
            vals.append(f"₹{v:,.0f}" if metric in ("mae","rmse") else f"{v:.4f}" if metric=="r2" else f"{v:.2f}%")
        row(f"  Global {name}", vals)

    # Overfitting
    row("  Overfit Gap (Train−Test R²)",
        [f"{r[k]['overfitting']['gap']:.4f} ({r[k]['overfitting']['status'][:12]})" for k in keys])

    # Ensemble weights
    print(f"\n  ENSEMBLE WEIGHTS")
    for m in ("catboost", "lightgbm", "xgboost"):
        row(f"  {m}", [f"{r[k]['ensemble_weights'].get(m, 0):.4f}" for k in keys])

    # Segment metrics
    print(f"\n  SEGMENT MODELS")
    for seg in ("economy", "premium", "luxury"):
        print(f"\n  [{seg.upper()}]")
        for metric, name in [("rows","Rows"), ("test_r2","R²"), ("test_mae","MAE (₹)"), ("test_mape","MAPE (%)")]:
            vals = []
            for k in keys:
                sm = r[k]["segment_metrics"].get(seg, {})
                v  = sm.get(metric, "n/a")
                if isinstance(v, float):
                    vals.append(f"₹{v:,.0f}" if metric=="test_mae" else f"{v:.4f}" if metric=="test_r2" else f"{v:.2f}%")
                else:
                    vals.append(f"{v:,}" if isinstance(v, int) else str(v))
            row(f"  {name}", vals)

    # Feature importance (top 5 from cell7 as reference)
    print(f"\n  TOP 5 FEATURE IMPORTANCE (CatBoost, cell7)")
    if "cell7" in r and r["cell7"].get("feature_importance"):
        fi = r["cell7"]["feature_importance"]
        for i, (feat, imp) in enumerate(list(fi.items())[:5]):
            oa_fi = r.get("owner_assumed", {}).get("feature_importance", {}).get(feat, 0)
            row(f"  {i+1}. {feat}", [f"{imp:.4f}%", f"{oa_fi:.4f}%"])

    # Winner
    print(f"\n  {'─'*70}")
    g0 = r[keys[0]]["global_metrics"]["test"]
    g1 = r[keys[1]]["global_metrics"]["test"]
    if g0["r2"] > g1["r2"] and g0["mape"] < g1["mape"]:
        winner = labels[0]
    elif g1["r2"] > g0["r2"] and g1["mape"] < g0["mape"]:
        winner = labels[1]
    else:
        winner = f"Mixed results (R²: {'→'.join(keys[0] if g0['r2']>g1['r2'] else keys[1])} wins; MAPE: {'→'.join(keys[0] if g0['mape']<g1['mape'] else keys[1])} wins)"
    print(f"  RECOMMENDED DATASET: {winner}")
    print(f"{'='*72}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print(DIV)
    print("PriceRef — Training Comparison Pipeline")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIV)

    all_results = {}

    for key, cfg in DATASETS.items():
        if not cfg["csv"].exists():
            print(f"\n[SKIP] {key} — file not found: {cfg['csv']}")
            continue
        result = run_training(key, cfg)
        all_results[key] = result

        # Save individual JSON
        out = cfg["art_dir"] / "model_metadata.json"
        print(f"  Metadata → {out}")

    if len(all_results) == 2:
        print_comparison(all_results)

        # Save combined comparison report
        report_path = ROOT / "model_artifacts_cell7" / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False,
                      default=lambda o: str(o))
        print(f"\n  Comparison JSON → {report_path}")

    print(f"\n{DIV}")
    print("All training runs complete.")
    print(DIV)


if __name__ == "__main__":
    main()
