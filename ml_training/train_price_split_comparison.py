"""
train_price_split_comparison.py
================================
Compares three segmentation strategies on processed_cell7_dataset.csv:

  Strategy A -- GLOBAL (no split)
      Single CatBoost+LightGBM+XGBoost ensemble over ALL rows.

  Strategy B -- THREE-BAND (0-6L / 6-12L / 12L+)
      band_low    :  selling_price < 600,000
      band_mid    :  600,000 <= selling_price < 1,200,000
      band_high   :  selling_price >= 1,200,000

  Strategy C -- TWO-BAND (0-10L / 10L+)
      band_mass   :  selling_price < 1,000,000
      band_premium:  selling_price >= 1,000,000

Each strategy trains CatBoost+LightGBM+XGBoost ensembles
(global or per-band) and prints a consolidated comparison table.

Usage:
    python ml_training/train_price_split_comparison.py

Outputs saved under:
    model_artifacts_split_compare/
        global/           -- Strategy A artifacts
        three_band/       -- Strategy B artifacts (per band)
        two_band/         -- Strategy C artifacts (per band)
        comparison_report.json
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

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
DATA_CSV = Path(__file__).resolve().parent / "data" / "processed_cell7_dataset.csv"
OUT_DIR  = ROOT / "model_artifacts_split_compare"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
DIV = "=" * 78

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

# ── Price band definitions ─────────────────────────────────────────────────────
THREE_BANDS = {
    "0-6L":  (0,           600_000),
    "6-12L": (600_000,   1_200_000),
    "12L+":  (1_200_000, float("inf")),
}

TWO_BANDS = {
    "0-10L": (0,           1_000_000),
    "10L+":  (1_000_000,  float("inf")),
}

MIN_BAND_ROWS = 300


# ── Helpers ────────────────────────────────────────────────────────────────────
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics on actual INR scale (expm1 of log-price predictions)."""
    yt = np.expm1(y_true)
    yp = np.expm1(y_pred)
    mae  = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100)
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
        return -r2_score(y_val, sum(w[i] * val_preds[names[i]] for i in range(n)))
    res = minimize(neg_r2, x0=[1.0/n]*n, method="SLSQP",
                   bounds=[(0, 1)]*n,
                   constraints={"type": "eq", "fun": lambda w: sum(w) - 1})
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
    cb    = frame.copy()
    lgb_f = frame.copy()
    for col in CAT_FEATURES:
        if col in lgb_f.columns:
            le = LabelEncoder()
            lgb_f[col] = le.fit_transform(lgb_f[col].astype(str))
    xgb_f = lgb_f.copy()
    return cb, lgb_f, xgb_f


# ── Data loader ────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"\n  Loading data from {DATA_CSV.name} ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  Raw rows: {len(df):,}  columns: {df.shape[1]}")

    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df.dropna(subset=["selling_price"], inplace=True)
    df = df[df["selling_price"].between(50_000, 20_000_000)].copy()

    df["vehicle_age"]      = pd.to_numeric(df["vehicle_age"],      errors="coerce").clip(0, 35).fillna(5)
    df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce").clip(0, 600_000).fillna(50_000)
    df["km_per_year"]      = pd.to_numeric(df["km_per_year"],      errors="coerce").clip(0, 100_000).fillna(10_000)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
        else:
            df[col] = "unknown"
    if "segment_class" not in df.columns:
        df["segment_class"] = "economy"

    df.dropna(subset=NUMERIC_FEATURES[:4], inplace=True)
    print(f"  Clean rows: {len(df):,}")
    return df


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


def _predict(name: str, model, frame) -> np.ndarray:
    if name == "catboost": return np.asarray(model.predict(frame))
    if name == "lightgbm": return np.asarray(model.predict(frame))
    if name == "xgboost":  return np.asarray(model.predict(xgb.DMatrix(frame)))
    raise ValueError(name)


# ── Core ensemble trainer (one band) ──────────────────────────────────────────
def train_ensemble(sub_df: pd.DataFrame, band_name: str, art_dir: Path) -> Optional[dict]:
    n_rows = len(sub_df)
    if n_rows < MIN_BAND_ROWS:
        print(f"    [SKIP] {band_name}: only {n_rows} rows (< {MIN_BAND_ROWS})")
        return None

    y = np.log1p(sub_df["selling_price"].values)
    X = sub_df[FEATURES]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_vl, X_te, y_vl, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    cat_levels = build_category_levels(sub_df.loc[X_tr.index])
    cb_tr, lgb_tr, xgb_tr = prepare_frames(sub_df.loc[X_tr.index], cat_levels)
    cb_vl, lgb_vl, xgb_vl = prepare_frames(sub_df.loc[X_vl.index], cat_levels)
    cb_te, lgb_te, xgb_te = prepare_frames(sub_df.loc[X_te.index], cat_levels)

    n_tr, n_vl, n_te = len(X_tr), len(X_vl), len(X_te)
    print(f"    Train:{n_tr:,}  Val:{n_vl:,}  Test:{n_te:,}", end="  ")

    print("CatBoost...", end=" ", flush=True)
    cb_m  = train_catboost(cb_tr, y_tr, cb_vl, y_vl)
    print("LightGBM...", end=" ", flush=True)
    lgb_m = train_lightgbm(lgb_tr, y_tr, lgb_vl, y_vl)
    print("XGBoost...", end=" ", flush=True)
    xgb_m = train_xgboost(xgb_tr, y_tr, xgb_vl, y_vl)
    print("done")

    models  = {"catboost": cb_m, "lightgbm": lgb_m, "xgboost": xgb_m}
    val_fr  = {"catboost": cb_vl,  "lightgbm": lgb_vl,  "xgboost": xgb_vl}
    test_fr = {"catboost": cb_te,  "lightgbm": lgb_te,  "xgboost": xgb_te}
    tr_fr   = {"catboost": cb_tr,  "lightgbm": lgb_tr,  "xgboost": xgb_tr}

    val_preds   = {n: _predict(n, m, val_fr[n])  for n, m in models.items()}
    weights     = optimise_weights(y_vl, val_preds)

    train_blend = blend(weights, {n: _predict(n, m, tr_fr[n])   for n, m in models.items()})
    val_blend   = blend(weights, {n: _predict(n, m, val_fr[n])  for n, m in models.items()})
    test_blend  = blend(weights, {n: _predict(n, m, test_fr[n]) for n, m in models.items()})

    tr_m   = calc_metrics(y_tr, train_blend)
    vl_m   = calc_metrics(y_vl, val_blend)
    te_m   = calc_metrics(y_te, test_blend)
    gap    = round(tr_m["r2"] - te_m["r2"], 4)
    status = ("healthy" if gap < 0.02 else "mild_overfit" if gap < 0.05 else "overfit")

    print(f"    -> Train R2:{tr_m['r2']:.4f}  Val R2:{vl_m['r2']:.4f}  "
          f"Test R2:{te_m['r2']:.4f}  MAPE:{te_m['mape']:.2f}%  "
          f"MAE:Rs{te_m['mae']:,.0f}  gap:{gap} ({status})")

    feat_imp: dict = {}
    try:
        imp      = cb_m.get_feature_importance()
        feat_imp = {FEATURES[i]: round(float(imp[i]), 4) for i in range(len(FEATURES))}
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass

    art_dir.mkdir(parents=True, exist_ok=True)
    safe = band_name.replace("+", "plus").replace("-", "_")
    artifact = {
        "catboost": cb_m, "lightgbm": lgb_m, "xgboost": xgb_m,
        "weights": weights, "features": FEATURES,
        "cat_features": CAT_FEATURES, "category_levels": cat_levels,
        "band": band_name, "rows": n_rows,
        "train_metrics": tr_m, "val_metrics": vl_m, "test_metrics": te_m,
        "overfit_gap": gap, "overfit_status": status,
        "feature_importance": feat_imp,
    }
    joblib.dump(artifact, art_dir / f"ensemble_{safe}.pkl")

    return {
        "band": band_name, "rows": n_rows,
        "train": tr_m, "val": vl_m, "test": te_m,
        "weights": weights,
        "overfit_gap": gap, "overfit_status": status,
        "feature_importance": feat_imp,
    }


# ── Strategy runners ───────────────────────────────────────────────────────────
def run_global(df: pd.DataFrame) -> dict:
    print(f"\n{DIV}")
    print("STRATEGY A  --  GLOBAL MODEL  (no price split)")
    print(DIV)
    t0     = time.time()
    result = train_ensemble(df, "global", OUT_DIR / "global")
    elapsed = round(time.time() - t0, 1)
    if result:
        result["strategy"]  = "global"
        result["elapsed_s"] = elapsed
    return result or {}


def run_three_band(df: pd.DataFrame) -> dict:
    print(f"\n{DIV}")
    print("STRATEGY B  --  THREE-BAND SPLIT  (0-6L | 6-12L | 12L+)")
    print(DIV)
    t0      = time.time()
    results = {}
    for band, (lo, hi) in THREE_BANDS.items():
        hi_str = "inf" if hi == float("inf") else f"{hi/1e5:.0f}L"
        mask   = (df["selling_price"] >= lo) & (df["selling_price"] < hi)
        sub_df = df[mask].copy()
        print(f"\n  -- Band: {band}  ({lo/1e5:.0f}L - {hi_str})  rows: {len(sub_df):,}")
        r = train_ensemble(sub_df, band, OUT_DIR / "three_band")
        if r:
            results[band] = r
    return {"strategy": "three_band", "bands": results,
            "elapsed_s": round(time.time() - t0, 1)}


def run_two_band(df: pd.DataFrame) -> dict:
    print(f"\n{DIV}")
    print("STRATEGY C  --  TWO-BAND SPLIT  (0-10L | 10L+)")
    print(DIV)
    t0      = time.time()
    results = {}
    for band, (lo, hi) in TWO_BANDS.items():
        hi_str = "inf" if hi == float("inf") else f"{hi/1e5:.0f}L"
        mask   = (df["selling_price"] >= lo) & (df["selling_price"] < hi)
        sub_df = df[mask].copy()
        print(f"\n  -- Band: {band}  ({lo/1e5:.0f}L - {hi_str})  rows: {len(sub_df):,}")
        r = train_ensemble(sub_df, band, OUT_DIR / "two_band")
        if r:
            results[band] = r
    return {"strategy": "two_band", "bands": results,
            "elapsed_s": round(time.time() - t0, 1)}


# ── Weighted-average metrics for split strategies ──────────────────────────────
def weighted_avg_metrics(band_results: dict) -> dict:
    total = sum(v["rows"] for v in band_results.values())
    if total == 0:
        return {}
    avg = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0}
    for v in band_results.values():
        w = v["rows"] / total
        for m in avg:
            avg[m] += w * v["test"][m]
    return {k: round(val, 4) for k, val in avg.items()}


# ── Comparison report ──────────────────────────────────────────────────────────
def print_comparison(ga: dict, tb: dict, twb: dict) -> None:
    print(f"\n\n{'='*78}")
    print("  FINAL COMPARISON REPORT")
    print(f"{'='*78}")

    tb_avg  = weighted_avg_metrics(tb.get("bands",  {}))
    twb_avg = weighted_avg_metrics(twb.get("bands", {}))
    ga_te   = ga.get("test", {})

    def row3(label, ga_v, tb_v, twb_v):
        print(f"  {label:<38} {ga_v:>13}  {tb_v:>13}  {twb_v:>13}")

    header = f"  {'Metric':<38} {'GLOBAL (A)':>13}  {'3-BAND (B)':>13}  {'2-BAND (C)':>13}"
    print(header)
    print(f"  {'-'*38} {'-'*13}  {'-'*13}  {'-'*13}")

    rows_ga  = ga.get("rows", 0)
    rows_tb  = sum(v["rows"] for v in tb.get("bands",  {}).values())
    rows_twb = sum(v["rows"] for v in twb.get("bands", {}).values())
    row3("Total rows used",  f"{rows_ga:,}",  f"{rows_tb:,}",  f"{rows_twb:,}")
    row3("Training time (s)",
         f"{ga.get('elapsed_s','?')}s",
         f"{tb.get('elapsed_s','?')}s",
         f"{twb.get('elapsed_s','?')}s")

    print("\n  WEIGHTED-AVERAGE TEST METRICS  (bands weighted by row count)")
    for metric, label, fmt in [
        ("r2",   "R2 Score",  lambda v: f"{v:.4f}"),
        ("mae",  "MAE (Rs)",  lambda v: f"Rs{v:,.0f}"),
        ("rmse", "RMSE (Rs)", lambda v: f"Rs{v:,.0f}"),
        ("mape", "MAPE (%)",  lambda v: f"{v:.2f}%"),
    ]:
        row3(f"  {label}",
             fmt(ga_te.get(metric,   0)),
             fmt(tb_avg.get(metric,  0)),
             fmt(twb_avg.get(metric, 0)))

    gap_str = (f"{ga.get('overfit_gap','?')} "
               f"({str(ga.get('overfit_status',''))[:10]})")
    tb_gaps = "/".join(f"{v['overfit_gap']:.4f}" for v in tb.get("bands", {}).values()) if tb.get("bands") else "--"
    twb_gaps = "/".join(f"{v['overfit_gap']:.4f}" for v in twb.get("bands", {}).values()) if twb.get("bands") else "--"
    row3("  Overfit Gap", gap_str, tb_gaps, twb_gaps)

    # Per-band detail B
    print(f"\n  STRATEGY B  --  THREE-BAND DETAIL")
    print(f"  {'Band':<12} {'Rows':>8}  {'R2':>8}  {'MAE (Rs)':>14}  {'MAPE':>7}  {'Gap':>7}  Status")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*14}  {'-'*7}  {'-'*7}  ------")
    for band, v in tb.get("bands", {}).items():
        print(f"  {band:<12} {v['rows']:>8,}  {v['test']['r2']:>8.4f}  "
              f"Rs{v['test']['mae']:>13,.0f}  {v['test']['mape']:>6.2f}%  "
              f"{v['overfit_gap']:>7.4f}  {v['overfit_status']}")

    # Per-band detail C
    print(f"\n  STRATEGY C  --  TWO-BAND DETAIL")
    print(f"  {'Band':<12} {'Rows':>8}  {'R2':>8}  {'MAE (Rs)':>14}  {'MAPE':>7}  {'Gap':>7}  Status")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*14}  {'-'*7}  {'-'*7}  ------")
    for band, v in twb.get("bands", {}).items():
        print(f"  {band:<12} {v['rows']:>8,}  {v['test']['r2']:>8.4f}  "
              f"Rs{v['test']['mae']:>13,.0f}  {v['test']['mape']:>6.2f}%  "
              f"{v['overfit_gap']:>7.4f}  {v['overfit_status']}")

    # Top-5 features from global model
    print(f"\n  TOP 5 FEATURES  (Global CatBoost feature importance)")
    fi = ga.get("feature_importance", {})
    for i, (feat, imp) in enumerate(list(fi.items())[:5], 1):
        print(f"  {i}. {feat:<30}  {imp:.4f}%")

    # Ensemble weights
    print(f"\n  ENSEMBLE WEIGHTS  (global model)")
    for name, w in ga.get("weights", {}).items():
        print(f"    {name:<12} {w:.4f}")

    # Winner
    best_r2    = max(ga_te.get("r2", 0),   tb_avg.get("r2", 0),    twb_avg.get("r2", 0))
    best_mape  = min(ga_te.get("mape", 99), tb_avg.get("mape", 99), twb_avg.get("mape", 99))
    r2_labels  = {ga_te.get("r2", 0): "GLOBAL (A)", tb_avg.get("r2", 0): "3-BAND (B)", twb_avg.get("r2", 0): "2-BAND (C)"}
    mape_labels= {ga_te.get("mape",99): "GLOBAL (A)", tb_avg.get("mape",99): "3-BAND (B)", twb_avg.get("mape",99): "2-BAND (C)"}

    r2_winner   = r2_labels[best_r2]
    mape_winner = mape_labels[best_mape]

    print(f"\n  {'_'*74}")
    print(f"  Best R2   -> {r2_winner}   ({best_r2:.4f})")
    print(f"  Best MAPE -> {mape_winner}  ({best_mape:.2f}%)")
    if r2_winner == mape_winner:
        print(f"\n  *** RECOMMENDED STRATEGY: {r2_winner} ***")
    else:
        print(f"\n  *** Mixed: Best R2 by {r2_winner} | Best MAPE by {mape_winner} ***")
        print(f"      Recommendation: prefer strategy with better MAPE for end-user accuracy.")
    print(f"{'='*78}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print(DIV)
    print("PriceRef  --  Price-Split Strategy Comparison")
    print(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATA_CSV}")
    print(DIV)

    if not DATA_CSV.exists():
        print(f"\n[ERROR] Dataset not found: {DATA_CSV}")
        sys.exit(1)

    df = load_data()

    # Show band distribution
    print("\n  Price band distribution:")
    total = len(df)
    seen_bands: dict = {**THREE_BANDS, **TWO_BANDS}
    for band, (lo, hi) in seen_bands.items():
        cnt = ((df["selling_price"] >= lo) & (df["selling_price"] < hi)).sum()
        print(f"    {band:<8}: {cnt:>7,} rows  ({cnt/total*100:.1f}%)")

    t_total = time.time()

    ga  = run_global(df)
    tb  = run_three_band(df)
    twb = run_two_band(df)

    print_comparison(ga, tb, twb)

    # Save JSON report
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "dataset":       str(DATA_CSV.name),
        "strategies": {
            "global":     ga,
            "three_band": tb,
            "two_band":   twb,
        },
    }
    report_path = OUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report saved -> {report_path}")
    print(f"  Total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"\n{DIV}")
    print("All training runs complete.")
    print(DIV)


if __name__ == "__main__":
    main()