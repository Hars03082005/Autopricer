"""
validate_models.py
==================
Validates ALL loadable models in model_artifacts/ against a processed dataset.
Derives missing features from dataset columns to match what each model expects.
Reports MAE, RMSE, R2, MAPE for each model on a sample of up to 5000 rows.

Usage:
    python scripts/validate_models.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model_artifacts"
DATA_DIR     = ROOT / "ml_training" / "data"
DATASET_PATH = DATA_DIR / "processed_overall.csv"
if not DATASET_PATH.exists():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        DATASET_PATH = csv_files[0]

SAMPLE_SIZE = 5_000
RANDOM_SEED = 42

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(t):   return f"{GREEN}{t}{RESET}"
def warn(t): return f"{YELLOW}{t}{RESET}"
def bad(t):  return f"{RED}{t}{RESET}"
def hdr(t):  return f"{BOLD}{CYAN}{t}{RESET}"

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_pred) & np.isfinite(y_true) & (y_true > 0)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(mae=float("nan"), rmse=float("nan"), r2=float("nan"), mape=float("nan"), n=0)
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - np.mean(yt)) ** 2)
    r2   = float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    return dict(mae=mae, rmse=rmse, r2=r2, mape=mape, n=int(len(yt)))

def print_result(name: str, m: dict, elapsed: float):
    def fmt(val, good, bad_, higher=True, pct=False):
        if math.isnan(val): return warn("N/A")
        s = f"{val:.2f}%" if pct else f"{val:.4f}"
        if higher: return ok(s) if val >= good else (bad(s) if val <= bad_ else warn(s))
        else:      return ok(s) if val <= good else (bad(s) if val >= bad_ else warn(s))

    print(f"\n{'─'*60}")
    print(f"{BOLD}{name}{RESET}")
    print(f"  R2  : {fmt(m['r2'],   0.95, 0.85,  higher=True)}")
    print(f"  MAE : {fmt(m['mae'],  50_000, 150_000, higher=False)} (Rs {m['mae']/1000:.1f}K)" if not math.isnan(m['mae']) else f"  MAE : {warn('N/A')}")
    print(f"  RMSE: {fmt(m['rmse'], 80_000, 200_000, higher=False)} (Rs {m['rmse']/1000:.1f}K)" if not math.isnan(m['rmse']) else f"  RMSE: {warn('N/A')}")
    print(f"  MAPE: {fmt(m['mape'], 8.0,  20.0, higher=False, pct=True)}")
    print(f"  n   : {m['n']:,}  |  time: {elapsed:.2f}s")

print(f"\n{'='*60}")
print("  PricerPoint -- Model Validation")
print(f"{'='*60}")
print(f"\nDataset : {DATASET_PATH.name}")
df = pd.read_csv(DATASET_PATH)
print(f"Total   : {len(df):,} rows  |  {len(df.columns)} columns")
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0].reset_index(drop=True)

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sample  : {len(df):,} rows")
else:
    print(f"Using   : {len(df):,} rows")

y_true = df["selling_price"].values

LUXURY_BRANDS = {"bmw","mercedes-benz","audi","jaguar","land rover","porsche",
                 "maserati","aston martin","bentley","rolls-royce","ferrari","lamborghini","hummer"}

BRAND_SEGMENT_MAP = {
    "maruti":"economy","maruti suzuki":"economy","datsun":"economy","bajaj":"economy",
    "chevrolet":"economy","fiat":"economy","opel":"economy","premier":"economy",
    "force":"economy","ashok leyland":"economy","ambassador":"economy",
    "hindustan motors":"economy","hyundai":"economy","honda":"economy","tata":"economy",
    "renault":"economy","nissan":"economy","ford":"economy","mitsubishi":"economy",
    "isuzu":"economy","citroen":"economy","dc":"economy",
    "volkswagen":"premium","skoda":"premium","toyota":"premium","mg":"premium",
    "jeep":"premium","kia":"premium","mini":"premium","volvo":"premium",
    "lexus":"premium","mahindra":"premium",
    "bmw":"luxury","mercedes-benz":"luxury","audi":"luxury","jaguar":"luxury",
    "land rover":"luxury","porsche":"luxury","maserati":"luxury","aston martin":"luxury",
    "bentley":"luxury","rolls-royce":"luxury","ferrari":"luxury","lamborghini":"luxury",
    "hummer":"luxury",
}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "rto_state" not in d.columns:
        d["rto_state"] = d["rto"].astype(str) if "rto" in d.columns else "unknown"
    if "color" not in d.columns:
        d["color"] = "unknown"
    if "luxury_brand" not in d.columns:
        d["luxury_brand"] = d["brand"].str.lower().map(lambda b: 1 if b in LUXURY_BRANDS else 0).astype(float)
    if "high_mileage" not in d.columns:
        d["high_mileage"] = (d["odometer_reading"] > 93_143).astype(float)
    if "inspected" not in d.columns:
        d["inspected"] = 0.0
    if "has_list_price" not in d.columns:
        d["has_list_price"] = 0.0
    if "segment_class" not in d.columns:
        d["segment_class"] = d["brand"].str.lower().map(lambda b: BRAND_SEGMENT_MAP.get(b, "economy"))
    for col in ["brand","model","variant","city","rto_state","color","segment_class","fuel_type","transmission","seller_type"]:
        if col in d.columns:
            d[col] = d[col].fillna("unknown").astype(str)
    return d

df = enrich(df)

with open(ARTIFACT_DIR / "model_metadata.json") as f:
    META = json.load(f)
MODEL_FEATURES = META.get("features", [])
CAT_FEATURES   = META.get("cat_features") or META.get("categorical_features") or []

def align_frame(df: pd.DataFrame, feature_names: list, cat_features: list, category_levels: dict = {}) -> pd.DataFrame:
    """Select & type-cast exactly the columns the model needs."""
    frame = {}
    for col in feature_names:
        if col in df.columns:
            frame[col] = df[col].copy()
        else:
            frame[col] = "unknown" if col in cat_features else 0.0
    frame_df = pd.DataFrame(frame, index=df.index)
    for col in cat_features:
        if col not in frame_df.columns:
            frame_df[col] = "unknown"
        raw = frame_df[col].fillna("unknown").astype(str)
        levels = category_levels.get(col, [])
        if levels:
            raw = raw.where(raw.isin(levels), "unknown")
        frame_df[col] = raw
    return frame_df[feature_names]

results: list[dict] = []

print(f"\n{hdr('Loading and evaluating models...')}")

p = ARTIFACT_DIR / "vehicle_price_catboost.cbm"
if p.exists():
    print(f"  + {p.name}")
    try:
        cb = CatBoostRegressor(); cb.load_model(str(p))
        frame = align_frame(df, list(cb.feature_names_), CAT_FEATURES)
        t0 = time.perf_counter()
        y_pred = np.expm1(cb.predict(frame))
        elapsed = time.perf_counter() - t0
        m = compute_metrics(y_true, y_pred)
        print_result("Global CatBoost  (vehicle_price_catboost.cbm)", m, elapsed)
        results.append({"model": "Global CatBoost", **m})
    except Exception as e:
        print(f"    FAILED: {e}")
else:
    print("  - vehicle_price_catboost.cbm  NOT FOUND")

for seg in ["economy", "premium", "luxury"]:
    p = ARTIFACT_DIR / f"ensemble_{seg}.pkl"
    if not p.exists():
        print(f"  - ensemble_{seg}.pkl  NOT FOUND")
        continue
    print(f"  + ensemble_{seg}.pkl")
    try:
        art = joblib.load(p)
        cat_f  = art.get("cat_features", CAT_FEATURES)
        levels = art.get("category_levels", {})
        cb     = art["catboost"]
        frame  = align_frame(df, list(cb.feature_names_), cat_f, levels)
        t0 = time.perf_counter()
        y_pred = np.expm1(cb.predict(frame))
        elapsed = time.perf_counter() - t0
        m = compute_metrics(y_true, y_pred)
        print_result(f"Segment PKL {seg}  (ensemble_{seg}.pkl -> CatBoost)", m, elapsed)
        results.append({"model": f"Segment PKL {seg}", **m})
    except Exception as e:
        print(f"    FAILED: {e}")

standalone = [
    ("segment_economy.cbm",      "segment_economy_levels.pkl"),
    ("segment_budget.cbm",       "segment_budget_levels.pkl"),
    ("segment_mid.cbm",          "segment_mid_levels.pkl"),
    ("segment_premium.cbm",      "segment_premium_levels.pkl"),
    ("segment_6_12_lakh.cbm",    "segment_6_12_lakh_levels.pkl"),
    ("segment_12_plus_lakh.cbm", "segment_12_plus_lakh_levels.pkl"),
    ("best_model.cbm",           None),
]
for cbm_name, lvl_name in standalone:
    p = ARTIFACT_DIR / cbm_name
    if not p.exists():
        continue
    print(f"  + {cbm_name}")
    try:
        cb = CatBoostRegressor(); cb.load_model(str(p))
        feat_names = list(cb.feature_names_)

        levels: dict = {}
        if lvl_name and (ARTIFACT_DIR / lvl_name).exists():
            levels = joblib.load(ARTIFACT_DIR / lvl_name)

        cat_idx  = cb.get_cat_feature_indices()
        cat_cols = [feat_names[i] for i in cat_idx] if cat_idx else CAT_FEATURES

        frame = align_frame(df, feat_names, cat_cols, levels)
        t0 = time.perf_counter()
        raw = cb.predict(frame)
        elapsed = time.perf_counter() - t0

        y_pred = np.expm1(raw) if float(np.median(raw)) < 20 else raw
        m = compute_metrics(y_true, y_pred)
        print_result(f"Standalone CatBoost  ({cbm_name})", m, elapsed)
        results.append({"model": cbm_name, **m})
    except Exception as e:
        print(f"    FAILED: {e}")

print(f"\n{'='*70}")
print(hdr("  SUMMARY  (sorted by R2 desc)"))
print(f"{'='*70}")
hline = f"{'Model':<48} {'R2':>7} {'MAE':>11} {'MAPE':>8} {'n':>7}"
print(hline)
print("-" * 84)
for r in sorted(results, key=lambda x: x.get("r2", -999), reverse=True):
    r2   = r["r2"];  mae  = r["mae"];  mape = r["mape"];  n = r["n"]
    r2_s   = f"{r2:.4f}"           if not math.isnan(r2)   else "N/A"
    mae_s  = f"Rs{mae/1000:.1f}K"  if not math.isnan(mae)  else "N/A"
    mape_s = f"{mape:.2f}%"        if not math.isnan(mape) else "N/A"
    r2_c = (ok(r2_s) if r2 >= 0.95 else (bad(r2_s) if r2 < 0.85 else warn(r2_s))) if not math.isnan(r2) else warn(r2_s)
    print(f"{r['model']:<48} {r2_c:>7} {mae_s:>11} {mape_s:>8} {n:>7,}")

print(f"\n  {ok('GREEN')} = R2 >= 0.95 (excellent)  |  {warn('YELLOW')} = 0.85-0.95 (acceptable)  |  {bad('RED')} = < 0.85 (poor)")
print()
