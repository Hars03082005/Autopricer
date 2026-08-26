#!/usr/bin/env python3
"""
test_50_vehicles.py — Dual-Mode Prediction Test
================================================
Samples ~50 vehicles from the HELD-OUT test split in data/data.csv,
covering EVERY brand proportionally (brands with more cars → more test samples),
with diversity in odometer, price, age, and variant.

Two prediction modes are compared against the actual selling price:
  1. Normal Prediction   — ChampionPredictor (5-Seed LightGBM + Luxury CatBoost Specialist,
                            Strategy D routing) <- production ML prediction.
  2. ML-Only Prediction  — Raw 5-Seed LightGBM champion average only (no routing / no specialist).

Error in INR and as a percentage of actual price is reported for both modes.

Usage:
    cd c:\\Users\\Harshavardhana\\Downloads\\Price-Prediction
    python scripts/test_50_vehicles.py
"""
from __future__ import annotations

import sys
import os

# Root-relative import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV    = Path(ROOT) / "data" / "data.csv"
BUNDLE_PATH = Path(ROOT) / "model_registry" / "final" / "ensemble_bundle.pkl"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = Path(ROOT) / "model_artifacts" / "ensemble_bundle.pkl"

TOTAL_SAMPLES = 50
RANDOM_SEED   = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load & filter the test split
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset ...")
df_full = pd.read_csv(DATA_CSV)
print(f"  Total rows: {len(df_full):,}")

if "split" in df_full.columns:
    df_test = df_full[df_full["split"] == "test"].copy()
    print(f"  Test split rows: {len(df_test):,}")
else:
    df_test = df_full.sample(frac=0.20, random_state=RANDOM_SEED).copy()
    print(f"  No 'split' column — using random 20% ({len(df_test):,} rows) as test proxy")

df_test = df_test.dropna(subset=["selling_price"])
df_test = df_test[df_test["selling_price"] > 0]
print(f"  Usable test rows (with price): {len(df_test):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Brand-proportional, diversity-aware sampling
# ══════════════════════════════════════════════════════════════════════════════
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df_test["brand_clean"] = df_test["brand"].fillna("unknown").str.strip().str.lower()

brand_counts = df_test["brand_clean"].value_counts()
all_brands   = brand_counts.index.tolist()
print(f"\n  Brands in test split ({len(all_brands)} total):")
for b, n in brand_counts.items():
    print(f"    {b:<22} {n:>5} rows")

# Allocate proportionally — every brand gets at least 1
raw_alloc = (brand_counts / brand_counts.sum() * TOTAL_SAMPLES).round().astype(int)
raw_alloc = raw_alloc.clip(lower=1)
diff = TOTAL_SAMPLES - raw_alloc.sum()
if diff > 0:
    for b in brand_counts.index:
        if diff == 0:
            break
        raw_alloc[b] += 1
        diff -= 1
elif diff < 0:
    for b in brand_counts.index[::-1]:
        if diff == 0:
            break
        if raw_alloc[b] > 1:
            raw_alloc[b] -= 1
            diff += 1

print(f"\n  Sample allocation per brand:")
for b, n in raw_alloc.items():
    print(f"    {b:<22} -> {n} sample(s)")

# Stratify by price within each brand for diversity
selected_rows = []
for brand, n_pick in raw_alloc.items():
    sub = df_test[df_test["brand_clean"] == brand].copy()
    if len(sub) == 0:
        continue
    n_pick = min(n_pick, len(sub))
    if len(sub) <= n_pick:
        selected_rows.append(sub)
    else:
        sub_sorted = sub.sort_values("selling_price").reset_index(drop=True)
        indices = np.linspace(0, len(sub_sorted) - 1, n_pick, dtype=int)
        selected_rows.append(sub_sorted.iloc[indices])

df_sampled = pd.concat(selected_rows, ignore_index=True)
df_sampled = df_sampled.sample(min(TOTAL_SAMPLES, len(df_sampled)), random_state=RANDOM_SEED)
df_sampled = df_sampled.reset_index(drop=True)
print(f"\n  Final sample: {len(df_sampled)} vehicles across {df_sampled['brand_clean'].nunique()} brands")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Load ChampionPredictor bundle
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading model bundle from:\n  {BUNDLE_PATH}")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

from backend.champion_predictor import ChampionPredictor
predictor = ChampionPredictor(bundle)
print("  Model loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _clean(v: object) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    return s if s else "unknown"


def _build_input(row: pd.Series) -> dict:
    """Convert a CSV row to the raw input dict the predictor expects."""
    def _num(key):
        v = row.get(key)
        return float(v) if pd.notna(v) else None

    return {
        "brand":            _clean(row.get("brand")),
        "model":            _clean(row.get("model")),
        "variant":          _clean(row.get("variant")),
        "locality":         _clean(row.get("locality")),
        "rto":              _clean(row.get("rto")),
        "fuel_type":        _clean(row.get("fuel_type")),
        "transmission":     _clean(row.get("transmission")),
        "seller_type":      _clean(row.get("seller_type")),
        "color":            _clean(row.get("color")),
        "vehicle_age":      _num("vehicle_age"),
        "odometer_reading": _num("odometer_reading"),
        "km_per_year":      _num("km_per_year"),
        "owner_count":      _num("owner_count"),
        "certified":        _num("certified"),
        "pincode":          _num("pincode"),
    }


def predict_both(record: dict) -> tuple[float, float, str]:
    """
    Returns:
        normal_pred   - Strategy D final prediction (LGBM or CatBoost specialist)
        ml_only_pred  - Raw 5-seed LGBM average only (no routing)
        routing       - 'champion' or 'specialist'
    """
    result       = predictor.predict_price(record)
    normal_pred  = float(result["predicted_price"])
    ml_only_pred = float(result["lgbm_prediction"])     # always the LGBM champion avg
    routing      = result.get("routing_decision", "champion")
    return normal_pred, ml_only_pred, routing


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Run predictions on all 50 vehicles
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning predictions on {len(df_sampled)} vehicles ...\n")

records = []
for i, (_, row) in enumerate(df_sampled.iterrows(), start=1):
    actual = float(row["selling_price"])
    inp    = _build_input(row)
    try:
        norm, ml_only, routing = predict_both(inp)
    except Exception as e:
        print(f"  WARNING row {i}: prediction failed -- {e}")
        continue

    norm_err_inr = norm - actual
    ml_err_inr   = ml_only - actual
    norm_err_pct = (norm_err_inr / actual) * 100
    ml_err_pct   = (ml_err_inr   / actual) * 100

    age_val = int(row["vehicle_age"]) if pd.notna(row.get("vehicle_age")) else -1
    odo_val = int(row["odometer_reading"]) if pd.notna(row.get("odometer_reading")) else -1

    records.append({
        "#":           i,
        "Brand":       _clean(row.get("brand")).title(),
        "Model":       _clean(row.get("model")).title(),
        "Variant":     (_clean(row.get("variant")))[:20],
        "Age_yr":      age_val,
        "Odo_km":      odo_val,
        "Actual":      actual,
        "Normal":      norm,
        "ML_Only":     ml_only,
        "N_Err_INR":   norm_err_inr,
        "N_Err_Pct":   norm_err_pct,
        "ML_Err_INR":  ml_err_inr,
        "ML_Err_Pct":  ml_err_pct,
        "Routing":     routing,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Print results table
# ══════════════════════════════════════════════════════════════════════════════
DASH = "=" * 190
THIN = "-" * 190


def fmt_inr(v: float) -> str:
    return f"Rs.{abs(v):>10,.0f}"


def fmt_pct(v: float, width: int = 8) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:{width}.2f}%"


# ANSI colours (works in most terminals including Windows Terminal)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"


def clr(v: float) -> str:
    s = fmt_pct(v)
    if abs(v) < 5:
        return f"{GREEN}{s}{RESET}"
    elif abs(v) < 10:
        return f"{YELLOW}{s}{RESET}"
    else:
        return f"{RED}{s}{RESET}"


print(DASH)
print(f"{BOLD}{'PriceRef — Dual-Mode Prediction Test (50 Vehicles)':^190}{RESET}")
print(f"{CYAN}{'Normal (Strategy D) vs  ML-Only (5-Seed LGBM Avg)':^190}{RESET}")
print(DASH)

HDR = (
    f"{'#':>3}  {'Brand':<14}{'Model':<14}{'Variant':<20}  "
    f"{'Age':>4}  {'Odo(km)':>8}  "
    f"{'Actual Price':>14}  "
    f"{'Normal Pred':>14}  {'Norm Err INR':>14}  {'Norm Err%':>10}  "
    f"{'ML-Only Pred':>14}  {'ML Err INR':>14}  {'ML Err%':>10}  "
    f"{'Routing':}"
)
print(HDR)
print(THIN)

for r in records:
    age = str(r["Age_yr"]) if r["Age_yr"] >= 0 else "?"
    odo = f"{r['Odo_km']:,}" if r["Odo_km"] >= 0 else "?"
    n_sign = "+" if r["N_Err_INR"] >= 0 else "-"
    m_sign = "+" if r["ML_Err_INR"] >= 0 else "-"

    print(
        f"{r['#']:>3}  {r['Brand']:<14}{r['Model']:<14}{r['Variant']:<20}  "
        f"{age:>4}  {odo:>8}  "
        f"Rs.{r['Actual']:>11,.0f}  "
        f"Rs.{r['Normal']:>11,.0f}  "
        f"{n_sign}Rs.{abs(r['N_Err_INR']):>10,.0f}  "
        f"{clr(r['N_Err_Pct']):}  "
        f"Rs.{r['ML_Only']:>11,.0f}  "
        f"{m_sign}Rs.{abs(r['ML_Err_INR']):>10,.0f}  "
        f"{clr(r['ML_Err_Pct']):}  "
        f"{r['Routing']}"
    )

print(DASH)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Summary statistics
# ══════════════════════════════════════════════════════════════════════════════
n_pct  = np.array([r["N_Err_Pct"]  for r in records])
m_pct  = np.array([r["ML_Err_Pct"] for r in records])
n_abs  = np.abs(n_pct)
m_abs  = np.abs(m_pct)
n_inr  = np.array([abs(r["N_Err_INR"])  for r in records])
m_inr  = np.array([abs(r["ML_Err_INR"]) for r in records])
n      = len(records)

print(f"\n{BOLD}{'SUMMARY STATISTICS':^80}{RESET}")
print("=" * 80)
print(f"{'Metric':<40}  {'Normal (Strategy D)':>18}  {'ML-Only (LGBM Avg)':>18}")
print("-" * 80)
print(f"{'MAE (Mean Absolute Error in Rs.)':<40}  Rs.{n_inr.mean():>14,.0f}  Rs.{m_inr.mean():>14,.0f}")
print(f"{'Median AE (Rs.)':<40}  Rs.{np.median(n_inr):>14,.0f}  Rs.{np.median(m_inr):>14,.0f}")
print(f"{'MAPE (Mean Abs % Error)':<40}  {n_abs.mean():>17.2f}%  {m_abs.mean():>17.2f}%")
print(f"{'Median APE':<40}  {np.median(n_abs):>17.2f}%  {np.median(m_abs):>17.2f}%")
print(f"{'Max APE':<40}  {n_abs.max():>17.2f}%  {m_abs.max():>17.2f}%")
print(f"{'Within 5%  (green zone)':<40}  {(n_abs < 5).sum():>17}/{n}  {(m_abs < 5).sum():>17}/{n}")
print(f"{'Within 10% (yellow zone)':<40}  {(n_abs < 10).sum():>17}/{n}  {(m_abs < 10).sum():>17}/{n}")
print(f"{'Within 15%':<40}  {(n_abs < 15).sum():>17}/{n}  {(m_abs < 15).sum():>17}/{n}")
print(f"{'Bias (mean signed % error)':<40}  {n_pct.mean():>17.2f}%  {m_pct.mean():>17.2f}%")
print("=" * 80)

routing_spec  = sum(1 for r in records if r["Routing"] == "specialist")
routing_champ = sum(1 for r in records if r["Routing"] == "champion")
print(f"\nRouting: {routing_champ} champion  |  {routing_spec} specialist (luxury-routed)")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Per-brand breakdown
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'PER-BRAND BREAKDOWN':^90}{RESET}")
print("=" * 90)
print(f"{'Brand':<20}  {'N':>3}  {'Normal MAPE':>12}  {'Normal MAE':>14}  {'ML MAPE':>10}  {'ML MAE':>14}")
print("-" * 90)

brand_grp: dict[str, list] = {}
for r in records:
    brand_grp.setdefault(r["Brand"], []).append(r)

for brand in sorted(brand_grp.keys()):
    rows  = brand_grp[brand]
    n_mape = np.mean([abs(r["N_Err_Pct"])  for r in rows])
    m_mape = np.mean([abs(r["ML_Err_Pct"]) for r in rows])
    n_mae  = np.mean([abs(r["N_Err_INR"])  for r in rows])
    m_mae  = np.mean([abs(r["ML_Err_INR"]) for r in rows])
    print(
        f"{brand:<20}  {len(rows):>3}  "
        f"{n_mape:>11.2f}%  Rs.{n_mae:>11,.0f}  "
        f"{m_mape:>9.2f}%  Rs.{m_mae:>11,.0f}"
    )

print("=" * 90)
print(f"\nLegend: {GREEN}Green{RESET} = |err| < 5%   {YELLOW}Yellow{RESET} = |err| < 10%   {RED}Red{RESET} = |err| >= 10%")
print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = Path(ROOT) / "scripts" / "test_50_results.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"Results saved: {out_csv}\n")
