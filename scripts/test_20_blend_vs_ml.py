#!/usr/bin/env python3
r"""
test_20_blend_vs_ml.py -- ML-Only vs ML+Comparable Blend
=========================================================
Takes 20 mainstream (well-represented) vehicles from the test split
and shows TWO genuinely different predictions:

  Mode A  [ML-Only]         -- Raw 5-Seed LightGBM champion average, no adjustments.
  Mode B  [ML + Comp Blend] -- ML prediction blended with real comparable vehicle
                               prices from the dataset (the production formula in
                               predict_market_value / main.py).

Blend formula (mirrors main.py predict_market_value):
  alpha = interpolated weight based on comparable similarity (0.5-0.7)
  final = alpha * comp_anchor + (1 - alpha) * ml_pred
  If no comparables found -> final == ml_pred (alpha=0)

Usage:
    cd Price-Prediction
    python scripts/test_20_blend_vs_ml.py
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle, random
from pathlib import Path
import numpy as np
import pandas as pd

DATA_CSV    = Path(ROOT) / "data" / "data.csv"
BUNDLE_PATH = Path(ROOT) / "model_registry" / "final" / "ensemble_bundle.pkl"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = Path(ROOT) / "model_artifacts" / "ensemble_bundle.pkl"

TOTAL_SAMPLES = 20
RANDOM_SEED   = 99

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset ...")
df_full = pd.read_csv(DATA_CSV)
if "split" in df_full.columns:
    df_test = df_full[df_full["split"] == "test"].copy()
else:
    df_test = df_full.sample(frac=0.20, random_state=RANDOM_SEED).copy()

df_test = df_test.dropna(subset=["selling_price"])
df_test = df_test[df_test["selling_price"] > 0].copy()
df_test["brand_clean"] = df_test["brand"].fillna("unknown").str.strip().str.lower()

# Pick brands that have plenty of test data (>15 rows) so comparables exist
WELL_REP = [b for b, n in df_test["brand_clean"].value_counts().items() if n >= 15]
df_pool = df_test[df_test["brand_clean"].isin(WELL_REP)].copy()

# Price-stratified sample: evenly across price range so we get budget, mid, premium
np.random.seed(RANDOM_SEED)
df_pool_sorted = df_pool.sort_values("selling_price").reset_index(drop=True)
indices = np.linspace(0, len(df_pool_sorted)-1, TOTAL_SAMPLES, dtype=int)
df_sampled = df_pool_sorted.iloc[indices].reset_index(drop=True)

print(f"  Sampled {len(df_sampled)} vehicles | price range: "
      f"Rs.{df_sampled['selling_price'].min():,.0f} — Rs.{df_sampled['selling_price'].max():,.0f}")

# ── 2. Load model ─────────────────────────────────────────────────────────────
print(f"\nLoading model bundle ...")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
from backend.champion_predictor import ChampionPredictor
predictor = ChampionPredictor(bundle)
print("  Model loaded OK")

# ── 3. Load decision engine (comparable search + blend config) ────────────────
from backend.decision_engine import get_comparable_anchor, get_blend_config
blend_cfg = get_blend_config()
SIM_LO   = blend_cfg["sim_lo"]   # 0.60
SIM_HI   = blend_cfg["sim_hi"]   # 0.75
ALPHA_LO = blend_cfg["alpha_lo"] # 0.50
ALPHA_HI = blend_cfg["alpha_hi"] # 0.70
print(f"  Blend config: sim_lo={SIM_LO}, sim_hi={SIM_HI}, alpha_lo={ALPHA_LO}, alpha_hi={ALPHA_HI}")

CURRENT_YEAR = 2026

# ── 4. Helpers ────────────────────────────────────────────────────────────────
def _clean(v):
    if v is None: return "unknown"
    s = str(v).strip().lower()
    return s if s else "unknown"

def _num(row, key):
    v = row.get(key)
    return float(v) if pd.notna(v) else None

def predict_ml_only(row):
    """Raw 5-Seed LightGBM champion average — no comp blending."""
    record = {
        "brand":            _clean(row.get("brand")),
        "model":            _clean(row.get("model")),
        "variant":          _clean(row.get("variant")),
        "locality":         _clean(row.get("locality")),
        "rto":              _clean(row.get("rto")),
        "fuel_type":        _clean(row.get("fuel_type")),
        "transmission":     _clean(row.get("transmission")),
        "seller_type":      _clean(row.get("seller_type")),
        "color":            _clean(row.get("color")),
        "vehicle_age":      _num(row, "vehicle_age"),
        "odometer_reading": _num(row, "odometer_reading"),
        "km_per_year":      _num(row, "km_per_year"),
        "owner_count":      _num(row, "owner_count"),
        "certified":        _num(row, "certified"),
        "pincode":          _num(row, "pincode"),
    }
    result = predictor.predict_price(record)
    return float(result["lgbm_prediction"])   # pure LGBM average


def predict_blend(row, ml_pred):
    """ML + comparable vehicle blend (same formula as main.py predict_market_value)."""
    age = int(row["vehicle_age"]) if pd.notna(row.get("vehicle_age")) else 5
    year = CURRENT_YEAR - age
    odo  = float(row["odometer_reading"]) if pd.notna(row.get("odometer_reading")) else 0.0
    owners = int(row["owner_count"]) if pd.notna(row.get("owner_count")) else 1

    comp_data = get_comparable_anchor(
        brand        = _clean(row.get("brand")),
        model        = _clean(row.get("model")),
        variant      = _clean(row.get("variant")),
        fuel         = _clean(row.get("fuel_type")),
        transmission = _clean(row.get("transmission")),
        year         = year,
        odometer     = odo,
        owner_count  = owners,
        seller_type  = _clean(row.get("seller_type")),
        locality     = _clean(row.get("locality")),
    )

    comp_anchor  = comp_data.get("comp_anchor")
    avg_sim      = comp_data.get("avg_similarity", 0.0)
    n_comps      = comp_data.get("n_comps", 0)
    conf_case    = comp_data.get("confidence_case", "low")

    # Exact blend formula from main.py
    if comp_anchor and avg_sim >= SIM_HI:
        alpha = ALPHA_HI
    elif comp_anchor and avg_sim >= SIM_LO:
        t     = (avg_sim - SIM_LO) / (SIM_HI - SIM_LO)
        alpha = ALPHA_LO + t * (ALPHA_HI - ALPHA_LO)
    else:
        alpha = 0.0

    if alpha > 0:
        blended = alpha * comp_anchor + (1.0 - alpha) * ml_pred
    else:
        blended = ml_pred

    return blended, comp_anchor, avg_sim, n_comps, conf_case, alpha


# ── 5. Run predictions ────────────────────────────────────────────────────────
print(f"\nRunning predictions on {len(df_sampled)} vehicles ...\n")

records = []
for i, (_, row) in enumerate(df_sampled.iterrows(), start=1):
    actual = float(row["selling_price"])
    try:
        ml_pred = predict_ml_only(row)
        blend_pred, comp_anchor, avg_sim, n_comps, conf_case, alpha = predict_blend(row, ml_pred)
    except Exception as e:
        print(f"  WARNING #{i}: {e}")
        continue

    ml_err_inr  = ml_pred   - actual
    bl_err_inr  = blend_pred - actual
    ml_err_pct  = (ml_err_inr  / actual) * 100
    bl_err_pct  = (bl_err_inr  / actual) * 100
    diff_inr    = blend_pred - ml_pred
    diff_pct    = (diff_inr / ml_pred) * 100

    records.append({
        "#":           i,
        "Brand":       _clean(row.get("brand")).title(),
        "Model":       _clean(row.get("model")).title(),
        "Variant":     _clean(row.get("variant"))[:18],
        "Age":         int(row["vehicle_age"]) if pd.notna(row.get("vehicle_age")) else "?",
        "Odo":         int(row["odometer_reading"]) if pd.notna(row.get("odometer_reading")) else "?",
        "Actual":      actual,
        "ML_Pred":     ml_pred,
        "Blend_Pred":  blend_pred,
        "Comp_Anchor": comp_anchor,
        "N_Comps":     n_comps,
        "Avg_Sim":     avg_sim,
        "Alpha":       alpha,
        "Conf":        conf_case,
        "ML_Err_INR":  ml_err_inr,
        "ML_Err_Pct":  ml_err_pct,
        "BL_Err_INR":  bl_err_inr,
        "BL_Err_Pct":  bl_err_pct,
        "Diff_INR":    diff_inr,
        "Diff_Pct":    diff_pct,
    })


# ── 6. Print results ──────────────────────────────────────────────────────────
G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
B  = "\033[94m"; C = "\033[96m"; BOLD = "\033[1m"; RST = "\033[0m"

def clr(v):
    s = (f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%")
    if abs(v) < 5:   return f"{G}{s:>9}{RST}"
    elif abs(v) < 10: return f"{Y}{s:>9}{RST}"
    else:             return f"{R}{s:>9}{RST}"

def inr(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}Rs.{abs(v):>9,.0f}"

DASH = "=" * 200
THIN = "-" * 200

print(DASH)
print(f"{BOLD}{'PriceRef — ML-Only  vs  ML + Comparable Vehicle Blend  (20 Vehicles)':^200}{RST}")
print(DASH)
print(
    f"{'#':>3}  {'Brand':<14}{'Model':<14}{'Variant':<18}  "
    f"{'Age':>4}  {'Odo(km)':>8}  "
    f"{'Actual':>14}  "
    f"{'ML-Only':>14}  {'ML Err':>12}  {'ML%':>9}  "
    f"{'Blend Pred':>14}  {'Blend Err':>12}  {'Blend%':>9}  "
    f"{'Diff(B-ML)':>12}  {'Diff%':>7}  "
    f"{'N':>3}  {'Sim':>5}  {'Alpha':>5}  {'Conf':>6}"
)
print(THIN)

for r in records:
    odo = f"{r['Odo']:,}" if isinstance(r["Odo"], int) else "?"
    blend_diff = r["Blend_Pred"] - r["ML_Pred"]
    diff_c = f"{G}" if abs(blend_diff) < 10000 else (f"{Y}" if abs(blend_diff) < 50000 else f"{R}")
    
    print(
        f"{r['#']:>3}  {r['Brand']:<14}{r['Model']:<14}{r['Variant']:<18}  "
        f"{str(r['Age']):>4}  {odo:>8}  "
        f"Rs.{r['Actual']:>11,.0f}  "
        f"Rs.{r['ML_Pred']:>11,.0f}  {inr(r['ML_Err_INR']):>12}  {clr(r['ML_Err_Pct'])}  "
        f"Rs.{r['Blend_Pred']:>11,.0f}  {inr(r['BL_Err_INR']):>12}  {clr(r['BL_Err_Pct'])}  "
        f"{diff_c}{inr(blend_diff):>12}{RST}  {diff_c}{r['Diff_Pct']:>+6.2f}%{RST}  "
        f"{r['N_Comps']:>3}  {r['Avg_Sim']:>5.2f}  {r['Alpha']:>5.2f}  {r['Conf']:>6}"
    )

print(DASH)

# ── 7. Summary ────────────────────────────────────────────────────────────────
ml_abs  = np.abs([r["ML_Err_Pct"]  for r in records])
bl_abs  = np.abs([r["BL_Err_Pct"]  for r in records])
ml_inr  = np.abs([r["ML_Err_INR"]  for r in records])
bl_inr  = np.abs([r["BL_Err_INR"]  for r in records])
ml_bias = np.mean([r["ML_Err_Pct"] for r in records])
bl_bias = np.mean([r["BL_Err_Pct"] for r in records])
n       = len(records)
blended_count = sum(1 for r in records if r["Alpha"] > 0)

print(f"\n{BOLD}{'SUMMARY':^70}{RST}")
print("=" * 70)
print(f"{'Metric':<38}  {'ML-Only':>14}  {'Blend':>14}")
print("-" * 70)
print(f"{'MAE (Rs.)':<38}  Rs.{np.mean(ml_inr):>10,.0f}  Rs.{np.mean(bl_inr):>10,.0f}")
print(f"{'Median AE (Rs.)':<38}  Rs.{np.median(ml_inr):>10,.0f}  Rs.{np.median(bl_inr):>10,.0f}")
print(f"{'MAPE':<38}  {np.mean(ml_abs):>13.2f}%  {np.mean(bl_abs):>13.2f}%")
print(f"{'Median APE':<38}  {np.median(ml_abs):>13.2f}%  {np.median(bl_abs):>13.2f}%")
print(f"{'Within 5%':<38}  {(ml_abs<5).sum():>13}/{n}  {(bl_abs<5).sum():>13}/{n}")
print(f"{'Within 10%':<38}  {(ml_abs<10).sum():>13}/{n}  {(bl_abs<10).sum():>13}/{n}")
print(f"{'Bias (signed mean %)':<38}  {ml_bias:>13.2f}%  {bl_bias:>13.2f}%")
print("=" * 70)
print(f"\nVehicles where blend differed from ML-only: {blended_count}/{n}")
print(f"  (blend kicks in only when >= 4 good comparables found with avg_sim >= 0.60)")
print(f"\nColumn guide:")
print(f"  N     = number of comparable vehicles found in dataset")
print(f"  Sim   = average similarity score of comparables (0-1)")
print(f"  alpha = blend weight given to comparables (0=pure ML, 0.7=70% comp price)")
print(f"  Conf  = high/medium/low comparable confidence")

out_csv = Path(ROOT) / "scripts" / "test_20_blend_results.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"\nResults saved: {out_csv}\n")
