from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
# Paths
HERE          = Path(__file__).resolve().parent
ROOT          = HERE.parent
DATA_DIR      = ROOT / "ml_training" / "data"
ARTIFACTS_DIR = ROOT / "model_artifacts"
OUT_PATH      = ARTIFACTS_DIR / "engine_config.json"
REPORT_PATH   = ARTIFACTS_DIR / "training_report.json"
DIV = "=" * 72
DATASET_CANDIDATES = [
    "processed_s1_s4_owner.csv",
    "processed_overall.csv",
    "processed_s1_s4_owner_1.csv",
]
# Minimum rows per category
MIN_MODEL_ROWS    = 20
MIN_LOCALITY_ROWS = 30
MIN_RTO_ROWS      = 20
# Brand → segment mapping
LUXURY_BRANDS = {
    "bmw", "mercedes-benz", "mercedes", "audi", "lexus", "volvo",
    "land rover", "jaguar", "porsche", "bentley", "rolls-royce",
    "ferrari", "lamborghini", "aston martin", "maserati",
}
PREMIUM_BRANDS = {
    "volkswagen", "skoda", "toyota", "mg", "jeep", "kia", "mini",
}
def get_segment(brand: str) -> str:
    b = (brand or "").strip().lower()
    if b in LUXURY_BRANDS:
        return "luxury"
    if b in PREMIUM_BRANDS:
        return "premium"
    return "economy"
# Load dataset and training report
def load_data() -> tuple[pd.DataFrame, str, dict]:
    chosen_path = None
    chosen_name = None
    for name in DATASET_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            chosen_path = p
            chosen_name = name
            break
    if chosen_path is None:
        raise FileNotFoundError(f"No processed CSV found in {DATA_DIR}")
    print(DIV)
    print("LOADING DATASET")
    print(DIV)
    df = pd.read_csv(chosen_path, low_memory=False)
    print(f"  Loaded : {chosen_name}  ({len(df):,} rows)")
    report = {}
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, encoding="utf-8") as f:
                report = json.load(f)
            print("  Loaded training_report.json")
        except Exception as e:
            print(f"  WARNING: could not read training_report.json: {e}")
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df.dropna(subset=["selling_price"])
    df = df[df["selling_price"].between(50_000, 20_000_000)]
    for col in ["brand", "model", "locality", "rto", "fuel_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["vehicle_age"], errors="coerce").fillna(0)
    elif "year" in df.columns:
        df["vehicle_age"] = (datetime.now().year - pd.to_numeric(df["year"], errors="coerce")).clip(lower=0)
    else:
        df["vehicle_age"] = 0
    if "odometer_reading" in df.columns:
        df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce").fillna(0)
    else:
        df["odometer_reading"] = 0
    safe_age = df["vehicle_age"].clip(lower=1)
    df["km_per_year"] = (df["odometer_reading"] / safe_age).clip(0, 80_000)
    print(f"\n  Working dataset    : {len(df):,} rows")
    print(f"  Selling price      : Rs.{df['selling_price'].min():,.0f} – Rs.{df['selling_price'].max():,.0f}")
    return df, chosen_name, report
# Compute model market bands
def compute_market_bands(df: pd.DataFrame) -> dict[str, list[float]]:
    print(f"\n{DIV}")
    print("COMPUTING MARKET BANDS (per model)")
    print(DIV)
    bands: dict[str, list[float]] = {}
    grouped = df.groupby("model")
    for model_name, group in grouped:
        if len(group) >= MIN_MODEL_ROWS:
            p10 = float(group["selling_price"].quantile(0.10))
            p90 = float(group["selling_price"].quantile(0.90))
            p10 = int(round(p10 / 500) * 500)
            p90 = int(round(p90 / 500) * 500)
            if p90 > p10:
                bands[str(model_name)] = [p10, p90]
    print(f"  Models with reliable bands : {len(bands)}")
    sample_keys = sorted(bands.keys())[:5]
    for k in sample_keys:
        print(f"    {k:<24}  Rs. {bands[k][0]//1000:>4}k – Rs. {bands[k][1]//1000:>4}k")
    return bands
# Compute segment market bands
def compute_segment_bands(df: pd.DataFrame) -> dict[str, list[float]]:
    print(f"\n{DIV}")
    print("COMPUTING SEGMENT BANDS")
    print(DIV)
    df["_segment"] = df["brand"].apply(get_segment)
    seg_bands: dict[str, list[float]] = {}
    for seg, group in df.groupby("_segment"):
        p5  = float(group["selling_price"].quantile(0.05))
        p95 = float(group["selling_price"].quantile(0.95))
        p5  = int(round(p5 / 1000) * 1000)
        p95 = int(round(p95 / 1000) * 1000)
        seg_bands[seg] = [p5, p95]
        print(f"  {seg:<10} Rs.{p5//1000}k – Rs.{p95//1000}k  ({len(group):,} rows)")
    return seg_bands
# Compute locality demand
def compute_locality_demand(df: pd.DataFrame) -> dict[str, float]:
    print(f"\n{DIV}")
    print("COMPUTING LOCALITY DEMAND")
    print(DIV)
    if "locality" not in df.columns:
        print("  locality column not found — skipping")
        return {}
    city_median = float(df["selling_price"].median())
    print(f"  Bangalore median selling price : Rs.{city_median:,.0f}")
    loc_demand: dict[str, float] = {}
    for loc, group in df.groupby("locality"):
        if loc in {"", "unknown", "nan"}:
            continue
        if len(group) >= MIN_LOCALITY_ROWS:
            loc_median = float(group["selling_price"].median())
            ratio = (loc_median - city_median) / city_median
            ratio = max(-0.25, min(0.35, ratio))
            loc_demand[str(loc)] = round(ratio, 4)
    print(f"  Localities computed : {len(loc_demand)}")
    sorted_locs = sorted(loc_demand.items(), key=lambda x: x[1], reverse=True)[:10]
    for loc, val in sorted_locs:
        print(f"    {loc:<24}  {val*100:+.2f}%")
    return loc_demand
# Compute RTO demand
def compute_rto_demand(df: pd.DataFrame) -> dict[str, float]:
    print(f"\n{DIV}")
    print("COMPUTING RTO DEMAND")
    print(DIV)
    if "rto" not in df.columns:
        print("  rto column not found — skipping")
        return {}
    city_median = float(df["selling_price"].median())
    rto_demand: dict[str, float] = {}
    for rto, group in df.groupby("rto"):
        if rto in {"", "unknown", "nan"}:
            continue
        if len(group) >= MIN_RTO_ROWS:
            rto_median = float(group["selling_price"].median())
            ratio = (rto_median - city_median) / city_median
            ratio = max(-0.30, min(0.40, ratio))
            rto_demand[str(rto).upper()] = round(ratio, 4)
    print(f"  RTO codes computed : {len(rto_demand)}")
    sorted_rtos = sorted(rto_demand.items(), key=lambda x: x[1], reverse=True)[:10]
    for rto, val in sorted_rtos:
        print(f"    {rto:<14}  {val*100:+.2f}%")
    return rto_demand
# Compute annual km tiers
def compute_annual_km_tiers(df: pd.DataFrame) -> dict[str, float]:
    print(f"\n{DIV}")
    print("COMPUTING ANNUAL KM TIERS")
    print(DIV)
    valid_km = df["km_per_year"].dropna()
    valid_km = valid_km[valid_km.between(500, 80_000)]
    tiers = {
        "very_low":  int(round(float(valid_km.quantile(0.10)))),
        "low":      int(round(float(valid_km.quantile(0.25)))),
        "moderate": int(round(float(valid_km.quantile(0.50)))),
        "high":     int(round(float(valid_km.quantile(0.75)))),
        "very_high": int(round(float(valid_km.quantile(0.95)))),
    }
    for k, v in tiers.items():
        print(f"  {k:<14} {v:,} km/yr")
    return tiers
# Compute certified vehicle premium
def compute_certified_premium(df: pd.DataFrame) -> float:
    print(f"\n{DIV}")
    print("COMPUTING CERTIFIED VEHICLE PREMIUM")
    print(DIV)
    if "certified" not in df.columns:
        print("  certified column not found — defaulting to 3.5%")
        return 0.035
    cert = df[df["certified"] == 1]["selling_price"]
    uncert = df[df["certified"] == 0]["selling_price"]
    if len(cert) < 50 or len(uncert) < 50:
        print("  Insufficient certified rows — defaulting to 3.5%")
        return 0.035
    c_med   = float(cert.median())
    u_med   = float(uncert.median())
    premium = (c_med - u_med) / u_med
    premium = max(0.01, min(0.20, premium))
    print(f"  Certified median   : Rs.{c_med:,.0f}")
    print(f"  Uncertified median : Rs.{u_med:,.0f}")
    print(f"  Premium            : {premium*100:.2f}%")
    return round(premium, 4)
# Compute owner statistics
def compute_owner_stats(df: pd.DataFrame) -> dict[str, float]:
    print(f"\n{DIV}")
    print("COMPUTING OWNER STATISTICS")
    print(DIV)
    if "owner_count" not in df.columns:
        return {"1": 1.0, "2": 0.94, "3": 0.88, "4": 0.80}
    med_1 = float(df[df["owner_count"] == 1]["selling_price"].median() or 1.0)
    stats: dict[str, float] = {}
    for owner in [1, 2, 3, 4, 5]:
        grp = df[df["owner_count"] == owner]["selling_price"]
        if len(grp) >= 20:
            ratio = float(grp.median()) / med_1
            stats[str(owner)] = round(ratio, 4)
            print(f"  owner={owner}   ratio={ratio:.4f}")
        else:
            stats[str(owner)] = round(max(0.35, 1.0 - (owner - 1) * 0.12), 4)
    return stats
# Compute seller type statistics
def compute_seller_stats(df: pd.DataFrame) -> dict[str, float]:
    print(f"\n{DIV}")
    print("COMPUTING SELLER TYPE STATISTICS")
    print(DIV)
    if "seller_type" not in df.columns:
        return {"dealer": 1.0, "individual": 0.96}
    dealer_grp = df[df["seller_type"] == "dealer"]["selling_price"]
    med_dealer = float(dealer_grp.median()) if len(dealer_grp) >= 20 else float(df["selling_price"].median())
    stats: dict[str, float] = {}
    for stype, group in df.groupby("seller_type"):
        if len(group) >= 20:
            ratio = float(group["selling_price"].median()) / max(med_dealer, 1.0)
            stats[str(stype)] = round(ratio, 4)
            print(f"  {stype:<20} ratio={ratio:.4f}  (n={len(group):,})")
    if "dealer" not in stats:
        stats["dealer"] = 1.0
    return stats
def compute_clamp_tolerances(report: dict) -> tuple[dict[str, list[float]], dict]:
    print(f"\n{DIV}")
    print("COMPUTING CLAMP TOLERANCE (from MAPE)")
    print(DIV)
    val_metrics = report.get("val_metrics", {})
    ens_metrics = val_metrics.get("Ensemble", {})
    mape        = float(ens_metrics.get("MAPE", 6.47))
    r2          = float(ens_metrics.get("R2", 0.976))
    print(f"  Ensemble MAPE      : {mape:.2f}%")
    mape_frac = max(0.04, min(0.20, mape / 100.0))
    tolerances = {
        "economy": [round(1.0 - mape_frac * 1.4, 4), round(1.0 + mape_frac * 1.4, 4)],
        "premium": [round(1.0 - mape_frac * 1.75, 4), round(1.0 + mape_frac * 1.75, 4)],
        "luxury":  [round(1.0 - mape_frac * 2.25, 4), round(1.0 + mape_frac * 2.25, 4)],
    }
    for seg, bounds in tolerances.items():
        width = bounds[1] - bounds[0]
        print(f"  {seg:<11} {bounds}  width={width:.4f}")
    mc_base = round(min(95.0, max(50.0, 50.0 + r2 * 40.0)), 1)
    bc_base = round(min(92.0, max(45.0, 50.0 + r2 * 35.0)), 1)
    conf_baseline = {
        "mc_base":   mc_base,
        "bc_base":   bc_base,
        "mape_frac": round(mape_frac, 4),
        "r2":        round(r2, 4),
    }
    print(f"\n  Confidence baseline — mc_base={mc_base}  bc_base={bc_base}  mape={mape:.2f}%")
    return tolerances, conf_baseline
# Compute comparable density
def compute_comparable_density(df: pd.DataFrame) -> int:
    if "locality" not in df.columns:
        return len(df.groupby("model"))
    valid = df.dropna(subset=["model", "locality"])
    valid = valid[~valid["locality"].isin(["", "unknown"])]
    n_pairs = len(valid.groupby(["model", "locality"]))
    print(f"\n  Comparable density — {n_pairs:,} model/locality pairs")
    return n_pairs
# Main generation flow
def generate_config() -> None:
    print("=" * 72)
    print("AutoPricer — generate_engine_config.py")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 72)
    df, dataset_name, report = load_data()
    market_bands    = compute_market_bands(df)
    segment_bands   = compute_segment_bands(df)
    locality_demand = compute_locality_demand(df)
    rto_demand      = compute_rto_demand(df)
    annual_km_tiers = compute_annual_km_tiers(df)
    cert_premium    = compute_certified_premium(df)
    owner_stats     = compute_owner_stats(df)
    seller_stats    = compute_seller_stats(df)
    clamp_tol, conf_base = compute_clamp_tolerances(report)
    comp_density    = compute_comparable_density(df)
    config = {
        "generated_at":               datetime.now().isoformat(),
        "dataset":                    dataset_name,
        "dataset_rows":               len(df),
        "market_bands":               market_bands,
        "segment_bands":              segment_bands,
        "locality_demand":            locality_demand,
        "rto_demand":                 rto_demand,
        "annual_km_tiers":            annual_km_tiers,
        "certified_vehicle_premium":  cert_premium,
        "owner_statistics":           owner_stats,
        "seller_type_statistics":     seller_stats,
        "clamp_tolerance":            clamp_tol,
        "confidence_baseline":        conf_base,
        "comparable_density_pairs":   comp_density,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n{DIV}")
    print("SAVED")
    print(DIV)
    print(f"  Output  : {OUT_PATH}")
    print(f"  Models  : {len(market_bands)} market bands")
    print(f"  Locs    : {len(locality_demand)} localities")
    print(f"  RTOs    : {len(rto_demand)} RTO codes")
    print("\n  Done — engine_config.json is ready.\n")
if __name__ == "__main__":
    generate_config()
