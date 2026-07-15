"""
preprocess_flat_csv.py
======================
Preprocessor for PriceRef flat-schema CSV datasets:
  - ml_training/data/cell7_dataset.csv
  - ml_training/data/owner assumed dataset.csv

Both files share the same 19-column flat CSV schema:
  city, year, make, model, trim, odometer, fuel, trans, rto, price,
  list price, segment, seller type, certified, locality, pincode,
  owner, color, age

Outputs (separate processed files):
  ml_training/data/processed_cell7_dataset.csv
  ml_training/data/processed_owner_assumed_dataset.csv

Engineered features produced (compatible with train_ml_model.py FEATURES):
  Core ML features:
    brand, model, variant, city, rto_state, color, segment_class,
    fuel_type, transmission, vehicle_age, odometer_reading, km_per_year,
    owner_count, ownership_trust_score, vehicle_health_score,
    inspected, high_mileage, luxury_brand, has_list_price, selling_price

  Enriched analytical features (from preprocess_2026.py v8.0):
    Km_Per_Year, Depreciation_Bucket, Mileage_Tier, Brand_Tier,
    Ownership_Category, Price_Segment, Is_Recent_Model, Seller_Type_Clean

Usage:
  python ml_training/preprocess_flat_csv.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
DATA_DIR     = HERE / "data"
CURRENT_YEAR = 2026
DIV = "=" * 72

INPUT_FILES = {
    "cell7_dataset":        DATA_DIR / "cell7_dataset.csv",
    "owner_assumed_dataset": DATA_DIR / "owner assumed dataset.csv",
}

# ── OEM allowlist (for brand validation) ─────────────────────────────────────
OEM_ALLOWLIST = {
    "maruti", "maruti suzuki", "hyundai", "honda", "tata", "mahindra",
    "toyota", "ford", "renault", "volkswagen", "skoda", "kia", "mg",
    "jeep", "nissan", "datsun", "fiat", "mitsubishi", "chevrolet",
    "isuzu", "bmw", "mercedes-benz", "audi", "volvo", "mini", "lexus",
    "porsche", "jaguar", "land rover", "maserati", "bentley",
    "rolls-royce", "ferrari", "lamborghini", "aston martin", "hummer",
    "opel", "ambassador", "premier", "hindustan motors", "force",
    "ashok leyland", "icml", "mahindra renault", "mahindra ssangyong",
    "dc", "citroen", "bajaj",
}

# ── Segment mapping ───────────────────────────────────────────────────────────
SEGMENT_MAP: dict[str, str] = {
    "mass market": "economy",
    "budget":      "economy",
    "unknown":     "economy",
    "assured":     "economy",
    "standard":    "premium",
    "premium":     "premium",
    "luxury":      "luxury",
    "luxe":        "luxury",
}

LUXURY_BRANDS = {
    "bmw", "mercedes-benz", "audi", "jaguar", "volvo", "lexus",
    "porsche", "land rover", "maserati", "bentley", "rolls-royce",
    "ferrari", "lamborghini", "aston martin", "hummer",
}

# ── Brand tier map ────────────────────────────────────────────────────────────
BRAND_TIER_MAP: dict[str, str] = {
    **{b: "budget"  for b in {"maruti", "maruti suzuki", "datsun", "bajaj", "chevrolet",
                               "fiat", "opel", "premier", "hindustan motors", "icml",
                               "force", "ashok leyland", "ambassador"}},
    **{b: "mid"     for b in {"hyundai", "honda", "tata", "renault", "nissan", "ford",
                               "mahindra", "mitsubishi", "isuzu", "citroen", "dc"}},
    **{b: "premium" for b in {"volkswagen", "skoda", "toyota", "mg", "jeep", "kia",
                               "mini", "volvo", "lexus"}},
    **{b: "luxury"  for b in {"bmw", "mercedes-benz", "audi", "jaguar", "land rover",
                               "porsche", "maserati", "aston martin", "bentley",
                               "rolls-royce", "ferrari", "lamborghini", "hummer"}},
}

# ── Brand → segment class (for segment_class column) ─────────────────────────
BRAND_SEGMENT_MAP: dict[str, str] = {
    **{b: "economy"  for b in {"maruti", "maruti suzuki", "datsun", "bajaj", "chevrolet",
                                "fiat", "opel", "premier", "hindustan motors", "force",
                                "ashok leyland", "ambassador", "hyundai", "honda", "tata",
                                "renault", "nissan", "ford", "mahindra", "mitsubishi",
                                "isuzu", "citroen", "dc"}},
    **{b: "premium"  for b in {"volkswagen", "skoda", "toyota", "mg", "jeep", "kia",
                                "mini", "volvo", "lexus"}},
    **{b: "luxury"   for b in {"bmw", "mercedes-benz", "audi", "jaguar", "land rover",
                                "porsche", "maserati", "aston martin", "bentley",
                                "rolls-royce", "ferrari", "lamborghini", "hummer"}},
}

VALID_FUEL  = {"petrol", "diesel", "electric", "cng", "lpg", "hybrid",
               "plug-in hybrid", "petrol+cng", "petrol+lpg"}
VALID_TRANS = {"manual", "automatic", "amt", "cvt", "dct", "imt"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(v, default: str = "unknown") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s if s and s not in {"nan", "none", ""} else default


def _num(v, default: float = np.nan) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return default


def _parse_owner(v) -> int:
    """Parse owner field to int. Handles numeric and 'Unknown'."""
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "unknown"}:
        return 0   # 0 = unknown, will be flagged
    try:
        n = int(float(s))
        return max(1, min(n, 6))
    except ValueError:
        return 0


def _parse_inspected(v) -> int:
    s = str(v).strip().lower()
    return 1 if s in {"yes", "true", "1", "certified", "inspected"} else 0


def _rto_state(rto: str) -> str:
    m = re.match(r"([A-Za-z]{2})", str(rto).strip())
    return m.group(1).lower() if m else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: LOAD & AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_audit(path: Path, label: str) -> pd.DataFrame:
    print(f"\n{DIV}")
    print(f"LOADING: {label}")
    print(f"  Path: {path}")
    print(DIV)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"  Columns: {list(df.columns)}")
    print("\n  Null counts per column:")
    null_counts = df.isna().sum()
    for col, cnt in null_counts[null_counts > 0].items():
        print(f"    {col:<25} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: COLUMN RENAME & CANONICAL TYPES
# ─────────────────────────────────────────────────────────────────────────────

COL_RENAME = {
    "city":        "city_raw",
    "year":        "year",
    "make":        "brand_raw",
    "model":       "model_raw",
    "trim":        "variant_raw",
    "odometer":    "odometer_reading",
    "fuel":        "fuel_raw",
    "trans":       "trans_raw",
    "rto":         "rto_raw",
    "price":       "selling_price",
    "list price":  "list_price",
    "segment":     "segment_raw",
    "seller type": "seller_type_raw",
    "certified":   "certified_raw",
    "locality":    "locality",
    "pincode":     "pincode",
    "owner":       "owner_raw",
    "color":       "color_raw",
    "age":         "vehicle_age_raw",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})
    print(f"\n  Columns after rename: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: CLEAN & VALIDATE CORE FIELDS
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 3 -- CLEAN & VALIDATE")
    print(DIV)
    before = len(df)

    # ── Selling price (target) ───────────────────────────────────────────────
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df[df["selling_price"].between(50_000, 20_000_000)]
    print(f"  Price filter [50k–2Cr]:  kept {len(df):,} / {before:,}")

    # ── Year ─────────────────────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(1990, CURRENT_YEAR)]
    df["year"] = df["year"].astype(int)
    print(f"  Year filter [1990–{CURRENT_YEAR}]: kept {len(df):,}")

    # ── Odometer ─────────────────────────────────────────────────────────────
    df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce")
    df = df[df["odometer_reading"].between(0, 600_000)]
    df["odometer_reading"] = df["odometer_reading"].clip(0, 600_000)
    print(f"  Odometer filter [0–600k]: kept {len(df):,}")

    # ── Brand (make) ─────────────────────────────────────────────────────────
    df["brand"] = df["brand_raw"].apply(lambda v: _norm(v))
    # Normalise known aliases
    df["brand"] = df["brand"].replace({
        "maruti suzuki": "maruti suzuki",
        "mercedes benz": "mercedes-benz",
        "land-rover":    "land rover",
    })
    # Keep only OEM allowlist brands
    before_brand = len(df)
    df = df[df["brand"].isin(OEM_ALLOWLIST)]
    print(f"  Brand filter (OEM allowlist): kept {len(df):,} / {before_brand:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    df["model_clean"] = df["model_raw"].apply(lambda v: _norm(v, "unknown"))
    # Brand-prefix model name (e.g. "honda city")
    df["model"] = df.apply(
        lambda r: f"{r['brand']} {r['model_clean']}"
        if r["brand"] not in r["model_clean"] else r["model_clean"],
        axis=1
    )
    df = df[df["model_clean"] != "unknown"]
    print(f"  Model null filter: kept {len(df):,}")

    # ── Fuel ──────────────────────────────────────────────────────────────────
    df["fuel_type"] = df["fuel_raw"].apply(lambda v: _norm(v))
    df["fuel_type"] = df["fuel_type"].replace({
        "petrol+cng": "cng", "petrol+lpg": "lpg", "plug-in hybrid": "hybrid",
    })
    df = df[df["fuel_type"].isin(VALID_FUEL)]
    print(f"  Fuel filter: kept {len(df):,}")

    # ── Transmission ──────────────────────────────────────────────────────────
    df["transmission"] = df["trans_raw"].apply(lambda v: _norm(v))
    df["transmission"] = df["transmission"].replace({"amt": "automatic", "cvt": "automatic", "dct": "automatic", "imt": "manual"})
    df = df[df["transmission"].isin({"manual", "automatic"})]
    print(f"  Transmission filter: kept {len(df):,}")

    # ── Variant ───────────────────────────────────────────────────────────────
    df["variant"] = df["variant_raw"].apply(lambda v: _norm(v, "unknown"))

    # ── City ──────────────────────────────────────────────────────────────────
    df["city"] = df["city_raw"].apply(lambda v: _norm(v, "unknown"))

    # ── RTO state ─────────────────────────────────────────────────────────────
    df["rto_state"] = df["rto_raw"].apply(lambda v: _rto_state(_norm(v, "un")))

    # ── Color ─────────────────────────────────────────────────────────────────
    df["color"] = df["color_raw"].apply(lambda v: _norm(v, "unknown"))

    # ── List price ────────────────────────────────────────────────────────────
    df["list_price"] = pd.to_numeric(df.get("list_price", np.nan), errors="coerce")
    df["has_list_price"] = df["list_price"].notna().astype(int)

    # ── Segment ───────────────────────────────────────────────────────────────
    df["segment_raw_norm"] = df["segment_raw"].apply(lambda v: _norm(v, "unknown"))
    df["segment_class"]    = df["segment_raw_norm"].map(SEGMENT_MAP).fillna("economy")
    # Override segment_class using brand if it's a luxury brand
    df.loc[df["brand"].isin(LUXURY_BRANDS), "segment_class"] = "luxury"
    # Also use brand→segment map for premium brands
    df["segment_class"] = df["brand"].map(BRAND_SEGMENT_MAP).combine_first(df["segment_class"]).fillna("economy")

    # ── Certified / Inspected ─────────────────────────────────────────────────
    df["inspected"] = df["certified_raw"].apply(_parse_inspected)

    # ── Owner count ───────────────────────────────────────────────────────────
    df["owner_count_raw"] = df["owner_raw"].apply(_parse_owner)
    df["owner_missing"]   = (df["owner_count_raw"] == 0).astype(int)
    # For unknown owners: assign 1 (conservative assumption)
    df["owner_count"]     = df["owner_count_raw"].replace(0, 1)

    # ── Seller type ───────────────────────────────────────────────────────────
    def _seller(v):
        s = _norm(v)
        if "dealer" in s:       return "dealer"
        if "direct" in s:       return "dealer"   # "Direct Seller"
        if "individual" in s:   return "individual"
        if "private" in s:      return "individual"
        return "unknown"
    df["seller_type"] = df.get("seller_type_raw", pd.Series("unknown", index=df.index)).apply(_seller)

    print(f"\n  Rows after all validation: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 4 -- FEATURE ENGINEERING")
    print(DIV)

    km   = df["odometer_reading"]
    year = df["year"]

    # ── Core ML features ─────────────────────────────────────────────────────
    df["vehicle_age"] = (CURRENT_YEAR - year).clip(lower=0)

    df["km_per_year"] = np.where(
        df["vehicle_age"] > 0,2
        (km / df["vehicle_age"]).clip(0, 100_000),
        km.clip(0, 100_000),
    ).round(1)

    df["ownership_trust_score"] = (
        (1 / df["owner_count"]) * 0.5
        + (1 - (df["vehicle_age"] / 35).clip(0, 1)) * 0.3
        + (1 - (km / 600_000).clip(0, 1)) * 0.2
    ).round(4)

    df["vehicle_health_score"] = (
        (1 - (km / 600_000).clip(0, 1)) * 0.5
        + (1 - (df["vehicle_age"] / 35).clip(0, 1)) * 0.3
        + (1 / df["owner_count"]) * 0.2
    ).round(4)

    odo_q75 = km.quantile(0.75)
    df["high_mileage"] = (km > odo_q75).astype(int)

    df["luxury_brand"] = df["brand"].isin(LUXURY_BRANDS).astype(int)

    # Negotiation features
    df["negotiation_margin"] = np.where(
        df["list_price"].notna(),
        df["list_price"] - df["selling_price"],
        np.nan,
    )
    df["negotiation_pct"] = np.where(
        df["list_price"].notna() & (df["list_price"] > 0),
        (df["list_price"] - df["selling_price"]) / df["list_price"],
        np.nan,
    )

    # ── Enriched analytical features (v8.0) ───────────────────────────────────

    # Km_Per_Year (already computed above, alias)
    df["Km_Per_Year"] = df["km_per_year"]

    # Depreciation_Bucket
    def _dep_bucket(age):
        if age <= 2:   return "new"
        if age <= 5:   return "recent"
        if age <= 9:   return "mid"
        if age <= 12:  return "old"
        return "very_old"
    df["Depreciation_Bucket"] = df["vehicle_age"].apply(_dep_bucket)

    # Mileage_Tier
    def _mileage_tier(km_val):
        if km_val < 30_000:   return "low"
        if km_val < 70_000:   return "moderate"
        if km_val < 120_000:  return "high"
        return "very_high"
    df["Mileage_Tier"] = df["odometer_reading"].apply(_mileage_tier)

    # Brand_Tier
    df["Brand_Tier"] = df["brand"].map(BRAND_TIER_MAP).fillna("mid")

    # Ownership_Category
    def _owner_cat(n):
        if n == 1:   return "single"
        if n == 2:   return "two_owner"
        if n == 3:   return "three_owner"
        return "four_plus"
    df["Ownership_Category"] = df["owner_count"].apply(_owner_cat)
    df.loc[df["owner_missing"] == 1, "Ownership_Category"] = "unknown"

    # Price_Segment
    def _price_seg(p):
        if p < 300_000:   return "budget"
        if p < 800_000:   return "mid"
        if p < 1_500_000: return "premium"
        return "luxury"
    df["Price_Segment"] = df["selling_price"].apply(_price_seg)

    # Is_Recent_Model
    df["Is_Recent_Model"] = (df["vehicle_age"] <= 3).astype(int)

    # Seller_Type_Clean (already computed as seller_type above)
    df["Seller_Type_Clean"] = df["seller_type"]

    print(f"\n  75th percentile odometer (high_mileage threshold): {odo_q75:,.0f} km")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: DEDUP
# ─────────────────────────────────────────────────────────────────────────────

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"\n  Dedup: removed {before - len(df):,} duplicates, kept {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: SELECT OUTPUT COLUMNS & SAVE
# ─────────────────────────────────────────────────────────────────────────────

# Columns that train_ml_model.py reads (canonical ML feature names)
ML_FEATURE_COLS = [
    "brand", "model", "variant", "city", "rto_state", "color",
    "segment_class", "fuel_type", "transmission",
    "vehicle_age", "odometer_reading", "km_per_year", "owner_count",
    "ownership_trust_score", "vehicle_health_score",
    "inspected", "high_mileage", "luxury_brand", "has_list_price",
    "selling_price",
]

# Extra cols for dataset analysis and business logic
ANALYSIS_COLS = [
    "year",
    "owner_missing", "seller_type",
    # Enriched v8.0 features
    "Km_Per_Year", "Depreciation_Bucket", "Mileage_Tier", "Brand_Tier",
    "Ownership_Category", "Price_Segment", "Is_Recent_Model", "Seller_Type_Clean",
]


def save_output(df: pd.DataFrame, out_path: Path, label: str) -> None:
    available_ml   = [c for c in ML_FEATURE_COLS  if c in df.columns]
    available_anal = [c for c in ANALYSIS_COLS     if c in df.columns]
    out_cols = available_ml + [c for c in available_anal if c not in available_ml]

    out_df = df[out_cols].copy()
    out_df.to_csv(out_path, index=False)

    print(f"\n{'─'*72}")
    print(f"SAVED: {label}")
    print(f"  Path:    {out_path}")
    print(f"  Shape:   {out_df.shape[0]:,} rows × {out_df.shape[1]} columns")
    print(f"  Memory:  {out_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  ML feature columns ({len(available_ml)}): {available_ml}")
    print(f"  Analysis columns ({len(available_anal)}): {available_anal}")

    null_rem = out_df.isna().sum()
    null_rem = null_rem[null_rem > 0]
    if not null_rem.empty:
        print(f"\n  Remaining nulls (intentional):")
        for col, cnt in null_rem.items():
            print(f"    {col:<30} {cnt:>8,}  ({cnt/len(out_df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{DIV}")
    print(f"DATASET SUMMARY: {label}")
    print(DIV)
    print(f"  Rows:            {len(df):,}")
    print(f"  Price range:     ₹{df['selling_price'].min():,.0f} — ₹{df['selling_price'].max():,.0f}")
    print(f"  Median price:    ₹{df['selling_price'].median():,.0f}")
    print(f"  Segment dist:    {df['segment_class'].value_counts().to_dict()}")
    print(f"  Fuel dist:       {df['fuel_type'].value_counts().to_dict()}")
    print(f"  Trans dist:      {df['transmission'].value_counts().to_dict()}")
    print(f"  Age range:       {df['vehicle_age'].min():.0f} — {df['vehicle_age'].max():.0f} yrs")
    print(f"  Odometer range:  {df['odometer_reading'].min():,.0f} — {df['odometer_reading'].max():,.0f} km")
    top5_brands = df["brand"].value_counts().head(5).to_dict()
    print(f"  Top 5 brands:    {top5_brands}")
    if "owner_count" in df.columns:
        print(f"  Owner dist:      {df['owner_count'].value_counts().sort_index().to_dict()}")
    if "Price_Segment" in df.columns:
        print(f"  Price segments:  {df['Price_Segment'].value_counts().to_dict()}")
    if "Depreciation_Bucket" in df.columns:
        print(f"  Dep. buckets:    {df['Depreciation_Bucket'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_file(key: str, in_path: Path) -> None:
    label    = key.replace("_", " ").title()
    out_path = DATA_DIR / f"processed_{key}.csv"

    df = load_and_audit(in_path, label)
    df = rename_columns(df)
    df = clean_and_validate(df)
    df = engineer_features(df)
    df = dedup(df)
    print_summary(df, label)
    save_output(df, out_path, label)


def main() -> None:
    print(DIV)
    print("PriceRef — Flat CSV Preprocessing Pipeline")
    print(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIV)

    for key, path in INPUT_FILES.items():
        if not path.exists():
            print(f"\n[SKIP] File not found: {path}")
            continue
        process_file(key, path)

    print(f"\n{DIV}")
    print("All files processed successfully.")
    print(f"Output directory: {DATA_DIR}")
    print(DIV)


if __name__ == "__main__":
    main()
