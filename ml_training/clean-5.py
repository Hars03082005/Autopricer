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

# CONFIGURATION

HERE         = Path(__file__).resolve().parent
DATA_DIR     = HERE / "data"
CURRENT_YEAR = 2026
DIV = "=" * 80

INPUT_FILES = {
    "pincode_with_owner1_filled-5": DATA_DIR / "pincode with owner1 filled-5.csv",
}

# BRAND VALIDATION

OEM_ALLOWLIST = {
    "maruti suzuki", "hyundai", "tata", "renault", "honda",
    "mahindra", "kia", "ford", "volkswagen", "skoda", "toyota",
    "nissan", "mg", "chevrolet", "datsun", "jeep", "bmw", "audi",
    "fiat", "mercedes-benz", "volvo", "land rover", "citroen",
    "bajaj", "jaguar", "mitsubishi", "mini", "lexus",
}

# BRAND TIER MAP  (0 budget → 4 luxury)

BRAND_TIER_MAP: dict[str, int] = {
    "datsun":        0,
    "maruti suzuki": 1, "renault": 1, "tata": 1,
    "chevrolet": 1, "fiat": 1, "bajaj": 1,
    "hyundai": 2, "honda": 2, "kia": 2, "ford": 2,
    "volkswagen": 2, "skoda": 2, "nissan": 2,
    "mitsubishi": 2, "mahindra": 2, "citroen": 2,
    "toyota": 3, "mg": 3, "jeep": 3,
    "bmw": 4, "mercedes-benz": 4, "audi": 4,
    "volvo": 4, "mini": 4, "lexus": 4,
    "jaguar": 4, "land rover": 4,
}

TIER_TO_SEGMENT = {
    0: "budget", 1: "economy", 2: "mid", 3: "premium", 4: "luxury",
}

def _price_to_segment(price: float) -> str:
    if price < 300_000:   return "budget"
    if price < 600_000:   return "economy"
    if price < 1_000_000: return "mid"
    if price < 2_000_000: return "premium"
    return "luxury"

VALID_FUEL = {
    "petrol", "diesel", "electric", "cng", "lpg",
    "hybrid", "plug-in hybrid", "petrol+cng", "petrol+lpg",
}

# HELPERS

def _norm(value, default="unknown") -> str:
    if pd.isna(value):
        return default
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    return s if s not in {"", "nan", "none"} else default

def _parse_owner(value) -> int:
    s = _norm(value)
    if s in {"unknown", "first", "1st"}:  return 1
    if s in {"second", "2nd"}:            return 2
    if s in {"third", "3rd"}:             return 3
    if s in {"fourth", "4th"}:            return 4
    if s in {"fifth", "5th"}:             return 5
    try:
        return max(1, min(int(float(s)), 6))
    except Exception:
        return 1

def _seller(value) -> str:
    s = _norm(value)
    if "dealer" in s or "direct" in s:
        return "dealer"
    if "individual" in s or "private" in s:
        return "individual"
    return "unknown"

def _usage_category(value) -> int:
    """Convert usage_category text to numeric."""
    mapping = {"low": 0, "medium": 1, "high": 2, "very high": 3}
    return mapping.get(_norm(value), 0)

# PHASE 1 — LOAD

def load_and_audit(path: Path, label: str) -> pd.DataFrame:
    print(f"\n{DIV}")
    print(f"LOADING : {label}")
    print(DIV)

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]}")
    print(f"\nColumn names found:")
    for c in df.columns:
        print(f"  {c}")

    print(f"\nMissing Values:")
    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col:<30} {count:>8,}  ({count/len(df)*100:.1f}%)")
    return df

# PHASE 2 — RENAME  (map new dataset columns to internal names)

COLUMN_MAPPING = {
    # New dataset column  →  internal name
    "make":               "brand_raw",
    "model":              "model_raw",
    "trim":               "variant_raw",
    "fuel":               "fuel_raw",
    "trans":              "trans_raw",
    "price":              "selling_price",
    "owner":              "owner_raw",
    "seller type":        "seller_type_raw",
    "year":               "year",
    "odometer":           "odometer_reading",
    "city":               "city",
    "locality":           "locality",
    "rto":                "rto",
    "segment":            "segment_raw",
    "age":                "vehicle_age_raw",
    "pincode":            "pincode",
    # enriched signals absent in this dataset — kept for forward-compat
    "avg_km_per_year":    "avg_km_per_year_raw",
    "usage_category":     "usage_category_raw",
    "price_per_year":     "price_per_year",
    "locality_density":   "locality_density",
    "popularity_score":   "popularity_score",
    "make_model_trim_combo":     "make_model_trim_combo",
}

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    print(f"\nRenamed {len(rename_map)} columns.")
    return df

# PHASE 2b — DROP LEAKAGE COLUMNS
# These columns are derived from the target (selling_price) and cause data
# leakage if left in the dataset. They must never reach the trained model.

LEAKAGE_COLS = [
    "make_model_year_avg_price",  # group mean of target → direct leakage
    "price_deviation_pct",        # pct deviation from group mean → direct leakage
    "is_above_market",            # binary flag derived from target → leakage
    "is_below_market",            # binary flag derived from target → leakage
]

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  [LEAKAGE DROP] Removed {len(to_drop)} leakage cols: {to_drop}")
    else:
        print("  [LEAKAGE DROP] No leakage columns present — OK")
    return df

# PHASE 3 — CLEAN & VALIDATE

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 3 : CLEAN & VALIDATE")
    print(DIV)

    before = len(df)

    # Selling price
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df[df["selling_price"].between(50_000, 20_000_000)]
    print(f"  Price filter       : {before:,} → {len(df):,}")

    # Year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(1990, CURRENT_YEAR)]
    df["year"] = df["year"].astype(int)
    print(f"  Year filter        : kept {len(df):,}")

    # Odometer
    df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce")
    df = df[df["odometer_reading"].between(0, 600_000)]
    print(f"  Odometer filter    : kept {len(df):,}")

    # Brand
    df["brand"] = df["brand_raw"].apply(_norm).replace({
        "mercedes benz": "mercedes-benz",
        "land-rover":    "land rover",
        "maruti":        "maruti suzuki",
    })
    before_brand = len(df)
    df = df[df["brand"].isin(OEM_ALLOWLIST)]
    print(f"  Brand filter       : {before_brand:,} → {len(df):,}")

    # Model
    df["model"] = df["model_raw"].apply(lambda x: _norm(x, "unknown"))
    df = df[df["model"] != "unknown"]
    print(f"  Model filter       : kept {len(df):,}")

    # Variant
    df["variant"] = df["variant_raw"].apply(lambda x: _norm(x, "unknown")) \
        if "variant_raw" in df.columns else "unknown"

    # Fuel
    df["fuel_type"] = df["fuel_raw"].apply(_norm).replace({
        "petrol+cng":     "cng",
        "petrol+lpg":     "lpg",
        "plug-in hybrid": "hybrid",
    }) if "fuel_raw" in df.columns else "petrol"
    df = df[df["fuel_type"].isin(VALID_FUEL)]
    print(f"  Fuel filter        : kept {len(df):,}")

    # Transmission
    df["transmission"] = df["trans_raw"].apply(_norm).replace({
        "amt": "automatic", "cvt": "automatic",
        "dct": "automatic", "imt": "manual",
    }) if "trans_raw" in df.columns else "manual"
    df = df[df["transmission"].isin({"manual", "automatic"})]
    print(f"  Trans filter       : kept {len(df):,}")

    # City
    df["city"] = df["city"].apply(_norm) if "city" in df.columns else "bangalore"

    # Locality
    df["locality"] = df["locality"].apply(lambda x: _norm(x, "unknown")) \
        if "locality" in df.columns else "unknown"

    # RTO
    df["rto"] = df["rto"].apply(lambda x: _norm(x, "unknown")) \
        if "rto" in df.columns else "unknown"

    # Owner count
    df["owner_count"] = df["owner_raw"].apply(_parse_owner) \
        if "owner_raw" in df.columns else 1

    # Seller type
    df["seller_type"] = df["seller_type_raw"].apply(_seller) \
        if "seller_type_raw" in df.columns else "unknown"

    # Numeric enriched columns from new dataset
    for col in ["locality_density", "popularity_score",
                "price_per_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Usage category
    df["usage_category_num"] = df["usage_category_raw"].apply(_usage_category) \
        if "usage_category_raw" in df.columns else 0

    print(f"\n  Rows after cleaning : {len(df):,}")
    return df

# PHASE 4 — SEGMENT CLASSIFICATION
# Brand tier first — cheap old BMW stays "luxury" not "economy"

def classify_segment(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 4 : SEGMENT CLASSIFICATION")
    print(DIV)

    def get_segment(row) -> str:
        tier = BRAND_TIER_MAP.get(row["brand"])
        if tier is not None:
            return TIER_TO_SEGMENT[tier]
        return _price_to_segment(row["selling_price"])

    df["segment_class"] = df.apply(get_segment, axis=1)

    print("\nSegment distribution:")
    for seg, count in df["segment_class"].value_counts().items():
        print(f"  {seg:<12} {count:>7,}  ({count/len(df)*100:.1f}%)")

    return df

# PHASE 5 — FEATURE ENGINEERING
# Uses pre-computed columns from new dataset + derives additional features

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 5 : FEATURE ENGINEERING")
    print(DIV)

    # Vehicle age
    # Use pre-computed 'age' from dataset if available, else compute
    if "vehicle_age_raw" in df.columns:
        df["vehicle_age"] = pd.to_numeric(
            df["vehicle_age_raw"], errors="coerce"
        ).fillna(CURRENT_YEAR - df["year"]).clip(lower=0)
    else:
        df["vehicle_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    print("  [1] vehicle_age             — from dataset or computed")

    # KM per year
    # Use pre-computed avg_km_per_year if available
    # FIXED — compute fallback first, then fillna with a Series
    fallback = np.where(
        df["vehicle_age"] > 0,
        df["odometer_reading"] / df["vehicle_age"],
        df["odometer_reading"]
    )
    if "avg_km_per_year_raw" in df.columns:
        df["km_per_year"] = (
            pd.to_numeric(df["avg_km_per_year_raw"], errors="coerce")
            .fillna(pd.Series(fallback, index=df.index))
            .clip(0, 50_000)
            .round(1)
        )
    else:
        df["km_per_year"] = pd.Series(fallback, index=df.index).clip(0, 50_000).round(1)
    print("  [2] km_per_year             — from dataset or computed")

    # Brand tier
    df["brand_tier"] = df["brand"].map(BRAND_TIER_MAP).fillna(1).astype(int)
    print("  [3] brand_tier              — prestige 0-4")

    # Age × KM interaction
    df["age_km_interaction"] = df["vehicle_age"] * df["odometer_reading"]
    print("  [4] age_km_interaction      — age × odometer")

    # Ownership trust score
    trust_map = {1: 100, 2: 75, 3: 50, 4: 25, 5: 10, 6: 10}
    df["ownership_trust_score"] = df["owner_count"].map(trust_map).fillna(10).astype(int)
    print("  [5] ownership_trust_score   — non-linear owner penalty")

    # Vehicle health score
    df["vehicle_health_score"] = (
        100
        - (df["vehicle_age"] * 3)
        - (df["odometer_reading"] / 10_000)
        - ((df["owner_count"] - 1) * 8)
    ).clip(0, 100).round(1)
    print("  [6] vehicle_health_score    — composite health 0-100")

    # High mileage flag
    df["is_high_mileage"] = (df["km_per_year"] > 15_000).astype(int)
    print("  [7] is_high_mileage         — flag >15K km/yr")

    # Locality tier (Bangalore-specific)
    # Uses locality_density if available, else manual tier mapping
    LOCALITY_TIER = {
        # Tier 3 — premium (buyers pay more)
        "koramangala": 3, "indiranagar": 3, "whitefield": 3,
        "jp nagar": 3, "jayanagar": 3, "hebbal": 3,
        "ulsoor": 3, "richmond town": 3, "sadashivanagar": 3,
        # Tier 2 — mid
        "electronic city": 2, "marathahalli": 2, "kr puram": 2,
        "bannerghatta": 2, "yeshwanthpur": 2, "rajajinagar": 2,
        "malleshwaram": 2, "bhoruka tech park": 2,
        # Tier 1 — budget
        "bommanahalli": 1, "begur": 1, "anekal": 1,
        "bellahalli": 1, "vega city mall": 1,
    }
    df["locality_tier"] = df["locality"].map(LOCALITY_TIER).fillna(2).astype(int)
    print("  [8] locality_tier           — Bangalore area premium 1-3")

    # Locality density (normalised)
    if "locality_density" in df.columns:
        max_density = df["locality_density"].max()
        df["locality_density_norm"] = (
            df["locality_density"] / max_density if max_density > 0 else 0
        ).round(4)
        print("  [9] locality_density_norm   — normalised area density")

    # Popularity score (log-scaled)
    if "popularity_score" in df.columns:
        df["popularity_score_log"] = np.log1p(df["popularity_score"]).round(4)
        print("  [10] popularity_score_log  — log popularity signal")

    print(f"\n  High mileage : {df['is_high_mileage'].mean()*100:.1f}% of dataset")
    print(f"  Avg health   : {df['vehicle_health_score'].mean():.1f}/100")


    return df

# PHASE 6 — DEDUPLICATION

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
   # df = df.drop_duplicates(
  #      subset=["brand", "model", "variant", "transmission",
  #              "fuel_type", "year", "odometer_reading", "selling_price"]
   # ).reset_index(drop=True)
    removed = before - len(df)
    print(f"\n  Duplicates removed : {removed:,}  ({removed/before*100:.1f}%)")
    print(f"  Remaining rows     : {len(df):,}")
    return df

# PHASE 7 — DISTRIBUTION REPORT

def distribution_report(df: pd.DataFrame) -> None:
    print(f"\n{DIV}")
    print("DISTRIBUTION REPORT")
    print(DIV)

    for col in ["segment_class", "fuel_type", "transmission",
                "seller_type", "brand_tier", "usage_category_num"]:
        if col in df.columns:
            print(f"\n{col.upper()}")
            print(df[col].value_counts(dropna=False).to_string())
            print("-" * 40)

    print("\nSELLING PRICE")
    print(df["selling_price"].describe().apply(lambda x: f"Rs.{x:,.0f}").to_string())

    print("\nPRICE BY SEGMENT (median)")
    print(
        df.groupby("segment_class")["selling_price"]
        .median()
        .sort_values()
        .apply(lambda x: f"Rs.{x:,.0f}")
        .to_string()
    )

    if "locality" in df.columns:
        print("\nTOP 15 LOCALITIES")
        print(df["locality"].value_counts().head(15).to_string())

    print("\nTOP 10 BRANDS")
    print(df["brand"].value_counts().head(10).to_string())

# PHASE 8 — SELECT OUTPUT COLUMNS & SAVE

ML_FEATURES = [
    # Categorical identifiers
    "brand",
    "model",
    "variant",
    "city",
    "locality",
    "rto",
    "segment_class",
    "fuel_type",
    "transmission",
    "seller_type",

    # Base numeric
    "vehicle_age",
    "odometer_reading",
    "km_per_year",
    "owner_count",

    # Derived features
    "brand_tier",
    "age_km_interaction",
    "ownership_trust_score",
    "vehicle_health_score",
    "is_high_mileage",
    "locality_tier",

    # Enriched signals (present if available in source)
    "usage_category_num",
    "locality_density_norm",
    "popularity_score_log",

    # Target
    "selling_price",
]

ANALYSIS_COLS = ["year", "pincode"]


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_cols = [c for c in ML_FEATURES + ANALYSIS_COLS if c in df.columns]
    out_df   = df[out_cols].copy()
    out_df.to_csv(out_path, index=False)

    print(f"\n{DIV}")
    print("SAVED")
    print(DIV)
    print(f"  Path    : {out_path}")
    print(f"  Rows    : {len(out_df):,}")
    print(f"  Columns : {len(out_df.columns)}")
    print(f"\n  Output columns:")
    for c in out_df.columns:
        print(f"    {c}")

    nulls = out_df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"\n  Remaining nulls:")
        for col, cnt in nulls.items():
            print(f"    {col:<30} {cnt:>7,}")
    else:
        print("\n  No nulls remaining.")

# PIPELINE

def process_file(name: str, in_path: Path) -> None:
    out_path = DATA_DIR / f"processed_{name}.csv"
    label    = name.replace("_", " ").title()

    df = load_and_audit(in_path, label)
    df = rename_columns(df)
    df = drop_leakage_columns(df)
    df = clean_and_validate(df)
    df = classify_segment(df)
    df = engineer_features(df)
    df = deduplicate(df)
    distribution_report(df)
    save_dataset(df, out_path)


def main() -> None:
    print(DIV)
    print("PricerPoint Bangalore — Preprocessing Pipeline")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(DIV)

    for name, path in INPUT_FILES.items():
        if not path.exists():
            print(f"\nFile not found: {path}")
            continue
        process_file(name, path)

    print(f"\n{DIV}")
    print("PREPROCESSING COMPLETE")
    print(DIV)


if __name__ == "__main__":
    main()