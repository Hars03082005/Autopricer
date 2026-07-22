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

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

CURRENT_YEAR = 2026

DIV = "=" * 80

INPUT_FILES = {
    "with owner filled":        DATA_DIR / "with owner filled.csv",
}

# BRAND VALIDATION

OEM_ALLOWLIST = {
    # All normalized to lowercase — raw variants are handled in clean_and_validate()
    "maruti suzuki",    # covers "Maruti Suzuki" and "Maruti" after normalization
    "hyundai",
    "tata",
    "renault",
    "honda",
    "mahindra",
    "kia",
    "ford",
    "volkswagen",
    "skoda",
    "toyota",
    "nissan",
    "mg",               # covers "MG" and "Mg Motors" after normalization
    "chevrolet",
    "datsun",
    "jeep",
    "bmw",
    "audi",
    "fiat",
    "mercedes-benz",    # covers "Mercedes-Benz" and "Mercedes" after normalization
    "volvo",
    "land rover",       # covers "Land Rover" and "Land" after normalization
    "citroen",
    "bajaj",
    "jaguar",
    "mitsubishi",
    "mini",             # covers "MINI" and "Mini" after normalization
    "lexus",
}

# BRAND → TIER  (0 budget · 1 economy · 2 mid · 3 premium · 4 luxury)
# Used for both brand_tier feature and segment fallback

BRAND_TIER_MAP: dict[str, int] = {
    # Budget (0)
    "datsun":           0,

    # Economy (1)
    "maruti suzuki":    1,
    "renault":          1,
    "tata":             1,
    "chevrolet":        1,
    "fiat":             1,
    "bajaj":            1,

    # Mid (2)
    "hyundai":          2,
    "honda":            2,
    "kia":              2,
    "ford":             2,
    "volkswagen":       2,
    "skoda":            2,
    "nissan":           2,
    "mitsubishi":       2,
    "mahindra":         2,
    "citroen":          2,

    # Premium (3)
    "toyota":           3,
    "mg":               3,
    "jeep":             3,

    # Luxury (4)
    "bmw":              4,
    "mercedes-benz":    4,
    "audi":             4,
    "volvo":            4,
    "mini":             4,
    "lexus":            4,
    "jaguar":           4,
    "land rover":       4,
}

TIER_TO_SEGMENT = {
    0: "budget",
    1: "economy",
    2: "mid",
    3: "premium",
    4: "luxury",
}

# Price-band fallback when brand is unknown
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

VALID_TRANS = {"manual", "automatic", "amt", "cvt", "dct", "imt"}

# HELPERS

def _norm(value, default="unknown"):
    if pd.isna(value):
        return default
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    if value in {"", "nan", "none"}:
        return default
    return value


def _num(value, default=np.nan):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def _parse_owner(value):
    value = _norm(value)
    if value == "unknown":
        return 1
    try:
        return max(1, min(int(float(value)), 6))
    except Exception:
        return 1


def _parse_inspected(value):
    return int(_norm(value) in {"yes", "true", "1", "certified", "inspected"})


# PHASE 1 — LOAD

def load_and_audit(path: Path, label: str) -> pd.DataFrame:
    print(f"\n{DIV}")
    print(f"LOADING : {label}")
    print(DIV)

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.shape[1]}")

    print("\nMissing Values")
    missing = df.isna().sum()
    for col, count in missing.items():
        if count == 0:
            continue
        print(f"  {col:<25} {count:>8}  ({count/len(df)*100:.1f}%)")

    return df


# PHASE 2 — RENAME

COLUMN_MAPPING = {
    "make":        "brand_raw",
    "model":       "model_raw",
    "trim":        "variant_raw",
    "fuel":        "fuel_raw",
    "trans":       "trans_raw",
    "segment":     "segment_raw",
    "seller type": "seller_type_raw",
    "certified":   "certified_raw",
    "owner":       "owner_raw",
    "color":       "color_raw",
    "odometer":    "odometer_reading",
    "price":       "selling_price",
    "year":        "year",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    )
    print("\nColumns renamed.")
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
    print("PHASE 3 : CLEANING")
    print(DIV)

    # Price
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df[df["selling_price"].between(50_000, 20_000_000)]

    # Year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(1990, CURRENT_YEAR)]
    df["year"] = df["year"].astype(int)

    # Odometer
    df["odometer_reading"] = pd.to_numeric(df["odometer_reading"], errors="coerce")
    df = df[df["odometer_reading"].between(0, 600_000)]

    # Brand
    df["brand"] = df["brand_raw"].apply(_norm).replace({
        "maruti suzuki": "maruti suzuki",
        "mercedes benz": "mercedes-benz",
        "land-rover":    "land rover",
    })
    before = len(df)
    df = df[df["brand"].isin(OEM_ALLOWLIST)]
    print(f"Brand validation  : {before:,} → {len(df):,}")

    # Model
    df["model"] = df["model_raw"].apply(lambda x: _norm(x, "unknown"))
    df = df[df["model"] != "unknown"]

    # Variant
    df["variant"] = df["variant_raw"].apply(lambda x: _norm(x, "unknown"))

    # Fuel
    df["fuel_type"] = df["fuel_raw"].apply(_norm).replace({
        "petrol+cng":    "cng",
        "petrol+lpg":    "lpg",
        "plug-in hybrid": "hybrid",
    })
    df = df[df["fuel_type"].isin(VALID_FUEL)]

    # Transmission
    df["transmission"] = df["trans_raw"].apply(_norm).replace({
        "amt": "automatic",
        "cvt": "automatic",
        "dct": "automatic",
        "imt": "manual",
    })
    df = df[df["transmission"].isin({"manual", "automatic"})]

    # Color (optional column)
    if "color_raw" in df.columns:
        df["color"] = df["color_raw"].apply(lambda x: _norm(x, "unknown"))
    else:
        df["color"] = "unknown"

    # Seller type
    def seller_mapping(value):
        value = _norm(value)
        if "dealer" in value or "direct" in value:
            return "dealer"
        if "individual" in value or "private" in value:
            return "individual"
        return "unknown"

    df["seller_type"] = df["seller_type_raw"].apply(seller_mapping)

    # Owner count
    df["owner_count"] = df["owner_raw"].apply(_parse_owner)

    # Inspection
    if "certified_raw" in df.columns:
        df["inspected"] = df["certified_raw"].apply(_parse_inspected)
    else:
        df["inspected"] = 0

    print(f"Rows after cleaning : {len(df):,}")
    return df


# PHASE 4 — SEGMENT CLASSIFICATION
# Segment is derived from brand tier first; price band is the fallback.
# This ensures a cheap old BMW stays "luxury", not "economy".

def classify_segment(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 4 : SEGMENT CLASSIFICATION")
    print(DIV)

    def get_segment(row) -> str:
        tier = BRAND_TIER_MAP.get(row["brand"])
        if tier is not None:
            return TIER_TO_SEGMENT[tier]
        # Unknown brand → fall back to price band
        return _price_to_segment(row["selling_price"])

    df["segment_class"] = df.apply(get_segment, axis=1)

    print("\nSegment distribution:")
    dist = df["segment_class"].value_counts()
    for seg, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {seg:<10} {count:>7,}  ({pct:.1f}%)")

    return df


# PHASE 5 — FEATURE ENGINEERING
# 5 derived features chosen for maximum ML value with zero extra data.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}")
    print("PHASE 5 : FEATURE ENGINEERING")
    print(DIV)

    # Base features needed by derived ones
    df["vehicle_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)

    df["km_per_year"] = np.where(
        df["vehicle_age"] > 0,
        df["odometer_reading"] / df["vehicle_age"],
        df["odometer_reading"],
    )
    df["km_per_year"] = df["km_per_year"].clip(0, 50_000).round(1)

    # DERIVED FEATURE 1: brand_tier
    # Numeric 0–4 encoding of brand prestige.
    # Gives the model an explicit depreciation-curve anchor without
    # requiring it to infer prestige purely from price patterns.
    df["brand_tier"] = df["brand"].map(BRAND_TIER_MAP).fillna(1).astype(int)
    print("  [1/5] brand_tier            — brand prestige 0–4")

    # DERIVED FEATURE 2: age_km_interaction
    # vehicle_age × odometer_reading
    # A 5-yr car with 20k km vs 5-yr car with 120k km are different
    # products. This interaction term captures that non-linearity.
    # Most impactful single addition for MAPE reduction.
    df["age_km_interaction"] = df["vehicle_age"] * df["odometer_reading"]
    print("  [2/5] age_km_interaction    — age × odometer (depreciation proxy)")

    # DERIVED FEATURE 3: ownership_trust_score
    # Non-linear penalty for owner count: 100 → 75 → 50 → 25 → 10
    # Raw owner_count treats 1→2 the same as 2→3. This curve reflects
    # that the first ownership change is the biggest value hit.
    trust_map = {1: 100, 2: 75, 3: 50, 4: 25, 5: 10, 6: 10}
    df["ownership_trust_score"] = df["owner_count"].map(trust_map).fillna(10).astype(int)
    print("  [3/5] ownership_trust_score — non-linear owner penalty 100→10")

    # DERIVED FEATURE 4: vehicle_health_score
    # Composite condition proxy from age + km + owners.
    # Gives the model one pre-computed health number instead of learning
    # all three relationships from scratch.
    # Clipped to [0, 100] — old/high-km cars don't go negative.
    df["vehicle_health_score"] = (
        100
        - (df["vehicle_age"] * 3)
        - (df["odometer_reading"] / 10_000)
        - ((df["owner_count"] - 1) * 8)
    ).clip(lower=0, upper=100).round(1)
    print("  [4/5] vehicle_health_score  — composite health 0–100")

    # DERIVED FEATURE 5: is_high_mileage
    # Binary flag for cars driven >15,000 km/year.
    # km_per_year handles mileage as a linear feature; this flag
    # explicitly marks outlier cases (e.g. the Seltos at 18k km/yr)
    # where the linear signal underestimates the risk.
    df["is_high_mileage"] = (df["km_per_year"] > 15_000).astype(int)
    print("  [5/5] is_high_mileage       — binary flag for >15k km/yr")

    high_pct = df["is_high_mileage"].mean() * 100
    print(f"\n  High-mileage vehicles : {high_pct:.1f}% of dataset")

    return df


# PHASE 6 — DEDUPLICATION

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # -- Spec-similarity analysis (REPORTING ONLY - no rows removed) ------
    _SPEC_COLS = [
        "brand", "model", "variant", "transmission",
        "fuel_type", "year", "odometer_reading", "selling_price",
    ]
    spec_dup_count = int(df.duplicated(subset=_SPEC_COLS, keep=False).sum())
    spec_groups    = before - df.drop_duplicates(subset=_SPEC_COLS).shape[0]

    print(f"\n  Spec-duplicate rows (analysis only, NOT removed): {spec_dup_count:,}")
    print("  These rows differ in locality / rto / owner_count / seller_type.")
    print(f"  Spec-duplicate groups: {spec_groups:,}")
    print("  -> Different market listings - keeping all.")

    # -- Exact duplicate removal (ALL columns must match) ------------------
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    exact_removed = before - len(df)


    # ---- Secondary dedup: true duplicates sharing same listing ---------------
    # Rows with same brand+model+year+odometer+price+locality are same car,
    # different scrape noise in other columns -> remove them
    _LISTING_COLS = [
        "brand", "model", "year", "odometer_reading", "selling_price", "locality",
    ]
    _present = [c for c in _LISTING_COLS if c in df.columns]
    before_2 = len(df)
    df = df.drop_duplicates(subset=_present, keep="first").reset_index(drop=True)
    true_dupes = before_2 - len(df)
    print(f"  True listing duplicates removed: {true_dupes:,}")
    print(f"  Final rows after both dedups   : {len(df):,}")

    print(f"\n  Original rows       : {before:,}")
    print(f"  Exact dupes removed : {exact_removed:,}  ({exact_removed/before*100:.1f}%)")
    print(f"  Final training rows : {len(df):,}")
    return df
# PHASE 7 — DISTRIBUTION REPORT

def distribution_report(df: pd.DataFrame):
    print(f"\n{DIV}")
    print("FEATURE DISTRIBUTIONS")
    print(DIV)

    for col in ["segment_class", "fuel_type", "transmission", "seller_type", "brand_tier"]:
        print(f"\n{col.upper()}")
        print(df[col].value_counts(dropna=False).to_string())
        print("-" * 40)

    print("\nSELLING PRICE (₹)")
    print(df["selling_price"].describe().apply(lambda x: f"{x:,.0f}").to_string())

    print("\nPRICE BY SEGMENT (₹ median)")
    print(
        df.groupby("segment_class")["selling_price"]
        .median()
        .sort_values()
        .apply(lambda x: f"₹{x:,.0f}")
        .to_string()
    )


# PHASE 8 — SAVE

ML_FEATURES = [
    # Identifiers
    "brand",
    "model",
    "variant",
    "color",

    # Segment
    "segment_class",

    # Vehicle specs
    "fuel_type",
    "transmission",

    # Base numeric features
    "vehicle_age",
    "odometer_reading",
    "km_per_year",
    "owner_count",

    # Derived features (the 5 additions)
    "brand_tier",
    "age_km_interaction",
    "ownership_trust_score",
    "vehicle_health_score",
    "is_high_mileage",

    # Contextual
    "seller_type",
    "inspected",

    # Target
    "selling_price",
]

ANALYSIS_COLUMNS = ["year"]


def save_dataset(df: pd.DataFrame, output_path: Path):
    output_columns = [c for c in ML_FEATURES + ANALYSIS_COLUMNS if c in df.columns]
    df = df[output_columns].copy()
    df.to_csv(output_path, index=False)

    print(f"\n{DIV}")
    print("DATASET SAVED")
    print(DIV)
    print(f"  Path    : {output_path}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")
    print(f"\n  Columns in output:")
    for c in df.columns:
        print(f"    {c}")


# PIPELINE

def process_file(name: str, input_path: Path):
    output_path = DATA_DIR / f"processed_{name}.csv"

    df = load_and_audit(input_path, name.replace("_", " ").title())
    df = rename_columns(df)
    df = drop_leakage_columns(df)
    df = clean_and_validate(df)
    df = classify_segment(df)
    df = engineer_features(df)
    df = deduplicate(df)
    distribution_report(df)
    save_dataset(df, output_path)


def main():
    print(DIV)
    print("PricerPoint Preprocessing Pipeline")
    print(datetime.now())
    print(DIV)

    for name, path in INPUT_FILES.items():
        if not path.exists():
            print(f"Missing file : {path}")
            continue
        process_file(name, path)

    print(f"\n{DIV}")
    print("PREPROCESSING COMPLETE")
    print(DIV)


if __name__ == "__main__":
    main()
