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

# =============================================================================
# CONFIGURATION
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

CURRENT_YEAR = 2026

DIV = "=" * 80

INPUT_FILES = {
    "cell7_dataset": DATA_DIR / "cell7_dataset.csv",
    "owner_assumed_dataset": DATA_DIR / "owner assumed dataset.csv",
}

# =============================================================================
# BRAND VALIDATION
# =============================================================================

OEM_ALLOWLIST = {
    "maruti",
    "maruti suzuki",
    "hyundai",
    "honda",
    "tata",
    "mahindra",
    "toyota",
    "ford",
    "renault",
    "volkswagen",
    "skoda",
    "kia",
    "mg",
    "jeep",
    "nissan",
    "datsun",
    "fiat",
    "mitsubishi",
    "chevrolet",
    "isuzu",
    "bmw",
    "mercedes-benz",
    "audi",
    "volvo",
    "mini",
    "lexus",
    "porsche",
    "jaguar",
    "land rover",
    "maserati",
    "bentley",
    "rolls-royce",
    "ferrari",
    "lamborghini",
    "aston martin",
    "hummer",
    "opel",
    "ambassador",
    "premier",
    "hindustan motors",
    "force",
    "ashok leyland",
    "icml",
    "mahindra renault",
    "mahindra ssangyong",
    "dc",
    "citroen",
    "bajaj",
}

# =============================================================================
# SEGMENT MAPPING
# =============================================================================

SEGMENT_MAP = {
    "budget": "economy",
    "mass market": "economy",
    "unknown": "economy",
    "assured": "economy",
    "standard": "premium",
    "premium": "premium",
    "luxury": "luxury",
    "luxe": "luxury",
}

VALID_FUEL = {
    "petrol",
    "diesel",
    "electric",
    "cng",
    "lpg",
    "hybrid",
    "plug-in hybrid",
    "petrol+cng",
    "petrol+lpg",
}

VALID_TRANS = {
    "manual",
    "automatic",
    "amt",
    "cvt",
    "dct",
    "imt",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _norm(value, default="unknown"):
    """
    Normalize text.
    """

    if pd.isna(value):
        return default

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    if value in {"", "nan", "none"}:
        return default

    return value


def _num(value, default=np.nan):
    """
    Safe numeric conversion.
    """

    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def _parse_owner(value):
    """
    Convert owner to integer.
    Unknown -> First Owner.
    """

    value = _norm(value)

    if value == "unknown":
        return 1

    try:
        return max(1, min(int(float(value)), 6))
    except Exception:
        return 1


def _parse_inspected(value):
    """
    Convert inspection status to binary.
    """

    return int(
        _norm(value)
        in {
            "yes",
            "true",
            "1",
            "certified",
            "inspected",
        }
    )


# =============================================================================
# PHASE 1
# LOAD DATA
# =============================================================================

def load_and_audit(path: Path, label: str):

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

        print(
            f"{col:<25}"
            f"{count:>10}"
            f" ({count/len(df)*100:.2f}%)"
        )

    return df


# =============================================================================
# PHASE 2
# RENAME COLUMNS
# =============================================================================

COLUMN_MAPPING = {

    "make": "brand_raw",

    "model": "model_raw",

    "trim": "variant_raw",

    "fuel": "fuel_raw",

    "trans": "trans_raw",

    "segment": "segment_raw",

    "seller type": "seller_type_raw",

    "certified": "certified_raw",

    "owner": "owner_raw",

    "color": "color_raw",

    "odometer": "odometer_reading",

    "price": "selling_price",

    "year": "year",

}


def rename_columns(df):

    df = df.rename(
        columns={
            k: v
            for k, v in COLUMN_MAPPING.items()
            if k in df.columns
        }
    )

    print("\nColumns renamed successfully.")

    return df


# =============================================================================
# PHASE 3
# CLEANING & VALIDATION
# =============================================================================

def clean_and_validate(df):

    print(f"\n{DIV}")
    print("PHASE 3 : CLEANING")
    print(DIV)

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------

    df["selling_price"] = pd.to_numeric(
        df["selling_price"],
        errors="coerce",
    )

    df = df[
        df["selling_price"].between(
            50_000,
            20_000_000,
        )
    ]

    # ---------------------------------------------------------
    # Year
    # ---------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df = df[
        df["year"].between(
            1990,
            CURRENT_YEAR,
        )
    ]

    df["year"] = df["year"].astype(int)

    # ---------------------------------------------------------
    # Odometer
    # ---------------------------------------------------------

    df["odometer_reading"] = pd.to_numeric(
        df["odometer_reading"],
        errors="coerce",
    )

    df = df[
        df["odometer_reading"].between(
            0,
            600000,
        )
    ]    # ---------------------------------------------------------
    # Brand
    # ---------------------------------------------------------

    df["brand"] = df["brand_raw"].apply(_norm)

    df["brand"] = df["brand"].replace({
        "maruti suzuki": "maruti suzuki",
        "mercedes benz": "mercedes-benz",
        "land-rover": "land rover",
    })

    before = len(df)

    df = df[df["brand"].isin(OEM_ALLOWLIST)]

    print(f"Brand validation : {before:,} -> {len(df):,}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    df["model"] = df["model_raw"].apply(
        lambda x: _norm(x, "unknown")
    )

    df = df[df["model"] != "unknown"]

    # ---------------------------------------------------------
    # Variant
    # ---------------------------------------------------------

    df["variant"] = df["variant_raw"].apply(
        lambda x: _norm(x, "unknown")
    )

    # ---------------------------------------------------------
    # Fuel
    # ---------------------------------------------------------

    df["fuel_type"] = df["fuel_raw"].apply(_norm)

    df["fuel_type"] = df["fuel_type"].replace({
        "petrol+cng": "cng",
        "petrol+lpg": "lpg",
        "plug-in hybrid": "hybrid",
    })

    df = df[df["fuel_type"].isin(VALID_FUEL)]

    # ---------------------------------------------------------
    # Transmission
    # ---------------------------------------------------------

    df["transmission"] = df["trans_raw"].apply(_norm)

    df["transmission"] = df["transmission"].replace({
        "amt": "automatic",
        "cvt": "automatic",
        "dct": "automatic",
        "imt": "manual",
    })

    df = df[df["transmission"].isin({"manual", "automatic"})]

    # ---------------------------------------------------------
    # Color
    # ---------------------------------------------------------

    df["color"] = df["color_raw"].apply(
        lambda x: _norm(x, "unknown")
    )

    # ---------------------------------------------------------
    # Segment
    # ---------------------------------------------------------

    df["segment_class"] = (
        df["segment_raw"]
        .apply(lambda x: _norm(x, "unknown"))
        .map(SEGMENT_MAP)
        .fillna("economy")
    )

    # ---------------------------------------------------------
    # Seller Type
    # ---------------------------------------------------------

    def seller_mapping(value):

        value = _norm(value)

        if "dealer" in value:
            return "dealer"

        if "direct" in value:
            return "dealer"

        if "individual" in value:
            return "individual"

        if "private" in value:
            return "individual"

        return "unknown"

    df["seller_type"] = df["seller_type_raw"].apply(
        seller_mapping
    )

    # ---------------------------------------------------------
    # Owner Count
    # ---------------------------------------------------------

    df["owner_count"] = df["owner_raw"].apply(
        _parse_owner
    )

    # ---------------------------------------------------------
    # Inspection
    # ---------------------------------------------------------

    df["inspected"] = df["certified_raw"].apply(
        _parse_inspected
    )

    print(f"\nRows after cleaning : {len(df):,}")

    return df


# =============================================================================
# PHASE 4
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):

    print(f"\n{DIV}")
    print("PHASE 4 : FEATURE ENGINEERING")
    print(DIV)

    df["vehicle_age"] = (
        CURRENT_YEAR - df["year"]
    ).clip(lower=0)

    df["km_per_year"] = np.where(
        df["vehicle_age"] > 0,
        df["odometer_reading"] / df["vehicle_age"],
        df["odometer_reading"],
    )

    df["km_per_year"] = (
        df["km_per_year"]
        .clip(0, 100000)
        .round(1)
    )

    print("Vehicle Age created")
    print("Average Annual Mileage created")

    return df


# =============================================================================
# PHASE 5
# DEDUPLICATION
# =============================================================================

def deduplicate(df):

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"\nDuplicates Removed : {before-len(df):,}"
    )

    print(
        f"Remaining Rows     : {len(df):,}"
    )

    return df


# =============================================================================
# PHASE 6
# FEATURE DISTRIBUTION
# =============================================================================

def distribution_report(df):

    print(f"\n{DIV}")
    print("FEATURE DISTRIBUTION")
    print(DIV)

    for column in [
        "color",
        "segment_class",
        "seller_type",
    ]:

        print(f"\n{column.upper()}")

        print(df[column].value_counts(dropna=False))

        print("-" * 50)


# =============================================================================
# PHASE 7
# SAVE DATASET
# =============================================================================

ML_FEATURES = [

    "brand",

    "model",

    "variant",

    "color",

    "segment_class",

    "fuel_type",

    "transmission",

    "vehicle_age",

    "odometer_reading",

    "km_per_year",

    "owner_count",

    "seller_type",

    "inspected",

    "selling_price",

]

ANALYSIS_COLUMNS = [

    "year",

]


def save_dataset(df, output_path):

    output_columns = ML_FEATURES + ANALYSIS_COLUMNS

    df = df[output_columns].copy()

    df.to_csv(output_path, index=False)

    print(f"\n{DIV}")
    print("DATASET SAVED")
    print(DIV)

    print(output_path)

    print(f"Rows : {len(df):,}")

    print(f"Columns : {len(df.columns)}")


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def process_file(name, input_path):

    output_path = DATA_DIR / f"processed_{name}.csv"

    df = load_and_audit(
        input_path,
        name.replace("_", " ").title(),
    )

    df = rename_columns(df)

    df = clean_and_validate(df)

    df = engineer_features(df)

    df = deduplicate(df)

    distribution_report(df)

    save_dataset(df, output_path)


def main():

    print(DIV)
    print("PriceRef Preprocessing Pipeline")
    print(datetime.now())
    print(DIV)

    for name, path in INPUT_FILES.items():

        if not path.exists():

            print(f"Missing file : {path}")

            continue

        process_file(name, path)

    print(f"\n{DIV}")
    print("PREPROCESSING COMPLETED")
    print(DIV)


if __name__ == "__main__":
    main()