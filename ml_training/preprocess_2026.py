
"""
preprocess_2026.py
==============================================================================
Production-grade preprocessing pipeline for PricerPoint used-car dataset.
Input  : ml_training/data/combined_2026.csv  (pipe-delimited, multi-schema)
Outputs:
  ml_training/data/cleaned_used_car_dataset.csv           <- full dataset
  ml_training/data/cleaned_used_car_dataset_no_listprice.csv

Schema detection by field count (see module docstring for full mapping).

Phases:
  1  - Initial Dataset Audit
  2  - Remove Invalid Records
  3  - Drop Unusable Features
  4  - Missing Value Handling
  5  - Data Standardization
  6  - Remove Duplicates
  7  - Feature Engineering
  8  - Outlier Detection (report only)
  9  - Preprocessing Summary
  10 - Final Dataset + Save
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE         = Path(__file__).resolve().parent
RAW_CSV      = HERE / "data" / "combined_2026.csv"
OUT_CSV      = HERE / "data" / "cleaned_used_car_dataset.csv"
OUT_CSV_NLP  = HERE / "data" / "cleaned_used_car_dataset_no_listprice.csv"
CURRENT_YEAR = datetime.now().year   # 2026

LUXURY_BRANDS = {
    "Bmw", "Mercedes-Benz", "Audi", "Jaguar",
    "Volvo", "Lexus", "Porsche", "Land Rover",
}

DIV = "=" * 72

# ---------------------------------------------------------------------------
# Raw parsing helpers
# ---------------------------------------------------------------------------

def _parse_full_model_str(full: str):
    """
    Extract (year, make, model, trim) from concatenated model strings
    such as '2016 Hyundai Grand i10 SPORTZ 1.2'.
    Returns (None, None, None, None) for empty input.
    """
    full = full.strip()
    parts = full.split()
    if not parts:
        return None, None, None, None
    year, offset = None, 0
    if re.match(r"^(19|20)\d{2}$", parts[0]):
        year = parts[0]
        offset = 1
    make, model, trim = None, None, None
    for make_len in (2, 1):
        if offset + make_len > len(parts):
            continue
        make = " ".join(parts[offset:offset + make_len])
        offset += make_len
        remaining = parts[offset:]
        model_parts, trim_parts = [], []
        for i, w in enumerate(remaining):
            if i < 3 and not w.isupper():
                model_parts.append(w)
            else:
                trim_parts = remaining[i:]
                break
        model = " ".join(model_parts).strip() or None
        trim  = " ".join(trim_parts).strip()  or None
        break
    return year, make, model, trim


def parse_raw_csv(path: Path) -> pd.DataFrame:
    """
    Parse the pipe-delimited combined_2026.csv with all schema variants into
    a single flat DataFrame with canonical column names.

    This function is reusable: call it on any future snapshot file with the
    same schema variants to get a consistently structured DataFrame.

    Canonical columns:
        LISTING_ID, CATEGORY, RECEIVED, CITY, YEAR, MAKE, MODEL, TRIM,
        ODOMETER, FUEL, TRANS, RTO, PRICE, LIST_PRICE,
        SEGMENT, CERTIFIED, LOCALITY, PINCODE, OWNER, COLOR,
        DRIVE, SELLER_TYPE, S2_YEAR_MAKE_MODEL_TRIM
    """
    NaN = np.nan

    EMPTY = {
        "LISTING_ID":            NaN,
        "CATEGORY":              NaN,
        "RECEIVED":              NaN,
        "CITY":                  NaN,
        "YEAR":                  NaN,
        "MAKE":                  NaN,
        "MODEL":                 NaN,
        "TRIM":                  NaN,
        "ODOMETER":              NaN,
        "FUEL":                  NaN,
        "TRANS":                 NaN,
        "RTO":                   NaN,
        "PRICE":                 NaN,
        "LIST_PRICE":            NaN,
        "SEGMENT":               NaN,
        "CERTIFIED":             NaN,
        "LOCALITY":              NaN,
        "PINCODE":               NaN,
        "OWNER":                 NaN,
        "COLOR":                 NaN,
        "DRIVE":                 NaN,
        "SELLER_TYPE":           NaN,
        "S2_YEAR_MAKE_MODEL_TRIM": NaN,
    }

    def opt(p, idx):
        """Return stripped field, or None if blank / index OOB."""
        if idx >= len(p):
            return None
        v = p[idx].strip()
        return v if v else None

    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        next(fh)  # skip header
        for line in fh:
            p = line.rstrip("\r\n").split("|")
            n = len(p)
            r = dict(EMPTY)  # fresh copy per row

            if n == 6:
                # RECEIVED | <YEAR MAKE MODEL TRIM> | ODO | FUEL | TRANS | PRICE
                r["RECEIVED"]                = opt(p, 0)
                r["S2_YEAR_MAKE_MODEL_TRIM"] = opt(p, 1)
                r["ODOMETER"]                = opt(p, 2)
                r["FUEL"]                    = opt(p, 3)
                r["TRANS"]                   = opt(p, 4)
                r["PRICE"]                   = opt(p, 5)
                yr, mk, md, tr = _parse_full_model_str(p[1])
                r["YEAR"], r["MAKE"], r["MODEL"], r["TRIM"] = yr, mk, md, tr

            elif n == 7:
                # LISTING_ID | ODO | FUEL | TRANS | RTO | PRICE | LIST_PRICE
                r["LISTING_ID"] = opt(p, 0)
                r["ODOMETER"]   = opt(p, 1)
                r["FUEL"]       = opt(p, 2)
                r["TRANS"]      = opt(p, 3)
                r["RTO"]        = opt(p, 4)
                r["PRICE"]      = opt(p, 5)
                r["LIST_PRICE"] = opt(p, 6)

            elif n == 8:
                # 8A: p[4] is a valid year (no price column)
                # 8B: p[3] is a full model string (has price in p[7])
                is_8a = False
                try:
                    yr_val = int(p[4].strip())
                    if 2000 <= yr_val <= CURRENT_YEAR:
                        is_8a = True
                except (ValueError, IndexError):
                    pass
                if is_8a:
                    r["LISTING_ID"] = opt(p, 0)
                    r["CATEGORY"]   = opt(p, 1)
                    r["RECEIVED"]   = opt(p, 2)
                    r["CITY"]       = opt(p, 3)
                    r["YEAR"]       = opt(p, 4)
                    r["MAKE"]       = opt(p, 5)
                    r["MODEL"]      = opt(p, 6)
                    r["TRIM"]       = opt(p, 7)
                else:
                    r["LISTING_ID"]              = opt(p, 0)
                    r["CATEGORY"]                = opt(p, 1)
                    r["RECEIVED"]                = opt(p, 2)
                    r["S2_YEAR_MAKE_MODEL_TRIM"] = opt(p, 3)
                    r["ODOMETER"]                = opt(p, 4)
                    r["FUEL"]                    = opt(p, 5)
                    r["TRANS"]                   = opt(p, 6)
                    r["PRICE"]                   = opt(p, 7)
                    yr, mk, md, tr = _parse_full_model_str(p[3])
                    r["YEAR"], r["MAKE"], r["MODEL"], r["TRIM"] = yr, mk, md, tr

            elif n == 11:
                r["LISTING_ID"] = opt(p, 0)
                r["CATEGORY"]   = opt(p, 1)
                r["RECEIVED"]   = opt(p, 2)
                r["YEAR"]       = opt(p, 3)
                r["MAKE"]       = opt(p, 4)
                r["MODEL"]      = opt(p, 5)
                r["TRIM"]       = opt(p, 6)
                r["ODOMETER"]   = opt(p, 7)
                r["FUEL"]       = opt(p, 8)
                r["TRANS"]      = opt(p, 9)
                r["PRICE"]      = opt(p, 10)

            elif n == 12:
                r["LISTING_ID"] = opt(p, 0)
                r["CATEGORY"]   = opt(p, 1)
                r["RECEIVED"]   = opt(p, 2)
                r["CITY"]       = opt(p, 3)
                r["YEAR"]       = opt(p, 4)
                r["MAKE"]       = opt(p, 5)
                r["MODEL"]      = opt(p, 6)
                r["TRIM"]       = opt(p, 7)
                r["ODOMETER"]   = opt(p, 8)
                r["FUEL"]       = opt(p, 9)
                r["TRANS"]      = opt(p, 10)
                r["PRICE"]      = opt(p, 11)

            elif n == 14:
                r["LISTING_ID"] = opt(p, 0)
                r["CATEGORY"]   = opt(p, 1)
                r["RECEIVED"]   = opt(p, 2)
                r["CITY"]       = opt(p, 3)
                r["YEAR"]       = opt(p, 4)
                r["MAKE"]       = opt(p, 5)
                r["MODEL"]      = opt(p, 6)
                r["TRIM"]       = opt(p, 7)
                r["ODOMETER"]   = opt(p, 8)
                r["FUEL"]       = opt(p, 9)
                r["TRANS"]      = opt(p, 10)
                r["RTO"]        = opt(p, 11)
                r["PRICE"]      = opt(p, 12)
                r["LIST_PRICE"] = opt(p, 13)

            elif n == 15:
                r["LISTING_ID"] = opt(p, 0)
                r["SEGMENT"]    = opt(p, 1)
                r["CATEGORY"]   = opt(p, 2)
                r["RECEIVED"]   = opt(p, 3)
                r["CITY"]       = opt(p, 4)
                r["YEAR"]       = opt(p, 5)
                r["MAKE"]       = opt(p, 6)
                r["MODEL"]      = opt(p, 7)
                r["TRIM"]       = opt(p, 8)
                r["ODOMETER"]   = opt(p, 9)
                r["FUEL"]       = opt(p, 10)
                r["TRANS"]      = opt(p, 11)
                r["RTO"]        = opt(p, 12)
                r["PRICE"]      = opt(p, 13)
                r["LIST_PRICE"] = opt(p, 14)

            elif n == 16:
                r["LISTING_ID"] = opt(p, 0)
                r["SEGMENT"]    = opt(p, 1)
                r["CATEGORY"]   = opt(p, 2)
                r["CERTIFIED"]  = opt(p, 3)
                r["RECEIVED"]   = opt(p, 4)
                r["CITY"]       = opt(p, 5)
                r["YEAR"]       = opt(p, 6)
                r["MAKE"]       = opt(p, 7)
                r["MODEL"]      = opt(p, 8)
                r["TRIM"]       = opt(p, 9)
                r["ODOMETER"]   = opt(p, 10)
                r["FUEL"]       = opt(p, 11)
                r["TRANS"]      = opt(p, 12)
                r["RTO"]        = opt(p, 13)
                r["PRICE"]      = opt(p, 14)
                r["LIST_PRICE"] = opt(p, 15)

            elif n == 18:
                r["LISTING_ID"] = opt(p, 0)
                r["SEGMENT"]    = opt(p, 1)
                r["CATEGORY"]   = opt(p, 2)
                r["CERTIFIED"]  = opt(p, 3)
                r["RECEIVED"]   = opt(p, 4)
                r["CITY"]       = opt(p, 5)
                r["YEAR"]       = opt(p, 6)
                r["MAKE"]       = opt(p, 7)
                r["MODEL"]      = opt(p, 8)
                r["TRIM"]       = opt(p, 9)
                r["ODOMETER"]   = opt(p, 10)
                r["FUEL"]       = opt(p, 11)
                r["TRANS"]      = opt(p, 12)
                r["RTO"]        = opt(p, 13)
                r["PRICE"]      = opt(p, 14)
                r["LIST_PRICE"] = opt(p, 15)
                r["LOCALITY"]   = opt(p, 16)
                r["PINCODE"]    = opt(p, 17)

            elif n == 19:
                r["LISTING_ID"] = opt(p, 0)
                r["SEGMENT"]    = opt(p, 1)
                r["CATEGORY"]   = opt(p, 2)
                r["CERTIFIED"]  = opt(p, 3)
                r["RECEIVED"]   = opt(p, 4)
                r["CITY"]       = opt(p, 5)
                r["YEAR"]       = opt(p, 6)
                r["MAKE"]       = opt(p, 7)
                r["MODEL"]      = opt(p, 8)
                r["TRIM"]       = opt(p, 9)
                r["ODOMETER"]   = opt(p, 10)
                r["FUEL"]       = opt(p, 11)
                r["TRANS"]      = opt(p, 12)
                r["RTO"]        = opt(p, 13)
                r["PRICE"]      = opt(p, 14)
                r["LIST_PRICE"] = opt(p, 15)
                r["LOCALITY"]   = opt(p, 16)
                r["PINCODE"]    = opt(p, 17)
                r["OWNER"]      = opt(p, 18)

            elif n == 21:
                # idx 2 = SELLER_TYPE (most complete schema variant)
                r["LISTING_ID"]  = opt(p, 0)
                r["SEGMENT"]     = opt(p, 1)
                r["SELLER_TYPE"] = opt(p, 2)
                r["CERTIFIED"]   = opt(p, 3)
                r["RECEIVED"]    = opt(p, 4)
                r["CITY"]        = opt(p, 5)
                r["YEAR"]        = opt(p, 6)
                r["MAKE"]        = opt(p, 7)
                r["MODEL"]       = opt(p, 8)
                r["TRIM"]        = opt(p, 9)
                r["ODOMETER"]    = opt(p, 10)
                r["FUEL"]        = opt(p, 11)
                r["TRANS"]       = opt(p, 12)
                r["RTO"]         = opt(p, 13)
                r["PRICE"]       = opt(p, 14)
                r["LIST_PRICE"]  = opt(p, 15)
                r["LOCALITY"]    = opt(p, 16)
                r["PINCODE"]     = opt(p, 17)
                r["OWNER"]       = opt(p, 18)
                r["COLOR"]       = opt(p, 19)
                r["DRIVE"]       = opt(p, 20)

            elif n == 23:
                r["LISTING_ID"]              = opt(p, 0)
                r["CATEGORY"]                = opt(p, 1)
                r["RECEIVED"]                = opt(p, 2)
                r["CITY"]                    = opt(p, 3)
                r["YEAR"]                    = opt(p, 4)
                r["MAKE"]                    = opt(p, 5)
                r["MODEL"]                   = opt(p, 6)
                r["TRIM"]                    = opt(p, 7)
                r["ODOMETER"]                = opt(p, 8)
                r["FUEL"]                    = opt(p, 9)
                r["TRANS"]                   = opt(p, 10)
                r["RTO"]                     = opt(p, 11)
                r["PRICE"]                   = opt(p, 12)
                r["LIST_PRICE"]              = opt(p, 13)
                r["SEGMENT"]                 = opt(p, 14)
                r["SELLER_TYPE"]             = opt(p, 15)
                r["CERTIFIED"]               = opt(p, 16)
                r["LOCALITY"]                = opt(p, 17)
                r["PINCODE"]                 = opt(p, 18)
                r["OWNER"]                   = opt(p, 19)
                r["COLOR"]                   = opt(p, 20)
                r["DRIVE"]                   = opt(p, 21)
                r["S2_YEAR_MAKE_MODEL_TRIM"] = opt(p, 22)

            # Any other field counts are unknown schema variants - skip
            records.append(r)

    df = pd.DataFrame(records)
    for col in ("YEAR", "PRICE", "ODOMETER", "LIST_PRICE", "OWNER"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ===========================================================================
# PHASE 1 -- Initial Dataset Audit
# ===========================================================================

def phase1_audit(df: pd.DataFrame) -> None:
    """
    Print a comprehensive pre-modification audit report.
    This function does NOT modify the dataframe.
    """
    print(DIV)
    print("PHASE 1 -- INITIAL DATASET AUDIT")
    print(DIV)

    print(f"\n[1a] Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("[1b] Column data types:")
    print(df.dtypes.to_string())

    dup_rows  = int(df.duplicated().sum())
    lid       = df["LISTING_ID"].dropna()
    dup_lids  = int(lid.duplicated().sum())
    lid_nulls = int(df["LISTING_ID"].isna().sum())
    print(f"\n[1c] Duplicate rows (full row match): {dup_rows:,}")
    print(f"[1d] Duplicate LISTING_IDs          : {dup_lids:,}  (null LISTING_IDs: {lid_nulls:,})")
    print("     NOTE: Same listing re-appears across multiple daily snapshot files.")
    print("           Full-row deduplication in Phase 6 resolves this correctly.")

    print("\n[1e] Invalid value audit (pre-cleaning):")
    yr_null = int(df["YEAR"].isna().sum())
    yr_low  = int((df["YEAR"] < 2000).sum())
    yr_high = int((df["YEAR"] > CURRENT_YEAR).sum())
    print(f"  YEAR     -- missing: {yr_null:,} | < 2000: {yr_low:,} | > {CURRENT_YEAR}: {yr_high:,}")

    pr_null = int(df["PRICE"].isna().sum())
    pr_low  = int((df["PRICE"] < 50_000).sum())
    pr_high = int((df["PRICE"] > 20_000_000).sum())
    print(f"  PRICE    -- missing: {pr_null:,} | < Rs 50,000: {pr_low:,} | > Rs 2 Cr: {pr_high:,}")

    od_null = int(df["ODOMETER"].isna().sum())
    od_low  = int((df["ODOMETER"] < 500).sum())
    od_high = int((df["ODOMETER"] > 400_000).sum())
    print(f"  ODOMETER -- missing: {od_null:,} | < 500 km: {od_low:,} | > 4,00,000 km: {od_high:,}")

    print("\n[1f] Missing value counts per column:")
    null_counts = df.isna().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({"Missing": null_counts, "Missing_%": null_pct})
    null_df = null_df[null_df["Missing"] > 0].sort_values("Missing", ascending=False)
    print(null_df.to_string())
    print()


# ===========================================================================
# PHASE 2 -- Remove Invalid Records
# ===========================================================================

def phase2_remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with physically impossible values.
    Rows where these columns are NaN are NOT touched (Phase 4 handles nulls).

    Constraints applied:
        PRICE    : < 50,000  OR  > 20,000,000 (Rs 2 crore)
        YEAR     : < 2000    OR  > CURRENT_YEAR
        ODOMETER : < 500     OR  > 400,000 km
    """
    print(DIV)
    print("PHASE 2 -- REMOVE INVALID RECORDS")
    print(DIV)

    mask = (
        (df["PRICE"].notna()    & ((df["PRICE"]    < 50_000)     | (df["PRICE"]    > 20_000_000))) |
        (df["YEAR"].notna()     & ((df["YEAR"]     < 2000)       | (df["YEAR"]     > CURRENT_YEAR))) |
        (df["ODOMETER"].notna() & ((df["ODOMETER"] < 500)        | (df["ODOMETER"] > 400_000)))
    )
    removed = int(mask.sum())
    df = df[~mask].reset_index(drop=True)

    print(f"\n  Rows removed (impossible values) : {removed:,}")
    print(f"  Rows remaining                   : {len(df):,}\n")
    return df


# ===========================================================================
# PHASE 3 -- Drop Unusable Features
# ===========================================================================

def phase3_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop four structurally unusable columns:
      DRIVE                  -- ~66% missing; insufficient coverage
      CATEGORY               -- internal dealership tag; not a vehicle attribute
      S2_YEAR_MAKE_MODEL_TRIM-- raw concat string; already parsed into columns
      LISTING_ID             -- identifier only; no predictive signal
    """
    print(DIV)
    print("PHASE 3 -- DROP UNUSABLE FEATURES")
    print(DIV)

    cols_to_drop = ["DRIVE", "CATEGORY", "S2_YEAR_MAKE_MODEL_TRIM", "LISTING_ID"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    print(f"\n  Dropped  : {cols_to_drop}")
    print(f"  New shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    return df


# ===========================================================================
# PHASE 4 -- Missing Value Handling
# ===========================================================================

def phase4_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column missing value strategy:

    Drop rows where:
        PRICE, YEAR, ODOMETER, MAKE, MODEL, FUEL, TRANS are null.

    Fill + create flag columns:
        OWNER      -> fill 'Unknown', add OWNER_MISSING (1/0)
        COLOR      -> fill 'Unknown', add COLOR_MISSING (1/0)
        LIST_PRICE -> leave NaN,      add HAS_LIST_PRICE (1/0)

    Fill only:
        LOCALITY   -> 'Unknown'
        CERTIFIED  -> 'False'
        CITY       -> 'Unknown'
        RTO        -> 'Unknown'
        SELLER_TYPE-> mode
        SEGMENT    -> mode

    Drop column:
        PINCODE    -> redundant with CITY + RTO
    """
    print(DIV)
    print("PHASE 4 -- MISSING VALUE HANDLING")
    print(DIV)
    initial = len(df)

    # Drop rows on critical columns
    critical = ["PRICE", "YEAR", "ODOMETER", "MAKE", "MODEL", "FUEL", "TRANS"]
    for col in critical:
        if col in df.columns:
            before = len(df)
            df     = df.dropna(subset=[col])
            after  = len(df)
            if before > after:
                print(f"  Dropped {before - after:>7,} rows -- missing {col}")

    rows_removed = initial - len(df)
    print(f"\n  Total rows dropped in Phase 4: {rows_removed:,}")
    print(f"  Rows remaining               : {len(df):,}\n")

    # OWNER: fill + flag
    df["OWNER_MISSING"] = df["OWNER"].isna().astype(int)
    df["OWNER"]         = df["OWNER"].fillna("Unknown")

    # COLOR: fill + flag
    df["COLOR_MISSING"] = df["COLOR"].isna().astype(int)
    df["COLOR"]         = df["COLOR"].fillna("Unknown")

    # LIST_PRICE: flag only, leave NaN intact
    df["HAS_LIST_PRICE"] = df["LIST_PRICE"].notna().astype(int)

    # LOCALITY
    df["LOCALITY"]  = df["LOCALITY"].fillna("Unknown")

    # CERTIFIED
    df["CERTIFIED"] = df["CERTIFIED"].fillna("False")

    # PINCODE -- drop column
    if "PINCODE" in df.columns:
        df = df.drop(columns=["PINCODE"])
        print("  PINCODE dropped (redundant with CITY + RTO).")

    # CITY
    df["CITY"] = df["CITY"].fillna("Unknown")

    # RTO
    df["RTO"] = df["RTO"].fillna("Unknown")

    # SELLER_TYPE -- mode imputation
    if "SELLER_TYPE" in df.columns:
        mode_val = df["SELLER_TYPE"].mode(dropna=True)
        if len(mode_val) > 0:
            df["SELLER_TYPE"] = df["SELLER_TYPE"].fillna(mode_val.iloc[0])
            print(f"  SELLER_TYPE null -> mode ('{mode_val.iloc[0]}')")

    # SEGMENT -- mode imputation
    if "SEGMENT" in df.columns:
        mode_val = df["SEGMENT"].mode(dropna=True)
        if len(mode_val) > 0:
            df["SEGMENT"] = df["SEGMENT"].fillna(mode_val.iloc[0])
            print(f"  SEGMENT null -> mode ('{mode_val.iloc[0]}')")

    print(f"\n  Shape after Phase 4: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    return df


# ===========================================================================
# PHASE 5 -- Data Standardization
# ===========================================================================

def _title_case_strip(series: pd.Series) -> pd.Series:
    """Trim whitespace and apply Title Case to a string Series."""
    return series.astype(str).str.strip().str.title()


def phase5_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categorical string columns:
      - Trim leading/trailing whitespace
      - Apply Title Case

    This resolves: petrol/PETROL/Petrol -> Petrol
                   manual/MANUAL        -> Manual
                   dealer/Dealer        -> Dealer
    """
    print(DIV)
    print("PHASE 5 -- DATA STANDARDIZATION")
    print(DIV)

    cat_cols = [
        "MAKE", "MODEL", "TRIM", "FUEL", "TRANS",
        "CITY", "RTO", "COLOR", "SEGMENT", "LOCALITY",
        "CERTIFIED", "SELLER_TYPE",
    ]
    applied = [c for c in cat_cols if c in df.columns]
    for col in applied:
        df[col] = _title_case_strip(df[col])

    print(f"\n  Title Case applied to {len(applied)} columns: {applied}\n")
    return df


# ===========================================================================
# PHASE 6 -- Remove Duplicates
# ===========================================================================

def phase6_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicate rows, keeping the first occurrence.
    The same vehicle listing re-appears in multiple daily snapshot files;
    full-row deduplication is the correct strategy here.
    """
    print(DIV)
    print("PHASE 6 -- REMOVE DUPLICATE LISTINGS")
    print(DIV)

    before  = len(df)
    df      = df.drop_duplicates(keep="first").reset_index(drop=True)
    after   = len(df)
    removed = before - after

    print(f"\n  Rows removed   : {removed:,}")
    print(f"  Rows remaining : {after:,}\n")
    return df


# ===========================================================================
# PHASE 7 -- Feature Engineering
# ===========================================================================

def phase7_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new predictive features:

    Vehicle_Age           = CURRENT_YEAR - YEAR
    Annual_Mileage        = ODOMETER / Vehicle_Age  (div-by-zero safe)
    Negotiation_Margin    = LIST_PRICE - PRICE       (NaN where LIST_PRICE absent)
    Negotiation_Percentage= Margin / LIST_PRICE      (NaN where LIST_PRICE absent)
    High_Mileage          = 1 if ODOMETER > 75th percentile of ODOMETER
    Luxury_Brand          = 1 if MAKE is in the luxury brand set
    """
    print(DIV)
    print("PHASE 7 -- FEATURE ENGINEERING")
    print(DIV)

    # Vehicle_Age
    df["Vehicle_Age"] = (CURRENT_YEAR - df["YEAR"]).clip(lower=0)

    # Annual_Mileage -- age=0 (brand new) -> use odometer directly
    df["Annual_Mileage"] = np.where(
        df["Vehicle_Age"] > 0,
        df["ODOMETER"] / df["Vehicle_Age"],
        df["ODOMETER"],
    ).round(2)

    # Negotiation features -- only where LIST_PRICE available
    df["Negotiation_Margin"] = np.where(
        df["LIST_PRICE"].notna(),
        df["LIST_PRICE"] - df["PRICE"],
        np.nan,
    )
    df["Negotiation_Percentage"] = np.where(
        df["LIST_PRICE"].notna() & (df["LIST_PRICE"] > 0),
        (df["LIST_PRICE"] - df["PRICE"]) / df["LIST_PRICE"],
        np.nan,
    )

    # High_Mileage -- binary flag at 75th percentile
    odo_q75 = df["ODOMETER"].quantile(0.75)
    df["High_Mileage"] = (df["ODOMETER"] > odo_q75).astype(int)

    # Luxury_Brand -- binary flag
    luxury_lower = {b.lower() for b in LUXURY_BRANDS}
    df["Luxury_Brand"] = df["MAKE"].str.strip().str.lower().isin(luxury_lower).astype(int)

    engineered = [
        "Vehicle_Age", "Annual_Mileage",
        "Negotiation_Margin", "Negotiation_Percentage",
        "High_Mileage", "Luxury_Brand",
    ]
    print(f"\n  Engineered features ({len(engineered)}):")
    for feat in engineered:
        non_null = df[feat].notna().sum()
        print(f"    {feat:<30} -> {non_null:,} non-null values")
    print(f"\n  ODOMETER 75th percentile (High_Mileage threshold): {odo_q75:,.0f} km\n")
    return df


# ===========================================================================
# PHASE 8 -- Outlier Detection (IQR, report only)
# ===========================================================================

def _iqr_stats(series: pd.Series) -> tuple:
    """Compute (q1, q3, iqr, lower_fence, upper_fence, outlier_count)."""
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower    = q1 - 1.5 * iqr
    upper    = q3 + 1.5 * iqr
    outliers = int(((series < lower) | (series > upper)).sum())
    return q1, q3, iqr, lower, upper, outliers


def phase8_outlier_report(df: pd.DataFrame) -> None:
    """
    Detect and report outliers using Tukey's IQR x 1.5 rule.
    NO rows are removed; this report informs downstream model decisions.
    """
    print(DIV)
    print("PHASE 8 -- OUTLIER DETECTION REPORT  (no rows removed)")
    print(DIV)

    targets = ["PRICE", "ODOMETER", "Vehicle_Age"]
    w = 13
    header = (f"  {'Feature':<18} {'Q1':>{w}} {'Q3':>{w}} {'IQR':>{w}} "
              f"{'Lower':>{w}} {'Upper':>{w}} {'Outliers':>{w}} {'%':>8}")
    print(f"\n{header}")
    print("  " + "-" * 107)

    for col in targets:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3, iqr, lower, upper, n_out = _iqr_stats(s)
        pct = n_out / len(s) * 100
        print(f"  {col:<18} {q1:>{w},.0f} {q3:>{w},.0f} {iqr:>{w},.0f} "
              f"{lower:>{w},.0f} {upper:>{w},.0f} {n_out:>{w},} {pct:>7.2f}%")

    print(
        "\n  RECOMMENDATION: Outliers detected but NOT removed.\n"
        "  Consider IQR capping or log-transform on PRICE/ODOMETER before training.\n"
    )


# ===========================================================================
# PHASE 9 -- Preprocessing Summary Table
# ===========================================================================

def phase9_summary() -> None:
    """
    Print a complete feature-level action and rationale table covering
    every original column and every engineered feature.
    """
    print(DIV)
    print("PHASE 9 -- PREPROCESSING SUMMARY TABLE")
    print(DIV)

    rows = [
        # (Feature,                    Action Taken,                               Reason)
        ("LISTING_ID",                "Dropped",                                  "Identifier only; no predictive signal"),
        ("CATEGORY",                  "Dropped",                                  "Internal dealership tag; not a vehicle attribute"),
        ("S2_YEAR_MAKE_MODEL_TRIM",   "Dropped",                                  "Raw concat string; fields already parsed"),
        ("DRIVE",                     "Dropped",                                  "~66% missing; insufficient coverage"),
        ("PINCODE",                   "Column dropped",                           "Redundant with CITY + RTO"),
        ("RECEIVED",                  "Retained as-is",                           "Timestamp; usable for seasonality if needed"),
        ("YEAR",                      "Rows dropped if null/out-of-range",        "Critical numeric predictor"),
        ("MAKE",                      "Rows dropped if null / Title Case",        "Core categorical predictor"),
        ("MODEL",                     "Rows dropped if null / Title Case",        "Core categorical predictor"),
        ("TRIM",                      "Retained / Title Case",                    "Variant-level detail; highly predictive"),
        ("FUEL",                      "Rows dropped if null / Title Case",        "Required feature"),
        ("TRANS",                     "Rows dropped if null / Title Case",        "Required feature"),
        ("ODOMETER",                  "Rows dropped if null/out-of-range",        "Critical numeric predictor"),
        ("PRICE",                     "Rows dropped if null/invalid (TARGET)",    "Target variable"),
        ("LIST_PRICE",                "Null left as NaN + HAS_LIST_PRICE flag",   "Imputing would introduce bias"),
        ("CITY",                      "Null -> 'Unknown' / Title Case",           "Geographic signal; safe placeholder"),
        ("RTO",                       "Null -> 'Unknown'",                        "State-level geographic signal"),
        ("LOCALITY",                  "Null -> 'Unknown'",                        "Sub-city detail; safe placeholder"),
        ("SEGMENT",                   "Null -> mode / Title Case",               "Mode imputation low-risk for segment"),
        ("CERTIFIED",                 "Null -> 'False' / Title Case",            "Absent certification implies not certified"),
        ("OWNER",                     "Null -> 'Unknown' + OWNER_MISSING flag",  "Flag preserves information about absence"),
        ("COLOR",                     "Null -> 'Unknown' + COLOR_MISSING flag",  "Color predictive; flag captures missingness"),
        ("SELLER_TYPE",               "Null -> mode / Title Case",               "Seller context; mode imputation safe"),
        # Engineered
        ("Vehicle_Age",               "Engineered: CURRENT_YEAR - YEAR",         "Captures depreciation directly"),
        ("Annual_Mileage",            "Engineered: ODOMETER / Vehicle_Age",      "Usage intensity; more informative than raw odometer"),
        ("Negotiation_Margin",        "Engineered: LIST_PRICE - PRICE",          "Absolute dealer discount signal"),
        ("Negotiation_Percentage",    "Engineered: Margin / LIST_PRICE",         "Relative discount rate (normalised)"),
        ("High_Mileage",              "Engineered: ODOMETER > Q75",              "Binary flag for high-usage vehicles"),
        ("Luxury_Brand",              "Engineered: MAKE in luxury set",          "Brand-class price premium flag"),
        ("OWNER_MISSING",             "Engineered: OWNER was null",              "Missingness pattern may itself predict"),
        ("COLOR_MISSING",             "Engineered: COLOR was null",              "Missingness pattern may itself predict"),
        ("HAS_LIST_PRICE",            "Engineered: LIST_PRICE not null",         "Listing type correlates with price disclosure"),
    ]

    print(f"\n  {'Feature':<30}  {'Action Taken':<44}  Reason")
    print("  " + "-" * 132)
    for feat, action, reason in rows:
        print(f"  {feat:<30}  {action:<44}  {reason}")
    print()


# ===========================================================================
# PHASE 10 -- Final Dataset Report + Save
# ===========================================================================

def phase10_final_report_and_save(df: pd.DataFrame) -> None:
    """
    Print final statistics and save two CSV outputs:
      1. cleaned_used_car_dataset.csv           -- full dataset with LIST_PRICE
      2. cleaned_used_car_dataset_no_listprice.csv -- without LIST_PRICE and negotiation cols
    """
    print(DIV)
    print("PHASE 10 -- FINAL DATASET")
    print(DIV)

    engineered = [
        "Vehicle_Age", "Annual_Mileage",
        "Negotiation_Margin", "Negotiation_Percentage",
        "High_Mileage", "Luxury_Brand",
        "OWNER_MISSING", "COLOR_MISSING", "HAS_LIST_PRICE",
    ]
    n_eng = sum(1 for f in engineered if f in df.columns)

    print(f"\n  Final shape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Engineered features : {n_eng}")
    print(f"  Memory usage        : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Remaining missing values
    null_rem = df.isna().sum()
    null_rem = null_rem[null_rem > 0]
    if null_rem.empty:
        print("\n  Remaining missing values: none")
    else:
        print("\n  Remaining missing values (intentional NaNs only):")
        for col, cnt in null_rem.items():
            pct = cnt / len(df) * 100
            print(f"    {col:<32} {cnt:>9,}  ({pct:.2f}%)")

    print("\n  Final column data types:")
    print(df.dtypes.to_string())

    # Save 1: with LIST_PRICE
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved (with list-price)    -> {OUT_CSV}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Save 2: without LIST_PRICE
    drop_lp = [c for c in ["LIST_PRICE", "Negotiation_Margin", "Negotiation_Percentage"]
               if c in df.columns]
    df_nolp = df.drop(columns=drop_lp)
    df_nolp.to_csv(OUT_CSV_NLP, index=False)
    print(f"\n  Saved (without list-price) -> {OUT_CSV_NLP}")
    print(f"  Shape: {df_nolp.shape[0]:,} rows x {df_nolp.shape[1]} columns")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print(DIV)
    print("PricerPoint -- Used Car Dataset Preprocessing Pipeline (2026)")
    print(f"Input  : {RAW_CSV}")
    print(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIV)

    print(f"\nParsing raw data from {RAW_CSV} ...")
    df = parse_raw_csv(RAW_CSV)
    print(f"Parsed {len(df):,} rows from raw CSV.\n")

    phase1_audit(df)
    df = phase2_remove_invalid(df)
    df = phase3_drop_features(df)
    df = phase4_missing_values(df)
    df = phase5_standardize(df)
    df = phase6_dedup(df)
    df = phase7_feature_engineering(df)
    phase8_outlier_report(df)
    phase9_summary()
    phase10_final_report_and_save(df)

    print(DIV)
    print("Pipeline complete.")
    print(DIV)


if __name__ == "__main__":
    main()

