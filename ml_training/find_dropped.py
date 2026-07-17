import pandas as pd
import numpy as np
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INPUT_PATH = DATA_DIR / "with owner filled.csv"
OUTPUT_PATH = DATA_DIR / "dropped_listings.csv"

# Load raw
df_raw = pd.read_csv(INPUT_PATH, low_memory=False)
df = df_raw.copy()

# Add a unique row identifier to track which row gets dropped
df["_raw_row_index"] = df.index

dropped_records = []

# Step 1: Base Numeric Outliers
# Price
price_num = pd.to_numeric(df["price"], errors="coerce")
invalid_price = ~price_num.between(50_000, 20_000_000)

# Year
year_num = pd.to_numeric(df["year"], errors="coerce")
invalid_year = ~year_num.between(1990, 2026)

# Odometer
odometer_num = pd.to_numeric(df["odometer"], errors="coerce")
invalid_odometer = ~odometer_num.between(0, 600_000)

outlier_mask = invalid_price | invalid_year | invalid_odometer
for idx in df[outlier_mask]["_raw_row_index"]:
    dropped_records.append({"row_index": idx, "reason": "Invalid Price/Year/Odometer"})
df = df[~outlier_mask].copy()

# Norm helper
def _norm(value, default="unknown"):
    if pd.isna(value):
        return default
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    if value in {"", "nan", "none"}:
        return default
    return value

OEM_ALLOWLIST = {
    "maruti suzuki", "hyundai", "tata", "renault", "honda", "mahindra", "kia", "ford", 
    "volkswagen", "skoda", "toyota", "nissan", "mg", "chevrolet", "datsun", "jeep", 
    "bmw", "audi", "fiat", "mercedes-benz", "volvo", "land rover", "citroen", "bajaj", 
    "jaguar", "mitsubishi", "mini", "lexus"
}

# Step 2: Brand validation
df["brand_norm"] = df["make"].apply(_norm).replace({
    "maruti suzuki": "maruti suzuki",
    "mercedes benz": "mercedes-benz",
    "land-rover":    "land rover",
})
invalid_brand = ~df["brand_norm"].isin(OEM_ALLOWLIST)
for idx in df[invalid_brand]["_raw_row_index"]:
    dropped_records.append({"row_index": idx, "reason": "Invalid Brand / Not in Allowlist"})
df = df[~invalid_brand].copy()

# Step 3: Model validation
df["model_norm"] = df["model"].apply(lambda x: _norm(x, "unknown"))
invalid_model = df["model_norm"] == "unknown"
for idx in df[invalid_model]["_raw_row_index"]:
    dropped_records.append({"row_index": idx, "reason": "Unknown Model"})
df = df[~invalid_model].copy()

# Step 4: Fuel / Trans validation
VALID_FUEL = {
    "petrol", "diesel", "electric", "cng", "lpg",
    "hybrid", "plug-in hybrid", "petrol+cng", "petrol+lpg",
}
df["fuel_norm"] = df["fuel"].apply(_norm).replace({
    "petrol+cng":    "cng",
    "petrol+lpg":    "lpg",
    "plug-in hybrid": "hybrid",
})
invalid_fuel = ~df["fuel_norm"].isin(VALID_FUEL)

df["trans_norm"] = df["trans"].apply(_norm).replace({
    "amt": "automatic",
    "cvt": "automatic",
    "dct": "automatic",
    "imt": "manual",
})
invalid_trans = ~df["trans_norm"].isin({"manual", "automatic"})

invalid_spec = invalid_fuel | invalid_trans
for idx in df[invalid_spec]["_raw_row_index"]:
    dropped_records.append({"row_index": idx, "reason": "Invalid Fuel or Transmission"})
df = df[~invalid_spec].copy()

# Step 5: Deduplication
# Identify duplicate records using: subset=["brand_norm", "model_norm", "year", "odometer", "price"]
# Keep the first, mark the rest as duplicates
duplicate_mask = df.duplicated(subset=["brand_norm", "model_norm", "year", "odometer", "price"], keep="first")
for idx in df[duplicate_mask]["_raw_row_index"]:
    dropped_records.append({"row_index": idx, "reason": "Duplicate Listing"})

# Create dropped dataframe
dropped_idx_to_reason = {r["row_index"]: r["reason"] for r in dropped_records}
df_dropped = df_raw.loc[df_raw.index.isin(dropped_idx_to_reason.keys())].copy()
df_dropped["drop_reason"] = df_dropped.index.map(dropped_idx_to_reason)

# Save to CSV
df_dropped.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df_dropped)} dropped rows to {OUTPUT_PATH}")

# Print summary
print("\nSummary of Dropped Listings:")
print(df_dropped["drop_reason"].value_counts())
print("\nTop 5 most frequent duplicate cars that were removed:")
dup_counts = df_dropped[df_dropped["drop_reason"] == "Duplicate Listing"].groupby(["make", "model", "year", "odometer", "price"]).size().reset_index(name="duplicate_count").sort_values(by="duplicate_count", ascending=False)
print(dup_counts.head(5).to_string(index=False))
