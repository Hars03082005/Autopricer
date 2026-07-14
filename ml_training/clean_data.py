"""
clean_data.py  —  PriceRef multi-schema dataset cleaner
Reads combined_2026.csv (pipe-delimited, ~35 MB, 7 schema variants) and writes cleaned.csv.

Detected schemas (by pipe/field count):
  7 fields  (6p)  : LISTING_ID|CATEGORY|RECEIVED|<full_model_string>|ODOMETER|FUEL|TRANS|PRICE
  8 fields  (7p)  : LISTING_ID|CATEGORY|RECEIVED|<full_model_string>|ODOMETER|FUEL|TRANS|PRICE
  11 fields (10p) : LISTING_ID|CATEGORY|RECEIVED|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|PRICE
  12 fields (11p) : LISTING_ID|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|PRICE
  14 fields (13p) : LISTING_ID|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE
  15 fields (14p) : LISTING_ID|SEGMENT|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE
  16 fields (15p) : LISTING_ID|SEGMENT|CATEGORY|INSPECTED|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE
  18 fields (17p) : + DEALER_BRANCH|DEALER_PINCODE
  19 fields (18p) : + DEALER_BRANCH|DEALER_PINCODE|OWNER_COUNT
  21 fields (20p) : + DEALER_BRANCH|DEALER_PINCODE|OWNER_COUNT|COLOR|EXTRA
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE    = Path(__file__).resolve().parent
RAW_CSV = HERE / "data" / "combined_2026.csv"
OUT_CSV = HERE / "data" / "cleaned.csv"

CURRENT_YEAR = 2026

# ── OEM allowlist ─────────────────────────────────────────────────────────────
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

VALID_FUEL  = {"petrol", "diesel", "electric", "cng", "lpg", "hybrid",
               "plug-in hybrid", "petrol+cng", "petrol+lpg"}
VALID_TRANS = {"manual", "automatic", "amt", "cvt", "dct", "imt"}

YEAR_RE   = re.compile(r"^(19[0-9]{2}|20[0-2][0-9]|2026)$")
PRICE_RE  = re.compile(r"^\d{4,10}$")
DATE_RE   = re.compile(r"^\d{4}-\d{2}-\d{2}")

BRAND_TIER_MAP: dict[str, str] = {
    **{b: "budget"  for b in {"maruti", "maruti suzuki", "datsun", "bajaj", "chevrolet",
                               "fiat", "opel", "premier", "hindustan motors", "icml",
                               "force", "ashok leyland", "ambassador"}},
    **{b: "mid"     for b in {"hyundai", "honda", "tata", "renault", "nissan", "ford",
                               "mahindra", "mahindra renault", "mahindra ssangyong",
                               "mitsubishi", "isuzu", "citroen", "dc"}},
    **{b: "premium" for b in {"volkswagen", "skoda", "toyota", "mg", "jeep", "kia",
                               "mini", "volvo", "lexus"}},
    **{b: "luxury"  for b in {"bmw", "mercedes-benz", "audi", "jaguar", "land rover",
                               "porsche", "maserati", "aston martin", "bentley",
                               "rolls-royce", "ferrari", "lamborghini", "hummer"}},
}

def _s(v: str) -> str:
    return v.strip()

def _norm(v: str, default: str = "unknown") -> str:
    s = re.sub(r"\s+", " ", v.strip().lower())
    return s if s else default

def _num(v: str) -> float:
    s = v.replace(",", "").strip()
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan

def _owner_from_category(cat: str) -> int:
    """Map category string to owner count number."""
    c = cat.strip().lower()
    for pat, n in [("s4", 4), ("s3", 3), ("s2", 2), ("s1", 1),
                   ("fourth", 4), ("third", 3), ("second", 2), ("first", 1)]:
        if pat in c:
            return n
    return 1

def _rto_state(rto: str) -> str:
    """Extract state prefix from RTO code e.g. KA-19 -> KA"""
    m = re.match(r"([A-Za-z]{2})", rto.strip())
    return m.group(1).upper() if m else "UN"

def _parse_full_model_str(full: str) -> tuple[str, str, str, str]:
    """
    Parse strings like '2016 Hyundai Grand i10 SPORTZ 1.2 KAPPA VTVT'
    Returns (year_str, make, model, trim).
    """
    full = full.strip()
    parts = full.split()
    if not parts:
        return ("", "", "", "")
    year_str = ""
    offset = 0
    if re.match(r"^(19|20)\d{2}$", parts[0]):
        year_str = parts[0]
        offset = 1
    # Try to match make (1 or 2 words) against OEM allowlist
    make, model, trim = "", "", ""
    for make_len in (2, 1):
        if offset + make_len > len(parts):
            continue
        candidate = " ".join(parts[offset:offset + make_len]).lower()
        if candidate in OEM_ALLOWLIST:
            make = candidate
            offset += make_len
            # Next word = model (could be multi-word, take rest as model+trim)
            remaining = parts[offset:]
            # Heuristic: model = first 1-2 non-uppercase words, rest = trim
            model_parts, trim_parts = [], []
            for i, w in enumerate(remaining):
                if i < 3 and not w.isupper():
                    model_parts.append(w)
                else:
                    trim_parts = remaining[i:]
                    break
            model = " ".join(model_parts).strip().lower()
            trim  = " ".join(trim_parts).strip().lower()
            break
    return (year_str, make, model, trim)


# ─── Schema parsers ──────────────────────────────────────────────────────────
# Each returns a dict with canonical keys, or None if row is invalid.

def parse_schema_7_8(p: list[str]) -> dict | None:
    """7-8 field: LISTING_ID|CATEGORY|RECEIVED|<full_model>|ODOMETER|FUEL|TRANS|PRICE[|LIST_PRICE]"""
    if len(p) < 8:
        return None
    n = len(p)
    year_str, make, model, trim = _parse_full_model_str(p[3])
    return dict(
        listing_id = _s(p[0]),
        category   = _s(p[1]),
        received   = _s(p[2]),
        city       = "unknown",
        year_str   = year_str,
        make       = make,
        model      = model,
        trim       = trim,
        odometer   = _s(p[4]),
        fuel       = _s(p[5]),
        trans      = _s(p[6]),
        rto        = "",
        price      = _s(p[7]),
        list_price = _s(p[8]) if n > 8 else "",
        segment    = "",
        inspected  = "",
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_11(p: list[str]) -> dict | None:
    """11 field: LISTING_ID|CATEGORY|RECEIVED|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|PRICE"""
    return dict(
        listing_id = _s(p[0]),
        category   = _s(p[1]),
        received   = _s(p[2]),
        city       = "unknown",
        year_str   = _s(p[3]),
        make       = _norm(p[4]),
        model      = _norm(p[5]),
        trim       = _norm(p[6]),
        odometer   = _s(p[7]),
        fuel       = _s(p[8]),
        trans      = _s(p[9]),
        rto        = "",
        price      = _s(p[10]),
        list_price = "",
        segment    = "",
        inspected  = "",
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_12(p: list[str]) -> dict | None:
    """12 field: LISTING_ID|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|PRICE"""
    return dict(
        listing_id = _s(p[0]),
        category   = _s(p[1]),
        received   = _s(p[2]),
        city       = _norm(p[3]),
        year_str   = _s(p[4]),
        make       = _norm(p[5]),
        model      = _norm(p[6]),
        trim       = _norm(p[7]),
        odometer   = _s(p[8]),
        fuel       = _s(p[9]),
        trans      = _s(p[10]),
        rto        = "",
        price      = _s(p[11]),
        list_price = "",
        segment    = "",
        inspected  = "",
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_14(p: list[str]) -> dict | None:
    """14 field: LISTING_ID|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE"""
    return dict(
        listing_id = _s(p[0]),
        category   = _s(p[1]),
        received   = _s(p[2]),
        city       = _norm(p[3]),
        year_str   = _s(p[4]),
        make       = _norm(p[5]),
        model      = _norm(p[6]),
        trim       = _norm(p[7]),
        odometer   = _s(p[8]),
        fuel       = _s(p[9]),
        trans      = _s(p[10]),
        rto        = _s(p[11]),
        price      = _s(p[12]),
        list_price = _s(p[13]),
        segment    = "",
        inspected  = "",
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_15(p: list[str]) -> dict | None:
    """15 field: LISTING_ID|SEGMENT|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE"""
    return dict(
        listing_id = _s(p[0]),
        segment    = _norm(p[1]),
        category   = _s(p[2]),
        received   = _s(p[3]),
        city       = _norm(p[4]),
        year_str   = _s(p[5]),
        make       = _norm(p[6]),
        model      = _norm(p[7]),
        trim       = _norm(p[8]),
        odometer   = _s(p[9]),
        fuel       = _s(p[10]),
        trans      = _s(p[11]),
        rto        = _s(p[12]),
        price      = _s(p[13]),
        list_price = _s(p[14]),
        inspected  = "",
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_16(p: list[str]) -> dict | None:
    """16 field: LISTING_ID|SEGMENT|CATEGORY|INSPECTED|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE"""
    return dict(
        listing_id = _s(p[0]),
        segment    = _norm(p[1]),
        category   = _s(p[2]),
        inspected  = _norm(p[3]),
        received   = _s(p[4]),
        city       = _norm(p[5]),
        year_str   = _s(p[6]),
        make       = _norm(p[7]),
        model      = _norm(p[8]),
        trim       = _norm(p[9]),
        odometer   = _s(p[10]),
        fuel       = _s(p[11]),
        trans      = _s(p[12]),
        rto        = _s(p[13]),
        price      = _s(p[14]),
        list_price = _s(p[15]),
        branch     = "",
        pincode    = "",
        owner_str  = "",
        color      = "",
    )

def parse_schema_18(p: list[str]) -> dict | None:
    """18 field: ...16... + BRANCH|PINCODE"""
    d = parse_schema_16(p)
    if d:
        d["branch"]  = _norm(p[16])
        d["pincode"] = _s(p[17])
    return d

def parse_schema_19(p: list[str]) -> dict | None:
    """19 field: ...18... + OWNER_COUNT"""
    d = parse_schema_18(p)
    if d:
        d["owner_str"] = _s(p[18])
    return d

def parse_schema_21(p: list[str]) -> dict | None:
    """21 field: ...19... + COLOR|EXTRA"""
    d = parse_schema_19(p)
    if d:
        d["color"] = _norm(p[19])
    return d

def parse_schema_23(p: list[str]) -> dict | None:
    """23 field: LISTING_ID|CATEGORY|RECEIVED|CITY|YEAR|MAKE|MODEL|TRIM|ODOMETER|FUEL|TRANS|RTO|PRICE|LIST_PRICE|SEGMENT|SELLER_TYPE|CERTIFIED|LOCALITY|PINCODE|OWNER|COLOR|DRIVE|YEAR_MAKE_MODEL_TRIM"""
    return dict(
        listing_id = _s(p[0]),
        category   = _s(p[1]),
        received   = _s(p[2]),
        city       = _norm(p[3]),
        year_str   = _s(p[4]),
        make       = _norm(p[5]),
        model      = _norm(p[6]),
        trim       = _norm(p[7]),
        odometer   = _s(p[8]),
        fuel       = _s(p[9]),
        trans      = _s(p[10]),
        rto        = _s(p[11]),
        price      = _s(p[12]),
        list_price = _s(p[13]),
        segment    = _norm(p[14]),
        inspected  = _norm(p[16]),  # CERTIFIED
        branch     = _norm(p[17]),  # LOCALITY
        pincode    = _s(p[18]),
        owner_str  = _s(p[19]),
        color      = _norm(p[20]),
    )

# Keys = number of FIELDS (len(parts)), not pipe count
SCHEMA_MAP = {
    7:  parse_schema_7_8,   # 6 pipes
    8:  parse_schema_7_8,   # 7 pipes
    11: parse_schema_11,    # 10 pipes
    12: parse_schema_12,    # 11 pipes
    14: parse_schema_14,    # 13 pipes
    15: parse_schema_15,    # 14 pipes
    16: parse_schema_16,    # 15 pipes
    18: parse_schema_18,    # 17 pipes
    19: parse_schema_19,    # 18 pipes
    21: parse_schema_21,    # 20 pipes
    23: parse_schema_23,    # 22 pipes
}



def parse_row(line: str) -> dict | None:
    parts = line.rstrip("\r\n").split("|")
    n = len(parts)
    parser = SCHEMA_MAP.get(n)
    if parser is None:
        return None
    return parser(parts)


def to_record(d: dict) -> dict | None:
    """Validate and convert parsed dict to clean feature record. Returns None if invalid."""
    # Year
    year = _num(d.get("year_str", ""))
    if np.isnan(year) or not (1990 <= year <= CURRENT_YEAR):
        return None
    year = int(year)

    # Make
    make = d.get("make", "").strip().lower()
    if make not in OEM_ALLOWLIST:
        return None

    # Fuel / Transmission
    fuel = _norm(d.get("fuel", ""), "petrol")
    trans = _norm(d.get("trans", ""), "manual")
    if fuel not in VALID_FUEL or trans not in VALID_TRANS:
        return None

    # Price / Odometer
    price = _num(d.get("price", ""))
    odo   = _num(d.get("odometer", ""))
    if np.isnan(price) or np.isnan(odo):
        return None
    if not (50_000 <= price <= 20_000_000):
        return None
    if not (100 <= odo <= 600_000):
        return None

    # List price
    list_price = _num(d.get("list_price", ""))

    # Owner count — try explicit owner_str first, fall back to category
    owner_str = d.get("owner_str", "").strip()
    if owner_str and owner_str.isdigit():
        owner_count = max(1, min(int(owner_str), 6))
    else:
        owner_count = _owner_from_category(d.get("category", "s1"))

    # Received date → listing month/year for seasonality
    received = d.get("received", "")
    try:
        rd = pd.Timestamp(received)
        listing_month = rd.month
        listing_year  = rd.year
    except Exception:
        listing_month = 0
        listing_year  = 0

    # RTO state
    rto      = d.get("rto", "").strip()
    rto_state = _rto_state(rto) if rto else "UN"

    # Derived features
    vehicle_age = max(0, CURRENT_YEAR - year)
    km_per_year = odo / max(vehicle_age, 0.5)
    km_per_year = min(km_per_year, 100_000)

    ownership_trust = (
        (1 / owner_count) * 0.5 +
        (1 - min(vehicle_age / 35, 1.0)) * 0.3 +
        (1 - min(odo / 600_000, 1.0)) * 0.2
    )
    vehicle_health = (
        (1 - min(odo / 600_000, 1.0)) * 0.5 +
        (1 - min(vehicle_age / 35, 1.0)) * 0.3 +
        (1 / owner_count) * 0.2
    )

    brand_tier = BRAND_TIER_MAP.get(make, "mid")

    model   = d.get("model", "").strip().lower() or "unknown"
    variant = d.get("trim", "").strip().lower()  or "unknown"
    city    = d.get("city", "").strip().lower()  or "unknown"
    segment = d.get("segment", "").strip().lower() or "unknown"
    color   = d.get("color", "").strip().lower()   or "unknown"
    branch  = d.get("branch", "").strip().lower()  or "unknown"
    inspected = 1 if d.get("inspected", "").strip().lower() in ("yes", "true", "1") else 0

    dep_ratio = round(price / list_price, 4) if (not np.isnan(list_price) and list_price > 0) else np.nan

    return {
        # Identity
        "brand":               make,
        "model":               model,
        "variant":             variant,
        # Time
        "year":                year,
        "vehicle_age":         vehicle_age,
        "listing_month":       listing_month,
        "listing_year":        listing_year,
        # Vehicle specs
        "fuel_type":           fuel,
        "transmission":        trans,
        "color":               color,
        # Usage
        "odometer_reading":    odo,
        "km_per_year":         round(km_per_year, 2),
        "owner_count":         owner_count,
        # Scores
        "ownership_trust_score": round(ownership_trust, 4),
        "vehicle_health_score":  round(vehicle_health, 4),
        # Location
        "city":                city,
        "rto_state":           rto_state,
        "branch":              branch,
        "pincode":             d.get("pincode", "").strip() or "unknown",
        # Metadata
        "segment":             segment,
        "inspected":           inspected,
        "brand_tier":          brand_tier,
        # Placeholders (trainer will median-impute from data)
        "fuel_efficiency":     np.nan,
        "engine_cc":           np.nan,
        # Target
        "selling_price":       price,
        "list_price":          list_price,
        "depreciation_ratio":  dep_ratio,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
print(f"Reading {RAW_CSV} …")

records = []
skipped_no_parser = 0
skipped_invalid   = 0

with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
    next(f)  # skip header
    for line in f:
        d = parse_row(line)
        if d is None:
            skipped_no_parser += 1
            continue
        rec = to_record(d)
        if rec is None:
            skipped_invalid += 1
            continue
        records.append(rec)

print(f"  Skipped (unknown schema): {skipped_no_parser:,}")
print(f"  Skipped (invalid values): {skipped_invalid:,}")
print(f"  Valid records:            {len(records):,}")

df = pd.DataFrame(records)

# Median-impute fuel_efficiency & engine_cc (not in raw data)
df["fuel_efficiency"] = df["fuel_efficiency"].fillna(15.0)
df["engine_cc"]       = df["engine_cc"].fillna(1200.0)

# Impute depreciation_ratio with median where list_price was missing
dep_med = df["depreciation_ratio"].median()
df["depreciation_ratio"] = df["depreciation_ratio"].fillna(dep_med)

df.reset_index(drop=True).to_csv(OUT_CSV, index=False)
print(f"\n✅  Saved {len(df):,} rows → {OUT_CSV}")

print("\n--- Column summary ---")
print(df.dtypes.to_string())
print("\n--- Numeric summary ---")
print(df.describe().T[["count", "mean", "min", "50%", "max"]].to_string())
print("\n--- Categorical value counts ---")
for col in ["brand_tier", "fuel_type", "transmission", "inspected", "segment", "color"]:
    if col in df.columns:
        print(f"\n{col}:\n{df[col].value_counts().head(8).to_string()}")
