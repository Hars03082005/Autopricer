"""
ml_training/clean-s5.py
S5 Quality Shop Data Preprocessing Pipeline

Sources : ml_training/data/s5.csv       (153 rows, has 'year' column)
          ml_training/data/s5_overall.csv (199 rows, has 'age' column)
Output  : ml_training/data/processed_s5.csv
          ml_training/data/processed_s5_report.json

Data profile (from analysis):
- Combined 352 rows, peak age 1-7 years
- 93% of records are age <= 7 years
- Price range: ₹3.14L - ₹51.44L, Median ₹11.89L
- All Petrol or Diesel, Manual or Automatic
- Brands: Hyundai, Kia, VW, Maruti, Skoda, MG, Tata, Mahindra, Jeep, Toyota, Honda, Ford, Audi, BMW, Mercedes-Benz, Renault
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
DATA_DIR     = HERE / "data"
CURRENT_YEAR = datetime.now().year
SCRIPT_NAME  = "clean-s5"
DIV          = "=" * 72

# S5 eligible age threshold (learned from data: 93% of records age <= 7)
S5_MAX_AGE   = 7

# ── COLUMN MAPS ──────────────────────────────────────────────────────────────
# s5.csv uses 'year'; s5_overall.csv uses 'age'
# Both share: segment, seller type, certified, city, make, model, trim,
#             odometer, fuel, trans, rto, selling price, owner, color

PRICE_MIN, PRICE_MAX = 50_000, 20_000_000
ODO_MIN,   ODO_MAX   = 0, 500_000
AGE_MIN,   AGE_MAX   = 0, 15

# ── BRAND NORMALIZATION ────────────────────────────────────────────────────────
BRAND_ALIAS = {
    "maruti":          "maruti suzuki",
    "maruti-suzuki":   "maruti suzuki",
    "marutisuzuki":    "maruti suzuki",
    "mercedes":        "mercedes-benz",
    "mercedes benz":   "mercedes-benz",
    "land-rover":      "land rover",
    "landrover":       "land rover",
    "vw":              "volkswagen",
    "volkswagon":      "volkswagen",
}

OEM_ALLOWLIST = {
    "maruti suzuki", "hyundai", "kia", "volkswagen", "skoda", "mg",
    "tata", "mahindra", "jeep", "toyota", "honda", "ford",
    "renault", "nissan", "bmw", "audi", "mercedes-benz", "volvo",
    "land rover", "jaguar", "mini", "lexus", "porsche",
    "citroen", "mitsubishi", "isuzu",
}

FUEL_ALIAS = {
    "petrol": "petrol",
    "diesel": "diesel",
    "electric": "electric",
    "ev": "electric",
    "cng": "cng",
    "hybrid": "hybrid",
    "mild hybrid": "petrol",
    "strong hybrid": "petrol",
}
VALID_FUEL = {"petrol", "diesel", "electric", "cng", "hybrid"}

TRANS_ALIAS = {
    "manual":    "manual",
    "automatic": "automatic",
    "auto":      "automatic",
    "amt":       "automatic",
    "cvt":       "automatic",
    "dct":       "automatic",
    "dsg":       "automatic",
    "imt":       "manual",
}
VALID_TRANS = {"manual", "automatic"}

DIV = "=" * 72


def _clean_str(v, default="unknown"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return str(v).strip().lower()


def _norm_brand(raw: str) -> str:
    b = _clean_str(raw)
    b = BRAND_ALIAS.get(b, b)
    return b if b in OEM_ALLOWLIST else b


def _norm_fuel(raw: str) -> str:
    f = _clean_str(raw)
    return FUEL_ALIAS.get(f, f if f in VALID_FUEL else "petrol")


def _norm_trans(raw: str) -> str:
    t = _clean_str(raw)
    return TRANS_ALIAS.get(t, t if t in VALID_TRANS else "manual")


def _norm_color(raw: str) -> str:
    c = _clean_str(raw, "unknown")
    map_ = {
        "white": "white", "silver": "silver", "grey": "grey", "gray": "grey",
        "black": "black", "blue": "blue", "navy blue": "blue", "royal blue": "blue",
        "red": "red", "maroon": "maroon", "brown": "brown",
        "beige": "beige", "cream": "beige", "champagne": "beige",
        "gold": "gold", "golden": "gold",
        "green": "green", "olive": "green",
        "orange": "orange", "yellow": "yellow",
        "purple": "purple", "violet": "purple",
    }
    return map_.get(c, "unknown")


def _norm_city(raw: str) -> str:
    c = _clean_str(raw, "unknown")
    aliases = {
        "bengaluru": "bangalore", "bengalooru": "bangalore", "bengalore": "bangalore",
        "new delhi": "delhi", "ncr": "delhi", "gurgaon": "delhi",
    }
    return aliases.get(c, c)


def _norm_rto(raw: str) -> str:
    r = _clean_str(raw, "unknown")
    r = re.sub(r"[^a-z0-9\-]", "", r)
    return r if len(r) >= 4 else "unknown"


def load_and_merge() -> pd.DataFrame:
    """Load s5.csv + s5_overall.csv, unify columns, compute age."""
    s5a = pd.read_csv(DATA_DIR / "s5.csv")
    s5b = pd.read_csv(DATA_DIR / "s5_overall.csv")

    # ── s5.csv: has 'year', no 'age' ─────────────────────────────────
    s5a["age"] = CURRENT_YEAR - pd.to_numeric(s5a["year"], errors="coerce").fillna(CURRENT_YEAR - 3)

    # ── s5_overall.csv: has 'age', no 'year' ─────────────────────────
    s5b["year"] = CURRENT_YEAR - pd.to_numeric(s5b["age"], errors="coerce").fillna(3).astype(int)

    combined = pd.concat([s5a, s5b], ignore_index=True)
    print(f"  Loaded: s5.csv={len(s5a)}, s5_overall.csv={len(s5b)}, combined={len(combined)}")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    stats = {"input_rows": len(df)}
    df = df.copy()

    # ── Rename to canonical column names ────────────────────────────
    col_map = {
        "make":          "brand",
        "model":         "model",
        "trim":          "variant",
        "fuel":          "fuel_type",
        "trans":         "transmission",
        "odometer":      "odometer_reading",
        "selling price": "selling_price",
        "owner":         "owner_count",
        "seller type":   "seller_type",
        "certified":     "certified",
        "city":          "city",
        "rto":           "rto",
        "color":         "color",
        "age":           "vehicle_age",
        "year":          "year",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Keep only needed columns ─────────────────────────────────────
    keep = ["brand", "model", "variant", "fuel_type", "transmission",
            "odometer_reading", "selling_price", "owner_count", "seller_type",
            "certified", "city", "rto", "color", "vehicle_age", "year"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # ── Numeric coercion ─────────────────────────────────────────────
    df["selling_price"]   = pd.to_numeric(df.get("selling_price"), errors="coerce")
    df["odometer_reading"]= pd.to_numeric(df.get("odometer_reading"), errors="coerce").fillna(0)
    df["owner_count"]     = pd.to_numeric(df.get("owner_count"), errors="coerce").fillna(1).clip(1, 5)
    df["vehicle_age"]     = pd.to_numeric(df.get("vehicle_age"), errors="coerce")
    df["year"]            = pd.to_numeric(df.get("year"), errors="coerce")

    # ── Recompute vehicle_age from year where missing ────────────────
    mask_age_missing = df["vehicle_age"].isna()
    df.loc[mask_age_missing, "vehicle_age"] = CURRENT_YEAR - df.loc[mask_age_missing, "year"].fillna(CURRENT_YEAR - 3)

    # ── Drop rows with missing price ─────────────────────────────────
    before = len(df)
    df.dropna(subset=["selling_price"], inplace=True)
    stats["dropped_no_price"] = before - len(df)

    # ── Price range filter ───────────────────────────────────────────
    before = len(df)
    df = df[(df["selling_price"] >= PRICE_MIN) & (df["selling_price"] <= PRICE_MAX)]
    stats["dropped_price_range"] = before - len(df)

    # ── Age filter (keep only S5-eligible records) ───────────────────
    before = len(df)
    df = df[(df["vehicle_age"] >= AGE_MIN) & (df["vehicle_age"] <= S5_MAX_AGE)]
    stats["dropped_age_range"] = before - len(df)

    # ── Odometer filter ──────────────────────────────────────────────
    df["odometer_reading"] = df["odometer_reading"].clip(ODO_MIN, ODO_MAX)

    # ── String normalization ─────────────────────────────────────────
    df["brand"]        = df["brand"].apply(_norm_brand)
    df["model"]        = df["model"].apply(lambda v: _clean_str(v, "unknown"))
    df["variant"]      = df["variant"].apply(lambda v: _clean_str(v, "unknown"))
    df["fuel_type"]    = df["fuel_type"].apply(_norm_fuel)
    df["transmission"] = df["transmission"].apply(_norm_trans)
    df["color"]        = df["color"].apply(_norm_color)
    df["city"]         = df["city"].apply(_norm_city)
    df["rto"]          = df["rto"].apply(_norm_rto) if "rto" in df.columns else "unknown"
    df["seller_type"]  = df.get("seller_type", pd.Series("s5", index=df.index)).apply(
                            lambda v: _clean_str(v, "s5"))
    df["certified"]    = df.get("certified", pd.Series("no", index=df.index)).apply(
                            lambda v: 0 if _clean_str(str(v)) in ("no", "false", "0", "nan", "") else 1)

    # ── Drop rows with unknown brand ─────────────────────────────────
    before = len(df)
    df = df[df["brand"] != "unknown"]
    stats["dropped_unknown_brand"] = before - len(df)

    # ── Engineered features ──────────────────────────────────────────
    df["vehicle_age"]   = df["vehicle_age"].fillna(3).astype(int)
    df["km_per_year"]   = np.where(
        df["vehicle_age"] > 0,
        df["odometer_reading"] / df["vehicle_age"],
        df["odometer_reading"],
    ).clip(0, 80_000)
    df["owner_count"]   = df["owner_count"].astype(int)

    # ── Log-price target ─────────────────────────────────────────────
    df["log_selling_price"] = np.log1p(df["selling_price"])

    # ── Deduplicate ──────────────────────────────────────────────────
    before = len(df)
    dup_cols = ["brand", "model", "variant", "vehicle_age", "odometer_reading",
                "fuel_type", "transmission", "selling_price"]
    df.drop_duplicates(subset=[c for c in dup_cols if c in df.columns], inplace=True)
    stats["dropped_duplicates"] = before - len(df)

    stats["output_rows"] = len(df)
    return df, stats


def main():
    t0 = time.time()
    print(DIV)
    print(f"  {SCRIPT_NAME} — S5 Quality Shop Preprocessing")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIV)

    print("\n[1/3] Loading datasets...")
    raw = load_and_merge()

    print("\n[2/3] Cleaning & normalizing...")
    df, stats = clean(raw)

    print(f"  Input rows   : {stats['input_rows']}")
    print(f"  No price     : -{stats.get('dropped_no_price', 0)}")
    print(f"  Price range  : -{stats.get('dropped_price_range', 0)}")
    print(f"  Age range    : -{stats.get('dropped_age_range', 0)} (kept age <= {S5_MAX_AGE})")
    print(f"  Unknown brand: -{stats.get('dropped_unknown_brand', 0)}")
    print(f"  Duplicates   : -{stats.get('dropped_duplicates', 0)}")
    print(f"  Output rows  : {stats['output_rows']}")

    print("\n[3/3] Age distribution in cleaned data:")
    age_dist = df["vehicle_age"].value_counts().sort_index()
    for age, cnt in age_dist.items():
        bar = "█" * int(cnt * 20 / age_dist.max())
        print(f"    Age {int(age):2d}: {cnt:3d} rows  {bar}")

    out_csv  = DATA_DIR / "processed_s5.csv"
    out_json = DATA_DIR / "processed_s5_report.json"

    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")

    report = {
        "script": SCRIPT_NAME,
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "age_distribution": age_dist.to_dict(),
        "price_stats": {
            "min": float(df["selling_price"].min()),
            "max": float(df["selling_price"].max()),
            "median": float(df["selling_price"].median()),
            "mean": float(df["selling_price"].mean()),
        },
        "brand_counts": df["brand"].value_counts().to_dict(),
        "s5_max_age_used": S5_MAX_AGE,
        "features": list(df.columns),
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report: {out_json}")

    elapsed = time.time() - t0
    print(f"\n{DIV}")
    print(f"  Done in {elapsed:.1f}s — {stats['output_rows']} clean rows ready for S5 training.")
    print(DIV)


if __name__ == "__main__":
    main()
