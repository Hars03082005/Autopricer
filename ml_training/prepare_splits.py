"""
ml_training/prepare_splits.py
AutoPricer — Dataset Preparation & Stratified Split Pipeline

Produces three processed + split dataset families:

  Dataset A  →  s1-s4_owner-filled.csv + s5_overall.csv  (merged)
  Dataset B  →  overall.csv            + s5_overall.csv  (merged)
  Dataset C  →  overall.csv                              (as-is)

For each dataset:
  1.  Merge sources (if applicable), deduplicate
  2.  Preprocess (same pipeline as clean-1.py)
  3.  Stratified train / valid / test split  (70 / 15 / 15)
      Stratification key: segment bucket
      Guarantees all buckets present in every split.

Outputs per dataset (written to ml_training/data/splits/<name>/):
  train.csv
  valid.csv
  test.csv
  split_report.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
DATA_DIR     = HERE / "data"
SPLITS_DIR   = DATA_DIR / "splits"
CURRENT_YEAR = datetime.now().year
DIV          = "=" * 80

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── SEGMENT BUCKET MAP ────────────────────────────────────────────────────────
SEGMENT_MAP = {
    "mass market": "mass_market",
    "massmarket":  "mass_market",
    "standard":    "standard",
    "luxury":      "luxury",
    "assured":     "assured",
    "budget":      "budget",
    "luxe":        "luxe",
    "premium":     "luxury",
    "s5":          "mass_market",
}

# ── PREPROCESSING CONSTANTS ───────────────────────────────────────────────────
PRICE_MIN, PRICE_MAX = 50_000, 20_000_000
AGE_MIN,   AGE_MAX   = 0, 30
ODO_MIN,   ODO_MAX   = 0, 600_000

OEM_ALLOWLIST = {
    "maruti suzuki", "hyundai", "tata", "renault", "honda",
    "mahindra", "kia", "ford", "volkswagen", "skoda", "toyota",
    "nissan", "mg", "chevrolet", "datsun", "jeep", "bmw", "audi",
    "fiat", "mercedes-benz", "volvo", "land rover", "citroen",
    "bajaj", "jaguar", "mitsubishi", "mini", "lexus", "isuzu",
    "porsche", "maserati", "lamborghini", "ferrari", "rolls-royce",
    "bentley", "aston martin",
}

BRAND_ALIAS = {
    "mercedes benz":  "mercedes-benz",
    "mercedes-benz":  "mercedes-benz",
    "merc":           "mercedes-benz",
    "land-rover":     "land rover",
    "landrover":      "land rover",
    "maruti":         "maruti suzuki",
    "suzuki":         "maruti suzuki",
    "marutisuzuki":   "maruti suzuki",
    "maruti-suzuki":  "maruti suzuki",
    "vw":             "volkswagen",
    "volkswagon":     "volkswagen",
    "tata motors":    "tata",
    "hyundai motor":  "hyundai",
    "honda cars":     "honda",
    "kia motors":     "kia",
    "general motors": "chevrolet",
    "chevy":          "chevrolet",
    "rolls royce":    "rolls-royce",
    "rollsroyce":     "rolls-royce",
    "aston-martin":   "aston martin",
}

VALID_FUEL = {"petrol", "diesel", "electric", "cng", "lpg", "hybrid"}
FUEL_ALIAS = {
    "petrol+cng":       "cng",   "cng+petrol":       "cng",
    "petrol + cng":     "cng",   "petrol+lpg":       "lpg",
    "lpg+petrol":       "lpg",   "petrol + lpg":     "lpg",
    "plug-in hybrid":   "hybrid","plugin hybrid":    "hybrid",
    "mild hybrid":      "hybrid","full hybrid":      "hybrid",
    "strong hybrid":    "hybrid","phev":             "hybrid",
    "ev":               "electric","bev":            "electric",
    "battery electric": "electric",
}

VALID_TRANS = {"manual", "automatic"}
TRANS_ALIAS = {
    "amt": "automatic", "cvt": "automatic", "dct": "automatic",
    "dsg": "automatic", "torque converter": "automatic",
    "imt": "manual",    "mt":  "manual",
    "at":  "automatic", "auto":"automatic",
}

VALID_COLORS = {"white", "black", "silver", "grey", "red", "blue",
                "brown", "orange", "green", "beige", "yellow", "purple"}

COLOR_ALIAS = {
    "pearl white":"white",  "ivory white":"white",   "solid white":"white",
    "arctic white":"white", "oxford white":"white",  "star white":"white",
    "mineral white":"white","celestial white":"white",
    "jet black":"black",    "midnight black":"black", "phantom black":"black",
    "carbon black":"black",
    "sangria red":"red",    "fiesta red":"red",       "crimson red":"red",
    "passion red":"red",    "lava red":"red",         "ember red":"red",
    "wine":"red",           "maroon":"red",           "burgundy":"red",
    "titan grey":"grey",    "typhoon grey":"grey",    "charcoal":"grey",
    "galaxy grey":"grey",   "sterling grey":"grey",   "ash":"grey",
    "granite":"grey",       "gray":"grey",
    "metallic silver":"silver","star dust":"silver",  "lunar silver":"silver",
    "stardust silver":"silver","silver metallic":"silver",
    "stargazing blue":"blue","sapphire blue":"blue",  "navy blue":"blue",
    "cobalt blue":"blue",   "electric blue":"blue",   "ocean blue":"blue",
    "sky blue":"blue",      "royal blue":"blue",      "dark blue":"blue",
    "techno blue":"blue",   "aqua blue":"blue",       "denim blue":"blue",
    "cerulean blue":"blue", "fiery blue":"blue",      "icy blue":"blue",
    "canyon orange":"orange","lava orange":"orange",  "amber orange":"orange",
    "champion yellow":"yellow","solar yellow":"yellow","acid yellow":"yellow",
    "sun kissed yellow":"yellow","golden":"yellow",   "gold":"yellow",
    "champagne":"beige",    "cream":"beige",          "ivory":"beige",
    "sand":"beige",
    "bronze":"brown",       "copper":"brown",         "chocolate":"brown",
    "chestnut":"brown",
    "lavender":"purple",    "violet":"purple",
    "forest green":"green", "mint":"green",           "olive":"green",
    "dark green":"green",   "emerald":"green",        "moss green":"green",
}

_DEALER_KEYWORDS     = {"dealer", "direct", "s1", "s2", "s3", "s4", "showroom", "authorized"}
_INDIVIDUAL_KEYWORDS = {"individual", "private", "owner", "person"}

LOCALITY_ALIAS: dict[str, str] = {
    "nexus whitefield":          "whitefield",
    "nexus whitefield mall":     "whitefield",
    "bhoruka tech park":         "whitefield",
    "itpl":                      "whitefield",
    "prestige shantiniketan":    "whitefield",
    "prestige lakeside habitat": "whitefield",
    "varthur":                   "whitefield",
    "thubarahalli":              "whitefield",
    "brookefield":               "whitefield",
    "mahadevapura":              "whitefield",
    "j p nagar":                 "jp nagar",
    "jp nagar 6th phase":        "jp nagar",
    "yeswanthpur":               "yeshwanthpur",
    "yeshwantpura":              "yeshwanthpur",
    "yeshwanthpura":             "yeshwanthpur",
    "naagarabhaavi":             "nagarbhavi",
    "nagarabhavi":               "nagarbhavi",
    "electronic city phase 1":   "electronic city",
    "electronic city phase 2":   "electronic city",
    "electronic city phase i":   "electronic city",
    "electronic city phase ii":  "electronic city",
    "btm 1st stage":             "btm layout",
    "btm 2nd stage":             "btm layout",
    "1st stage btm":             "btm layout",
    "marathalli":                "marathahalli",
    "bannerghatta road":         "bannerghatta",
    "bannerghatta main road":    "bannerghatta",
    "koramangala 1st block":     "koramangala",
    "koramangala 5th block":     "koramangala",
    "koramangala 6th block":     "koramangala",
    "koramangala 7th block":     "koramangala",
    "koramangala 8th block":     "koramangala",
    "bellahalli":                "hebbal",
    "hunasamaranahalli":         "hebbal",
    "kogilu":                    "hebbal",
    "indira nagar":              "indiranagar",
    "rajaji nagar":              "rajajinagar",
    "basavana gudi":             "basavanagudi",
    "vega city mall":            "bannerghatta",
    "mantri commercio":          "hebbal",
    "gt world mall":             "hebbal",
    "krishnarajapuram":          "kr puram",
    "rajarajeshwari nagar":      "mysore road",
    "subramanyapura":            "mysore road",
    "nagasandra":                "peenya",
    "vidyaranyapura":            "yelahanka",
    "singasandra":               "begur",
    "akshayanagar":              "begur",
    "laggere":                   "rajajinagar",
    "jigani":                    "electronic city",
    "mysore":                    "mysuru",
}

_KEYWORD_MAP = [
    ("nexus whitefield",      "whitefield"),   ("shriram wytfield",    "whitefield"),
    ("bhoruka tech park",     "whitefield"),   ("prestige shantiniketan","whitefield"),
    ("prestige lakeside",     "whitefield"),   ("thubarahalli",         "whitefield"),
    ("brookefield",           "whitefield"),   ("whitefield",           "whitefield"),
    ("varthur",               "whitefield"),   ("itpl",                 "whitefield"),
    ("mahadevapura",          "whitefield"),
    ("jp nagar",              "jp nagar"),     ("j p nagar",            "jp nagar"),
    ("jigani",                "electronic city"),("electronic city",    "electronic city"),
    ("koramangala",           "koramangala"),  ("indiranagar",          "indiranagar"),
    ("indira nagar",          "indiranagar"),  ("marathahalli",         "marathahalli"),
    ("bellandur",             "bellandur"),    ("yeshwanthpur",         "yeshwanthpur"),
    ("yeswanthpur",           "yeshwanthpur"), ("malleshwaram",         "malleshwaram"),
    ("rajajinagar",           "rajajinagar"),  ("rajaji nagar",         "rajajinagar"),
    ("hebbal",                "hebbal"),       ("bellahalli",           "hebbal"),
    ("hunasamaranahalli",     "hebbal"),       ("kogilu",               "hebbal"),
    ("mantri commercio",      "hebbal"),       ("gt world mall",        "hebbal"),
    ("bannerghatta",          "bannerghatta"), ("vega city mall",       "bannerghatta"),
    ("jayanagar",             "jayanagar"),    ("sadashivanagar",       "sadashivanagar"),
    ("basavanagudi",          "basavanagudi"), ("basavana gudi",        "basavanagudi"),
    ("yelahanka",             "yelahanka"),    ("vidyaranyapura",       "yelahanka"),
    ("jakkur",                "yelahanka"),    ("devanahalli",          "yelahanka"),
    ("horamavu",              "horamavu"),     ("nagarbhavi",           "nagarbhavi"),
    ("nagarabhavi",           "nagarbhavi"),   ("naagarabhaavi",        "nagarbhavi"),
    ("hsr layout",            "hsr layout"),   ("btm layout",           "btm layout"),
    ("btm ",                  "btm layout"),   ("bommanahalli",         "bommanahalli"),
    ("singasandra",           "begur"),        ("akshayanagar",         "begur"),
    ("begur",                 "begur"),        ("anekal",               "anekal"),
    ("peenya",                "peenya"),       ("nagasandra",           "peenya"),
    ("kengeri",               "kengeri"),      ("mysore road",          "mysore road"),
    ("rajarajeshwari nagar",  "mysore road"),  ("subramanyapura",       "mysore road"),
    ("kr puram",              "kr puram"),     ("krishnarajapuram",     "kr puram"),
    ("ramamurthy nagar",      "ramamurthy nagar"),
    ("vijayanagar",           "vijayanagar"),  ("gandhi nagar",         "gandhi nagar"),
    ("laggere",               "rajajinagar"),  ("gorguntepalya",        "gorguntepalya"),
    ("hosur road",            "hosur road"),   ("hosur main road",      "hosur road"),
    ("hennur",                "hennur"),       ("kalyan nagar",         "kalyan nagar"),
    ("banaswadi",             "banaswadi"),    ("thanisandra",          "thanisandra"),
    ("sahakara nagar",        "sahakara nagar"),("domlur",              "domlur"),
    ("frazer town",           "frazer town"),  ("cox town",             "cox town"),
    ("shivajinagar",          "shivajinagar"), ("cunningham road",      "shivajinagar"),
    ("richmond road",         "richmond road"),("mg road",              "mg road"),
    ("brigade road",          "mg road"),      ("sanjaynagar",          "sanjaynagar"),
    ("mathikere",             "yeshwanthpur"), ("srirampura",           "yeshwanthpur"),
    ("chord road",            "yeshwanthpur"), ("tumkur road",          "yeshwanthpur"),
    ("dasarahalli",           "yeshwanthpur"), ("jalahalli",            "yeshwanthpur"),
    ("mysuru",                "mysuru"),       ("mysore",               "mysuru"),
]

ML_FEATURES = [
    "brand", "model", "variant",
    "locality", "rto",
    "fuel_type", "transmission", "seller_type", "color",
    "vehicle_age", "odometer_reading", "km_per_year",
    "owner_count", "certified", "pincode",
    "selling_price",
]

RAW_KEEP = [
    "segment", "seller type", "certified", "make", "model", "trim",
    "odometer", "fuel", "trans", "rto", "selling price",
    "locality", "pincode", "owner", "color", "year", "age",
]

RENAME_MAP = {
    "make":          "brand_raw",
    "model":         "model_raw",
    "trim":          "variant_raw",
    "fuel":          "fuel_raw",
    "trans":         "trans_raw",
    "selling price": "selling_price",
    "owner":         "owner_raw",
    "seller type":   "seller_type_raw",
    "odometer":      "odometer_raw",
}

DUPLICATE_COLUMNS = [
    "brand", "model", "variant",
    "vehicle_age", "odometer_reading", "owner_count",
    "fuel_type", "transmission", "seller_type",
    "locality", "selling_price",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _norm(value, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    return s if s not in {"", "nan", "none", "null"} else default


def _normalize_locality(raw) -> str:
    if pd.isna(raw):
        return "unknown"
    s = re.sub(r"\s+", " ", str(raw).strip().lower())
    if s in {"", "nan", "none", "null"}:
        return "unknown"
    if s in LOCALITY_ALIAS:
        return LOCALITY_ALIAS[s]
    for keyword, canonical in _KEYWORD_MAP:
        if keyword in s:
            return canonical
    return s


# ── STAGE 1: MERGE SOURCES ────────────────────────────────────────────────────
def merge_sources(paths: list[Path], dataset_name: str) -> pd.DataFrame:
    print(f"\n{DIV}\nMERGE — {dataset_name}\n{DIV}")
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        print(f"  Loaded {p.name:40s}  ->  {len(df):>7,} rows x {df.shape[1]} cols")
        frames.append(df)

    if len(frames) == 1:
        merged = frames[0].copy()
    else:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        print(f"\n  Combined                                    ->  {len(merged):>7,} rows")

    # Drop duplicate column names, keep first
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


# ── STAGE 2: NORMALISE SEGMENT BUCKET ────────────────────────────────────────
def normalise_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw segment labels to canonical bucket names."""
    print(f"\n{DIV}\nNORMALISE SEGMENT BUCKET\n{DIV}")
    if "segment" in df.columns:
        df["segment_bucket"] = (
            df["segment"]
            .apply(lambda x: _norm(x, "unknown"))
            .map(lambda s: SEGMENT_MAP.get(s, s))
        )
    else:
        df["segment_bucket"] = "mass_market"

    print("  Bucket distribution (pre-preprocessing):")
    print(df["segment_bucket"].value_counts().to_string())
    return df


# ── STAGE 3: PREPROCESS ───────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    # Select & rename
    available = [c for c in RAW_KEEP if c in df.columns]
    # Always keep segment_bucket through selection
    df_seg = df["segment_bucket"].copy() if "segment_bucket" in df.columns else None
    df = df[available].copy()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    if df_seg is not None:
        df["segment_bucket"] = df_seg.values

    audit.append({"step": "raw_after_merge", "rows": len(df)})

    # Price
    b = len(df)
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df[df["selling_price"].notna() & df["selling_price"].between(PRICE_MIN, PRICE_MAX)]
    audit.append({"step": "price_filter", "dropped": b - len(df), "remaining": len(df)})

    # Age
    b = len(df)
    if "age" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["vehicle_age"] = np.nan
    if "year" in df.columns:
        year_num = pd.to_numeric(df["year"], errors="coerce")
        mask = df["vehicle_age"].isna() & year_num.between(1990, CURRENT_YEAR)
        df.loc[mask, "vehicle_age"] = (CURRENT_YEAR - year_num[mask]).clip(lower=0)
    df = df[df["vehicle_age"].notna() & df["vehicle_age"].between(AGE_MIN, AGE_MAX)]
    audit.append({"step": "age_filter", "dropped": b - len(df), "remaining": len(df)})

    # Odometer
    b = len(df)
    df["odometer_reading"] = pd.to_numeric(df["odometer_raw"], errors="coerce")
    df = df[df["odometer_reading"].notna() & df["odometer_reading"].between(ODO_MIN, ODO_MAX)]
    audit.append({"step": "odometer_filter", "dropped": b - len(df), "remaining": len(df)})

    # Brand
    b = len(df)
    df["brand"] = df["brand_raw"].apply(_norm).replace(BRAND_ALIAS)
    df = df[df["brand"].isin(OEM_ALLOWLIST)]
    audit.append({"step": "brand_filter", "dropped": b - len(df), "remaining": len(df)})

    # Model
    b = len(df)
    df["model"] = df["model_raw"].apply(_norm)
    df = df[df["model"] != "unknown"]
    audit.append({"step": "model_filter", "dropped": b - len(df), "remaining": len(df)})

    # Variant
    df["variant"] = df["variant_raw"].apply(_norm) if "variant_raw" in df.columns else "unknown"

    # Fuel
    b = len(df)
    df["fuel_type"] = df["fuel_raw"].apply(_norm).replace(FUEL_ALIAS) if "fuel_raw" in df.columns else "petrol"
    df = df[df["fuel_type"].isin(VALID_FUEL)]
    audit.append({"step": "fuel_filter", "dropped": b - len(df), "remaining": len(df)})

    # Transmission
    b = len(df)
    df["transmission"] = df["trans_raw"].apply(_norm).replace(TRANS_ALIAS) if "trans_raw" in df.columns else "manual"
    df = df[df["transmission"].isin(VALID_TRANS)]
    audit.append({"step": "trans_filter", "dropped": b - len(df), "remaining": len(df)})

    # Seller type
    def _seller(v) -> str:
        s = _norm(v)
        if any(k in s for k in _DEALER_KEYWORDS):    return "dealer"
        if any(k in s for k in _INDIVIDUAL_KEYWORDS): return "individual"
        return "unknown"
    df["seller_type"] = df["seller_type_raw"].apply(_seller) if "seller_type_raw" in df.columns else "unknown"

    # Locality / RTO / Color
    df["locality"] = df["locality"].apply(_normalize_locality) if "locality" in df.columns else "unknown"
    df["rto"]      = df["rto"].apply(lambda x: _norm(x, "unknown")) if "rto" in df.columns else "unknown"

    def _color(v) -> str:
        s = _norm(v, "unknown")
        if s == "unknown": return "unknown"
        if s in VALID_COLORS: return s
        if s in COLOR_ALIAS: return COLOR_ALIAS[s]
        for base in VALID_COLORS:
            if base in s: return base
        return "unknown"
    df["color"] = df["color"].apply(_color) if "color" in df.columns else "unknown"

    # Certified
    _CERT = {"yes":1.0,"1":1.0,"true":1.0,"y":1.0,"no":0.0,"0":0.0,"false":0.0,"n":0.0}
    df["certified"] = df["certified"].apply(_norm).map(_CERT) if "certified" in df.columns else np.nan

    # Pincode
    if "pincode" in df.columns:
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
        bad = df["pincode"].notna() & ~df["pincode"].between(100_000, 999_999)
        df.loc[bad, "pincode"] = np.nan
    else:
        df["pincode"] = np.nan

    # Owner count
    def _owner(v) -> int:
        s = _norm(v)
        if s == "unknown": return 1
        try: return max(1, min(int(float(s)), 6))
        except Exception: return 1
    df["owner_count"] = df["owner_raw"].apply(_owner) if "owner_raw" in df.columns else 1

    # km_per_year
    safe_age = df["vehicle_age"].clip(lower=0.5)
    df["km_per_year"] = (df["odometer_reading"] / safe_age).clip(0, 100_000).round(1)

    # Deduplication
    b = len(df)
    dedup_cols = [c for c in DUPLICATE_COLUMNS if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    audit.append({"step": "deduplication", "dropped": b - len(df), "remaining": len(df)})

    print(f"  Final rows after preprocessing: {len(df):,}")
    print("  Segment bucket distribution (post-preprocessing):")
    if "segment_bucket" in df.columns:
        print(df["segment_bucket"].value_counts().to_string())
    return df


# ── STAGE 4: STRATIFIED SPLIT ─────────────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / valid / test split (70/15/15) on segment_bucket.
    Buckets with < 3 rows are handled separately to avoid sklearn ValueError.
    """
    print(f"\n{DIV}\nSTRATIFIED SPLIT — {dataset_name}\n{DIV}")
    strat_col = "segment_bucket"
    bucket_counts = df[strat_col].value_counts()
    print("  Bucket distribution:")
    print(bucket_counts.to_string())
    print(f"\n  Total rows  : {len(df):,}")
    print(f"  Split ratio : {TRAIN_RATIO:.0%} train / {VALID_RATIO:.0%} valid / {TEST_RATIO:.0%} test")

    # Separate tiny buckets that sklearn can't stratify
    small_buckets = bucket_counts[bucket_counts < 3].index.tolist()
    main_df  = df[~df[strat_col].isin(small_buckets)]
    small_df = df[df[strat_col].isin(small_buckets)]

    train_list, val_list, test_list = [], [], []

    if len(main_df) > 0:
        tr, vt = train_test_split(
            main_df, test_size=(VALID_RATIO + TEST_RATIO),
            stratify=main_df[strat_col], random_state=RANDOM_SEED
        )
        val, tst = train_test_split(
            vt, test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
            stratify=vt[strat_col], random_state=RANDOM_SEED
        )
        train_list.append(tr)
        val_list.append(val)
        test_list.append(tst)

    # Distribute tiny-bucket rows: guarantee at least 1 in val & test
    if len(small_df) > 0:
        print(f"\n  Tiny-bucket rows (handled separately): {len(small_df)}")
        small_shuffled = small_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        n      = len(small_shuffled)
        n_val  = max(1, round(n * VALID_RATIO))
        n_test = max(1, round(n * TEST_RATIO))
        n_val  = min(n_val,  n - 2)
        n_test = min(n_test, n - n_val - 1)
        train_list.append(small_shuffled.iloc[n_val + n_test:])
        val_list.append(small_shuffled.iloc[:n_val])
        test_list.append(small_shuffled.iloc[n_val: n_val + n_test])

    train = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    val   = pd.concat(val_list,   ignore_index=True)
    test  = pd.concat(test_list,  ignore_index=True)

    print(f"\n  Train : {len(train):>7,} rows")
    print(f"  Valid : {len(val):>7,} rows")
    print(f"  Test  : {len(test):>7,} rows")

    # Coverage check
    expected = set(bucket_counts.index)
    for split_name, split_df in [("train", train), ("valid", val), ("test", test)]:
        present = set(split_df[strat_col].unique())
        missing = expected - present
        status  = "OK  all buckets present" if not missing else f"WARN missing: {missing}"
        print(f"  [{split_name:5s}] {status}")

    return train, val, test


# ── STAGE 5: SAVE SPLITS ──────────────────────────────────────────────────────
def save_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    out_dir: Path, dataset_name: str, audit: list,
    sources: list[str], duration: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [c for c in ML_FEATURES if c in train.columns]

    train[out_cols].to_csv(out_dir / "train.csv", index=False)
    val[out_cols].to_csv(out_dir / "valid.csv",   index=False)
    test[out_cols].to_csv(out_dir / "test.csv",   index=False)

    report = {
        "dataset":         dataset_name,
        "sources":         sources,
        "processed_at":    datetime.now().isoformat(),
        "current_year":    CURRENT_YEAR,
        "duration_sec":    round(duration, 2),
        "split_ratios":    {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
        "random_seed":     RANDOM_SEED,
        "output_features": out_cols,
        "row_counts": {
            "train": len(train), "valid": len(val), "test": len(test),
            "total": len(train) + len(val) + len(test),
        },
        "bucket_distribution": {
            "train": train["segment_bucket"].value_counts().to_dict() if "segment_bucket" in train.columns else {},
            "valid": val["segment_bucket"].value_counts().to_dict()   if "segment_bucket" in val.columns   else {},
            "test":  test["segment_bucket"].value_counts().to_dict()  if "segment_bucket" in test.columns  else {},
        },
        "pipeline_audit": audit,
    }

    with open(out_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to : {out_dir}")
    print(f"    train.csv          ->  {len(train):,} rows x {len(out_cols)} cols")
    print(f"    valid.csv          ->  {len(val):,} rows x {len(out_cols)} cols")
    print(f"    test.csv           ->  {len(test):,} rows x {len(out_cols)} cols")
    print(f"    split_report.json")


# ── PIPELINE ──────────────────────────────────────────────────────────────────
def run_pipeline(dataset_name: str, source_paths: list[Path], out_dir: Path) -> None:
    t0    = time.perf_counter()
    audit: list[dict] = []

    print(f"\n{'#' * 80}")
    print(f"  DATASET  : {dataset_name}")
    print(f"  SOURCES  : {[p.name for p in source_paths]}")
    print(f"  STARTED  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")

    df = merge_sources(source_paths, dataset_name)
    df = normalise_segment(df)

    print(f"\n{DIV}\nPREPROCESS — {dataset_name}\n{DIV}")
    df = preprocess(df, audit)

    train, val, test = stratified_split(df, dataset_name)

    duration = time.perf_counter() - t0
    save_splits(train, val, test, out_dir, dataset_name, audit,
                sources=[p.name for p in source_paths], duration=duration)

    print(f"\n{'#' * 80}")
    print(f"  PIPELINE COMPLETE  —  {dataset_name}  |  {duration:.2f}s")
    print(f"{'#' * 80}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    PIPELINES = [
        {
            "name":    "s1s4_plus_s5",
            "sources": [
                DATA_DIR / "s1-s4_owner-filled.csv",
                DATA_DIR / "s5_overall.csv",
            ],
            "out_dir": SPLITS_DIR / "s1s4_plus_s5",
        },
        {
            "name":    "overall_plus_s5",
            "sources": [
                DATA_DIR / "overall.csv",
                DATA_DIR / "s5_overall.csv",
            ],
            "out_dir": SPLITS_DIR / "overall_plus_s5",
        },
        {
            "name":    "overall_only",
            "sources": [
                DATA_DIR / "overall.csv",
            ],
            "out_dir": SPLITS_DIR / "overall_only",
        },
    ]

    for p in PIPELINES:
        missing = [src for src in p["sources"] if not src.exists()]
        if missing:
            print(f"\nWARN  Skipping '{p['name']}' — missing source files: {[m.name for m in missing]}")
            continue
        run_pipeline(p["name"], p["sources"], p["out_dir"])

    print("\n" + "=" * 80)
    print("  ALL PIPELINES COMPLETE")
    print(f"  Outputs in: {SPLITS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
