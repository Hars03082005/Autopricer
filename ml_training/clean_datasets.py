"""Cleans, normalizes variants/models, deduplicates, and re-splits all three dataset folders."""
from __future__ import annotations

import re

# ── ENGINE DISPLACEMENT / FUEL TECH TOKENS TO STRIP FROM VARIANTS ─────────
_ENG_DISP_RE = re.compile(
    r"""
    (?:
        \b\d+\.\d+\s*[lL]\b       # 1.2l, 1.5L, 2.0l
        | \b\d+\.\d+\b            # 1.2, 1.5, 2.0 (bare float)
        | \b\d\s*\d[lL]\b         # 1 2l, 1 5l (spaced)
        | \(\s*[pPdD]\s*\)        # (p), (P), (d), (D)
        | \b(?:petrol|diesel|lpg|cng|electric|hybrid)\b
        | \b(?:mpi|tsi|tdi|tgi|crdi|crtd|dci|ddis|vtvt|ivtec|i-vtec|i-dtec|crde|vcdi|dohc)\b
        | \b(?:turbo(?!\s*(?:sport|edition)))\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
_COSMETIC_RE = re.compile(
    r"""
    \b(?:\d+\s*)?alloy\b          # alloy, 16 alloy
    | \bdual\s+tone\b
    | \bspecial\s+edition\b
    | \banniversary\s+edition\b
    | \blimited\s+edition\b
    | \bopt(?:ion(?:al)?)?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s{2,}")

# Known i20 generation model aliases → canonical
_MODEL_ALIASES: dict[str, str] = {
    "elite i20":         "i20",
    "new i20":           "i20",
    "i20 active":        "i20 active",
    "active i20":        "i20 active",
    "i20 elite":         "i20",
    "grand i10 nios":    "grand i10 nios",
    "grand i10":         "grand i10",
    "xcent prime":       "xcent",
    "swift dzire":       "dzire",
    "dzire tour":        "dzire",
    "wagon r 1.0":       "wagon r",
    "wagon r stingray":  "wagon r",
    "alto 800":          "alto",
    "alto k10":          "alto k10",
    "omni e":            "omni",
    "eeco cargo":        "eeco",
    "scorpio classic":   "scorpio",
    "scorpio n":         "scorpio n",
    "safari storme":     "safari",
    "safari dicor":      "safari",
    "sumo gold":         "sumo",
    "sumo victa":        "sumo",
    "innova crysta":     "innova crysta",
    "urban cruiser hyryder": "hyryder",
    "fortuner legender": "fortuner",
    "hilux std":         "hilux",
    "glanza g":          "glanza",
    "nexon ev":          "nexon ev",
    "nexon ev prime":    "nexon ev",
    "nexon ev max":      "nexon ev",
    "tigor ev":          "tigor ev",
    "tiago nrg":         "tiago",
    "altroz racer":      "altroz",
    "punch ev":          "punch ev",
    "curvv ev":          "curvv ev",
    "hector plus":       "hector plus",
    "astor neo":         "astor",
    "comet ev":          "comet",
    "windsor ev":        "windsor",
    "zs ev":             "zs ev",
    "seltos x line":     "seltos",
    "sonet x line":      "sonet",
    "carens limousine":  "carens",
    "ev6 gt":            "ev6",
    "thar roxx":         "thar",
    "xuv 3xo":           "xuv300",
    "xuv400 ec":         "xuv400",
    "xuv400 el":         "xuv400",
    "be 6e":             "be 6",
    "polo gt":           "polo",
    "polo comfortline":  "polo",
    "polo highline":     "polo",
    "polo trendline":    "polo",
    "vento highline":    "vento",
    "ameo":              "ameo",
    "kushaq onyx":       "kushaq",
    "slavia style":      "slavia",
    "rapid style":       "rapid",
    "wr-v s":            "wr-v",
    "jazz zx":           "jazz",
    "amaze s":           "amaze",
    "city zx":           "city",
    "city e cvt":        "city",
    "endeavour sport":   "endeavour",
    "ecosport s":        "ecosport",
    "aspire titanium":   "aspire",
    "figo s":            "figo",
    "freestyle titanium":"freestyle",
    "kiger rxz":         "kiger",
    "magnite rxz":       "magnite",
    "triber rxe":        "triber",
    "kwid rxt":          "kwid",
    "duster rxs":        "duster",
    "kicks xv":          "kicks",
    "terrano xv":        "terrano",
    "micra active":      "micra",
    "redi-go t":         "redi-go",
    "c-class cabriolet": "c class",
    "e-class cabriolet": "e class",
    "a-class limousine": "a class",
    "glc coupe":         "glc",
    "gle coupe":         "gle",
    "x3 xdrive":         "x3",
    "x5 xdrive":         "x5",
    "3 series gran limousine": "3 series",
    "5 series gran turismo":   "5 series",
}

def normalize_variant(variant: str) -> str:
    """Generalize variant name: strip engine displacement, fuel type, and cosmetic suffixes."""
    if not isinstance(variant, str):
        return "unknown"
    v = variant.strip().lower()
    if v in ("", "unknown", "nan", "none"):
        return "unknown"
    # Strip engine displacement / fuel tech tokens
    v = _ENG_DISP_RE.sub(" ", v)
    # Strip cosmetic tokens
    v = _COSMETIC_RE.sub(" ", v)
    # Collapse whitespace
    v = _WHITESPACE_RE.sub(" ", v).strip(" -/()")
    return v if v else "unknown"


def normalize_model(model: str, brand: str = "") -> str:
    """Normalize model name: resolve generation aliases and strip engine size."""
    if not isinstance(model, str):
        return "unknown"
    m = model.strip().lower()
    # Check alias map first (most specific wins)
    if m in _MODEL_ALIASES:
        return _MODEL_ALIASES[m]
    # Strip trailing engine displacement from model name (e.g. "wagon r 1.0")
    m = re.sub(r"\s+\d+\.\d+[lL]?\s*$", "", m).strip()
    m = re.sub(r"\s+\d+[lL]\s*$", "", m).strip()
    return m if m else "unknown"


def normalize_brand(brand: str) -> str:
    """Normalize brand aliases to canonical form."""
    if not isinstance(brand, str):
        return "unknown"
    b = brand.strip().lower()
    _ALIASES = {
        "maruti":        "maruti suzuki",
        "maruti-suzuki": "maruti suzuki",
        "suzuki":        "maruti suzuki",
        "mercedes":      "mercedes-benz",
        "mercedes benz": "mercedes-benz",
        "land-rover":    "land rover",
        "vw":            "volkswagen",
    }
    return _ALIASES.get(b, b)


def _bucket_label(price: float) -> str:
    for label, lo, hi in PRICE_BUCKETS:
        if lo <= price < hi:
            return label
    return "15L_plus"


import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATASETS = ["overall_only"]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

CORE_DUP_COLS = [
    "brand", "model", "variant",
    "vehicle_age", "odometer_reading", "owner_count",
    "fuel_type", "transmission",
    "selling_price",
]

NEAR_DUP_COLS = [
    "brand", "model", "variant",
    "vehicle_age", "owner_count",
    "fuel_type", "transmission",
    "selling_price",
    "_odo_round1k",
]

PRICE_BUCKETS = [
    ("0_3L",     0,           300_000),
    ("3_5L",     300_000,     500_000),
    ("5_10L",    500_000,   1_000_000),
    ("10_15L",   1_000_000, 1_500_000),
    ("15L_plus", 1_500_000, 20_000_000),
]

ML_FEATURES = [
    "brand", "model", "variant",
    "locality", "rto",
    "fuel_type", "transmission", "seller_type", "color",
    "vehicle_age", "odometer_reading", "km_per_year",
    "owner_count", "certified", "pincode",
    "selling_price",
]

def _bucket_label(price: float) -> str:
    for label, lo, hi in PRICE_BUCKETS:
        if lo <= price < hi:
            return label
    return "15L_plus"

def deduplicate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(df)

    # Pass 1: Exact match on core vehicle identity + price
    exact_cols = [c for c in CORE_DUP_COLS if c in df.columns]
    df_exact = df.drop_duplicates(subset=exact_cols, keep="first").reset_index(drop=True)
    pass1_rows = len(df_exact)
    exact_dropped = initial_rows - pass1_rows

    # Pass 2: Near match (odometer rounded to nearest 1,000 km)
    df_exact["_odo_round1k"] = (df_exact["odometer_reading"] / 1_000).round(0) * 1_000
    near_cols = [c for c in NEAR_DUP_COLS if c in df_exact.columns]
    df_clean = df_exact.drop_duplicates(subset=near_cols, keep="first").reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_odo_round1k"], errors="ignore")
    final_rows = len(df_clean)
    near_dropped = pass1_rows - final_rows

    stats = {
        "initial_rows": initial_rows,
        "exact_duplicates_dropped": exact_dropped,
        "near_duplicates_dropped": near_dropped,
        "total_duplicates_dropped": initial_rows - final_rows,
        "clean_rows": final_rows,
    }
    return df_clean, stats

def leak_free_stratified_split(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()
    df["_bucket"] = df["selling_price"].apply(_bucket_label)

    # Group by vehicle identity key to ensure identical vehicle profiles stay together
    df["_group_key"] = df["brand"].astype(str) + "_" + df["model"].astype(str) + "_" + df["variant"].astype(str) + "_" + df["vehicle_age"].astype(str) + "_" + df["selling_price"].astype(str)

    # Aggregate groups to ensure no group crosses train/valid/test
    grouped = df.groupby("_group_key").agg({
        "_bucket": "first"
    }).reset_index()

    train_groups, valid_groups, test_groups = [], [], []
    bucket_dist = {"train": {}, "valid": {}, "test": {}}

    for label, lo, hi in PRICE_BUCKETS:
        stratum_groups = grouped[grouped["_bucket"] == label]["_group_key"].values
        n = len(stratum_groups)
        if n == 0:
            bucket_dist["train"][label] = 0
            bucket_dist["valid"][label] = 0
            bucket_dist["test"][label] = 0
            continue

        if n < 6:
            train_groups.extend(stratum_groups)
            bucket_dist["train"][label] = len(df[df["_group_key"].isin(stratum_groups)])
            bucket_dist["valid"][label] = 0
            bucket_dist["test"][label] = 0
            continue

        tr_g, tmp_g = train_test_split(
            stratum_groups, test_size=(VALID_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED, shuffle=True
        )
        vl_g, te_g = train_test_split(
            tmp_g, test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED, shuffle=True
        )

        train_groups.extend(tr_g)
        valid_groups.extend(vl_g)
        test_groups.extend(te_g)

        bucket_dist["train"][label] = len(df[df["_group_key"].isin(tr_g)])
        bucket_dist["valid"][label] = len(df[df["_group_key"].isin(vl_g)])
        bucket_dist["test"][label]  = len(df[df["_group_key"].isin(te_g)])

    df_train = df[df["_group_key"].isin(train_groups)].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_valid = df[df["_group_key"].isin(valid_groups)].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_test  = df[df["_group_key"].isin(test_groups)].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df_train = df_train.drop(columns=["_bucket", "_group_key"], errors="ignore")
    df_valid = df_valid.drop(columns=["_bucket", "_group_key"], errors="ignore")
    df_test  = df_test.drop(columns=["_bucket", "_group_key"], errors="ignore")

    # Leakage check
    key_cols = [c for c in ["brand", "model", "variant", "vehicle_age", "odometer_reading", "selling_price"] if c in df_train.columns]
    tr_keys = set(df_train[key_cols].apply(tuple, axis=1))
    vl_keys = set(df_valid[key_cols].apply(tuple, axis=1))
    te_keys = set(df_test[key_cols].apply(tuple, axis=1))

    leakage_stats = {
        "train_valid_overlap": len(tr_keys & vl_keys),
        "train_test_overlap":  len(tr_keys & te_keys),
        "valid_test_overlap":  len(vl_keys & te_keys),
    }

    return df_train, df_valid, df_test, {"bucket_distribution": bucket_dist, "leakage_stats": leakage_stats}

def process_dataset_folder(dataset_name: str) -> None:
    folder = DATA_DIR / dataset_name
    if not folder.exists():
        print(f"Skipping {dataset_name}: Folder not found.")
        return

    print("\n==================================================")
    print(f" PROCESSING DATASET: {dataset_name}")
    print("==================================================")

    # Load train, valid, test if present
    parts = []
    for split in ["train", "valid", "test"]:
        fpath = folder / f"{split}.csv"
        if fpath.exists():
            df_part = pd.read_csv(fpath, low_memory=False)
            parts.append(df_part)

    if not parts:
        print(f"No CSV split files found in {folder}")
        return

    combined_df = pd.concat(parts, ignore_index=True)
    initial_count = len(combined_df)
    print(f"Combined initial row count: {initial_count:,}")

    # Normalize brand, model, variant before deduplication
    if "brand" in combined_df.columns:
        combined_df["brand"] = combined_df["brand"].apply(normalize_brand)
    if "model" in combined_df.columns:
        combined_df["model"] = combined_df.apply(
            lambda r: normalize_model(str(r["model"]), str(r.get("brand", ""))), axis=1
        )
    if "variant" in combined_df.columns:
        combined_df["variant"] = combined_df["variant"].apply(normalize_variant)

    before_norm = initial_count
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    after_norm = len(combined_df)
    print(f"After normalization + re-dedup: {after_norm:,} rows ({before_norm - after_norm:,} collapsed by normalization)")

    # Deduplicate
    clean_df, dedup_stats = deduplicate_dataframe(combined_df)
    print("Deduplication summary:")
    print(f"  Exact duplicates removed : {dedup_stats['exact_duplicates_dropped']:,}")
    print(f"  Near duplicates removed  : {dedup_stats['near_duplicates_dropped']:,}")
    print(f"  Total duplicates removed : {dedup_stats['total_duplicates_dropped']:,}")
    print(f"  Remaining unique rows    : {dedup_stats['clean_rows']:,}")

    # Split
    df_train, df_valid, df_test, split_info = leak_free_stratified_split(clean_df, dataset_name)
    leakage = split_info["leakage_stats"]
    print("Stratified leak-free split results:")
    print(f"  Train : {len(df_train):,} rows")
    print(f"  Valid : {len(df_valid):,} rows")
    print(f"  Test  : {len(df_test):,} rows")
    print("Leakage validation:")
    print(f"  train & valid overlap : {leakage['train_valid_overlap']}")
    print(f"  train & test overlap  : {leakage['train_test_overlap']}")
    print(f"  valid & test overlap  : {leakage['valid_test_overlap']}")

    # Ensure output columns ordering
    out_cols = [c for c in ML_FEATURES if c in clean_df.columns]

    df_train[out_cols].to_csv(folder / "train.csv", index=False)
    df_valid[out_cols].to_csv(folder / "valid.csv", index=False)
    df_test[out_cols].to_csv(folder / "test.csv", index=False)

    report = {
        "dataset": dataset_name,
        "processed_at": datetime.now().isoformat(),
        "initial_total_rows": initial_count,
        "dedup_stats": dedup_stats,
        "split_row_counts": {
            "train": len(df_train),
            "valid": len(df_valid),
            "test": len(df_test),
            "total": len(clean_df),
        },
        "bucket_distribution": split_info["bucket_distribution"],
        "leakage_stats": leakage,
        "features": out_cols,
    }

    with open(folder / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Updated {dataset_name} files: train.csv, valid.csv, test.csv, split_report.json")

def main():
    t0 = time.perf_counter()
    for ds in DATASETS:
        process_dataset_folder(ds)
    print(f"\nAll datasets cleaned and re-split in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
