"""
ml_training/prepare_splits.py

Robust data preparation & splitting pipeline:
1. Loads raw source CSVs if present, or falls back to existing pre-split CSVs in data/<dataset>/.
2. Applies brand, model, and variant normalization (stripping engine size, fuel tech badges, cosmetic suffixes).
3. Applies strict two-pass deduplication and quality filters.
4. Performs leak-free group-stratified 70/15/15 splitting based on price buckets.
5. Saves clean train.csv, valid.csv, test.csv and split_report.json directly to data/<dataset>/ and data/splits/<dataset>/.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from ml_training.clean_datasets import (
    normalize_brand,
    normalize_model,
    normalize_variant,
)

DATA_DIR     = HERE / "data"
SPLITS_DIR   = DATA_DIR / "splits"
CURRENT_YEAR = datetime.now().year
DIV          = "=" * 80

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

PRICE_MIN, PRICE_MAX = 50_000, 20_000_000
AGE_MIN,   AGE_MAX   = 0, 30
ODO_MIN,   ODO_MAX   = 0, 600_000

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

def _bucket_label(price: float) -> str:
    for label, lo, hi in PRICE_BUCKETS:
        if lo <= price < hi:
            return label
    return "15L_plus"

# ── STAGE 1: LOAD & MERGE SOURCES ────────────────────────────────────────────
def load_dataset_sources(dataset_name: str, raw_sources: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    existing_raw = [p for p in raw_sources if p.exists()]
    sources_used = []

    if existing_raw:
        frames = []
        for p in existing_raw:
            df_part = pd.read_csv(p, low_memory=False)
            frames.append(df_part)
            sources_used.append(p.name)
        combined = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {len(existing_raw)} raw source file(s) for '{dataset_name}': {len(combined):,} rows")
        return combined, sources_used

    # Fallback to existing split files in data/<dataset_name>/
    folder = DATA_DIR / dataset_name
    split_frames = []
    for split in ["train", "valid", "test"]:
        sp = folder / f"{split}.csv"
        if sp.exists():
            df_part = pd.read_csv(sp, low_memory=False)
            split_frames.append(df_part)
            sources_used.append(f"{dataset_name}/{split}.csv")

    if split_frames:
        combined = pd.concat(split_frames, ignore_index=True)
        print(f"  Fallback: Loaded {len(split_frames)} pre-split CSVs from {folder.name}/: {len(combined):,} rows")
        return combined, sources_used

    return pd.DataFrame(), []

# ── STAGE 2: PREPROCESS & NORMALIZE ──────────────────────────────────────────
def preprocess_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    initial_count = len(df)
    if initial_count == 0:
        return df, {}

    df = df.copy()

    # Price filter
    if "selling_price" in df.columns:
        df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
        df = df[df["selling_price"].notna() & df["selling_price"].between(PRICE_MIN, PRICE_MAX)]

    # Age filter
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["vehicle_age"], errors="coerce")
    elif "age" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["age"], errors="coerce")
    elif "year" in df.columns:
        df["vehicle_age"] = CURRENT_YEAR - pd.to_numeric(df["year"], errors="coerce")
    df = df[df["vehicle_age"].notna() & df["vehicle_age"].between(AGE_MIN, AGE_MAX)]

    # Odometer filter
    odo_col = "odometer_reading" if "odometer_reading" in df.columns else "odometer" if "odometer" in df.columns else None
    if odo_col:
        df["odometer_reading"] = pd.to_numeric(df[odo_col], errors="coerce")
        df = df[df["odometer_reading"].notna() & df["odometer_reading"].between(ODO_MIN, ODO_MAX)]

    # Normalize Brand, Model, Variant
    if "brand" in df.columns:
        df["brand"] = df["brand"].apply(normalize_brand)
    elif "make" in df.columns:
        df["brand"] = df["make"].apply(normalize_brand)

    if "model" in df.columns:
        df["model"] = df.apply(lambda r: normalize_model(str(r["model"]), str(r.get("brand", ""))), axis=1)

    variant_col = "variant" if "variant" in df.columns else "trim" if "trim" in df.columns else None
    if variant_col:
        df["variant"] = df[variant_col].apply(normalize_variant)

    # 2-Pass Deduplication
    exact_cols = [c for c in CORE_DUP_COLS if c in df.columns]
    df_exact = df.drop_duplicates(subset=exact_cols, keep="first").reset_index(drop=True)

    df_exact["_odo_round1k"] = (df_exact["odometer_reading"] / 1_000).round(0) * 1_000
    near_cols = [c for c in NEAR_DUP_COLS if c in df_exact.columns]
    df_clean = df_exact.drop_duplicates(subset=near_cols, keep="first").reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_odo_round1k"], errors="ignore")

    stats = {
        "initial_rows": initial_count,
        "clean_rows": len(df_clean),
        "dropped_rows": initial_count - len(df_clean),
    }

    return df_clean, stats

# ── STAGE 3: LEAK-FREE GROUP STRATIFIED SPLIT ─────────────────────────────────
def leak_free_stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()
    df["_bucket"] = df["selling_price"].apply(_bucket_label)
    df["_group_key"] = (
        df["brand"].astype(str) + "_" +
        df["model"].astype(str) + "_" +
        df["variant"].astype(str) + "_" +
        df["vehicle_age"].astype(str) + "_" +
        df["selling_price"].astype(str)
    )

    grouped = df.groupby("_group_key").agg({"_bucket": "first"}).reset_index()

    train_groups, valid_groups, test_groups = [], [], []
    bucket_dist = {"train": {}, "valid": {}, "test": {}}

    for label, lo, hi in PRICE_BUCKETS:
        stratum_groups = grouped[grouped["_bucket"] == label]["_group_key"].values
        n = len(stratum_groups)
        if n == 0:
            bucket_dist["train"][label] = 0
            bucket_dist["valid"][label] = 0
            bucket_dist["test"][label]  = 0
            continue

        if n < 6:
            train_groups.extend(stratum_groups)
            bucket_dist["train"][label] = len(df[df["_group_key"].isin(stratum_groups)])
            bucket_dist["valid"][label] = 0
            bucket_dist["test"][label]  = 0
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

# ── STAGE 4: RUN PIPELINE ─────────────────────────────────────────────────────
def run_pipeline(dataset_name: str, raw_sources: list[Path]) -> None:
    t0 = time.perf_counter()
    print(f"\n{DIV}")
    print(f" PIPELINE: {dataset_name}")
    print(f"{DIV}")

    df, sources_used = load_dataset_sources(dataset_name, raw_sources)
    if df.empty:
        print(f"  SKIP: No source files or existing pre-split CSVs found for '{dataset_name}'.")
        return

    clean_df, prep_stats = preprocess_and_normalize(df)
    df_train, df_valid, df_test, split_info = leak_free_stratified_split(clean_df)
    leakage = split_info["leakage_stats"]

    print(f"  Rows: initial={prep_stats['initial_rows']:,} -> clean={prep_stats['clean_rows']:,}")
    print(f"  Splits: Train={len(df_train):,} | Valid={len(df_valid):,} | Test={len(df_test):,}")
    print(f"  Leakage Check: train-valid={leakage['train_valid_overlap']} | train-test={leakage['train_test_overlap']} | valid-test={leakage['valid_test_overlap']}")

    out_cols = [c for c in ML_FEATURES if c in clean_df.columns]

    # Target folders: data/<dataset_name>/ and data/splits/<dataset_name>/
    target_dirs = [DATA_DIR / dataset_name, SPLITS_DIR / dataset_name]

    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        df_train[out_cols].to_csv(target_dir / "train.csv", index=False)
        df_valid[out_cols].to_csv(target_dir / "valid.csv", index=False)
        df_test[out_cols].to_csv(target_dir / "test.csv", index=False)

        report = {
            "dataset": dataset_name,
            "sources_used": sources_used,
            "processed_at": datetime.now().isoformat(),
            "prep_stats": prep_stats,
            "split_row_counts": {
                "train": len(df_train),
                "valid": len(df_valid),
                "test": len(df_test),
                "total": len(clean_df),
            },
            "bucket_distribution": split_info["bucket_distribution"],
            "leakage_stats": leakage,
            "features": out_cols,
            "duration_sec": round(time.perf_counter() - t0, 2),
        }

        with open(target_dir / "split_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"  SAVED to {DATA_DIR / dataset_name} & {SPLITS_DIR / dataset_name} ({time.perf_counter() - t0:.2f}s)")

def main() -> None:
    PIPELINES = [
        {
            "name": "overall_only",
            "sources": [DATA_DIR / "overall.csv"],
        },
        {
            "name": "overall_plus_s5",
            "sources": [DATA_DIR / "overall.csv", DATA_DIR / "s5_overall.csv"],
        },
        {
            "name": "s1s4_plus_s5",
            "sources": [DATA_DIR / "s1-s4_owner-filled.csv", DATA_DIR / "s5_overall.csv"],
        },
    ]

    for p in PIPELINES:
        run_pipeline(p["name"], p["sources"])

    print(f"\n{DIV}")
    print(" ALL PREPARE SPLITS PIPELINES COMPLETE")
    print(f"{DIV}")

if __name__ == "__main__":
    main()
