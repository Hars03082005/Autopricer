"""
compute_s5_premium.py
Computes statistical premium multipliers from S5 quality shop data vs Variant 1 baseline.
Outputs a config table ready to paste into valuation_config.json.
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd

import numpy as np

from backend.main import VehicleInput, predict_base_market_value

s5 = pd.read_csv("ml_training/data/processed_s5.csv")

print(f"S5 dataset: {len(s5)} rows")

results = []

errors = 0

for i, row in s5.iterrows():

    brand  = str(row["brand"])

    model  = str(row["model"])

    age    = int(row["vehicle_age"])

    odo    = float(row["odometer_reading"])

    fuel   = str(row["fuel_type"])

    trans  = str(row["transmission"])

    owners = int(row["owner_count"])

    actual = float(row["selling_price"])

    year   = 2026 - age

    try:

        v = VehicleInput(

            brand=brand, model=model, variant="unknown",

            year=year, fuel_type=fuel, transmission=trans,

            odometer_reading=int(odo), owner_count=owners,

            city="bangalore", condition="Good",

        )

        pred, _, _ = predict_base_market_value(v, model_variant="variant_1")

        ratio = actual / max(pred, 1)

        results.append({

            "brand": brand, "model": model, "age": age,

            "actual": actual, "v1_pred": pred, "ratio": ratio,

        })

    except Exception as e:

        errors += 1

df = pd.DataFrame(results)

print(f"Computed {len(df)} comparisons  ({errors} errors)\n")

print("Overall S5 premium (actual / Variant1 prediction):")

print(f"  Mean   : {df['ratio'].mean():.4f}  ({(df['ratio'].mean()-1)*100:.1f}% premium)")

print(f"  Median : {df['ratio'].median():.4f}  ({(df['ratio'].median()-1)*100:.1f}% premium)")

print(f"  Std    : {df['ratio'].std():.4f}")

print(f"  P25    : {df['ratio'].quantile(0.25):.4f}")

print(f"  P75    : {df['ratio'].quantile(0.75):.4f}")

print()

print("By age bracket:")

brackets = [(0, 2, "0-2"), (3, 4, "3-4"), (5, 7, "5-7")]

age_multipliers = {}

for lo, hi, label in brackets:

    sub = df[(df["age"] >= lo) & (df["age"] <= hi)]

    if len(sub) == 0:

        print(f"  Age {label}: no data")

        continue

    med = sub["ratio"].median()

    mean = sub["ratio"].mean()

    std  = sub["ratio"].std()

    print(f"  Age {label}:  n={len(sub):3d}  mean={mean:.4f}  median={med:.4f}  std={std:.4f}  ({(med-1)*100:.1f}% premium)")

    age_multipliers[label] = round(med, 4)

print()

print("By brand (median ratio):")

brand_stats = df.groupby("brand")["ratio"].agg(["median", "mean", "count"]).round(4)

brand_stats = brand_stats.sort_values("median", ascending=False)

print(brand_stats.to_string())

print()

print("By model (median ratio, min 2 rows):")

model_stats = df.groupby("model")["ratio"].agg(["median", "count"]).round(4)

model_stats = model_stats[model_stats["count"] >= 2].sort_values("median", ascending=False)

print(model_stats.head(20).to_string())

print()

s5_catalog = {}

for _, row in df.iterrows():

    b = row["brand"]

    m = row["model"]

    s5_catalog.setdefault(b, set()).add(m)

s5_catalog = {b: sorted(list(v)) for b, v in s5_catalog.items()}

config = {

    "_comment": "S5 Quality Shop premium multipliers derived from S5 dataset vs Variant 1 baseline. Updated by compute_s5_premium.py",

    "enabled": True,

    "max_age": 7,

    "default_multiplier": round(df["ratio"].median(), 4),

    "age_bracket_multipliers": {

        "0_to_2": age_multipliers.get("0-2", 1.10),

        "3_to_4": age_multipliers.get("3-4", 1.08),

        "5_to_7": age_multipliers.get("5-7", 1.05),

    },

    "brand_multipliers": {

        b: round(float(brand_stats.loc[b, "median"]), 4)

        for b in brand_stats.index

        if brand_stats.loc[b, "count"] >= 3

    },

    "s5_catalog": s5_catalog,

}

print("\n=== Generated S5 Config ===")

print(json.dumps(config, indent=2))

out = Path(__file__).parent / "s5_premium_config.json"

with open(out, "w") as f:

    json.dump(config, f, indent=2)

print(f"\nSaved to: {out}")

