"""Combine train, valid, and test splits into a single reference dataset.

This script is a one-time utility. The output must NOT be used for training.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

BASE = Path("ml_training/data/overall_only")
OUT  = Path("data")
OUT.mkdir(exist_ok=True)

train = pd.read_csv(BASE / "train.csv")
valid = pd.read_csv(BASE / "valid.csv")
test  = pd.read_csv(BASE / "test.csv")

train["split"] = "train"
valid["split"] = "valid"
test["split"]  = "test"

combined = pd.concat([train, valid, test], ignore_index=True)
out_path = OUT / "data.csv"
combined.to_csv(out_path, index=False)

sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

counts = {
    "train": len(train),
    "valid": len(valid),
    "test":  len(test),
    "total": len(combined),
}

manifest = {
    "description": (
        "Combined full dataset (train + valid + test). "
        "READ-ONLY reference file — do NOT use for training."
    ),
    "warning": (
        "This file contains all splits including the held-out test set. "
        "It must NEVER be used as training input, validation input, or for "
        "any form of model fitting or hyperparameter search."
    ),
    "row_counts": counts,
    "columns":    list(combined.columns),
    "sha256":     sha256,
    "source":     "ml_training/data/overall_only/{train,valid,test}.csv",
}

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

readme = f"""# data/

This folder contains the **full combined dataset** for reference and analysis only.

## data.csv

| Split      | Rows         |
|------------|--------------|
| train      | {counts['train']:,} |
| valid      | {counts['valid']:,} |
| test       | {counts['test']:,}  |
| **Total**  | **{counts['total']:,}** |

> **WARNING**: This file contains ALL splits including the held-out test set.
> It **must not** be used for model training, hyperparameter tuning, or any
> other form of model fitting. Use only for exploratory analysis, reporting,
> or the production comparable-search engine.

A `split` column is included on every row to indicate its original split origin.

SHA-256: `{sha256}`
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

print(f"Combined {counts['total']:,} rows  ->  data/data.csv")
print(f"SHA-256 : {sha256}")
print("Splits  :", counts)
