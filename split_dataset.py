"""
split_dataset.py
----------------
Reads cleaned.csv, drops listing_month and listing_year,
then groups rows by their "non-null column signature"
(i.e. the exact set of columns that have data for each row)
and writes each group to its own CSV file under:
    ml_training/data/splits/
"""

import pandas as pd
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "ml_training/data/cleaned.csv"
OUTPUT_DIR  = "ml_training/data/splits"
DROP_COLS   = ["listing_month", "listing_year"]   # columns to exclude entirely

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Shape before drop: {df.shape}")

# Drop unwanted columns (if they exist)
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"  Dropped columns  : {cols_to_drop}")
print(f"  Shape after drop : {df.shape}")
print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
print()

# ── Build non-null signature per row ─────────────────────────────────────────
# A column is considered "present" for a row when it is not NaN and not empty string.
def non_null_signature(row):
    return frozenset(col for col, val in row.items()
                     if pd.notna(val) and str(val).strip() not in ('', 'nan', 'NaN', 'unknown'))

print("Computing column signatures … (this may take a moment on 200k+ rows)")
sig_series = df.apply(non_null_signature, axis=1)
print("Done.")

# ── Group by signature ────────────────────────────────────────────────────────
from collections import defaultdict
groups = defaultdict(list)
for idx, sig in sig_series.items():
    groups[sig].append(idx)

# Sort groups by descending row count so group_01 is the largest
sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

print(f"\nFound {len(sorted_groups)} distinct column-signature groups.\n")

# ── Write output ──────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []

for rank, (sig, indices) in enumerate(sorted_groups, start=1):
    subset = df.loc[indices]

    # Only keep the columns that are actually in this signature (drop fully-NaN columns)
    cols_present = [c for c in df.columns if c in sig]
    # Always preserve column order from original dataframe
    cols_present = [c for c in df.columns if c in sig]
    subset = subset[cols_present].copy()

    # Build a short readable name from the non-core extra columns
    core = {"brand", "model", "year", "vehicle_age", "fuel_type",
            "transmission", "odometer_reading", "km_per_year",
            "owner_count", "ownership_trust_score", "vehicle_health_score",
            "selling_price"}
    extra_cols = sorted([c for c in cols_present if c not in core])
    short_name = "_".join(extra_cols[:7])          # keep name reasonable length
    if len(extra_cols) > 7:
        short_name += f"_plus{len(extra_cols)-7}more"

    filename = f"group_{rank:02d}_{short_name}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    subset.to_csv(filepath, index=False)

    print(f"  Group {rank:02d} | {len(indices):>6} rows | {len(cols_present):>2} cols | {filename}")

    summary_rows.append({
        "group"    : rank,
        "rows"     : len(indices),
        "n_columns": len(cols_present),
        "columns"  : ", ".join(cols_present),
        "extra_cols": ", ".join(extra_cols),
        "file"     : filename,
    })

# ── Write summary ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "_split_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary written → {summary_path}")
print("All done ✓")
