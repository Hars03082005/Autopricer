"""
PriceRef – Price-Band & Brand-Wise Complete Analysis Report Generator
Reads data/data.csv and produces a rich Markdown + JSON report.
"""
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(ROOT, "data", "data.csv")
OUT_MD = os.path.join(ROOT, "analysis", "price_brand_report.md")
OUT_JSON = os.path.join(ROOT, "analysis", "price_brand_report.json")

# ── Price Bands ─────────────────────────────────────────────────────────────────
PRICE_BANDS = [
    ("0–1L",   0,          100_000),
    ("1–2L",   100_000,    200_000),
    ("2–3L",   200_000,    300_000),
    ("3–4L",   300_000,    400_000),
    ("4–5L",   400_000,    500_000),
    ("5–6L",   500_000,    600_000),
    ("6–8L",   600_000,    800_000),
    ("8–10L",  800_000,  1_000_000),
    ("10–12L", 1_000_000, 1_200_000),
    ("12–15L", 1_200_000, 1_500_000),
    ("15–20L", 1_500_000, 2_000_000),
    ("20–30L", 2_000_000, 3_000_000),
    ("30L+",   3_000_000, math.inf),
]

# ── Helper utilities ────────────────────────────────────────────────────────────
def fmt_inr(v: float) -> str:
    if v >= 100_000:
        return f"Rs.{v/100_000:.2f}L"
    return f"Rs.{v/1_000:.1f}K"

def compute_band_stats(prices: np.ndarray) -> dict:
    n = len(prices)
    if n == 0:
        return {}
    return {
        "count": int(n),
        "mean":  float(np.mean(prices)),
        "median": float(np.median(prices)),
        "std":   float(np.std(prices)),
        "min":   float(np.min(prices)),
        "max":   float(np.max(prices)),
        "p10":   float(np.percentile(prices, 10)),
        "p25":   float(np.percentile(prices, 25)),
        "p75":   float(np.percentile(prices, 75)),
        "p90":   float(np.percentile(prices, 90)),
        "p95":   float(np.percentile(prices, 95)),
    }

def fuel_mix(sub: pd.DataFrame) -> str:
    counts = sub["fuel_type"].value_counts(normalize=True) * 100
    return ", ".join(f"{k} {v:.0f}%" for k, v in counts.items())

def tx_mix(sub: pd.DataFrame) -> str:
    counts = sub["transmission"].value_counts(normalize=True) * 100
    return ", ".join(f"{k} {v:.0f}%" for k, v in counts.items())

# ── Load Data ──────────────────────────────────────────────────────────────────
print(f"Loading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")

df["brand"] = df["brand"].str.strip().str.title()
df["model"] = df["model"].str.strip().str.title()
df["fuel_type"] = df["fuel_type"].str.strip().str.lower()
df["transmission"] = df["transmission"].str.strip().str.lower()
df["seller_type"] = df["seller_type"].str.strip().str.lower()

prices = df["selling_price"].values
total = len(df)

# ── 1. Overall Stats ────────────────────────────────────────────────────────────
overall = compute_band_stats(prices)

# ── 2. Price-Band Analysis ─────────────────────────────────────────────────────
band_rows = []
for label, lo, hi in PRICE_BANDS:
    mask = (df["selling_price"] >= lo) & (df["selling_price"] < hi)
    sub = df[mask]
    p = sub["selling_price"].values
    if len(p) == 0:
        continue
    row = compute_band_stats(p)
    row["band"] = label
    row["share_pct"] = 100.0 * len(p) / total
    row["fuel_mix"] = fuel_mix(sub)
    row["tx_mix"] = tx_mix(sub)
    top_brands = sub["brand"].value_counts().head(3)
    row["top_brands"] = ", ".join(f"{b}({c})" for b, c in top_brands.items())
    if "certified" in sub.columns:
        cert_col = pd.to_numeric(sub["certified"], errors="coerce")
        row["certified_pct"] = 100.0 * cert_col.eq(1.0).sum() / len(p)
    else:
        row["certified_pct"] = None
    owner_pct = sub["owner_count"].value_counts(normalize=True) * 100
    row["single_owner_pct"] = float(owner_pct.get(1, 0.0))
    band_rows.append(row)

# ── 3. Brand Analysis ──────────────────────────────────────────────────────────
brand_stats = []
for brand, grp in df.groupby("brand"):
    p = grp["selling_price"].values
    n = len(p)
    if n < 5:
        continue
    s = compute_band_stats(p)
    s["brand"] = brand
    s["count"] = n
    s["share_pct"] = 100.0 * n / total
    s["top_model"] = grp["model"].value_counts().idxmax()
    s["fuel_mix"] = fuel_mix(grp)
    s["tx_mix"] = tx_mix(grp)
    s["budget_count"]  = int(((p >= 0) & (p < 300_000)).sum())
    s["economy_count"] = int(((p >= 300_000) & (p < 600_000)).sum())
    s["mid_count"]     = int(((p >= 600_000) & (p < 1_200_000)).sum())
    s["premium_count"] = int(((p >= 1_200_000) & (p < 2_000_000)).sum())
    s["luxury_count"]  = int((p >= 2_000_000).sum())
    brand_stats.append(s)

brand_df = pd.DataFrame(brand_stats).sort_values("count", ascending=False)

# ── 4. Heat-Map ────────────────────────────────────────────────────────────────
top_brands_list = brand_df.head(15)["brand"].tolist()
heat_map = {}
for brand in top_brands_list:
    heat_map[brand] = {}
    bsub = df[df["brand"] == brand]
    for label, lo, hi in PRICE_BANDS:
        cnt = int(((bsub["selling_price"] >= lo) & (bsub["selling_price"] < hi)).sum())
        heat_map[brand][label] = cnt

# ── 5. Save JSON ───────────────────────────────────────────────────────────────
def safe_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out

report_json = {
    "generated_at": datetime.now().isoformat(),
    "total_listings": total,
    "overall": safe_dict(overall),
    "price_bands": [safe_dict(r) for r in band_rows],
    "brands": [safe_dict(r) for r in brand_df.to_dict("records")],
    "heat_map": heat_map,
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report_json, f, indent=2, ensure_ascii=False)
print(f"JSON saved -> {OUT_JSON}")

# ── 6. Markdown Report ─────────────────────────────────────────────────────────
lines = []
a = lines.append

a("# PriceRef - Complete Vehicle Price & Brand Analysis Report")
a("")
a(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
a(f"**Dataset:** data/data.csv | **Total Listings:** {total:,}")
a("")
a("---")
a("")

# Overall
a("## 1. Overall Dataset Summary")
a("")
a("| Metric | Value |")
a("| :--- | :--- |")
a(f"| Total Listings | **{total:,}** |")
a(f"| Mean Price | **{fmt_inr(overall['mean'])}** |")
a(f"| Median Price | **{fmt_inr(overall['median'])}** |")
a(f"| Std Deviation | **{fmt_inr(overall['std'])}** |")
a(f"| Min Price | **{fmt_inr(overall['min'])}** |")
a(f"| Max Price | **{fmt_inr(overall['max'])}** |")
a(f"| P10 | **{fmt_inr(overall['p10'])}** |")
a(f"| P25 | **{fmt_inr(overall['p25'])}** |")
a(f"| P75 | **{fmt_inr(overall['p75'])}** |")
a(f"| P90 | **{fmt_inr(overall['p90'])}** |")
a(f"| P95 | **{fmt_inr(overall['p95'])}** |")
a("")

splits = df["split"].value_counts()
a("### Dataset Splits")
a("")
a("| Split | Count | Share |")
a("| :--- | :---: | :---: |")
for s_name, s_cnt in splits.items():
    a(f"| {s_name.title()} | {s_cnt:,} | {100*s_cnt/total:.1f}% |")
a("")
a("---")
a("")

# Price Bands
a("## 2. Price-Band Breakdown")
a("")
a("### 2a. Volume & Pricing Stats")
a("")
a("| Band | Count | Share | Mean | Median | Std Dev | Min | Max |")
a("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for row in band_rows:
    a(f"| **{row['band']}** | {row['count']:,} | {row['share_pct']:.1f}% | {fmt_inr(row['mean'])} | {fmt_inr(row['median'])} | {fmt_inr(row['std'])} | {fmt_inr(row['min'])} | {fmt_inr(row['max'])} |")
a("")

a("### 2b. Percentile Distribution per Band")
a("")
a("| Band | P10 | P25 | P50 (Median) | P75 | P90 | P95 |")
a("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
for row in band_rows:
    a(f"| **{row['band']}** | {fmt_inr(row['p10'])} | {fmt_inr(row['p25'])} | {fmt_inr(row['median'])} | {fmt_inr(row['p75'])} | {fmt_inr(row['p90'])} | {fmt_inr(row['p95'])} |")
a("")

a("### 2c. Fuel & Transmission Mix per Band")
a("")
a("| Band | Count | Fuel Mix | Transmission Mix | Top Brands |")
a("| :--- | :---: | :--- | :--- | :--- |")
for row in band_rows:
    a(f"| **{row['band']}** | {row['count']:,} | {row['fuel_mix']} | {row['tx_mix']} | {row['top_brands']} |")
a("")

a("### 2d. Ownership & Certification per Band")
a("")
a("| Band | Count | Single-Owner % | Certified % |")
a("| :--- | :---: | :---: | :---: |")
for row in band_rows:
    cert = f"{row['certified_pct']:.1f}%" if row.get("certified_pct") is not None else "N/A"
    a(f"| **{row['band']}** | {row['count']:,} | {row['single_owner_pct']:.1f}% | {cert} |")
a("")
a("---")
a("")

# Brand Analysis
a("## 3. Brand-Wise Analysis")
a("")
a("### 3a. All Brands - Volume & Pricing (sorted by listing count)")
a("")
a("| Rank | Brand | Count | Share | Mean | Median | Min | Max | Top Model |")
a("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
for rank, (_, row) in enumerate(brand_df.iterrows(), 1):
    a(f"| {rank} | **{row['brand']}** | {row['count']:,} | {row['share_pct']:.1f}% | {fmt_inr(row['mean'])} | {fmt_inr(row['median'])} | {fmt_inr(row['min'])} | {fmt_inr(row['max'])} | {row['top_model']} |")
a("")

a("### 3b. Brand Segment Distribution (Count of listings per segment)")
a("")
a("| Brand | Budget (0-3L) | Economy (3-6L) | Mid (6-12L) | Premium (12-20L) | Luxury (20L+) | Total |")
a("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
for _, row in brand_df.iterrows():
    a(f"| **{row['brand']}** | {row['budget_count']} | {row['economy_count']} | {row['mid_count']} | {row['premium_count']} | {row['luxury_count']} | {row['count']} |")
a("")

a("### 3c. Brand Fuel & Transmission Mix")
a("")
a("| Brand | Count | Fuel Mix | Transmission Mix |")
a("| :--- | :---: | :--- | :--- |")
for _, row in brand_df.iterrows():
    a(f"| **{row['brand']}** | {row['count']:,} | {row['fuel_mix']} | {row['tx_mix']} |")
a("")
a("---")
a("")

# Heat Map
a("## 4. Top-15 Brands x Price Band Heat-Map (Listing Counts)")
a("")
band_labels = [b[0] for b in PRICE_BANDS]
header = "| Brand | " + " | ".join(band_labels) + " |"
sep    = "| :--- | " + " | ".join(":---:" for _ in band_labels) + " |"
a(header)
a(sep)
for brand in top_brands_list:
    cells = " | ".join(str(heat_map[brand].get(bl, 0)) for bl in band_labels)
    a(f"| **{brand}** | {cells} |")
a("")
a("---")
a("")

# Key Findings
a("## 5. Key Findings & Insights")
a("")

most_popular_band = max(band_rows, key=lambda r: r["count"])
cheapest_band = min(band_rows, key=lambda r: r["mean"])
most_expensive_band = max(band_rows, key=lambda r: r["mean"])
top_brand = brand_df.iloc[0]
highest_median_brand = brand_df.sort_values("median", ascending=False).iloc[0]
lowest_median_brand = brand_df.sort_values("median").iloc[0]

a(f"1. **Most popular price band:** **{most_popular_band['band']}** with {most_popular_band['count']:,} listings ({most_popular_band['share_pct']:.1f}% of all cars).")
a(f"2. **Most affordable band (by mean):** **{cheapest_band['band']}** — mean price {fmt_inr(cheapest_band['mean'])}.")
a(f"3. **Highest avg price band:** **{most_expensive_band['band']}** — mean {fmt_inr(most_expensive_band['mean'])}.")
a(f"4. **Largest brand by volume:** **{top_brand['brand']}** ({top_brand['count']:,} cars, {top_brand['share_pct']:.1f}% share).")
a(f"5. **Highest median price brand (>=5 cars):** **{highest_median_brand['brand']}** — median {fmt_inr(highest_median_brand['median'])}.")
a(f"6. **Lowest median price brand (>=5 cars):** **{lowest_median_brand['brand']}** — median {fmt_inr(lowest_median_brand['median'])}.")
a("")

# Fuel overall
a("### Fuel Type Distribution (All Listings)")
a("")
a("| Fuel Type | Count | Share |")
a("| :--- | :---: | :---: |")
fuel_counts = df["fuel_type"].value_counts()
for ft, cnt in fuel_counts.items():
    a(f"| {ft.title()} | {cnt:,} | {100*cnt/total:.1f}% |")
a("")

# Transmission overall
a("### Transmission Distribution (All Listings)")
a("")
a("| Transmission | Count | Share |")
a("| :--- | :---: | :---: |")
tx_counts = df["transmission"].value_counts()
for tx, cnt in tx_counts.items():
    a(f"| {tx.title()} | {cnt:,} | {100*cnt/total:.1f}% |")
a("")

# Owner count
a("### Owner Count Distribution")
a("")
a("| Owner Count | Listings | Share |")
a("| :---: | :---: | :---: |")
oc_counts = df["owner_count"].value_counts().sort_index()
for oc, cnt in oc_counts.items():
    a(f"| {oc} | {cnt:,} | {100*cnt/total:.1f}% |")
a("")

# Age distribution
a("### Vehicle Age Distribution")
a("")
age_bins = [0, 3, 5, 8, 10, 15, 99]
age_labels_list = ["0-3 yrs", "4-5 yrs", "6-8 yrs", "9-10 yrs", "11-15 yrs", "16+ yrs"]
df["age_bin"] = pd.cut(df["vehicle_age"], bins=age_bins, labels=age_labels_list, right=True)
a("| Age Group | Listings | Share | Mean Price | Median Price |")
a("| :--- | :---: | :---: | :---: | :---: |")
for ag in age_labels_list:
    sub = df[df["age_bin"] == ag]
    if len(sub) == 0:
        continue
    mp = sub["selling_price"].mean()
    mdp = sub["selling_price"].median()
    a(f"| {ag} | {len(sub):,} | {100*len(sub)/total:.1f}% | {fmt_inr(mp)} | {fmt_inr(mdp)} |")
a("")
a("---")
a("")

# Top 20 models appendix
a("## 6. Appendix - Top 20 Models by Listing Volume")
a("")
model_stats = (
    df.groupby(["brand", "model"])["selling_price"]
    .agg(count="count", mean="mean", median="median", min="min", max="max")
    .reset_index()
    .sort_values("count", ascending=False)
    .head(20)
)
a("| Rank | Brand | Model | Count | Mean | Median | Min | Max |")
a("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for i, (_, row) in enumerate(model_stats.iterrows(), 1):
    a(f"| {i} | {row['brand']} | {row['model']} | {row['count']:,} | {fmt_inr(row['mean'])} | {fmt_inr(row['median'])} | {fmt_inr(row['min'])} | {fmt_inr(row['max'])} |")
a("")
a("---")
a("")
a("*Report generated automatically by PriceRef Analysis Suite.*")

md_text = "\n".join(lines)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md_text)
print(f"Markdown saved -> {OUT_MD}")
print("Done!")
