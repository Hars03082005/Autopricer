"""
PriceRef — R2 / MAE / MAPE by Price Band & Brand
Runs the production champion ensemble on test + validation sets
and reports only R², MAE, MAPE per fine-grained band and per brand.
"""
import sys
import os
import json
import math
import pickle
from pathlib import Path

# ── stdout UTF-8 (Windows) ────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "model_registry" / "final" / "ensemble_bundle.pkl"
DATA_DIR    = ROOT / "ml_training" / "data" / "overall_only"
OUT_MD      = ROOT / "analysis" / "model_metrics_report.md"

# ── Price bands ────────────────────────────────────────────────────────────────
PRICE_BANDS = [
    ("0–1L",    0,          100_000),
    ("1–2L",    100_000,    200_000),
    ("2–3L",    200_000,    300_000),
    ("3–4L",    300_000,    400_000),
    ("4–5L",    400_000,    500_000),
    ("5–6L",    500_000,    600_000),
    ("6–8L",    600_000,    800_000),
    ("8–10L",   800_000,  1_000_000),
    ("10–12L", 1_000_000, 1_200_000),
    ("12–15L", 1_200_000, 1_500_000),
    ("15–20L", 1_500_000, 2_000_000),
    ("20–30L", 2_000_000, 3_000_000),
    ("30L+",   3_000_000, math.inf),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def metrics(y_true, y_pred):
    return r2(y_true, y_pred), mae(y_true, y_pred), mape(y_true, y_pred)

def fmt_inr(v):
    if v >= 100_000:
        return f"₹{v/100_000:.2f}L"
    return f"₹{v/1_000:.1f}K"

# ── Load champion predictor ────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "backend"))
from champion_predictor import ChampionPredictor  # noqa: E402

print(f"Loading bundle from {BUNDLE_PATH} ...")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
predictor = ChampionPredictor(bundle)
print("  Bundle loaded OK.")

# ── Load test + validation data ────────────────────────────────────────────────
print("Loading test + validation data ...")
dfs = []
for split_name, fname in [("validation", "valid.csv"), ("test", "test.csv")]:
    p = DATA_DIR / fname
    if p.exists():
        d = pd.read_csv(p)
        d["_split"] = split_name
        dfs.append(d)
        print(f"  Loaded {split_name}: {len(d):,} rows")

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0].reset_index(drop=True)
print(f"  Combined: {len(df):,} rows after cleaning")

# ── Run predictions in batches ─────────────────────────────────────────────────
print("Running predictions ...")
BATCH = 500
preds = []
for i in range(0, len(df), BATCH):
    chunk = df.iloc[i : i + BATCH].copy()
    chunk_result = predictor.predict_batch_df(chunk)
    preds.extend(chunk_result["predicted_price"].tolist())
    if (i // BATCH) % 5 == 0:
        print(f"  {i+len(chunk):,}/{len(df):,} done")

df["predicted_price"] = preds
print("  Predictions done.")

y_true_all = df["selling_price"].values
y_pred_all = df["predicted_price"].values

# ── Overall metrics ────────────────────────────────────────────────────────────
r2_all, mae_all, mape_all = metrics(y_true_all, y_pred_all)
print(f"\nOverall  R²={r2_all:.4f}  MAE={fmt_inr(mae_all)}  MAPE={mape_all:.2f}%\n")

# ── Per Price Band ─────────────────────────────────────────────────────────────
band_results = []
for label, lo, hi in PRICE_BANDS:
    mask = (df["selling_price"] >= lo) & (df["selling_price"] < hi)
    sub = df[mask]
    n = len(sub)
    if n < 5:
        continue
    yt = sub["selling_price"].values
    yp = sub["predicted_price"].values
    r2_v, mae_v, mape_v = metrics(yt, yp)
    band_results.append({"band": label, "n": n, "r2": r2_v, "mae": mae_v, "mape": mape_v})
    print(f"  {label:8s}  n={n:5d}  R²={r2_v:.4f}  MAE={fmt_inr(mae_v)}  MAPE={mape_v:.2f}%")

# ── Per Brand ──────────────────────────────────────────────────────────────────
df["brand_clean"] = df["brand"].str.strip().str.title()
brand_results = []
for brand, grp in df.groupby("brand_clean"):
    n = len(grp)
    if n < 10:
        continue
    yt = grp["selling_price"].values
    yp = grp["predicted_price"].values
    r2_v, mae_v, mape_v = metrics(yt, yp)
    avg_price = float(np.mean(yt))
    brand_results.append({
        "brand": brand, "n": n,
        "r2": r2_v, "mae": mae_v, "mape": mape_v,
        "avg_price": avg_price,
    })

brand_results.sort(key=lambda x: x["n"], reverse=True)
print("\nBrand results computed for", len(brand_results), "brands")

# ── Build Markdown ─────────────────────────────────────────────────────────────
lines = []
a = lines.append

a("# 📊 PriceRef — Model Performance Report")
a("")
a(f"**Dataset:** Test + Validation sets | **Total predictions:** {len(df):,}")
a(f"**Model:** 5-Seed LightGBM + Luxury CatBoost Specialist (Strategy D Routing)")
a("")
a("---")
a("")

# Overall
a("## Overall Model Metrics")
a("")
a("| Metric | Value |")
a("| :--- | :--- |")
a(f"| **R² Score** | **{r2_all:.4f}** |")
a(f"| **MAE** | **{fmt_inr(mae_all)}** |")
a(f"| **MAPE** | **{mape_all:.2f}%** |")
a(f"| **Sample Size** | **{len(df):,} cars** |")
a("")
a("---")
a("")

# Price Band
a("## Price Band–Wise Results")
a("")
a("| Price Band | N | R² | MAE | MAPE |")
a("| :--- | :---: | :---: | :---: | :---: |")
for row in band_results:
    r2_str = f"{row['r2']:.4f}" if not math.isnan(row['r2']) else "N/A"
    a(f"| **{row['band']}** | {row['n']:,} | {r2_str} | {fmt_inr(row['mae'])} | {row['mape']:.2f}% |")
a("")

# Observations
best_band = min((r for r in band_results if not math.isnan(r["r2"])), key=lambda r: r["mape"])
worst_band = max((r for r in band_results if not math.isnan(r["r2"])), key=lambda r: r["mape"])
a("> **Best MAPE band:** "
  f"**{best_band['band']}** → MAPE = {best_band['mape']:.2f}%, R² = {best_band['r2']:.4f}")
a("")
a("> **Worst MAPE band:** "
  f"**{worst_band['band']}** → MAPE = {worst_band['mape']:.2f}%, R² = {worst_band['r2']:.4f}")
a("")
a("---")
a("")

# Brand
a("## Brand–Wise Results")
a("")
a("| Rank | Brand | N | Avg Price | R² | MAE | MAPE |")
a("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
for i, row in enumerate(brand_results, 1):
    r2_str = f"{row['r2']:.4f}" if not math.isnan(row['r2']) else "N/A"
    a(f"| {i} | **{row['brand']}** | {row['n']:,} | {fmt_inr(row['avg_price'])} "
      f"| {r2_str} | {fmt_inr(row['mae'])} | {row['mape']:.2f}% |")
a("")

# Brand observations
valid_brands = [r for r in brand_results if not math.isnan(r["r2"])]
best_mape_brand  = min(valid_brands, key=lambda r: r["mape"])
worst_mape_brand = max(valid_brands, key=lambda r: r["mape"])
best_r2_brand    = max(valid_brands, key=lambda r: r["r2"])

a("> **Best MAPE brand:** "
  f"**{best_mape_brand['brand']}** → MAPE = {best_mape_brand['mape']:.2f}%, "
  f"R² = {best_mape_brand['r2']:.4f} (N={best_mape_brand['n']:,})")
a("")
a("> **Worst MAPE brand:** "
  f"**{worst_mape_brand['brand']}** → MAPE = {worst_mape_brand['mape']:.2f}%, "
  f"R² = {worst_mape_brand['r2']:.4f} (N={worst_mape_brand['n']:,})")
a("")
a("> **Best R² brand:** "
  f"**{best_r2_brand['brand']}** → R² = {best_r2_brand['r2']:.4f}, "
  f"MAPE = {best_r2_brand['mape']:.2f}% (N={best_r2_brand['n']:,})")
a("")
a("---")
a("")
a("*Generated by PriceRef metrics script from live model inference.*")

md_text = "\n".join(lines)
OUT_MD.write_text(md_text, encoding="utf-8")
print(f"\nSaved → {OUT_MD}")
print("Done!")
