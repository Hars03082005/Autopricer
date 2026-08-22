import json
import math
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model_artifacts"
DATA_DIR = ROOT / "ml_training" / "data"

# Find best available dataset
DATASET_PATH = None
for p in [
    DATA_DIR / "overall_only" / "test.csv",
    DATA_DIR / "overall_only" / "valid.csv",
    DATA_DIR / "processed_overall.csv",
]:
    if p.exists():
        DATASET_PATH = p
        break

if DATASET_PATH is None:
    csv_files = list(DATA_DIR.glob("**/*.csv"))
    if csv_files:
        DATASET_PATH = csv_files[0]

print(f"Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Clean target
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0].reset_index(drop=True)
y_true = df["selling_price"].values

# Load CatBoost Model
model_path = ARTIFACT_DIR / "vehicle_price_catboost.cbm"
if not model_path.exists():
    print(f"Model not found at {model_path}")
    sys.exit(1)

cb = CatBoostRegressor()
cb.load_model(str(model_path))

# Load metadata
with open(ARTIFACT_DIR / "model_metadata.json") as f:
    meta = json.load(f)

cat_features = meta.get("cat_features") or meta.get("categorical_features") or [
    "brand", "model", "variant", "fuel_type", "transmission", "seller_type", "locality", "rto", "color"
]

LUXURY_BRANDS = {"bmw","mercedes-benz","audi","jaguar","land rover","porsche",
                 "maserati","aston martin","bentley","rolls-royce","ferrari","lamborghini","hummer"}

BRAND_SEGMENT_MAP = {
    "maruti":"economy","maruti suzuki":"economy","datsun":"economy","bajaj":"economy",
    "chevrolet":"economy","fiat":"economy","opel":"economy","premier":"economy",
    "force":"economy","ashok leyland":"economy","ambassador":"economy",
    "hindustan motors":"economy","hyundai":"economy","honda":"economy","tata":"economy",
    "renault":"economy","nissan":"economy","ford":"economy","mitsubishi":"economy",
    "isuzu":"economy","citroen":"economy","dc":"economy",
    "volkswagen":"premium","skoda":"premium","toyota":"premium","mg":"premium",
    "jeep":"premium","kia":"premium","mini":"premium","volvo":"premium",
    "lexus":"premium","mahindra":"premium",
    "bmw":"luxury","mercedes-benz":"luxury","audi":"luxury","jaguar":"luxury",
    "land rover":"luxury","porsche":"luxury","maserati":"luxury","aston martin":"luxury",
    "bentley":"luxury","rolls-royce":"luxury","ferrari":"luxury","lamborghini":"luxury",
    "hummer":"luxury",
}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "rto_state" not in d.columns:
        d["rto_state"] = d["rto"].astype(str) if "rto" in d.columns else "unknown"
    if "color" not in d.columns:
        d["color"] = "unknown"
    if "luxury_brand" not in d.columns:
        d["luxury_brand"] = d["brand"].str.lower().map(lambda b: 1 if b in LUXURY_BRANDS else 0).astype(float)
    if "high_mileage" not in d.columns:
        d["high_mileage"] = (d["odometer_reading"] > 93_143).astype(float)
    if "inspected" not in d.columns:
        d["inspected"] = 0.0
    if "has_list_price" not in d.columns:
        d["has_list_price"] = 0.0
    if "segment_class" not in d.columns:
        d["segment_class"] = d["brand"].str.lower().map(lambda b: BRAND_SEGMENT_MAP.get(b, "economy"))
    for col in ["brand","model","variant","city","locality","rto","rto_state","color","segment_class","fuel_type","transmission","seller_type"]:
        if col in d.columns:
            d[col] = d[col].fillna("unknown").astype(str)
    return d

df_enriched = enrich(df)
feat_names = list(cb.feature_names_)

def align_frame(df_in: pd.DataFrame, feature_names: list, cat_cols: list) -> pd.DataFrame:
    frame = {}
    for col in feature_names:
        if col in df_in.columns:
            frame[col] = df_in[col].copy()
        else:
            frame[col] = "unknown" if col in cat_cols else 0.0
    frame_df = pd.DataFrame(frame, index=df_in.index)
    for col in cat_cols:
        if col in frame_df.columns:
            frame_df[col] = frame_df[col].fillna("unknown").astype(str)
    return frame_df[feature_names]

eval_frame = align_frame(df_enriched, feat_names, cat_features)
raw_pred = cb.predict(eval_frame)

# Model was trained on log1p(selling_price)
y_pred = np.expm1(raw_pred) if np.median(raw_pred) < 25 else raw_pred
y_pred = np.round(y_pred, -2) # round to nearest 100

abs_error = np.abs(y_true - y_pred)
pct_error = (abs_error / y_true) * 100
diff = y_pred - y_true

# Overall statistics
mae = np.mean(abs_error)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mape = np.mean(pct_error)
med_ape = np.median(pct_error)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

within_5 = np.mean(pct_error <= 5) * 100
within_10 = np.mean(pct_error <= 10) * 100
within_15 = np.mean(pct_error <= 15) * 100
within_20 = np.mean(pct_error <= 20) * 100

print(f"\n===== EVALUATION METRICS ON TEST SET ({len(df):,} cars) =====")
print(f"R² Score              : {r2:.4f}")
print(f"MAE (Mean Abs Error)  : ₹{mae:,.0f}")
print(f"RMSE (Root Mean Sq)   : ₹{rmse:,.0f}")
print(f"MAPE (Mean Abs % Err) : {mape:.2f}%")
print(f"Median Abs % Error    : {med_ape:.2f}%")
print(f"Accuracy within ±5%   : {within_5:.2f}%")
print(f"Accuracy within ±10%  : {within_10:.2f}%")
print(f"Accuracy within ±15%  : {within_15:.2f}%")
print(f"Accuracy within ±20%  : {within_20:.2f}%")

# Create complete results DataFrame
res_df = df.copy()
res_df["predicted_price"] = y_pred
res_df["absolute_error"] = abs_error
res_df["difference"] = diff
res_df["percentage_error"] = np.round(pct_error, 2)

# Format for output table
out_df = pd.DataFrame({
    "Brand": res_df["brand"].str.title(),
    "Model": res_df["model"].str.title(),
    "Variant": res_df["variant"].str.upper(),
    "Age (Yrs)": res_df["vehicle_age"].astype(float),
    "Odometer (KM)": res_df["odometer_reading"].astype(int),
    "Fuel": res_df["fuel_type"].str.capitalize(),
    "Transmission": res_df["transmission"].str.capitalize(),
    "Actual Price (₹)": res_df["selling_price"].astype(int),
    "Predicted Price (₹)": res_df["predicted_price"].astype(int),
    "Difference (₹)": res_df["difference"].astype(int),
    "Error (%)": res_df["percentage_error"]
})

# Save full output CSV
output_csv_path = ARTIFACT_DIR / "validation_actual_vs_predicted_3750_cars.csv"
out_df.to_csv(output_csv_path, index=False)
print(f"\nSaved full validation dataset ({len(out_df)} records) to:\n{output_csv_path}")

# Print JSON summary stats for output generation
summary_dict = {
    "count": int(len(df)),
    "r2": round(float(r2), 4),
    "mae": round(float(mae), 2),
    "rmse": round(float(rmse), 2),
    "mape": round(float(mape), 2),
    "median_ape": round(float(med_ape), 2),
    "within_5": round(float(within_5), 2),
    "within_10": round(float(within_10), 2),
    "within_15": round(float(within_15), 2),
    "within_20": round(float(within_20), 2),
}

with open(ARTIFACT_DIR / "validation_summary_stats.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

print("\nSample top 20 results:")
print(out_df.head(20).to_string(index=False))
