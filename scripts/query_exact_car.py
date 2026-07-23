import pandas as pd
from pathlib import Path

csv_path = Path("ml_training/data/processed_pincode without owner-4.csv")
if not csv_path.exists():
    import glob
    files = glob.glob("**/*.csv", recursive=True)
    csv_path = Path(files[0]) if files else None

if csv_path and csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path} with {len(df):,} rows.")
    
    # Clean string columns for matching
    df["brand_clean"] = df["brand"].astype(str).str.strip().str.lower()
    df["model_clean"] = df["model"].astype(str).str.strip().str.lower()
    df["variant_clean"] = df["variant"].astype(str).str.strip().str.lower() if "variant" in df.columns else ""
    df["fuel_clean"] = df["fuel_type"].astype(str).str.strip().str.lower() if "fuel_type" in df.columns else ""
    
    # Match Maruti / Swift / 2017
    b_mask = df["brand_clean"].str.contains("maruti")
    m_mask = df["model_clean"] == "swift"
    y_mask = df["year"] == 2017
    
    match_swift_2017 = df[b_mask & m_mask & y_mask]
    print(f"\nTotal 2017 Maruti Swift rows in dataset: {len(match_swift_2017)}")
    
    # Print all 17 listings for 2017 Maruti Swift in the dataset
    print("\n" + "="*80)
    print("  ALL 2017 MARUTI SWIFT LISTINGS IN DATASET")
    print("="*80)
    
    cols = ["brand", "model", "variant", "year", "fuel_type", "transmission", "odometer_reading", "selling_price", "city", "locality"]
    
    for idx, row in match_swift_2017.iterrows():
        p = row["selling_price"]
        odo = row["odometer_reading"]
        v = row.get("variant", "unknown")
        f = row.get("fuel_type", "unknown")
        t = row.get("transmission", "unknown")
        loc = row.get("locality", "Bangalore")
        print(f"Row {idx:<6} | {row['brand']} {row['model']} {v:<8} | {row['year']} | Fuel: {f:<6} | Trans: {t:<9} | Odo: {odo:>7,} km | Price: Rs. {p:>7,} ({p/100000:.2f}L) | Locality: {loc}")
