import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── CLEAN CARDEKHO (Indian dataset) ─────────────────────────
def clean_cardekho(filepath):
    df = pd.read_csv(filepath)

    # Extract brand and model from name
    df['brand'] = df['name'].str.split().str[0]
    df['model'] = (df['name']
                     .str.split()
                     .str[1:3]
                     .str.join(' '))

    # Rename columns to standard
    df = df.rename(columns={
        'fuel':          'fuel_type',
        'km_driven':     'km_driven',
        'selling_price': 'price_inr',
        'owner':         'owner'
    })

    # Clean owner
    owner_map = {
        'First Owner':          1,
        'Second Owner':         2,
        'Third Owner':          3,
        'Fourth & Above Owner': 4,
        'Test Drive Car':       1
    }
    df['owner_num'] = df['owner'].map(owner_map).fillna(2)

    # Standardise fuel type
    df['fuel_type'] = (df['fuel_type']
                         .str.strip()
                         .str.title())

    # Standardise transmission
    df['transmission'] = (df['transmission']
                            .str.strip()
                            .str.title())

    # Standardise seller type
    df['seller_type'] = (df['seller_type']
                           .str.strip()
                           .str.title())

    # Feature engineering
    current_year      = 2026
    df['age']         = current_year - df['year']
    df['age']         = df['age'].clip(lower=0)
    df['km_per_year'] = (df['km_driven'] /
                          df['age'].replace(0, 0.5))
    df['age_km']      = df['age'] * df['km_driven']
    df['is_low_mileage'] = (
        df['km_per_year'] < 5000).astype(int)

    # Brand tier
    df['brand_tier'] = df['brand'].map({
        'Maruti':     'Budget',
        'Datsun':     'Budget',
        'Renault':    'Budget',
        'Hyundai':    'Mid',
        'Honda':      'Mid',
        'Tata':       'Mid',
        'Kia':        'Mid',
        'Nissan':     'Mid',
        'Ford':       'Mid',
        'Mahindra':   'Mid',
        'Volkswagen': 'Premium',
        'Skoda':      'Premium',
        'Toyota':     'Premium',
        'MG':         'Premium',
        'Jeep':       'Premium',
        'BMW':        'Luxury',
        'Mercedes':   'Luxury',
        'Audi':       'Luxury',
        'Volvo':      'Luxury',
        'Jaguar':     'Luxury',
        'Land':       'Luxury',
    }).fillna('Mid')

    # Remove outliers
    df = df[df['price_inr'] > 50000]
    df = df[df['price_inr'] < 10000000]
    df = df[df['km_driven'] > 100]
    df = df[df['km_driven'] < 500000]
    df = df[df['age'] >= 0]
    df = df[df['age'] <= 30]

    # Drop duplicates and nulls
    df = df.drop_duplicates()
    df = df.dropna(subset=['price_inr',
                            'km_driven',
                            'fuel_type',
                            'transmission'])

    # Select final columns
    df = df[[
        'brand', 'model', 'year', 'age',
        'km_driven', 'km_per_year', 'age_km',
        'is_low_mileage', 'fuel_type',
        'transmission', 'seller_type',
        'owner_num', 'brand_tier', 'price_inr'
    ]]

    df['source'] = 'cardekho'
    df = df.reset_index(drop=True)
    return df


# ── CLEAN VEHICLES/CRAIGSLIST (US dataset) ──────────────────
def clean_vehicles(filepath):
    df = pd.read_csv(filepath,
                     low_memory=False)

    # Keep only useful columns
    keep = ['price', 'year', 'manufacturer',
            'model', 'condition', 'cylinders',
            'fuel', 'odometer', 'title_status',
            'transmission', 'drive', 'type',
            'paint_color', 'state']
    df = df[[c for c in keep
             if c in df.columns]]

    # Rename to standard
    df = df.rename(columns={
        'price':        'price_usd',
        'manufacturer': 'brand',
        'fuel':         'fuel_type',
        'odometer':     'km_driven',
        'title_status': 'title'
    })

    # Convert USD to INR (approx rate)
    USD_TO_INR = 83.5
    df['price_inr'] = df['price_usd'] * USD_TO_INR

    # Convert miles to KM
    df['km_driven'] = df['km_driven'] * 1.60934

    # Remove outliers
    df = df[df['price_usd'] > 500]
    df = df[df['price_usd'] < 150000]
    df = df[df['km_driven'] > 100]
    df = df[df['km_driven'] < 500000]
    df = df.dropna(subset=['price_usd',
                            'year',
                            'km_driven'])

    # Standardise
    df['brand'] = (df['brand']
                     .str.strip()
                     .str.title()
                     .fillna('Unknown'))
    df['fuel_type'] = (df['fuel_type']
                         .str.strip()
                         .str.title()
                         .fillna('Petrol'))
    df['transmission'] = (df['transmission']
                            .str.strip()
                            .str.title()
                            .fillna('Manual'))

    # Map US fuel to Indian standard
    fuel_map = {
        'Gas':      'Petrol',
        'Diesel':   'Diesel',
        'Electric': 'Electric',
        'Hybrid':   'Hybrid',
        'Other':    'Petrol'
    }
    df['fuel_type'] = (df['fuel_type']
                         .map(fuel_map)
                         .fillna('Petrol'))

    # Feature engineering
    current_year      = 2026
    df['age']         = current_year - df['year']
    df['age']         = df['age'].clip(lower=0)
    df['km_per_year'] = (df['km_driven'] /
                          df['age'].replace(0, 0.5))
    df['age_km']      = df['age'] * df['km_driven']
    df['is_low_mileage'] = (
        df['km_per_year'] < 5000).astype(int)

    df['seller_type'] = 'Dealer'
    df['owner_num']   = 1
    df['brand_tier']  = 'Mid'
    df['model']       = df['model'].fillna(
                            'Unknown').str.title()

    # Remove duplicates
    df = df.drop_duplicates()

    # Select final columns
    df = df[[
        'brand', 'model', 'year', 'age',
        'km_driven', 'km_per_year', 'age_km',
        'is_low_mileage', 'fuel_type',
        'transmission', 'seller_type',
        'owner_num', 'brand_tier', 'price_inr'
    ]]

    df['source'] = 'craigslist'
    df = df.reset_index(drop=True)
    return df


# ── MERGE BOTH DATASETS ──────────────────────────────────────
def merge_datasets(cardekho_path,
                   vehicles_path):
    print("Loading CarDekho...")
    cd = clean_cardekho(cardekho_path)
    print(f"CarDekho clean: {len(cd):,} rows")

    print("Loading Vehicles (large)...")
    veh = clean_vehicles(vehicles_path)
    print(f"Vehicles clean: {len(veh):,} rows")

    merged = pd.concat([cd, veh],
                        ignore_index=True)
    merged = merged.dropna()
    merged = merged.reset_index(drop=True)

    print(f"\nMerged dataset: {len(merged):,} rows")
    print(f"Features: {list(merged.columns)}")
    return merged, cd, veh


# ── CLEANING REPORT ──────────────────────────────────────────
def get_report(cd_raw, veh_raw,
               cd_clean, veh_clean,
               merged):
    return {
        'cardekho_raw':     len(cd_raw),
        'cardekho_clean':   len(cd_clean),
        'vehicles_raw':     len(veh_raw),
        'vehicles_clean':   len(veh_clean),
        'merged_total':     len(merged),
        'features':         list(
            merged.columns),
        'cardekho_removed': (
            len(cd_raw) - len(cd_clean)),
        'vehicles_removed': (
            len(veh_raw) - len(veh_clean)),
    }


if __name__ == "__main__":
    merged, cd, veh = merge_datasets(
        'data/cardekho.csv',
        'data/vehicles.csv'
    )
    merged.to_csv(
        'data/merged_clean.csv',
        index=False)
    print("\nSaved to data/merged_clean.csv")
    print(merged.describe())