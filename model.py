import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import (r2_score,
                              mean_absolute_error,
                              mean_squared_error)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'age', 'km_driven', 'km_per_year',
    'age_km', 'is_low_mileage',
    'owner_num', 'fuel_type',
    'transmission', 'brand_tier',
    'brand', 'seller_type'
]

CAT_COLS = [
    'fuel_type', 'transmission',
    'brand_tier', 'brand', 'seller_type'
]


def prepare(df):
    data   = df.copy()
    le_map = {}

    for col in CAT_COLS:
        if col in data.columns:
            le        = LabelEncoder()
            data[col] = le.fit_transform(
                data[col].astype(str))
            le_map[col] = le

    available = [f for f in FEATURES
                 if f in data.columns]
    X = data[available]
    y = np.log1p(data['price_inr'])
    return X, y, le_map, available


def train(df):
    print("Preparing features...")
    X, y, le_map, available = prepare(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("Training base models...")

    # Base models
    xgb = XGBRegressor(
        n_estimators     = 1000,
        learning_rate    = 0.05,
        max_depth        = 7,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        verbosity        = 0
    )

    lgbm = LGBMRegressor(
        n_estimators  = 1000,
        learning_rate = 0.05,
        max_depth     = 7,
        subsample     = 0.8,
        random_state  = 42,
        verbose       = -1
    ) 

    cat = CatBoostRegressor(
        iterations    = 1000,
        learning_rate = 0.05,
        depth         = 7,
        verbose       = 0,
        random_seed   = 42
    )

    # Stacking ensemble
    print("Training stacking ensemble...")
    stack = StackingRegressor(
        estimators=[
            ('xgb',  xgb),
            ('lgbm', lgbm),
            ('cat',  cat)
        ],
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )

    stack.fit(X_tr, y_tr)

    # Evaluate
    y_pred        = stack.predict(X_te)
    y_actual      = np.expm1(y_te)
    y_pred_actual = np.expm1(y_pred)

    metrics = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(
                    y_actual,
                    y_pred_actual), 0),
        'rmse': round(np.sqrt(mean_squared_error(
                    y_actual,
                    y_pred_actual)), 0),
    }

    print(f"\nR2   : {metrics['r2']}")
    print(f"MAE  : Rs.{int(metrics['mae']):,}")
    print(f"RMSE : Rs.{int(metrics['rmse']):,}")

    # Save
    with open('models/model_artifacts.pkl',
              'wb') as f:
        pickle.dump({
            'model':    stack,
            'le_map':   le_map,
            'features': available,
            'metrics':  metrics
        }, f)

    print("\nModel saved to models/")
    return stack, metrics, le_map, available


def predict(input_dict):
    with open('models/model_artifacts.pkl',
              'rb') as f:
        art = pickle.load(f)

    model    = art['model']
    le_map   = art['le_map']
    features = art['features']

    inp = pd.DataFrame([input_dict])

    # Engineer features
    inp['age'] = 2026 - inp.get('year', 2020)
    inp['km_per_year'] = (
        inp['km_driven'] /
        inp['age'].replace(0, 0.5))
    inp['age_km'] = (
        inp['age'] * inp['km_driven'])
    inp['is_low_mileage'] = (
        (inp['km_per_year'] < 5000)
        .astype(int))

    for col, le in le_map.items():
        if col in inp.columns:
            val = str(inp[col].iloc[0])
            if val in le.classes_:
                inp[col] = le.transform([val])
            else:
                inp[col] = le.transform(
                    [le.classes_[0]])

    X         = inp[features].fillna(0)
    log_pred  = model.predict(X)[0]
    price     = np.expm1(log_pred)

    return {
        'predicted': round(price, 0),
        'low':       round(price * 0.92, 0),
        'high':      round(price * 1.08, 0),
        'metrics':   art['metrics']
    }


if __name__ == "__main__":
    print("Loading merged dataset...")
    df = pd.read_csv('data/merged_clean.csv')
    print(f"Training on {len(df):,} records...")
    train(df)