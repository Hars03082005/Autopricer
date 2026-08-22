"""Final Production Model Retraining & Packaging Script.

Architecture:
  - 5-Seed LightGBM Global Champion (Seeds: 42, 123, 456, 789, 2024)
  - Luxury CatBoost Specialist (depth=8, lr=0.04, l2=3.0)
  - Strategy D Brand-Aware Routing (Thresholds: Rs16L luxury brands, Rs22L global)
  - Output Bundle: model_registry/final/ensemble_bundle.pkl
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import pickle
import random
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT            = Path(__file__).resolve().parents[1]
DATA_DIR        = ROOT / 'ml_training' / 'data' / 'overall_only'
LUXURY_DIR      = ROOT / 'ml_training' / 'data' / 'luxury_augmented'
TRAIN_FILE      = DATA_DIR / 'train.csv'
VALID_FILE      = DATA_DIR / 'valid.csv'
TEST_FILE       = DATA_DIR / 'test.csv'
LUXURY_POOL     = LUXURY_DIR / 'luxury_train_pool.csv'
LUXURY_HOLDOUT  = LUXURY_DIR / 'luxury_holdout_untouched.csv'
OUTPUT_DIR      = ROOT / 'model_registry' / 'final'
OUTPUT_BUNDLE   = OUTPUT_DIR / 'ensemble_bundle.pkl'
LOG_FILE        = OUTPUT_DIR / 'training.log'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding='utf-8'),
    ],
)
log = logging.getLogger('final_train')

# 17-feature production schema
CAT_FEATURES = [
    'brand', 'model', 'variant',
    'locality', 'rto',
    'fuel_type', 'transmission', 'seller_type', 'color',
    'brand_model', 'model_variant',
]
NUM_FEATURES = [
    'vehicle_age', 'odometer_reading', 'km_per_year',
    'owner_count', 'certified', 'pincode',
]
RAW_CAT_FEATURES = [c for c in CAT_FEATURES if c not in ('brand_model', 'model_variant')]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = 'selling_price'

SEEDS = [42, 123, 456, 789, 2024]

LGB_BASE_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'learning_rate':     0.03,
    'num_leaves':        48,
    'min_child_samples': 25,
    'feature_fraction':  0.70,
    'bagging_fraction':  0.90,
    'bagging_freq':      5,
    'lambda_l1':         0,
    'lambda_l2':         3,
    'verbosity':         -1,
}
LGB_ROUNDS     = 5000
LGB_EARLY_STOP = 150

CB_LUXURY_PARAMS = {
    'iterations':            4000,
    'learning_rate':         0.04,
    'depth':                 8,
    'l2_leaf_reg':           3.0,
    'loss_function':         'RMSE',
    'eval_metric':           'RMSE',
    'random_seed':           42,
    'early_stopping_rounds': 100,
}

LUXURY_BRANDS = frozenset({
    'bmw', 'mercedes-benz', 'audi', 'volvo',
    'land rover', 'porsche', 'jaguar', 'lexus', 'mini',
})
LUXURY_BRAND_THRESHOLD  = 1_600_000.0
GLOBAL_LUXURY_THRESHOLD = 2_200_000.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _clean_str(s: pd.Series) -> pd.Series:
    return s.fillna('unknown').astype(str).str.strip().str.lower().replace('', 'unknown')


def _metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_raw)
    y_pred = np.expm1(y_pred_raw)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(np.log1p(y_true), np.log1p(y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    bias = float(np.mean(y_pred - y_true))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    p90  = float(np.percentile(np.abs(y_true - y_pred), 90))
    p95  = float(np.percentile(np.abs(y_true - y_pred), 95))
    p99  = float(np.percentile(np.abs(y_true - y_pred), 99))
    max_ae = float(np.max(np.abs(y_true - y_pred)))
    return {
        'MAE': round(mae, 2), 'MAPE': round(mape, 4), 'RMSE': round(rmse, 2),
        'R2': round(r2, 6), 'Bias': round(bias, 2), 'MedianAE': round(med_ae, 2),
        'P90': round(p90, 2), 'P95': round(p95, 2), 'P99': round(p99, 2),
        'MaxAE': round(max_ae, 2),
    }


def _price_band_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> dict[str, dict[str, Any]]:
    bands = [
        ('0-3L',   0,         300_000),
        ('3-7L',   300_000,   700_000),
        ('7-10L',  700_000,  1_000_000),
        ('10-15L', 1_000_000, 1_500_000),
        ('15-20L', 1_500_000, 2_000_000),
        ('20-30L', 2_000_000, 3_000_000),
        ('30-50L', 3_000_000, 5_000_000),
        ('50L+',   5_000_000, 999_000_000),
    ]
    result = {}
    for name, lo, hi in bands:
        mask = (y_true_price >= lo) & (y_true_price < hi)
        n = int(mask.sum())
        if n == 0:
            result[name] = {'n': 0}
            continue
        yt = y_true_price[mask]
        yp = y_pred_price[mask]
        ae = np.abs(yt - yp)
        result[name] = {
            'n':        n,
            'MAE':      round(float(ae.mean()), 2),
            'MAPE':     round(float(np.mean(ae / (yt + 1e-8)) * 100), 4),
            'RMSE':     round(float(math.sqrt(mean_squared_error(yt, yp))), 2),
            'Bias':     round(float((yp - yt).mean()), 2),
            'MedianAE': round(float(np.median(ae)), 2),
        }
    return result


def step1_verify_schema():
    log.info('='*72)
    log.info('STEP 1 -- FEATURE SCHEMA VERIFICATION')
    log.info('='*72)
    log.info(f'  Categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}')
    log.info(f'  Numerical features  ({len(NUM_FEATURES)}): {NUM_FEATURES}')
    log.info(f'  Total features      : {len(ALL_FEATURES)}')
    log.info(f'  Target              : {TARGET}  (transformed: log1p)')
    assert len(ALL_FEATURES) == 17, f'Expected 17 features, got {len(ALL_FEATURES)}'
    assert len(CAT_FEATURES) == 11, f'Expected 11 cat features, got {len(CAT_FEATURES)}'
    assert len(NUM_FEATURES) == 6,  f'Expected 6 num features, got {len(NUM_FEATURES)}'
    log.info('  OK Schema verified: 17 features (11 cat + 6 num)')


def step2_data_integrity():
    log.info('='*72)
    log.info('STEP 2 -- DATA INTEGRITY CHECKS')
    log.info('='*72)

    def _check(df: pd.DataFrame, name: str) -> pd.DataFrame:
        n0 = len(df)
        df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET].between(50_000, 20_000_000)]
        for c in NUM_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in RAW_CAT_FEATURES:
            if c in df.columns:
                df[c] = df[c].fillna('unknown').astype(str).str.strip().str.lower()
        required = [c for c in ['vehicle_age', 'odometer_reading', 'km_per_year', 'owner_count'] if c in df.columns]
        df = df.dropna(subset=required)
        dups = df.duplicated().sum()
        log.info(f'  [{name}] {n0:,} raw -> {len(df):,} clean rows  |  {dups} duplicate rows')
        if dups > 0:
            df = df.drop_duplicates()
        leakage_cols = {'price', 'sale_price', 'label', 'resale_price'}
        found = leakage_cols & set(df.columns)
        if found:
            raise ValueError(f'[{name}] LEAKAGE COLUMNS DETECTED: {found}')
        log.info(f'  [{name}] OK  target range: Rs{df[TARGET].min():,.0f} - Rs{df[TARGET].max():,.0f}')
        return df

    df_train = _check(pd.read_csv(TRAIN_FILE, low_memory=False), 'train')
    df_valid  = _check(pd.read_csv(VALID_FILE, low_memory=False), 'valid')
    log.info('  OK Data integrity checks passed')
    return df_train, df_valid


def step3_verify_test_set():
    log.info('='*72)
    log.info('STEP 3 -- VERIFY TEST SET INTEGRITY')
    log.info('='*72)
    if not TEST_FILE.exists():
        raise FileNotFoundError(f'Test file not found: {TEST_FILE}')
    with open(TEST_FILE, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    log.info(f'  Test set: {len(df_test):,} rows  |  SHA-256: {sha[:24]}...')
    log.info('  OK Test set verified (not used in training)')
    return df_test


def _engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in RAW_CAT_FEATURES:
        if c in df.columns:
            df[c] = _clean_str(df[c])
        else:
            df[c] = 'unknown'
    df['brand_model']   = df['brand'] + '__' + df['model']
    df['model_variant'] = df['model'] + '__' + df['variant']
    return df


def _build_encoders_and_levels(df_train: pd.DataFrame) -> tuple[dict, dict, dict]:
    cat_levels = {}
    encoders = {}
    for col in CAT_FEATURES:
        if col in df_train.columns:
            vals = df_train[col].astype(str).fillna('unknown').unique().tolist()
        else:
            vals = []
        if 'unknown' not in vals:
            vals.append('unknown')
        cat_levels[col] = sorted(vals)
        enc = LabelEncoder()
        enc.fit(cat_levels[col])
        encoders[col] = enc
    medians = {}
    for col in NUM_FEATURES:
        if col in df_train.columns:
            m = df_train[col].median()
            medians[col] = 0.0 if pd.isna(m) else float(m)
        else:
            medians[col] = 0.0
    return cat_levels, encoders, medians


def _prepare_lgb_frame(df: pd.DataFrame, cat_levels: dict, encoders: dict, medians: dict) -> pd.DataFrame:
    df = _engineer_interactions(df)
    X = pd.DataFrame(index=df.index)
    for col in CAT_FEATURES:
        known = set(cat_levels.get(col, ['unknown']))
        if col in df.columns:
            s = df[col].apply(lambda x: x if x in known else 'unknown')
        else:
            s = pd.Series(['unknown'] * len(df), index=df.index)
        X[col] = encoders[col].transform(s)
    for col in NUM_FEATURES:
        med = medians.get(col, 0.0)
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce').fillna(med)
        else:
            X[col] = float(med)
    return X[ALL_FEATURES]


def step4_train_lightgbm(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> dict[str, Any]:
    log.info('='*72)
    log.info('STEP 4 -- TRAINING 5-SEED LIGHTGBM ENSEMBLE')
    log.info('='*72)
    log.info(f"  Seeds: {SEEDS}")
    log.info(f"  num_leaves={LGB_BASE_PARAMS['num_leaves']}  ff={LGB_BASE_PARAMS['feature_fraction']}  bf={LGB_BASE_PARAMS['bagging_fraction']}  l2={LGB_BASE_PARAMS['lambda_l2']}")

    df_tr = _engineer_interactions(df_train)
    cat_levels, encoders, medians = _build_encoders_and_levels(df_tr)

    X_tr = _prepare_lgb_frame(df_train, cat_levels, encoders, medians)
    X_v  = _prepare_lgb_frame(df_valid, cat_levels, encoders, medians)
    y_tr = np.log1p(df_train[TARGET].values)
    y_v  = np.log1p(df_valid[TARGET].values)

    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_v  = lgb.Dataset(X_v,  label=y_v, reference=ds_tr)

    seed_models = []
    seed_metrics = []
    seed_preds_val = []

    for seed in SEEDS:
        _set_seed(seed)
        t0 = time.perf_counter()
        params = {**LGB_BASE_PARAMS, 'seed': seed}
        model = lgb.train(
            params, ds_tr,
            num_boost_round=LGB_ROUNDS,
            valid_sets=[ds_v],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        elapsed = time.perf_counter() - t0
        preds_v = model.predict(X_v)
        m = _metrics(y_v, preds_v)
        seed_metrics.append(m)
        seed_models.append(model)
        seed_preds_val.append(preds_v)
        log.info(f"  Seed {seed:4d}: trees={model.num_trees():4d}  MAE=Rs{m['MAE']:,.0f}  MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  ({elapsed:.1f}s)")

    ens_preds_val = np.mean(seed_preds_val, axis=0)
    ens_m = _metrics(y_v, ens_preds_val)
    log.info(f"  5-Seed Ensemble: MAE=Rs{ens_m['MAE']:,.0f}  MAPE={ens_m['MAPE']:.2f}%  R2={ens_m['R2']:.4f}  Bias=Rs{ens_m['Bias']:,.0f}")
    log.info('  OK 5-seed LightGBM training complete')

    return {
        'models':               seed_models,
        'cat_levels':           cat_levels,
        'encoders':             encoders,
        'medians':              medians,
        'seed_metrics':         seed_metrics,
        'ensemble_val_metrics': ens_m,
        'val_preds':            ens_preds_val,
        'y_val':                y_v,
        'X_val':                X_v,
        'df_valid':             df_valid,
    }


def step5_train_luxury_specialist() -> dict[str, Any]:
    log.info('='*72)
    log.info('STEP 5 -- TRAINING LUXURY CATBOOST SPECIALIST')
    log.info('='*72)

    if not LUXURY_POOL.exists():
        raise FileNotFoundError(f'Luxury pool not found: {LUXURY_POOL}')
    if not LUXURY_HOLDOUT.exists():
        raise FileNotFoundError(f'Luxury holdout not found: {LUXURY_HOLDOUT}')

    df_pool = pd.read_csv(LUXURY_POOL, low_memory=False)
    log.info(f'  Luxury pool: {len(df_pool):,} rows')

    df_holdout = pd.read_csv(LUXURY_HOLDOUT, low_memory=False)
    log.info(f'  Luxury holdout: {len(df_holdout):,} rows  (UNTOUCHED -- evaluation only)')

    df_pool[TARGET] = pd.to_numeric(df_pool[TARGET], errors='coerce')
    df_pool = df_pool.dropna(subset=[TARGET])
    df_pool = df_pool[df_pool[TARGET] >= 1_000_000]

    _set_seed(42)
    idx = np.random.permutation(len(df_pool))
    split = int(len(df_pool) * 0.80)
    df_lux_tr = df_pool.iloc[idx[:split]].reset_index(drop=True)
    df_lux_v  = df_pool.iloc[idx[split:]].reset_index(drop=True)
    log.info(f'  Luxury split: train={len(df_lux_tr):,}  val={len(df_lux_v):,}')

    def _prep_cb(df: pd.DataFrame) -> pd.DataFrame:
        df = _engineer_interactions(df)
        out = pd.DataFrame(index=df.index)
        for c in CAT_FEATURES:
            out[c] = df[c].astype(str) if c in df.columns else 'unknown'
        for c in NUM_FEATURES:
            out[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0) if c in df.columns else 0.0
        return out[ALL_FEATURES]

    X_lux_tr = _prep_cb(df_lux_tr)
    X_lux_v  = _prep_cb(df_lux_v)
    y_lux_tr = np.log1p(df_lux_tr[TARGET].values)
    y_lux_v  = np.log1p(df_lux_v[TARGET].values)

    pool_tr = Pool(X_lux_tr, y_lux_tr, cat_features=CAT_FEATURES)
    pool_v  = Pool(X_lux_v,  y_lux_v,  cat_features=CAT_FEATURES)

    t0 = time.perf_counter()
    specialist = CatBoostRegressor(**CB_LUXURY_PARAMS, verbose=200)
    specialist.fit(pool_tr, eval_set=pool_v, use_best_model=True)
    elapsed = time.perf_counter() - t0

    sp_val_preds = specialist.predict(X_lux_v)
    sp_val_m = _metrics(y_lux_v, sp_val_preds)
    log.info(f"  Specialist val: trees={specialist.tree_count_}  MAE=Rs{sp_val_m['MAE']:,.0f}  MAPE={sp_val_m['MAPE']:.2f}%  R2={sp_val_m['R2']:.4f}  ({elapsed:.1f}s)")

    X_holdout = _prep_cb(df_holdout)
    y_holdout = np.log1p(pd.to_numeric(df_holdout[TARGET], errors='coerce').fillna(0).values)
    ho_preds  = specialist.predict(X_holdout)
    ho_m      = _metrics(y_holdout, ho_preds)
    log.info(f"  Specialist holdout: MAE=Rs{ho_m['MAE']:,.0f}  MAPE={ho_m['MAPE']:.2f}%  R2={ho_m['R2']:.4f}  Bias=Rs{ho_m['Bias']:,.0f}")
    log.info('  OK Luxury specialist training complete')

    return {
        'model':           specialist,
        'val_metrics':     sp_val_m,
        'holdout_metrics': ho_m,
        'prep_fn':         _prep_cb,
        'df_holdout':      df_holdout,
    }


def _apply_strategy_d(df_raw: pd.DataFrame, lgb_preds_log: np.ndarray, specialist_model: CatBoostRegressor, prep_fn: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_cb = prep_fn(df_raw)
    spec_preds = np.expm1(specialist_model.predict(X_cb))
    champ_preds = np.expm1(lgb_preds_log)
    brands = df_raw['brand'].fillna('unknown').astype(str).str.strip().str.lower().tolist() if 'brand' in df_raw.columns else ['unknown'] * len(df_raw)
    routing = []
    final = champ_preds.copy()
    for i in range(len(df_raw)):
        b  = brands[i]
        cp = float(champ_preds[i])
        if (b in LUXURY_BRANDS and cp >= LUXURY_BRAND_THRESHOLD) or (cp >= GLOBAL_LUXURY_THRESHOLD):
            final[i] = float(spec_preds[i])
            routing.append('specialist')
        else:
            routing.append('champion')
    return final, spec_preds, routing


def step7_evaluate_full_system_val(lgb_result: dict, lux_result: dict) -> dict[str, Any]:
    log.info('='*72)
    log.info('STEP 7 -- FULL SYSTEM EVALUATION (VALIDATION SET)')
    log.info('='*72)
    df_valid = lgb_result['df_valid']
    y_v      = lgb_result['y_val']
    lgb_preds_log = lgb_result['val_preds']
    final_preds, _, routing = _apply_strategy_d(df_valid, lgb_preds_log, lux_result['model'], lux_result['prep_fn'])
    y_true_price = np.expm1(y_v)
    n_spec  = routing.count('specialist')
    n_champ = routing.count('champion')
    log.info(f"  Routing (val): specialist={n_spec} ({100*n_spec/len(routing):.1f}%)  champion={n_champ}")
    final_log = np.log1p(np.clip(final_preds, 1, None))
    m = _metrics(y_v, final_log)
    log.info(f"  Full system val: MAE=Rs{m['MAE']:,.0f}  MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  Bias=Rs{m['Bias']:,.0f}")
    bands = _price_band_metrics(y_true_price, final_preds)
    for band, bm in bands.items():
        if bm.get('n', 0) > 0:
            log.info(f"    {band:8s}: N={bm['n']:5d}  MAE=Rs{bm['MAE']:>10,.0f}  MAPE={bm['MAPE']:5.2f}%")
    log.info('  OK Validation system evaluation complete')
    return {'metrics': m, 'band_metrics': bands, 'routing': {'specialist': n_spec, 'champion': n_champ}}


def step8_evaluate_test(lgb_result: dict, lux_result: dict, df_test_raw: pd.DataFrame) -> dict[str, Any]:
    log.info('='*72)
    log.info('STEP 8 -- FINAL EVALUATION ON UNTOUCHED TEST SET')
    log.info('='*72)
    cat_levels = lgb_result['cat_levels']
    encoders   = lgb_result['encoders']
    medians    = lgb_result['medians']
    models     = lgb_result['models']
    df_test = df_test_raw.copy()
    df_test[TARGET] = pd.to_numeric(df_test[TARGET], errors='coerce')
    df_test = df_test.dropna(subset=[TARGET])
    df_test = df_test[df_test[TARGET].between(50_000, 20_000_000)]
    X_test = _prepare_lgb_frame(df_test, cat_levels, encoders, medians)
    y_test = np.log1p(df_test[TARGET].values)
    lgb_preds_log = np.mean([m.predict(X_test) for m in models], axis=0)
    final_preds, spec_preds, routing = _apply_strategy_d(df_test, lgb_preds_log, lux_result['model'], lux_result['prep_fn'])
    y_true_price = df_test[TARGET].values
    n_spec  = routing.count('specialist')
    n_champ = routing.count('champion')
    log.info(f"  Test set: {len(df_test):,} rows  |  specialist={n_spec} ({100*n_spec/len(routing):.2f}%)  champion={n_champ} ({100*n_champ/len(routing):.2f}%)")
    final_log = np.log1p(np.clip(final_preds, 1, None))
    m = _metrics(y_test, final_log)
    log.info('')
    log.info('  FINAL SYSTEM -- TEST SET METRICS')
    log.info(f"    MAE       = Rs{m['MAE']:>12,.2f}")
    log.info(f"    MAPE      = {m['MAPE']:>10.4f}%")
    log.info(f"    RMSE      = Rs{m['RMSE']:>12,.2f}")
    log.info(f"    R2        = {m['R2']:>10.6f}")
    log.info(f"    Bias      = Rs{m['Bias']:>12,.2f}")
    log.info(f"    Median AE = Rs{m['MedianAE']:>12,.2f}")
    log.info(f"    P90       = Rs{m['P90']:>12,.2f}")
    log.info(f"    P95       = Rs{m['P95']:>12,.2f}")
    log.info(f"    P99       = Rs{m['P99']:>12,.2f}")
    log.info(f"    MaxAE     = Rs{m['MaxAE']:>12,.2f}")
    log.info('')
    OLD = {'MAE': 38980.36, 'MAPE': 6.74, 'R2': 0.9674}
    log.info(f"  Delta vs old variant_1:  MAE={m['MAE']-OLD['MAE']:+,.0f}  MAPE={m['MAPE']-OLD['MAPE']:+.2f}%  R2={m['R2']-OLD['R2']:+.4f}")

    bands = _price_band_metrics(y_true_price, final_preds)
    log.info('')
    log.info('  Price-Band Breakdown:')
    for band, bm in bands.items():
        if bm.get('n', 0) > 0:
            log.info(f"    {band:8s}: N={bm['n']:5d}  MAE=Rs{bm['MAE']:>10,.0f}  MAPE={bm['MAPE']:5.2f}%  RMSE=Rs{bm['RMSE']:>10,.0f}  Bias=Rs{bm['Bias']:>10,.0f}  MedianAE=Rs{bm['MedianAE']:>10,.0f}")

    luxury_tier_metrics = {}
    log.info('')
    log.info('  Luxury Tier Breakdown:')
    for name, thresh in [('>=15L', 1_500_000), ('>=20L', 2_000_000), ('>=30L', 3_000_000), ('>=50L', 5_000_000)]:
        mask = y_true_price >= thresh
        n = int(mask.sum())
        if n == 0:
            luxury_tier_metrics[name] = {'n': 0}
            continue
        yt = y_true_price[mask]
        yp = final_preds[mask]
        ae = np.abs(yt - yp)
        luxury_tier_metrics[name] = {
            'n':        n,
            'MAE':      round(float(ae.mean()), 2),
            'MAPE':     round(float(np.mean(ae / (yt + 1e-8)) * 100), 4),
            'RMSE':     round(float(math.sqrt(mean_squared_error(yt, yp))), 2),
            'Bias':     round(float((yp - yt).mean()), 2),
            'MedianAE': round(float(np.median(ae)), 2),
        }
        log.info(f"    {name:6s}: N={n:4d}  MAE=Rs{ae.mean():>10,.0f}  MAPE={luxury_tier_metrics[name]['MAPE']:5.2f}%  RMSE=Rs{luxury_tier_metrics[name]['RMSE']:>10,.0f}  Bias=Rs{luxury_tier_metrics[name]['Bias']:>10,.0f}  MedianAE=Rs{luxury_tier_metrics[name]['MedianAE']:>10,.0f}")

    # Mass-market performance (< 15L)
    mass_mask = y_true_price < 1_500_000
    mass_yt = y_true_price[mass_mask]
    mass_yp = final_preds[mass_mask]
    mass_ae = np.abs(mass_yt - mass_yp)
    mass_metrics = {
        'n':        int(mass_mask.sum()),
        'MAE':      round(float(mass_ae.mean()), 2),
        'MAPE':     round(float(np.mean(mass_ae / (mass_yt + 1e-8)) * 100), 4),
        'RMSE':     round(float(math.sqrt(mean_squared_error(mass_yt, mass_yp))), 2),
        'Bias':     round(float((mass_yp - mass_yt).mean()), 2),
        'MedianAE': round(float(np.median(mass_ae)), 2),
    }
    log.info('')
    log.info(f"  Mass-Market Tier (<15L): N={mass_metrics['n']}  MAE=Rs{mass_metrics['MAE']:,.0f}  MAPE={mass_metrics['MAPE']:.2f}%  RMSE=Rs{mass_metrics['RMSE']:,.0f}  Bias=Rs{mass_metrics['Bias']:,.0f}")

    # Brand Breakdown
    brand_metrics = {}
    log.info('')
    log.info('  Brand Breakdown:')
    brands_raw = df_test['brand'].fillna('unknown').astype(str).str.strip().str.lower()
    target_brands = ['bmw', 'mercedes-benz', 'audi', 'volvo', 'toyota', 'land rover', 'jeep', 'hyundai', 'maruti']
    for b in target_brands:
        mask = (brands_raw == b)
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_true_price[mask]
        yp = final_preds[mask]
        ae = np.abs(yt - yp)
        bm = {
            'n':        n,
            'MAE':      round(float(ae.mean()), 2),
            'MAPE':     round(float(np.mean(ae / (yt + 1e-8)) * 100), 4),
            'RMSE':     round(float(math.sqrt(mean_squared_error(yt, yp))), 2),
            'Bias':     round(float((yp - yt).mean()), 2),
            'MedianAE': round(float(np.median(ae)), 2),
        }
        brand_metrics[b] = bm
        log.info(f"    {b:15s}: N={n:4d}  MAE=Rs{bm['MAE']:>10,.0f}  MAPE={bm['MAPE']:5.2f}%  Bias=Rs{bm['Bias']:>10,.0f}")

    # Error Tail Analysis
    tail_metrics = {}
    log.info('')
    log.info('  Catastrophic Error Tail Analysis:')
    ae_arr = np.abs(y_true_price - final_preds)
    for thresh, label in [(200_000, '>2L'), (500_000, '>5L'), (1_000_000, '>10L'), (2_000_000, '>20L')]:
        count = int((ae_arr > thresh).sum())
        pct = round(100.0 * count / len(ae_arr), 2)
        tail_metrics[label] = {'count': count, 'pct': pct}
        log.info(f"    |Error| {label}: {count} cars ({pct:.2f}%)")

    log.info('  OK Test set evaluation complete')
    return {
        'metrics':             m,
        'band_metrics':        bands,
        'luxury_tier_metrics': luxury_tier_metrics,
        'mass_metrics':        mass_metrics,
        'brand_metrics':       brand_metrics,
        'tail_metrics':        tail_metrics,
        'n_test':              len(df_test),
        'routing':             {'specialist': n_spec, 'champion': n_champ, 'specialist_pct': round(100*n_spec/len(routing), 2), 'champion_pct': round(100*n_champ/len(routing), 2)},
    }


def package_bundle(lgb_result: dict, lux_result: dict, test_results: dict, val_results: dict) -> None:
    log.info('='*72)
    log.info('PACKAGING FINAL BUNDLE')
    log.info('='*72)
    import importlib
    lib_versions = {}
    for lib in ['lightgbm', 'catboost', 'numpy', 'pandas', 'sklearn', 'scipy']:
        try:
            mod = importlib.import_module(lib if lib != 'sklearn' else 'sklearn')
            lib_versions[lib] = getattr(mod, '__version__', 'unknown')
        except Exception:
            lib_versions[lib] = 'not installed'

    metadata = {
        'model_version':           'final',
        'training_timestamp':      datetime.now(timezone.utc).isoformat(),
        'python_version':          platform.python_version(),
        'library_versions':        lib_versions,
        'feature_schema':          ALL_FEATURES,
        'categorical_features':    CAT_FEATURES,
        'numerical_features':      NUM_FEATURES,
        'target':                  TARGET,
        'target_transform':        'log1p',
        'inverse_transform':       'expm1',
        'lightgbm_seeds':          SEEDS,
        'lightgbm_parameters':     LGB_BASE_PARAMS,
        'catboost_parameters':     CB_LUXURY_PARAMS,
        'luxury_brand_list':       sorted(list(LUXURY_BRANDS)),
        'luxury_threshold':        LUXURY_BRAND_THRESHOLD,
        'global_threshold':        GLOBAL_LUXURY_THRESHOLD,
        'training_dataset':        str(DATA_DIR),
        'luxury_training_dataset': str(LUXURY_POOL),
        'validation_dataset':      str(VALID_FILE),
        'evaluation_dataset':      str(TEST_FILE),
        'metrics': {
            'validation':          val_results['metrics'],
            'test':                test_results['metrics'],
            'price_bands':         test_results['band_metrics'],
            'luxury_tiers':        test_results['luxury_tier_metrics'],
            'mass_market':         test_results['mass_metrics'],
            'brands':              test_results['brand_metrics'],
            'error_tails':         test_results['tail_metrics'],
            'luxury_holdout':      lux_result['holdout_metrics'],
        },
        'routing_stats': {
            'test':                test_results['routing'],
            'validation':          val_results['routing'],
        },
        'architecture': '5-Seed LightGBM + Luxury CatBoost Specialist + Strategy D Routing',
        'variant_id':   'final',
    }

    bundle = {
        'lgb_models':        lgb_result['models'],
        'cat_levels':        lgb_result['cat_levels'],
        'encoders':          lgb_result['encoders'],
        'medians':           lgb_result['medians'],
        'luxury_specialist': lux_result['model'],
        'routing': {
            'luxury_brands':          sorted(list(LUXURY_BRANDS)),
            'luxury_brand_threshold': LUXURY_BRAND_THRESHOLD,
            'global_threshold':       GLOBAL_LUXURY_THRESHOLD,
            'strategy':               'D',
        },
        'metadata': metadata,
    }

    OUTPUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(OUTPUT_BUNDLE, 'wb') as f:
        pickle.dump(bundle, f, protocol=5)
    save_time = time.perf_counter() - t0
    size_mb = OUTPUT_BUNDLE.stat().st_size / 1e6
    log.info(f'  Bundle saved: {OUTPUT_BUNDLE}')
    log.info(f'  Size: {size_mb:.1f} MB  |  Save time: {save_time:.1f}s')

    meta_path = OUTPUT_DIR / 'model_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f'  Metadata JSON: {meta_path}')

    log.info('  Verifying bundle round-trip load...')
    t1 = time.perf_counter()
    with open(OUTPUT_BUNDLE, 'rb') as f:
        loaded = pickle.load(f)
    load_time = time.perf_counter() - t1
    assert len(loaded['lgb_models']) == 5
    assert 'luxury_specialist' in loaded
    assert 'routing' in loaded
    assert 'metadata' in loaded
    log.info(f'  OK Round-trip load verified ({load_time:.2f}s)')
    log.info(f"  OK Bundle: {len(loaded['lgb_models'])} LightGBM models + CatBoost specialist + routing + metadata")


def benchmark_latency(bundle_path: Path, df_test_raw: pd.DataFrame) -> dict[str, float]:
    log.info('='*72)
    log.info('BENCHMARKING INFERENCE LATENCY')
    log.info('='*72)
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    lgb_models = bundle['lgb_models']
    cat_levels = bundle['cat_levels']
    encoders   = bundle['encoders']
    medians    = bundle['medians']
    specialist = bundle['luxury_specialist']
    lux_brands = set(bundle['routing']['luxury_brands'])
    lux_thresh = bundle['routing']['luxury_brand_threshold']
    glob_thresh = bundle['routing']['global_threshold']

    def _predict_single(rec: dict) -> float:
        df = pd.DataFrame([rec])
        df = _engineer_interactions(df)
        X_lgb = pd.DataFrame(index=df.index)
        for col in CAT_FEATURES:
            known = set(cat_levels.get(col, ['unknown']))
            s = df[col].apply(lambda x: x if x in known else 'unknown')
            X_lgb[col] = encoders[col].transform(s)
        for col in NUM_FEATURES:
            med = medians.get(col, 0.0)
            X_lgb[col] = pd.to_numeric(df[col], errors='coerce').fillna(med)
        
        preds = [m.predict(X_lgb[ALL_FEATURES])[0] for m in lgb_models]
        cp = float(np.expm1(np.mean(preds)))
        b = str(rec.get('brand', 'unknown')).strip().lower()
        if (b in lux_brands and cp >= lux_thresh) or (cp >= glob_thresh):
            X_cb = pd.DataFrame(index=df.index)
            for c in CAT_FEATURES:
                X_cb[c] = df[c].astype(str) if c in df.columns else 'unknown'
            for c in NUM_FEATURES:
                X_cb[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0) if c in df.columns else 0.0
            sp = float(np.expm1(specialist.predict(X_cb[ALL_FEATURES])[0]))
            return sp
        return cp

    sample_record = df_test_raw.iloc[0].to_dict()
    # Warmup
    for _ in range(20):
        _predict_single(sample_record)

    # Measure single prediction
    n_single = 200
    t0 = time.perf_counter()
    for _ in range(n_single):
        _predict_single(sample_record)
    single_ms = ((time.perf_counter() - t0) / n_single) * 1000.0

    # Measure batch prediction (e.g. 100 records)
    batch_df = df_test_raw.iloc[:100]
    t0 = time.perf_counter()
    X_lgb = _prepare_lgb_frame(batch_df, cat_levels, encoders, medians)
    lgb_preds_log = np.mean([m.predict(X_lgb) for m in lgb_models], axis=0)
    def _prep_cb(df):
        d = _engineer_interactions(df)
        out = pd.DataFrame(index=d.index)
        for c in CAT_FEATURES:
            out[c] = d[c].astype(str) if c in d.columns else 'unknown'
        for c in NUM_FEATURES:
            out[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0) if c in d.columns else 0.0
        return out[ALL_FEATURES]
    _apply_strategy_d(batch_df, lgb_preds_log, specialist, _prep_cb)
    batch_ms = (time.perf_counter() - t0) * 1000.0

    log.info(f"  Single prediction latency : {single_ms:.2f} ms / request")
    log.info(f"  Batch 100 prediction time : {batch_ms:.2f} ms ({batch_ms/100:.2f} ms / record)")
    return {'single_ms': round(single_ms, 2), 'batch_100_ms': round(batch_ms, 2)}


def main():
    t_total = time.perf_counter()
    log.info('='*72)
    log.info('FINAL PRODUCTION TRAINING -- START')
    log.info(f'Output: {OUTPUT_BUNDLE}')
    log.info('='*72)

    step1_verify_schema()
    df_train, df_valid = step2_data_integrity()
    df_test_raw        = step3_verify_test_set()
    lgb_result         = step4_train_lightgbm(df_train, df_valid)
    lux_result         = step5_train_luxury_specialist()

    log.info('='*72)
    log.info('STEP 6 -- COMPONENT VALIDATION')
    log.info('='*72)
    ens_m = lgb_result['ensemble_val_metrics']
    log.info(f"  LightGBM 5-seed: MAE=Rs{ens_m['MAE']:,.0f}  MAPE={ens_m['MAPE']:.2f}%  R2={ens_m['R2']:.4f}")
    sp_m  = lux_result['val_metrics']
    log.info(f"  Luxury spec val: MAE=Rs{sp_m['MAE']:,.0f}  MAPE={sp_m['MAPE']:.2f}%  R2={sp_m['R2']:.4f}")
    ho_m  = lux_result['holdout_metrics']
    log.info(f"  Luxury holdout : MAE=Rs{ho_m['MAE']:,.0f}  MAPE={ho_m['MAPE']:.2f}%  R2={ho_m['R2']:.4f}")

    val_results  = step7_evaluate_full_system_val(lgb_result, lux_result)
    test_results = step8_evaluate_test(lgb_result, lux_result, df_test_raw)

    package_bundle(lgb_result, lux_result, test_results, val_results)
    benchmark_latency(OUTPUT_BUNDLE, df_test_raw)

    elapsed = time.perf_counter() - t_total
    log.info('='*72)
    log.info(f'FINAL PRODUCTION TRAINING -- COMPLETE  ({elapsed:.1f}s)')
    log.info(f'Bundle: {OUTPUT_BUNDLE}')
    log.info('='*72)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log.critical('Training failed', exc_info=True)
        sys.exit(1)
