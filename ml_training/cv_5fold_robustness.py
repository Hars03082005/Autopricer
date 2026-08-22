"""5-Fold Cross-Validation Robustness Study — Final Production Architecture.

ROBUSTNESS EXPERIMENT ONLY.

This script:
  - Performs 5-fold CV on the development data (train.csv + valid.csv combined)
  - Trains the EXACT approved architecture on each fold:
      * 5-Seed LightGBM champion
      * Luxury CatBoost specialist
      * Strategy D routing
  - Fits preprocessing (encoders, medians, cat_levels) INSIDE each fold (no leakage)
  - Generates OOF predictions for every development record
  - Saves results to: ml_training/cv_results/final_architecture_5fold/

DOES NOT:
  - Touch model_registry/final/ensemble_bundle.pkl
  - Use the official test set (test.csv)
  - Use luxury_holdout_untouched.csv
  - Tune hyperparameters
  - Modify production configuration
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT          = Path(__file__).resolve().parents[1]
DATA_DIR      = ROOT / "ml_training" / "data" / "overall_only"
LUXURY_DIR    = ROOT / "ml_training" / "data" / "luxury_augmented"
TRAIN_FILE    = DATA_DIR / "train.csv"
VALID_FILE    = DATA_DIR / "valid.csv"
TEST_FILE     = DATA_DIR / "test.csv"
LUXURY_POOL   = LUXURY_DIR / "luxury_train_pool.csv"
PROD_BUNDLE   = ROOT / "model_registry" / "final" / "ensemble_bundle.pkl"
CV_DIR        = ROOT / "ml_training" / "cv_results" / "final_architecture_5fold"
CV_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE      = CV_DIR / "cv_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("cv_study")

SEEDS = [42, 123, 456, 789, 2024]
LGB_BASE_PARAMS = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.03,
    "num_leaves": 48, "min_child_samples": 25, "feature_fraction": 0.70,
    "bagging_fraction": 0.90, "bagging_freq": 5, "lambda_l1": 0, "lambda_l2": 3,
    "verbosity": -1,
}
LGB_ROUNDS     = 5000
LGB_EARLY_STOP = 150
CB_LUXURY_PARAMS = {
    "iterations": 4000, "learning_rate": 0.04, "depth": 8, "l2_leaf_reg": 3.0,
    "loss_function": "RMSE", "eval_metric": "RMSE", "random_seed": 42,
    "early_stopping_rounds": 100, "verbose": False,
}
LUXURY_BRANDS           = frozenset({"bmw","mercedes-benz","audi","volvo","land rover","porsche","jaguar","lexus","mini"})
LUXURY_BRAND_THRESHOLD  = 1_600_000.0
GLOBAL_LUXURY_THRESHOLD = 2_200_000.0
RAW_CAT_FEATURES = ["brand","model","variant","locality","rto","fuel_type","transmission","seller_type","color"]
CAT_FEATURES     = RAW_CAT_FEATURES + ["brand_model","model_variant"]
NUM_FEATURES     = ["vehicle_age","odometer_reading","km_per_year","owner_count","certified","pincode"]
ALL_FEATURES     = CAT_FEATURES + NUM_FEATURES
TARGET           = "selling_price"
N_FOLDS, CV_SEED = 5, 42
PRICE_BANDS = [
    ("0-3L",0,300000),("3-7L",300000,700000),("7-10L",700000,1000000),
    ("10-15L",1000000,1500000),("15-20L",1500000,2000000),("20-30L",2000000,3000000),
    ("30-50L",3000000,5000000),("50L+",5000000,999000000),
]
OFFICIAL_BENCHMARK = {"MAE":39969.55,"MAPE":6.7342,"RMSE":97821.45,"R2":0.967513,"Bias":-2987.86,"MedianAE":20880.69}

def _set_seed(s): random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)

def _clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("unknown").astype(str).str.strip().str.lower().replace("","unknown")

def _clean_df(df):
    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(50000, 20000000)].copy()
    for c in NUM_FEATURES:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in RAW_CAT_FEATURES:
        df[c] = _clean_str(df[c]) if c in df.columns else "unknown"
    req = [c for c in ["vehicle_age","odometer_reading","km_per_year","owner_count"] if c in df.columns]
    return df.dropna(subset=req).reset_index(drop=True)

def _engineer(df):
    df = df.copy()
    df["brand_model"]   = df["brand"] + "__" + df["model"]
    df["model_variant"] = df["model"] + "__" + df["variant"]
    return df

def _fit_preprocessing(df_tr):
    cat_levels, encoders, medians = {}, {}, {}
    for col in CAT_FEATURES:
        vals = df_tr[col].astype(str).fillna("unknown").unique().tolist() if col in df_tr.columns else []
        if "unknown" not in vals: vals.append("unknown")
        cat_levels[col] = sorted(vals)
        enc = LabelEncoder(); enc.fit(cat_levels[col]); encoders[col] = enc
    for col in NUM_FEATURES:
        m = df_tr[col].median() if col in df_tr.columns else 0.0
        medians[col] = 0.0 if pd.isna(m) else float(m)
    return cat_levels, encoders, medians

def _encode_lgb(df, cat_levels, encoders, medians):
    df = _engineer(df); X = pd.DataFrame(index=df.index)
    for col in CAT_FEATURES:
        known = set(cat_levels.get(col, ["unknown"]))
        s = df[col].apply(lambda x: x if x in known else "unknown") if col in df.columns else pd.Series(["unknown"]*len(df), index=df.index)
        X[col] = encoders[col].transform(s)
    for col in NUM_FEATURES:
        med = medians.get(col, 0.0)
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(med) if col in df.columns else float(med)
    return X[ALL_FEATURES]

def _lux_frame(df, medians):
    df = _engineer(df); X = pd.DataFrame(index=df.index)
    for col in CAT_FEATURES:
        X[col] = df[col].astype(str) if col in df.columns else "unknown"
    for col in NUM_FEATURES:
        med = medians.get(col, 0.0)
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(med) if col in df.columns else float(med)
    y = np.log1p(df[TARGET].values)
    return X[ALL_FEATURES], y

def _metrics(yt, yp):
    ae = np.abs(yt - yp)
    return {
        "MAE": round(float(ae.mean()),2), "MAPE": round(float(np.mean(ae/(yt+1e-8))*100),4),
        "RMSE": round(float(math.sqrt(mean_squared_error(yt,yp))),2),
        "R2": round(float(r2_score(np.log1p(yt), np.log1p(np.maximum(yp,1.0)))),6),
        "Bias": round(float((yp-yt).mean()),2), "MedianAE": round(float(np.median(ae)),2),
        "P90": round(float(np.percentile(ae,90)),2), "P95": round(float(np.percentile(ae,95)),2),
        "P99": round(float(np.percentile(ae,99)),2), "MaxAE": round(float(ae.max()),2),
    }

def _bands(yt, yp):
    out = {}
    for name,lo,hi in PRICE_BANDS:
        mask = (yt>=lo)&(yt<hi); n = int(mask.sum())
        if n==0: out[name]={"n":0}; continue
        y1,y2 = yt[mask],yp[mask]; ae=np.abs(y1-y2)
        out[name]={"n":n,"MAE":round(float(ae.mean()),2),"MAPE":round(float(np.mean(ae/(y1+1e-8))*100),4),
                   "RMSE":round(float(math.sqrt(mean_squared_error(y1,y2))),2),"Bias":round(float((y2-y1).mean()),2)}
    return out

def main():
    log.info("="*72)
    log.info("5-FOLD CV — FINAL PRODUCTION ARCHITECTURE — ROBUSTNESS EXPERIMENT")
    log.info("="*72)
    if not PROD_BUNDLE.exists(): raise FileNotFoundError(f"Production bundle missing: {PROD_BUNDLE}")
    prod_size = PROD_BUNDLE.stat().st_size
    log.info(f"  Production bundle: {prod_size/1e6:.1f} MB — WILL NOT BE MODIFIED")

    df_train_raw = pd.read_csv(TRAIN_FILE, low_memory=False)
    df_valid_raw = pd.read_csv(VALID_FILE, low_memory=False)
    df_dev = _clean_df(pd.concat([df_train_raw, df_valid_raw], ignore_index=True))
    N_DEV  = len(df_dev)
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    df_lux  = pd.read_csv(LUXURY_POOL, low_memory=False)
    df_lux[TARGET] = pd.to_numeric(df_lux[TARGET], errors="coerce")
    df_lux = df_lux.dropna(subset=[TARGET])
    df_lux = df_lux[df_lux[TARGET]>=1000000].reset_index(drop=True)
    for c in RAW_CAT_FEATURES:
        if c in df_lux.columns: df_lux[c] = _clean_str(df_lux[c])
    N_LUX = len(df_lux)

    log.info(f"  Dev set: {N_DEV:,} rows (train+valid)  |  test set: {len(df_test):,} ISOLATED  |  luxury pool: {N_LUX:,}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    folds = list(kf.split(df_dev))
    log.info("\nFOLD SIZES:")
    for fi,(tr,va) in enumerate(folds,1):
        log.info(f"  Fold {fi}: train={len(tr):,}  val={len(va):,}")
        assert not (set(tr)&set(va)), f"Fold {fi} overlap!"
    assert sum(len(va) for _,va in folds)==N_DEV

    oof_champ   = np.full(N_DEV, np.nan)
    oof_spec    = np.full(N_DEV, np.nan)
    oof_final   = np.full(N_DEV, np.nan)
    oof_routing = np.full(N_DEV, "", dtype=object)
    oof_fold_id = np.full(N_DEV, -1, dtype=int)
    fold_results = []

    for fi,(tr_idx,va_idx) in enumerate(folds,1):
        log.info(f"\n{'='*72}\nFOLD {fi}/5\n{'='*72}")
        t0 = time.perf_counter()
        df_tr = df_dev.iloc[tr_idx].reset_index(drop=True)
        df_va = df_dev.iloc[va_idx].reset_index(drop=True)

        # Fit preprocessing on fold training data ONLY
        df_tr_eng = _engineer(df_tr)
        cat_levels, encoders, medians = _fit_preprocessing(df_tr_eng)
        log.info(f"  Preprocessing fitted on {len(df_tr):,} training rows")

        X_tr = _encode_lgb(df_tr, cat_levels, encoders, medians)
        X_va = _encode_lgb(df_va, cat_levels, encoders, medians)
        y_tr = np.log1p(df_tr[TARGET].values)
        y_va = np.log1p(df_va[TARGET].values)
        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

        seed_preds = []
        for seed in SEEDS:
            _set_seed(seed)
            m = lgb.train({**LGB_BASE_PARAMS,"seed":seed}, ds_tr, num_boost_round=LGB_ROUNDS,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(LGB_EARLY_STOP,verbose=False), lgb.log_evaluation(0)])
            seed_preds.append(m.predict(X_va))
            log.info(f"    Seed {seed}: {m.num_trees()} trees")

        champ_log   = np.mean(seed_preds, axis=0)
        champ_price = np.expm1(champ_log)

        # Luxury specialist — fold-specific 80/20 split of luxury pool
        _set_seed(42+fi)
        lux_idx  = np.random.permutation(N_LUX)
        lux_spl  = int(N_LUX*0.80)
        lux_tr   = df_lux.iloc[lux_idx[:lux_spl]].reset_index(drop=True)
        lux_va_d = df_lux.iloc[lux_idx[lux_spl:]].reset_index(drop=True)
        log.info(f"  Luxury pool: {len(lux_tr):,} train / {len(lux_va_d):,} val (fold-specific split)")
        X_ltr, y_ltr = _lux_frame(lux_tr, medians)
        X_lva, y_lva = _lux_frame(lux_va_d, medians)
        cat_idx = list(range(len(CAT_FEATURES)))
        cb = CatBoostRegressor(**CB_LUXURY_PARAMS)
        cb.fit(Pool(X_ltr,y_ltr,cat_features=cat_idx), eval_set=Pool(X_lva,y_lva,cat_features=cat_idx), verbose=False)
        log.info(f"  CatBoost best_iter={cb.best_iteration_}")

        X_va_cb, _ = _lux_frame(df_va, medians)
        spec_price  = np.expm1(cb.predict(X_va_cb))

        va_brands   = df_va["brand"].fillna("unknown").astype(str).str.strip().str.lower().values
        n_va        = len(df_va)
        final_price = champ_price.copy()
        routing     = np.array(["champion"]*n_va, dtype=object)
        for i in range(n_va):
            b,cp = va_brands[i], float(champ_price[i])
            if (b in LUXURY_BRANDS and cp>=LUXURY_BRAND_THRESHOLD) or cp>=GLOBAL_LUXURY_THRESHOLD:
                final_price[i]=spec_price[i]; routing[i]="specialist"

        n_spec = int((routing=="specialist").sum())
        n_chmp = n_va - n_spec
        y_va_price = np.expm1(y_va)
        gm = _metrics(y_va_price, final_price)
        bm = _bands(y_va_price, final_price)
        log.info(f"  Fold {fi}: MAE=Rs{gm['MAE']:,.0f}  MAPE={gm['MAPE']:.2f}%  R2={gm['R2']:.4f}  Bias=Rs{gm['Bias']:,.0f}  Specialist={n_spec} ({n_spec/n_va*100:.2f}%)")

        oof_champ[va_idx]   = champ_price
        oof_spec[va_idx]    = spec_price
        oof_final[va_idx]   = final_price
        oof_routing[va_idx] = routing
        oof_fold_id[va_idx] = fi

        fold_results.append({
            "fold":fi,"n_train":len(df_tr),"n_val":n_va,
            "global":gm,"price_bands":bm,
            "routing":{"n_champion":n_chmp,"n_specialist":n_spec,"specialist_pct":round(n_spec/n_va*100,4)},
        })
        log.info(f"  Fold {fi} done in {time.perf_counter()-t0:.1f}s")

    # OOF verification
    assert not np.any(np.isnan(oof_final))
    assert not np.any(oof_fold_id==-1)
    assert len(oof_final)==N_DEV
    y_dev = df_dev[TARGET].values.astype(float)
    oof_gm = _metrics(y_dev, oof_final)
    oof_bm = _bands(y_dev, oof_final)

    fold_maes  = [r["global"]["MAE"] for r in fold_results]
    fold_mapes = [r["global"]["MAPE"] for r in fold_results]
    fold_rmses = [r["global"]["RMSE"] for r in fold_results]
    fold_r2s   = [r["global"]["R2"]   for r in fold_results]
    fold_bias  = [r["global"]["Bias"] for r in fold_results]
    fold_meds  = [r["global"]["MedianAE"] for r in fold_results]
    fold_specs = [r["routing"]["specialist_pct"] for r in fold_results]
    summary = {k:{"mean":round(np.mean(v),4),"std":round(np.std(v),4)} for k,v in [
        ("MAE",fold_maes),("MAPE",fold_mapes),("RMSE",fold_rmses),("R2",fold_r2s),
        ("Bias",fold_bias),("MedianAE",fold_meds),("SpecPct",fold_specs)]}
    mae_cv  = summary["MAE"]["std"]  / summary["MAE"]["mean"]  * 100
    mape_cv = summary["MAPE"]["std"] / summary["MAPE"]["mean"] * 100
    stability = "STABLE" if (mae_cv<5 and mape_cv<5) else ("MODERATELY VARIABLE" if (mae_cv<15 and mape_cv<15) else "UNSTABLE")

    # Save OOF CSV
    oof_df = pd.DataFrame({
        "record_id": range(N_DEV), "fold": oof_fold_id,
        "actual_selling_price": y_dev,
        "champion_prediction": oof_champ.round(2),
        "luxury_specialist_prediction": oof_spec.round(2),
        "final_prediction": oof_final.round(2),
        "routing_decision": oof_routing,
    })
    oof_df.to_csv(CV_DIR/"oof_predictions.csv", index=False)

    results = {
        "experiment":"5-Fold CV Robustness — Final Production Architecture",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_folds":N_FOLDS,"cv_seed":CV_SEED,"n_dev":N_DEV,
        "n_test_isolated":len(df_test),"n_luxury_pool":N_LUX,
        "official_benchmark":OFFICIAL_BENCHMARK,
        "fold_results":fold_results,
        "cross_fold_summary":{k:{"mean":round(v["mean"],4),"std":round(v["std"],4)} for k,v in summary.items()},
        "oof_global_metrics":oof_gm,"oof_price_band_metrics":oof_bm,
        "best_fold":int(np.argmin(fold_maes))+1,"worst_fold":int(np.argmax(fold_maes))+1,
        "stability_verdict":stability,"mae_cv_pct":round(mae_cv,2),"mape_cv_pct":round(mape_cv,2),
        "production_decision":"PRODUCTION ARTIFACT REMAINS UNCHANGED.",
    }
    with open(CV_DIR/"cv_results.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("\n"+"="*72)
    log.info("CROSS-FOLD SUMMARY")
    log.info("="*72)
    log.info(f"  {'Fold':>4} {'MAE':>10} {'MAPE':>8} {'RMSE':>12} {'R2':>8} {'Bias':>10} {'MedAE':>10} {'Spec%':>7}")
    for r in fold_results:
        g=r["global"]; rt=r["routing"]
        log.info(f"  {r['fold']:>4} {g['MAE']:>10,.0f} {g['MAPE']:>7.2f}% {g['RMSE']:>12,.0f} {g['R2']:>8.4f} {g['Bias']:>10,.0f} {g['MedianAE']:>10,.0f} {rt['specialist_pct']:>6.2f}%")
    log.info(f"\n  MAE  : {summary['MAE']['mean']:,.0f} ± {summary['MAE']['std']:,.0f}")
    log.info(f"  MAPE : {summary['MAPE']['mean']:.2f}% ± {summary['MAPE']['std']:.3f}%")
    log.info(f"  RMSE : {summary['RMSE']['mean']:,.0f} ± {summary['RMSE']['std']:,.0f}")
    log.info(f"  R2   : {summary['R2']['mean']:.4f} ± {summary['R2']['std']:.4f}")
    log.info(f"  Bias : {summary['Bias']['mean']:,.0f} ± {summary['Bias']['std']:,.0f}")
    log.info(f"  MedAE: {summary['MedianAE']['mean']:,.0f} ± {summary['MedianAE']['std']:,.0f}")
    log.info(f"\n  OOF Global: MAE=Rs{oof_gm['MAE']:,.0f}  MAPE={oof_gm['MAPE']:.2f}%  R2={oof_gm['R2']:.4f}")
    log.info(f"\n  Official Test Benchmark (IMMUTABLE): MAE=Rs{OFFICIAL_BENCHMARK['MAE']:,.0f}  MAPE={OFFICIAL_BENCHMARK['MAPE']:.2f}%  R2={OFFICIAL_BENCHMARK['R2']:.4f}")
    log.info(f"\n  MAE CV%={mae_cv:.1f}%  MAPE CV%={mape_cv:.1f}% → Stability: {stability}")
    log.info(f"\n  PRODUCTION DECISION: THE PRODUCTION ARTIFACT REMAINS UNCHANGED.")
    log.info("\n"+"="*72)
    log.info("CV EXPERIMENT COMPLETE")
    log.info("="*72)
    return results

if __name__ == "__main__":
    main()
