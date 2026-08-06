from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import math
import re
import gc
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.decision_engine import (
    calculate_decision,
    check_disqualifier,
    get_seasonal_multiplier,
    get_wheelr_risk_deductions,
    get_recon_cost,
    get_negotiation_trio,
    get_deal_health,
    shap_explanation,
    generate_similar_cars,
    get_market_range_result,
    get_locality_demand,
)
from backend.ensemble_predictor import EnsemblePredictor
from backend.brand_catalog import build_brand_catalog
from backend import model_registry
ROOT         = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model_artifacts"
_BRAND_TIER_MAP: dict[str, int] = {
    "datsun":        0,
    "maruti suzuki": 1, "renault": 1, "tata": 1, "chevrolet": 1,
    "fiat":          1, "bajaj":   1,
    "hyundai":       2, "honda":   2, "kia":  2, "ford":       2,
    "volkswagen":    2, "skoda":   2, "nissan": 2, "mitsubishi": 2,
    "mahindra":      2, "citroen": 2,
    "toyota":        3, "mg":      3, "jeep":  3,
    "bmw":           4, "mercedes-benz": 4, "audi":    4, "volvo": 4,
    "mini":          4, "lexus":   4, "jaguar": 4, "land rover": 4,
}
def resolve_variant_data(variant_id: Optional[str] = None) -> tuple[EnsemblePredictor, dict, dict, dict, str]:
    active_id = variant_id or model_registry.get_default_variant_id()
    if active_id:
        try:
            vdata = model_registry.get_variant(active_id)
            return (
                vdata["predictor"],
                vdata["segment_models"],
                vdata["metadata"],
                vdata["catalog"],
                active_id,
            )
        except FileNotFoundError as exc:
            if variant_id:
                log.error("Variant '%s' not found in registry: %s", variant_id, exc)
                raise HTTPException(
                    status_code=404,
                    detail=f"Model variant '{variant_id}' not found in registry. "
                           f"Available variants: {[v['variant_id'] for v in model_registry.list_variants()]}"
                )
            log.warning("Default variant '%s' not found in registry — falling back to model_artifacts/", active_id)
        except Exception as exc:
            log.error("Failed to load variant '%s': %s", active_id, exc, exc_info=True)
            if variant_id:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model variant '{variant_id}': {exc}"
                )
    meta_path = ARTIFACT_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"model_name": "CatBoostRegressor", "features": [], "categorical_features": []}
    pred = EnsemblePredictor.from_artifact_dir(ARTIFACT_DIR)
    seg_mods = {}
    for _seg in ["economy", "premium", "luxury"]:
        _p = ARTIFACT_DIR / f"ensemble_{_seg}.pkl"
        if _p.exists():
            seg_mods[_seg] = joblib.load(_p)
    cat_data = {}
    cat_p = ARTIFACT_DIR / "dataset_catalog.json"
    if cat_p.exists():
        with open(cat_p, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
    return pred, seg_mods, meta, cat_data, active_id or "default"
predictor       = None
SEGMENT_MODELS  = {}
METADATA        = {}
DATASET_CATALOG = {}
ACTIVE_VARIANT_ID = None
FEATURES        = []
CAT_FEATURES    = []
CURRENT_YEAR    = datetime.now().year
CONDITION_MULTIPLIERS = {
    "excellent": 1.05,
    "good":      1.00,
    "average":   0.92,
    "poor":      0.82,
}
BRAND_CATALOG   = {}
BRAND_SEGMENT_MAP: dict = {
    # Tier 0 → budget
    "datsun": "budget",
    # Tier 1 → economy
    "maruti suzuki": "economy", "maruti": "economy",
    "renault": "economy", "tata": "economy", "chevrolet": "economy",
    "fiat": "economy", "bajaj": "economy",
    "opel": "economy", "premier": "economy", "force": "economy",
    "ashok leyland": "economy", "ambassador": "economy", "dc": "economy",
    "hyundai": "mid", "honda": "mid", "kia": "mid", "ford": "mid",
    "volkswagen": "mid", "skoda": "mid", "nissan": "mid",
    "mitsubishi": "mid", "mahindra": "mid", "citroen": "mid", "isuzu": "mid",
    # Tier 3 → premium
    "toyota": "premium", "mg": "premium", "jeep": "premium",
    "mini": "premium", "volvo": "premium", "lexus": "premium",
    # Tier 4 → luxury
    "bmw": "luxury", "mercedes-benz": "luxury", "audi": "luxury",
    "jaguar": "luxury", "land rover": "luxury", "porsche": "luxury",
    "maserati": "luxury", "aston martin": "luxury", "bentley": "luxury",
    "rolls-royce": "luxury", "ferrari": "luxury", "lamborghini": "luxury",
    "hummer": "luxury",
}
log = logging.getLogger("priceref")
def _validate_variant_features(variant_id: str, metadata: dict) -> None:
    try:
        dummy = VehicleInput(
            brand="Honda", model="City", year=2021,
            fuel_type="Petrol", transmission="Manual", odometer_reading=30000,
        )
        frame    = build_features(dummy)
        expected = set(metadata.get("features", []))
        actual   = set(frame.columns)
        missing  = expected - actual
        extra    = actual   - expected
        if missing:
            log.warning(
                "[FEATURE DRIFT] Variant '%s': %d feature(s) expected by model "
                "but NOT produced by build_features(): %s",
                variant_id, len(missing), sorted(missing),
            )
        if extra:
            log.info(
                "[FEATURE DRIFT] Variant '%s': %d extra feature(s) in build_features() "
                "output that this model ignores: %s",
                variant_id, len(extra), sorted(extra),
            )
        if not missing and not extra:
            log.info("[FEATURE DRIFT] Variant '%s': feature schema OK (%d features).",
                     variant_id, len(expected))
    except Exception as exc:
        log.warning("[FEATURE DRIFT] Could not validate features for variant '%s': %s",
                    variant_id, exc)
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global predictor, SEGMENT_MODELS, METADATA, DATASET_CATALOG, ACTIVE_VARIANT_ID
    global FEATURES, CAT_FEATURES, CURRENT_YEAR, BRAND_CATALOG, BRAND_SEGMENT_MAP
    log.info("==> Loading ML models…")
    gc.collect()
    predictor, SEGMENT_MODELS, METADATA, DATASET_CATALOG, ACTIVE_VARIANT_ID = resolve_variant_data()
    gc.collect()
    FEATURES    = METADATA.get("features", [])
    CAT_FEATURES = METADATA.get("categorical_features", [])
    CURRENT_YEAR = METADATA.get("current_year_used_for_age", datetime.now().year)
    BRAND_CATALOG = build_brand_catalog()
    BRAND_SEGMENT_MAP = METADATA.get("brand_segment_map", BRAND_SEGMENT_MAP)
    _validate_variant_features(ACTIVE_VARIANT_ID, METADATA)
    log.info("==> ML models loaded. Active variant: %s", ACTIVE_VARIANT_ID)
    yield
    log.info("==> Shutting down.")
# FastAPI app
app = FastAPI(title="PriceRef ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Pydantic models
class VehicleInput(BaseModel):
    brand: str = "Honda"
    model: str = "City"
    variant: str = "unknown"
    year: int = 2021
    fuel_type: str = "Petrol"
    transmission: str = "Manual"
    odometer_reading: int = Field(28000, ge=0)
    fuel_efficiency: float = 17.5
    owner_count: int = Field(1, ge=1)
    engine_cc: int = Field(1497, ge=0)
    city: str = "Bangalore"
    locality: str = "Indiranagar"
    color: str = "unknown"
    inspected: bool = False
    condition: str = "Good"
    seller_asking_price: float = 0
    target_margin_pct: float = 10
    repair_buffer: float = 0
    seller_type: str = "dealer"
    pincode: Optional[str] = None
    model_variant: Optional[str] = None
DEFAULT_VENDOR_TYPE = {
    "engine":    "vendor",
    "tyre":      "vendor",
    "body":      "vendor",
    "interior":  "vendor",
    "electrical":"vendor",
}
class EnhancedEvaluateRequest(VehicleInput):
    accident_history: str = "none"
    registration_state: str = ""
    sale_state: str = ""
    loan_outstanding: bool = False
    seller_reason: str = "upgrading"
    engine_grade: str = "good"
    tyre_grade: str = "good"
    body_grade: str = "clean"
    interior_grade: str = "clean"
    electrical_grade: str = "all_good"
    vendor_type: dict = Field(default_factory=lambda: dict(DEFAULT_VENDOR_TYPE))
    rc_transfer_cost: float = Field(3500, ge=0, description="RC transfer cost entered by dealer")
    idv_value: float = Field(0, ge=0, description="Insurance Declared Value from policy (optional)")
class ReverseCalculateRequest(BaseModel):
    expected_sell_price: int = Field(..., ge=0)
    year: int = Field(2021, ge=1990)
    accident_history: str = "none"
    registration_state: str = ""
    sale_state: str = ""
    loan_outstanding: bool = False
    seller_reason: str = "upgrading"
    engine_grade: str = "good"
    tyre_grade: str = "good"
    body_grade: str = "clean"
    interior_grade: str = "clean"
    electrical_grade: str = "all_good"
    vendor_type: dict = Field(default_factory=lambda: dict(DEFAULT_VENDOR_TYPE))
    owner_count: int = Field(1, ge=1)
    odometer: int = Field(0, ge=0)
    target_margin_pct: float = Field(0.10, ge=0, le=1)
# Helper functions
def clean_text(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text if text else default
# Model-year alias map
# looking up the sanity-clamp band.
MODEL_YEAR_ALIASES: dict[str, list[tuple[int, int, str]]] = {
    "innova":        [(2014, 9999, "innova crysta")],
    "creta":         [(2020, 9999, "creta")],
    "nexon":         [(2020, 9999, "nexon")],
    "vitara brezza": [(2022, 9999, "brezza")],
    "brezza":        [(2022, 9999, "brezza")],
    "scorpio":       [(2022, 9999, "scorpio n")],
    "safari":        [(1900, 2020, "safari classic")],
    "thar":          [(1900, 2019, "thar gen1")],
    # i10 generations:
    "i10":           [(2020, 9999, "grand i10 nios"),
                      (2014, 2019, "grand i10")],
    "grand i10":     [(2020, 9999, "grand i10 nios")],
}
def _resolve_model_alias(model: str, year: int) -> str:
    model_key = clean_text(model)
    rules = MODEL_YEAR_ALIASES.get(model_key)
    if rules:
        for min_yr, max_yr, canonical in rules:
            if min_yr <= year <= max_yr:
                return canonical
    return model_key
_BRAND_ALIAS_MAP: dict[str, str] = {
    # Maruti variants
    "maruti":               "maruti suzuki",
    "marutisuzuki":         "maruti suzuki",
    "maruti-suzuki":        "maruti suzuki",
    "suzuki":               "maruti suzuki",
    # Mercedes variants
    "mercedes":             "mercedes-benz",
    "mercedes benz":        "mercedes-benz",
    "mercedesbenz":         "mercedes-benz",
    "merc":                 "mercedes-benz",
    "mercedes-benz":        "mercedes-benz",
    # Land Rover
    "land-rover":           "land rover",
    "landrover":            "land rover",
    "range rover":          "land rover",
    # Volkswagen
    "vw":                   "volkswagen",
    "volkswagon":           "volkswagen",
    # Others
    "hyundai motor":        "hyundai",
    "tata motors":          "tata",
    "honda cars":           "honda",
    "general motors":       "chevrolet",
    "chevy":                "chevrolet",
    "bajaj auto":           "bajaj",
    "fiat chrysler":        "fiat",
    "citroen":              "citroen",
}
def _normalize_brand(brand: str) -> str:
    b = clean_text(brand)
    return _BRAND_ALIAS_MAP.get(b, b)
def normalize_model_name(brand: str, model_name: str, year: int = 0) -> str:
    brand_clean = _normalize_brand(brand)
    model_clean = _resolve_model_alias(model_name, year)
    for b in [brand_clean] + list(_BRAND_ALIAS_MAP.values()):
        prefix = f"{b} "
        if model_clean.startswith(prefix):
            model_clean = model_clean[len(prefix):].strip()
            break
    _KNOWN_VARIANT_TOKENS = {
        "lxi", "vxi", "zxi", "zxi+", "vxi+", "ldi", "vdi", "zdi", "zdi+",
        "sx", "sx+", "sx(o)", "ex", "ex+", "s", "se", "sv", "sv+",
        "base", "top", "plus", "amt", "ags", "cvt", "ivtec", "dtec",
        "xm", "xz", "xz+", "xe", "xt", "xta",
        "magna", "sportz", "asta", "era", "d-lite", "d lite",
        "xl", "xls", "xxl",
        "lx", "vx", "zx",
        "g", "v", "z", "rs", "gt",
        "prestige", "luxury", "prime",
        "anniversary", "limited", "special", "edition",
        "4wd", "awd", "4x4", "2wd",
    }
    parts = model_clean.split()
    if len(parts) > 1 and parts[-1] in _KNOWN_VARIANT_TOKENS:
        candidate = " ".join(parts[:-1])
        if len(candidate) >= 3:
            model_clean = candidate
    return model_clean
def condition_to_score(condition: str) -> int:
    return {"excellent": 90, "good": 75, "average": 58, "poor": 38}.get(clean_text(condition), 65)
_CITY_FEATURE_MAP: dict[str, tuple[int, float, str, str]] = {
    "bangalore":  (2, 1.000, "ka-03", "bellahalli"),
    "bengaluru":  (2, 1.000, "ka-03", "bellahalli"),
    "mysuru":     (2, 0.071, "ka-09", "mysuru"),
    "mysore":     (2, 0.071, "ka-09", "mysuru"),
    "mumbai":     (1, 0.950, "mh-02", "andheri"),
    "pune":       (1, 0.720, "mh-12", "kothrud"),
    "delhi":      (1, 0.980, "dl-05", "lajpat nagar"),
    "new delhi":  (1, 0.980, "dl-05", "lajpat nagar"),
    "ncr":        (1, 0.900, "hr-26", "gurgaon"),
    "gurgaon":    (1, 0.750, "hr-26", "gurgaon"),
    "noida":      (1, 0.730, "up-16", "noida sector 18"),
    "hyderabad":  (1, 0.830, "ts-09", "banjara hills"),
    "chennai":    (1, 0.800, "tn-09", "anna nagar"),
    "kolkata":    (1, 0.780, "wb-02", "salt lake"),
    "ahmedabad":  (1, 0.700, "gj-01", "sg highway"),
    "surat":      (2, 0.420, "gj-05", "surat"),
    "jaipur":     (2, 0.380, "rj-14", "jaipur"),
    "lucknow":    (2, 0.320, "up-32", "lucknow"),
    "chandigarh": (2, 0.410, "ch-01", "chandigarh"),
    "kochi":      (2, 0.360, "kl-07", "edapally"),
    "coimbatore": (2, 0.290, "tn-37", "coimbatore"),
    "nagpur":     (2, 0.280, "mh-31", "nagpur"),
    "bhubaneswar": (3, 0.190, "od-02", "bhubaneswar"),
    "patna":      (3, 0.170, "br-01", "patna"),
    "indore":     (2, 0.250, "mp-09", "indore"),
    "bhopal":     (2, 0.220, "mp-04", "bhopal"),
    "visakhapatnam": (2, 0.230, "ap-31", "visakhapatnam"),
    "vadodara":   (2, 0.260, "gj-06", "vadodara"),
    "agra":       (3, 0.180, "up-80", "agra"),
    "varanasi":   (3, 0.160, "up-65", "varanasi"),
}
_BRAND_POPULARITY_LOG: dict[str, float] = {
    "renault":       6.28, "hyundai":      6.13, "honda":        6.06,
    "maruti suzuki": 5.89, "maruti":       5.89, "kia":          5.85,
    "ford":          5.64, "volkswagen":   5.54, "tata":         5.49,
    "jeep":          5.31, "mahindra":     5.08, "skoda":        4.76,
    "mg":            4.67, "nissan":       4.63, "datsun":       4.41,
    "toyota":        3.86, "chevrolet":    3.62, "fiat":         3.46,
    "bmw":           3.45, "audi":         3.43, "jaguar":       2.64,
    "mitsubishi":    2.56, "citroen":      2.53, "bajaj":        2.40,
    "volvo":         2.34, "mercedes-benz": 2.32, "land rover":  1.99,
    "lexus":         1.79, "mini":         1.78,
}
def _usage_category_num(km_per_year: float) -> float:
    if km_per_year < 8_000:  return 0.0
    if km_per_year < 15_000: return 1.0
    if km_per_year < 25_000: return 2.0
    return 3.0
def build_features(vehicle: VehicleInput) -> pd.DataFrame:
    vehicle_age = max(0, CURRENT_YEAR - int(vehicle.year))
    km          = max(0, float(vehicle.odometer_reading or 0))
    owner       = max(1, int(vehicle.owner_count or 1))
    km_per_year = min(km / max(vehicle_age, 0.5), 100_000)
    seg_class    = get_segment_class(vehicle.brand)
    color        = clean_text(getattr(vehicle, "color", None) or "unknown")
    high_mileage = 1 if km > 93_143 else 0
    luxury_brand = 1 if seg_class == "luxury" else 0
    inspected    = 1 if getattr(vehicle, "inspected", False) else 0
    brand_clean = _normalize_brand(vehicle.brand)
    brand_tier  = _BRAND_TIER_MAP.get(brand_clean, 1)
    city_clean  = clean_text(vehicle.city)
    city_info   = _CITY_FEATURE_MAP.get(city_clean)
    if city_info:
        locality_tier_val, density_norm, rto_val, _city_locality = city_info
    else:
        locality_tier_val, density_norm, rto_val, _city_locality = 2, 0.20, "unknown", city_clean
    _user_locality = clean_text(getattr(vehicle, "locality", None) or "")
    locality_val   = _user_locality if _user_locality and _user_locality != "unknown" else _city_locality
    # ── Core identifiers ─────────────────────────────────────────────────────
    variant_clean = clean_text(vehicle.variant or "unknown")
    model_clean   = normalize_model_name(vehicle.brand, vehicle.model, int(vehicle.year))
    certified_val = 1.0 if getattr(vehicle, "inspected", False) else 0.0
    raw_pin = getattr(vehicle, "pincode", None)
    pincode_val = float(raw_pin) if raw_pin and 100_000 <= float(raw_pin) <= 999_999 else np.nan
    ownership_trust_score = float({1: 100, 2: 75, 3: 50, 4: 25, 5: 10, 6: 10}.get(owner, 10))
    vehicle_health_score  = float(max(0.0, min(100.0,
        100.0 - (vehicle_age * 3) - (km / 10_000) - ((owner - 1) * 8))))
    age_km_interaction = float(vehicle_age) * float(km)
    is_high_mileage    = 1 if km_per_year > 15_000 else 0
    brand_age_penalty  = float(brand_tier) * float(vehicle_age)
    brand_km_penalty   = float(brand_tier) * (km / 10_000)
    pop_log            = _BRAND_POPULARITY_LOG.get(brand_clean, 3.50)
    usage_cat_num      = _usage_category_num(km_per_year)
    row = {
        "brand":        brand_clean,
        "model":        model_clean,
        "variant":      variant_clean,
        "locality":     locality_val,
        "rto":          rto_val,
        "fuel_type":    clean_text(vehicle.fuel_type),
        "transmission": clean_text(vehicle.transmission),
        "seller_type":  clean_text(getattr(vehicle, "seller_type", "dealer") or "dealer"),
        "color":        color,
        "vehicle_age":      float(vehicle_age),
        "odometer_reading": float(km),
        "km_per_year":      float(km_per_year),
        "owner_count":      float(owner),
        "certified":        certified_val,
        "pincode":          pincode_val,
        "segment_class":         seg_class,
        "rto_state":             rto_val,
        "brand_tier":            float(brand_tier),
        "age_km_interaction":    float(age_km_interaction),
        "ownership_trust_score": float(ownership_trust_score),
        "vehicle_health_score":  float(vehicle_health_score),
        "is_high_mileage":       float(is_high_mileage),
        "locality_tier":         float(locality_tier_val),
        "usage_category_num":    float(usage_cat_num),
        "locality_density_norm": float(density_norm),
        "popularity_score_log":  float(pop_log),
        "brand_age_penalty":     brand_age_penalty,
        "brand_km_penalty":      brand_km_penalty,
        "inspected":             float(inspected),
        "high_mileage":          float(high_mileage),
        "luxury_brand":          float(luxury_brand),
        # "has_list_price": 0.0   <- intentionally omitted
    }
    df = pd.DataFrame([row])
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df
def condition_multiplier(condition: str) -> float:
    return float(
        CONDITION_MULTIPLIERS.get(
            clean_text(condition, "good"),
            CONDITION_MULTIPLIERS.get("good", 1.0),
        )
    )
# Segment routing helpers
def get_segment_class(brand: str) -> str:
    return BRAND_SEGMENT_MAP.get(_normalize_brand(brand), "economy")
_MEDIAN_DEP_RATIO = 0.75
def _run_class_model(features: pd.DataFrame, artifact: dict) -> float:
    cb_f  = features.copy()
    lgb_f = features.copy()
    xgb_f = features.copy()
    for col in artifact.get("cat_features", []):
        if col not in features.columns:
            continue
        cat_levels = artifact.get("category_levels", {}).get(col, [])
        raw = features[col].astype(str)
        if cat_levels:
            known_levels = cat_levels if "unknown" in cat_levels else cat_levels + ["unknown"]
            raw = raw.where(raw.isin(cat_levels), "unknown")
            # CatBoost: plain string
            cb_f[col] = raw.astype(str)
            # LightGBM: pd.Categorical with explicit categories
            lgb_f[col] = pd.Categorical(raw, categories=known_levels)
            # XGBoost: integer label
            mapping = {cat: idx for idx, cat in enumerate(known_levels)}
            xgb_f[col] = raw.map(mapping).fillna(len(known_levels)).astype(int)
        else:
            cb_f[col]  = raw.astype(str)
            lgb_f[col] = raw.astype("category")
            # XGBoost: encode by category code
            cat_series = raw.astype("category")
            xgb_f[col] = cat_series.cat.codes.astype(int)
    weights = artifact["weights"]
    preds = {}
    if weights.get("catboost", 0) > 0:
        cb_model = artifact["catboost"]
        cb_cols   = cb_model.feature_names_
        # Fill any missing expected columns
        for _col in cb_cols:
            if _col not in cb_f.columns:
                cb_f[_col] = "unknown" if _col in artifact.get("cat_features", []) else 0.0
        cb_f_aligned = cb_f[cb_cols]
        for _col in artifact.get("cat_features", []):
            if _col in cb_f_aligned.columns:
                cb_f_aligned = cb_f_aligned.copy()
                cb_f_aligned[_col] = cb_f_aligned[_col].astype(str)
        preds["catboost"] = float(cb_model.predict(cb_f_aligned)[0])
    if weights.get("lightgbm", 0) > 0:
        preds["lightgbm"] = float(artifact["lightgbm"].predict(lgb_f)[0])
    if weights.get("xgboost", 0) > 0:
        preds["xgboost"]  = float(artifact["xgboost"].predict(xgb_f)[0])
    if not preds:
        cb_model = artifact["catboost"]
        cb_cols   = cb_model.feature_names_
        for _col in cb_cols:
            if _col not in cb_f.columns:
                cb_f[_col] = "unknown" if _col in artifact.get("cat_features", []) else 0.0
        cb_f_aligned = cb_f[cb_cols]
        for _col in artifact.get("cat_features", []):
            if _col in cb_f_aligned.columns:
                cb_f_aligned = cb_f_aligned.copy()
                cb_f_aligned[_col] = cb_f_aligned[_col].astype(str)
        preds["catboost"] = float(cb_model.predict(cb_f_aligned)[0])
        weights = {"catboost": 1.0}
    return sum(weights[k] * preds[k] for k in preds)
def _is_s5_model_known(brand: str, raw_model: str, year: int, cat_data: dict) -> bool:
    if not cat_data:
        return True
    b_key = clean_text(brand)
    m_key = clean_text(_resolve_model_alias(raw_model, year))
    brand_models = cat_data.get(b_key) or {}
    if not brand_models:
        for b_name in cat_data:
            if b_name in b_key or b_key in b_name:
                brand_models = cat_data[b_name]
                break
    if not brand_models:
        return False
    if m_key in brand_models:
        return True
    return any(m_key in m or m in m_key for m in brand_models)
# Core prediction
def predict_base_market_value(vehicle: VehicleInput, model_variant: Optional[str] = None) -> tuple[int, str, float]:
    var_id = model_variant or vehicle.model_variant
    pred_obj, seg_models, meta, cat_data, active_id = resolve_variant_data(var_id)
    resolved_model = _resolve_model_alias(vehicle.model, int(vehicle.year))
    if active_id in ("variant_4", "variant_s5"):
        if not _is_s5_model_known(vehicle.brand, resolved_model, int(vehicle.year), cat_data):
            base_v1, _, _ = predict_base_market_value(vehicle, model_variant="variant_1")
            s5_fallback_val = int(round((base_v1 * 1.08) / 500) * 500)
            return s5_fallback_val, f"s5 quality shop fallback (+8%) [{active_id}]", 0.0
    features  = build_features(vehicle)
    seg_class = get_segment_class(vehicle.brand)
    try:
        global_result = pred_obj.predict_with_variance(features)
        global_log    = global_result["log_price"]
        ensemble_variance = global_result.get("variance", 0.0)
        rough_price   = float(np.expm1(global_log))
        if not math.isfinite(rough_price) or rough_price <= 0:
            rough_price = 0
    except Exception:
        rough_price = 0
        ensemble_variance = 0.0
    def _pick_band(price: float) -> str:
        if price <= 0:
            if seg_class in ("budget", "economy"):
                return "0_6_lakh"
            if seg_class == "luxury":
                return "12_plus_lakh"
            return "0_6_lakh"
        if price < 600_000:
            return "0_6_lakh"
        elif price < 1_200_000:
            return "6_12_lakh"
        else:
            return "12_plus_lakh"
    price_band = _pick_band(rough_price)
    # the ML-first design goal.
    artifact   = seg_models.get(price_band)
    log_price    = global_log if rough_price > 0 else 0.0
    routing_note = f"global model ({seg_class}) [{active_id}]"
    if artifact and isinstance(artifact, dict) and "model" in artifact:
        try:
            seg_m      = artifact["model"]
            cat_levels = artifact.get("cat_levels", {})
            cb_f = features.copy()
            cat_cols = [c for c in CAT_FEATURES if c in cb_f.columns]
            # Normalise unseen categories
            for col in cat_cols:
                known = set(cat_levels.get(col, ["unknown"]))
                cb_f[col] = cb_f[col].astype(str).apply(lambda x: x if x in known else "unknown")
            for col in seg_m.feature_names_:
                if col not in cb_f.columns:
                    cb_f[col] = "unknown" if col in artifact.get("cat_features", []) or col in CAT_FEATURES else 0.0
                elif col in artifact.get("cat_features", []) or col in CAT_FEATURES:
                    cb_f[col] = cb_f[col].astype(str)
            band_log_price = float(seg_m.predict(cb_f[seg_m.feature_names_])[0])
            log_price    = band_log_price
            routing_note = f"segment model '{price_band}' used [{active_id}]"
        except Exception as e:
            log.warning("Segment model failed (%s), falling back to global: %s", price_band, e)
            routing_note = f"segment model error — global fallback [{active_id}]"
    elif rough_price > 0:
        routing_note = f"no segment model for '{price_band}' — global [{active_id}]"
    market_value = float(np.expm1(log_price))
    if not math.isfinite(market_value):
        market_value = 0
    market_value = max(50_000, min(market_value, 20_000_000))
    return int(round(market_value / 500) * 500), routing_note, ensemble_variance
def predict_market_value(vehicle: VehicleInput, model_variant: Optional[str] = None) -> dict:
    resolved_model = _resolve_model_alias(vehicle.model, int(vehicle.year))
    base_value, routing_note, ensemble_variance = predict_base_market_value(vehicle, model_variant=model_variant)
    seg_class = get_segment_class(vehicle.brand)
    age       = max(0, CURRENT_YEAR - int(vehicle.year))
    user_loc    = clean_text(getattr(vehicle, "locality", "") or vehicle.city or "")
    loc_uplift  = get_locality_demand(user_loc, segment=seg_class)
    adj_base    = base_value * (1.0 + loc_uplift)
    final_value = int(round(max(50_000, min(adj_base, 20_000_000)) / 500) * 500)
    variant_clean    = clean_text(getattr(vehicle, "variant", None) or "")
    variant_is_known = variant_clean not in ("", "unknown")
    irdai_note = ""
    if not variant_is_known:
        if age <= 0:    _irdai_dep = 5
        elif age == 1:  _irdai_dep = 15
        elif age == 2:  _irdai_dep = 20
        elif age == 3:  _irdai_dep = 30
        elif age == 4:  _irdai_dep = 40
        elif age == 5:  _irdai_dep = 50
        else:           _irdai_dep = min(80, 50 + (age - 5) * 4)
        irdai_note = (
            f"variant unknown — IRDAI schedule reference: "
            f"{_irdai_dep}% depreciation for {age}-yr vehicle (informational only)"
        )
    _meta_mape = METADATA.get("metrics", {}).get("mape", 0.0628) / 100.0 \
                 if METADATA.get("metrics", {}).get("mape", 0.0628) > 1 \
                 else METADATA.get("metrics", {}).get("mape", 0.0628)
    try:
        mr_result = get_market_range_result(
            brand              = str(vehicle.brand),
            model              = normalize_model_name(vehicle.brand, vehicle.model, int(vehicle.year)),
            variant            = variant_clean,
            fuel               = clean_text(vehicle.fuel_type),
            transmission       = clean_text(vehicle.transmission),
            year               = int(vehicle.year),
            odometer           = float(vehicle.odometer_reading or 0),
            owner_count        = int(vehicle.owner_count or 1),
            prediction         = float(final_value),
            model_mape         = float(_meta_mape),
            ensemble_variance  = float(ensemble_variance),
            seller_type        = clean_text(getattr(vehicle, "seller_type", "") or ""),
            locality           = clean_text(getattr(vehicle, "locality", "") or getattr(vehicle, "city", "") or ""),
        )
    except Exception as _mr_exc:
        import logging as _logging
        _logging.getLogger("uvicorn.error").warning(
            f"[predict_market_value] market_range_result failed for "
            f"{vehicle.brand} {vehicle.model} {vehicle.year}: {_mr_exc}"
        )
        _mape = float(_meta_mape)
        mr_result = {
            "price_min":               int(round(final_value * (1 - _mape) / 500) * 500),
            "price_max":               int(round(final_value * (1 + _mape) / 500) * 500),
            "price_median":            final_value,
            "market_range_comp_count": 0,
            "market_range_stage":      0,
            "market_range_stage_label": "error_fallback",
            "market_range_source":     "mape_fallback",
            "similar_cars":            [],
            "confidence":              "Low",
            "confidence_score":        0.0,
            "market_support":          "Weak",
            "comparables_used":        0,
            "average_similarity":      0.0,
            "ensemble_variance":       round(ensemble_variance, 6),
            "expected_model_error":    round(_mape * 100, 2),
            "confidence_case":         "low",
            "comp_p25":                int(round(final_value * (1 - _mape) / 500) * 500),
            "comp_p75":                int(round(final_value * (1 + _mape) / 500) * 500),
        }
    similar_cars = mr_result.get("similar_cars", [])
    outlier_flagged = False
    outlier_note    = ""
    if mr_result["market_range_source"] == "dataset" and mr_result["market_range_comp_count"] >= 5:
        comp_p25    = mr_result.get("comp_p25", mr_result["price_min"])
        comp_p75    = mr_result.get("comp_p75", mr_result["price_max"])
        iqr         = max(comp_p75 - comp_p25, 1)
        lower_fence = comp_p25 - 1.5 * iqr
        upper_fence = comp_p75 + 1.5 * iqr
        if not (lower_fence <= final_value <= upper_fence):
            outlier_flagged = True
            outlier_note = (
                f"ML prediction Rs.{final_value/1e5:.2f}L is outside the dataset "
                f"IQR fence Rs.{lower_fence/1e5:.2f}L – Rs.{upper_fence/1e5:.2f}L "
                f"({mr_result['market_range_comp_count']} comps, "
                f"comp P25=Rs.{comp_p25/1e5:.2f}L, P75=Rs.{comp_p75/1e5:.2f}L)"
            )
    var_id = model_variant or vehicle.model_variant
    _, seg_models, meta, _, active_id = resolve_variant_data(var_id)
    return {
        "base_market_value":          int(base_value),
        "market_value":               final_value,
        # ── Diagnostic / informational ──
        "condition_multiplier":       1.0,
        "condition_adjustment":       0,
        "condition_score":            condition_to_score(vehicle.condition),
        "segment_class":              seg_class,
        "segment_model_used":         seg_class in seg_models,
        "routing_note":               routing_note,
        "sanity_clamped":             False,
        "sanity_note":                "ML-first: no sanity clamp applied",
        "similar_anchor_note":        "",
        "irdai_note":                 irdai_note,
        "variant_is_known":           variant_is_known,
        "resolved_model":             resolved_model,
        "model_variant":              active_id,
        "outlier_flagged":            outlier_flagged,
        "outlier_note":               outlier_note,
        "price_min":                  mr_result["price_min"],
        "price_max":                  mr_result["price_max"],
        "price_median":               mr_result["price_median"],
        "market_range_comp_count":    mr_result["market_range_comp_count"],
        "market_range_stage":         mr_result["market_range_stage"],
        "market_range_stage_label":   mr_result["market_range_stage_label"],
        "market_range_source":        mr_result["market_range_source"],
        "similar_cars":               similar_cars,
        "confidence":                 mr_result.get("confidence", "Low"),
        "confidence_score":           mr_result.get("confidence_score", 0.0),
        "market_support":             mr_result.get("market_support", "Weak"),
        "comparables_used":           mr_result.get("comparables_used", 0),
        "average_similarity":         mr_result.get("average_similarity", 0.0),
        "ensemble_variance":          mr_result.get("ensemble_variance", 0.0),
        "expected_model_error":       mr_result.get("expected_model_error", 0.0),
        "confidence_case":            mr_result.get("confidence_case", "low"),
    }
def shap_like_explanation(vehicle: VehicleInput, market_value: int) -> list[dict]:
    age       = max(0, CURRENT_YEAR - int(vehicle.year))
    km        = max(0, int(vehicle.odometer_reading or 0))
    seg_class = get_segment_class(vehicle.brand)
    return shap_explanation(
        market_value    = float(market_value),
        vehicle_age     = age,
        km              = float(km),
        owner_count     = int(vehicle.owner_count or 1),
        condition       = str(vehicle.condition or "Good"),
        fuel            = str(vehicle.fuel_type or "Petrol"),
        transmission    = str(vehicle.transmission or "Manual"),
        city            = str(vehicle.city or ""),
        inspected       = bool(getattr(vehicle, "inspected", False)),
        fuel_efficiency = float(vehicle.fuel_efficiency or 0),
        brand           = str(vehicle.brand or ""),
        segment         = seg_class,
    )
def warnings_for(vehicle: VehicleInput, decision: dict, prediction: dict = None) -> list[str]:
    warnings = []
    age = max(0, CURRENT_YEAR - int(vehicle.year))
    if vehicle.odometer_reading > 100000:
        warnings.append("High odometer reading detected (>1L km)")
    if age > 8:
        warnings.append("Vehicle age exceeds 8 years")
    if clean_text(vehicle.condition) == "poor":
        warnings.append("Poor condition — reconditioning advised")
    if decision["risk_score"] >= 65:
        warnings.append("High acquisition risk — manual inspection recommended")
    if decision["confidence_score"] < 60:
        warnings.append("Lower confidence because of risk or missing data")
    # Commercial / taxi variant detection
    variant_raw = str(getattr(vehicle, "variant", "") or "").lower().strip()
    _COMMERCIAL_KEYWORDS = {"tour", "tour s", "tour h", "taxi", "fleet", "cng fleet"}
    if any(kw in variant_raw for kw in _COMMERCIAL_KEYWORDS):
        warnings.append(
            "⚠️ Commercial/taxi variant detected ('Tour'/'Taxi' registration). "
            "These have commercial (yellow plate) registration — lower resale value "
            "and a different buyer pool than personal variants. Verify plate type before acquiring."
        )
    if prediction:
        avg_sim = prediction.get("average_similarity", 0)
        comp_count = prediction.get("market_range_comp_count", 0)
        if comp_count > 0 and avg_sim < 65:
            warnings.append(
                f"Comparable match quality is low ({avg_sim:.1f}% avg). "
                f"Price range based mainly on ML model — fewer reliable market comps found."
            )
        if prediction.get("outlier_flagged"):
            warnings.append(f"ML Outlier: {prediction.get('outlier_note', 'Prediction outside comp IQR')}")
    return warnings
def evaluate_vehicle(vehicle: VehicleInput, model_variant: Optional[str] = None) -> dict:
    prediction   = predict_market_value(vehicle, model_variant=model_variant)
    market_value = prediction["market_value"]
    decision     = calculate_decision(vehicle, market_value)
    var_id = model_variant or vehicle.model_variant
    _, _, meta, _, active_id = resolve_variant_data(var_id)
    # ── Market-range price anchoring ─────────────────────────────────────────
    price_min    = prediction.get("price_min", 0)
    price_max    = prediction.get("price_max", 0)
    price_median = prediction.get("price_median", 0)
    mrange_src   = prediction.get("market_range_source", "mape_fallback")
    comp_count   = prediction.get("market_range_comp_count", 0)
    if mrange_src == "dataset" and comp_count >= 1 and price_median > 0:
        recon_cost   = decision.get("recon_cost",   18_000)
        holding_cost = decision.get("holding_cost",  5_000)
        doc_cost     = decision.get("doc_cost",      4_500)
        risk_buffer  = decision.get("risk_buffer",   3_000)
        veh_cat = decision.get("vehicle_category", "economy")
        _REAL_PROFIT_CAPS = {
            "economy":       40_000,
            "premium_hatch": 50_000,
            "compact_suv":   60_000,
            "mid_suv":       70_000,
            "luxury":        85_000,
        }
        real_max_profit = _REAL_PROFIT_CAPS.get(veh_cat, 55_000)
        # Sell: top of market range
        anchored_sell = int(round(price_max / 500) * 500)
        real_max_profit = min(real_max_profit, int(anchored_sell * 0.04))
        real_max_profit = max(real_max_profit, 15_000)
        total_non_profit_costs = recon_cost + holding_cost + doc_cost + risk_buffer
        anchored_buy = int(round(
            (anchored_sell - total_non_profit_costs - real_max_profit) / 500
        ) * 500)
        buy_floor    = int(round(price_min * 0.75 / 500) * 500)
        anchored_buy = max(anchored_buy, buy_floor)
        soft_ceil = int(round(price_min * 0.97 / 500) * 500)
        if anchored_buy > soft_ceil:
            profit_at_ceil = anchored_sell - soft_ceil - recon_cost - holding_cost - doc_cost
            if profit_at_ceil <= real_max_profit:
                anchored_buy = soft_ceil
        anchored_profit = max(0, anchored_sell - anchored_buy - recon_cost - holding_cost - doc_cost)
        anchored_margin = round((anchored_profit / max(anchored_buy, 1)) * 100, 1)
        nego_room    = int(anchored_buy * 0.04)
        opening_off  = int(round(max(0, anchored_buy - nego_room) / 500) * 500)
        target_off   = int(round(max(0, anchored_buy - nego_room * 0.35) / 500) * 500)
        walk_away    = int(round((anchored_buy * 1.01) / 500) * 500)
        walk_away    = min(walk_away, int(round(anchored_sell * 0.97 / 500) * 500))
        blend_weight = min(comp_count / 3.0, 1.0)
        if blend_weight < 1.0:
            wf_buy  = decision.get("recommended_buy_price",  anchored_buy)
            wf_sell = decision.get("recommended_sell_price", anchored_sell)
            anchored_buy  = int(round(
                (blend_weight * anchored_buy  + (1 - blend_weight) * wf_buy)  / 500) * 500)
            anchored_sell = int(round(
                (blend_weight * anchored_sell + (1 - blend_weight) * wf_sell) / 500) * 500)
            anchored_profit = max(0, anchored_sell - anchored_buy - recon_cost - holding_cost - doc_cost)
            anchored_margin = round((anchored_profit / max(anchored_buy, 1)) * 100, 1)
            nego_room    = int(anchored_buy * 0.04)
            opening_off  = int(round(max(0, anchored_buy - nego_room) / 500) * 500)
            target_off   = int(round(max(0, anchored_buy - nego_room * 0.35) / 500) * 500)
            walk_away    = int(round((anchored_buy * 1.01) / 500) * 500)
            walk_away    = min(walk_away, int(round(anchored_sell * 0.97 / 500) * 500))
        decision.update({
            "recommended_buy_price":  anchored_buy,
            "recommended_sell_price": anchored_sell,
            "dealer_acq_price":       anchored_buy,
            "suggested_sell_price":   anchored_sell,
            "expected_profit":        anchored_profit,
            "expected_margin_pct":    anchored_margin,
            "margin_pct":             anchored_margin,
            "margin_amt":             anchored_profit,
            "opening_offer":          opening_off,
            "target_offer":           target_off,
            "max_offer":              walk_away,
        })
        for row in decision.get("waterfall", []):
            if row.get("label") == "Recommended Buy Price":
                row["value"] = anchored_buy
            if row.get("label") == "ML Market Value":
                row["value"] = int(price_median)
                row["note"]  = f"Dataset-anchored median ({comp_count} comps)"
    return {
        **prediction,
        "model_name":         meta.get("model_name", "CatBoostRegressor"),
        "is_ml_powered":      True,
        "metrics":            meta.get("metrics", {}),
        "train_metrics":      meta.get("train_metrics", {}),
        "validation_metrics": meta.get("validation_metrics", {}),
        "test_metrics":       meta.get("test_metrics", {}),
        "overfitting_check":  meta.get("overfitting_check", {}),
        "shap":               shap_like_explanation(vehicle, market_value),
        "warnings":           warnings_for(vehicle, decision, prediction=prediction),
        "model_variant":      active_id,
        **decision,
    }
# API routes
@app.get("/health")
def health():
    pred, seg_mods, meta, _, active_id = resolve_variant_data()
    cbm_exists = (ARTIFACT_DIR / "vehicle_price_catboost.cbm").exists() or (model_registry.get_variant_path(active_id) is not None)
    return {
        "status":              "ok",
        "model_loaded":        cbm_exists,
        "ensemble_enabled":    meta.get("ensemble", {}).get("enabled", False),
        "model_name":          meta.get("model_name", "CatBoostRegressor"),
        "segmentation":        "segment_class",
        "segments_loaded":     list(seg_mods.keys()),
        "active_variant":      active_id,
    }
@app.get("/metadata")
def metadata(model_variant: Optional[str] = None):
    _, _, meta, _, _ = resolve_variant_data(model_variant)
    return meta
@app.get("/api/registry")
def get_registry():
    return {
        "default": model_registry.get_default_variant_id(),
        "variants": model_registry.list_variants(),
    }
@app.post("/api/registry/{variant_id}/activate")
def activate_variant_endpoint(variant_id: str):
    success = model_registry.activate_variant(variant_id)
    if not success:
        return {"status": "error", "message": f"Variant '{variant_id}' not found in registry"}
    global predictor, SEGMENT_MODELS, METADATA, DATASET_CATALOG, ACTIVE_VARIANT_ID
    predictor, SEGMENT_MODELS, METADATA, DATASET_CATALOG, ACTIVE_VARIANT_ID = resolve_variant_data()
    return {"status": "success", "active_variant": variant_id}
@app.get("/api/brands")
def get_brands():
    return {"brands": BRAND_CATALOG}
@app.get("/api/catalog")
def get_catalog(model_variant: Optional[str] = None):
    _, _, _, catalog, _ = resolve_variant_data(model_variant)
    return {"catalog": catalog or DATASET_CATALOG}
@app.get("/api/catalog/{brand}")
def get_catalog_brand(brand: str, model_variant: Optional[str] = None):
    _, _, _, catalog, _ = resolve_variant_data(model_variant)
    cat = catalog or DATASET_CATALOG
    key = brand.strip().lower()
    models_map = cat.get(key)
    if models_map is None:
        for cat_brand in cat:
            if cat_brand.startswith(key) or key.startswith(cat_brand.split()[0]):
                models_map = cat[cat_brand]
                break
    if models_map is None:
        return {"brand": brand, "models": {}}
    return {"brand": brand, "models": models_map}
@app.get("/api/options")
def get_vehicle_options(
    brand: Optional[str] = None,
    model: Optional[str] = None,
    variant: Optional[str] = None,
):
    from backend.decision_engine import _load_dataset_df
    CURRENT = datetime.now().year
    all_years = [str(y) for y in range(CURRENT, CURRENT - 25, -1)]
    # Defaults when no dataset available
    default = {
        "fuel_types":     ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"],
        "transmissions":  ["Manual", "Automatic", "AMT", "CVT", "DCT", "IMT"],
        "years":          all_years,
    }
    if not brand:
        return default
    df = _load_dataset_df()
    if df is None or df.empty:
        return default
    # Normalise inputs
    bk = str(brand).strip().lower()
    mk = str(model).strip().lower() if model else None
    vk = str(variant).strip().lower() if variant else None
    # Filter by brand
    mask = df["brand"].str.lower().str.strip().eq(bk)
    if not mask.any():
        # Try partial match
        mask = df["brand"].str.lower().str.strip().str.contains(bk, na=False)
    sub = df[mask]
    if sub.empty:
        return default
    # Filter by model
    if mk:
        # Try exact, then resolve alias
        resolved_mk = normalize_model_name(brand, model, 0)
        m_mask = sub["model"].str.lower().str.strip().eq(mk)
        if not m_mask.any():
            m_mask = sub["model"].str.lower().str.strip().eq(resolved_mk.lower())
        if not m_mask.any():
            m_mask = sub["model"].str.lower().str.strip().str.contains(mk, na=False)
        if m_mask.any():
            sub = sub[m_mask]
    if vk and vk not in ("", "unknown"):
        vk_tokens = set(vk.lower().split())
        def _variant_overlap(cell):
            cell_tokens = set(str(cell).lower().split())
            return bool(vk_tokens & cell_tokens)
        if "variant" in sub.columns:
            v_mask = sub["variant"].apply(_variant_overlap)
            if v_mask.any():
                sub = sub[v_mask]
    # ── Fuel types ────────────────────────────────────────────────────────────
    fuel_col = "fuel_type" if "fuel_type" in sub.columns else None
    if fuel_col:
        raw_fuels = sub[fuel_col].dropna().str.strip().str.lower().unique().tolist()
        fuel_map = {
            "petrol": "Petrol", "diesel": "Diesel", "cng": "CNG",
            "electric": "Electric", "hybrid": "Hybrid", "lpg": "LPG",
        }
        fuels = sorted({fuel_map.get(f, f.title()) for f in raw_fuels if f})
    else:
        fuels = default["fuel_types"]
    # ── Transmissions ─────────────────────────────────────────────────────────
    tx_col = "transmission" if "transmission" in sub.columns else None
    if tx_col:
        raw_tx = sub[tx_col].dropna().str.strip().str.lower().unique().tolist()
        tx_map = {
            "manual": "Manual", "automatic": "Automatic",
            "amt": "AMT", "cvt": "CVT", "dct": "DCT", "imt": "IMT",
        }
        txs = sorted({tx_map.get(t, t.title()) for t in raw_tx if t})
    else:
        txs = default["transmissions"]
    # ── Years ─────────────────────────────────────────────────────────────────
    _PROD = {
        # Maruti Suzuki
        "alto":            (2000, 9999), "alto k10":       (2010, 2022),
        "swift":           (2005, 9999), "swift dzire":    (2008, 9999),
        "dzire":           (2008, 9999), "baleno":         (2015, 9999),
        "wagon r":         (1999, 9999), "celerio":        (2014, 9999),
        "ignis":           (2017, 9999), "s-cross":        (2015, 9999),
        "vitara brezza":   (2016, 2022), "brezza":         (2016, 9999),
        "ertiga":          (2012, 9999), "xl6":            (2019, 9999),
        "ciaz":            (2014, 9999), "s-presso":       (2019, 9999),
        "jimny":           (2023, 9999), "fronx":          (2023, 9999),
        "invicto":         (2023, 9999), "grand vitara":   (2022, 9999),
        "omni":            (1984, 2019), "eeco":           (2010, 9999),
        # Hyundai
        "i10":             (2007, 9999), "grand i10":      (2013, 9999),
        "i20":             (2008, 9999), "aura":           (2020, 9999),
        "verna":           (2006, 9999), "creta":          (2015, 9999),
        "venue":           (2019, 9999), "tucson":         (2016, 9999),
        "elantra":         (2012, 2023), "sonata":         (2001, 2014),
        "santro":          (1998, 2022), "xcent":          (2014, 2022),
        "ioniq":           (2021, 9999), "alcazar":        (2021, 9999),
        "kona":            (2019, 9999),
        # Tata
        "nexon":           (2017, 9999), "harrier":        (2019, 9999),
        "safari":          (2021, 9999), "punch":          (2021, 9999),
        "tiago":           (2016, 9999), "tigor":          (2017, 9999),
        "altroz":          (2020, 9999), "zest":           (2014, 2020),
        "bolt":            (2015, 2019), "manza":          (2009, 2016),
        "indica":          (1998, 2018), "indigo":         (2002, 2018),
        "sumo":            (1994, 2019), "safari dicor":   (2004, 2021),
        "hexa":            (2017, 2020), "aria":           (2010, 2017),
        "nano":            (2009, 2018), "curvv":          (2024, 9999),
        # Mahindra
        "xuv500":          (2011, 2021), "xuv300":         (2019, 9999),
        "xuv400":          (2023, 9999), "xuv700":         (2021, 9999),
        "scorpio":         (2002, 9999), "scorpio n":      (2022, 9999),
        "thar":            (2010, 9999), "bolero":         (2000, 9999),
        "bolero neo":      (2021, 9999), "marazzo":        (2018, 9999),
        "alturas":         (2018, 2023), "kuv100":         (2016, 9999),
        "be 6":            (2024, 9999), "xuv 3xo":        (2024, 9999),
        # Honda
        "city":            (1998, 9999), "amaze":          (2013, 9999),
        "jazz":            (2009, 2022), "wr-v":           (2017, 2023),
        "cr-v":            (2001, 9999), "hr-v":           (2022, 9999),
        "elevate":         (2023, 9999), "civic":          (2006, 2021),
        "accord":          (2001, 2017), "mobilio":        (2014, 2017),
        "br-v":            (2016, 2021), "brio":           (2011, 2019),
        # Toyota
        "innova":          (2004, 9999), "innova crysta":  (2016, 9999),
        "fortuner":        (2009, 9999), "hilux":          (2021, 9999),
        "corolla":         (1999, 2022), "yaris":          (2018, 2022),
        "glanza":          (2019, 9999), "urban cruiser":  (2020, 9999),
        "rumion":          (2023, 9999), "hyryder":        (2022, 9999),
        "camry":           (2002, 9999), "vellfire":       (2020, 9999),
        "land cruiser":    (2003, 9999), "prius":          (2010, 9999),
        # Kia
        "seltos":          (2019, 9999), "sonet":          (2020, 9999),
        "carnival":        (2020, 9999), "ev6":            (2022, 9999),
        "carens":          (2022, 9999), "clavis":         (2025, 9999),
        # Volkswagen
        "polo":            (2010, 2022), "vento":          (2010, 2022),
        "taigun":          (2021, 9999), "virtus":         (2022, 9999),
        "tiguan":          (2017, 9999), "t-roc":          (2020, 9999),
        # Skoda
        "rapid":           (2011, 2022), "octavia":        (2001, 9999),
        "superb":          (2009, 9999), "kushaq":         (2021, 9999),
        "slavia":          (2022, 9999), "kodiaq":         (2017, 9999),
        "karoq":           (2020, 9999),
        # Jeep
        "compass":         (2017, 9999), "wrangler":       (2015, 9999),
        "meridian":        (2022, 9999), "grand cherokee": (2012, 9999),
        # MG
        "hector":          (2019, 9999), "zs ev":          (2020, 9999),
        "gloster":         (2020, 9999), "astor":          (2021, 9999),
        "comet":           (2023, 9999), "windsor":        (2024, 9999),
        # Renault
        "kwid":            (2015, 9999), "duster":         (2012, 9999),
        "triber":          (2019, 9999), "kiger":          (2021, 9999),
        "lodgy":           (2015, 2019), "captur":         (2017, 2020),
        # Nissan / Datsun
        "magnite":         (2020, 9999), "kicks":          (2019, 2022),
        "terrano":         (2013, 2022), "micra":          (2010, 2020),
        "sunny":           (2011, 2019), "go":             (2014, 2020),
        "redi-go":         (2016, 2022),
        "ecosport":        (2013, 2022), "endeavour":      (2003, 2022),
        "figo":            (2010, 2022), "aspire":         (2015, 2022),
        "freestyle":       (2018, 2022), "mustang":        (2016, 2022),
        # Chevrolet (exited India 2017)
        "beat":            (2010, 2017), "cruze":          (2009, 2017),
        "tavera":          (2004, 2017), "spark":          (2007, 2015),
        "sail":            (2012, 2017), "enjoy":          (2013, 2017),
        # BMW
        "3 series":        (2005, 9999), "5 series":       (2003, 9999),
        "7 series":        (2008, 9999), "x1":             (2011, 9999),
        "x3":              (2011, 9999), "x5":             (2014, 9999),
        "x7":              (2019, 9999), "2 series":       (2022, 9999),
        "m3":              (2014, 9999), "m5":             (2012, 9999),
        "i4":              (2022, 9999), "ix":             (2022, 9999),
        # Mercedes-Benz
        "c class":         (2000, 9999), "e class":        (2002, 9999),
        "s class":         (2006, 9999), "glc":            (2016, 9999),
        "gle":             (2016, 9999), "gls":            (2016, 9999),
        "a class":         (2019, 9999), "cla":            (2020, 9999),
        "amg":             (2016, 9999), "eqb":            (2022, 9999),
        # Audi
        "a4":              (2008, 9999), "a6":             (2011, 9999),
        "q3":              (2013, 9999), "q5":             (2009, 9999),
        "q7":              (2007, 9999), "a8":             (2012, 9999),
        "rs":              (2013, 9999), "e-tron":         (2021, 9999),
        # Volvo
        "xc40":            (2019, 9999), "xc60":          (2010, 9999),
        "xc90":            (2015, 9999), "s60":           (2011, 9999),
        "s90":             (2017, 9999),
        # Jaguar / Land Rover
        "xe":              (2016, 9999), "xf":            (2010, 9999),
        "xj":              (2010, 2019), "f-pace":        (2017, 9999),
        "defender":        (2020, 9999), "discovery":     (2010, 9999),
        "range rover":     (2010, 9999), "range rover sport": (2012, 9999),
        "evoque":          (2012, 9999), "velar":         (2018, 9999),
    }
    def _get_prod_range(model_str: str) -> tuple[int, int] | None:
        m = (model_str or "").lower().strip()
        best_key, best_len = None, 0
        for key in _PROD:
            if key in m and len(key) > best_len:
                best_key, best_len = key, len(key)
        if best_key:
            return _PROD[best_key]
        for word in m.split():
            if word in _PROD and len(word) > best_len:
                best_key, best_len = word, len(word)
        return _PROD[best_key] if best_key else None
    if "vehicle_age" in sub.columns:
        ages = sub["vehicle_age"].dropna().astype(int).unique().tolist()
        years_from_data = {str(CURRENT - a) for a in ages if 0 <= a <= 30}
    elif "year" in sub.columns:
        years_from_data = {str(int(y)) for y in sub["year"].dropna().unique()
                          if 1990 <= int(y) <= CURRENT}
    else:
        years_from_data = set()
    # Fill gaps using production catalogue
    prod_range = _get_prod_range(model or "")
    if prod_range:
        first_yr, last_yr = prod_range
        last_yr = min(last_yr, CURRENT)
        catalogue_years = {str(y) for y in range(first_yr, last_yr + 1)}
        years_from_data = years_from_data | catalogue_years
    for y in [str(CURRENT), str(CURRENT - 1), str(CURRENT - 2)]:
        years_from_data.add(y)
    # Sort descending (newest first)
    years_from_data = sorted(years_from_data, reverse=True)
    return {
        "fuel_types":    fuels or default["fuel_types"],
        "transmissions": txs or default["transmissions"],
        "years":         years_from_data or all_years,
    }
@app.post("/predict")
def predict(vehicle: VehicleInput, model_variant: Optional[str] = None):
    prediction = predict_market_value(vehicle, model_variant=model_variant)
    var_id = model_variant or vehicle.model_variant
    _, _, meta, _, _ = resolve_variant_data(var_id)
    return {
        **prediction,
        "model_name":    meta.get("model_name", "CatBoostRegressor"),
        "is_ml_powered": True,
    }
@app.post("/evaluate")
def evaluate(vehicle: VehicleInput, model_variant: Optional[str] = None):
    return evaluate_vehicle(vehicle, model_variant=model_variant)
def _wheelr_enrichment(
    *,
    market_value: int,
    recommended_buy_price: int,
    profit_target: int,
    vehicle_age: int,
    odometer: int,
    owner_count: int,
    accident_history: str,
    registration_state: str,
    sale_state: str,
    loan_outstanding: bool,
    seller_reason: str,
    engine_grade: str,
    tyre_grade: str,
    body_grade: str,
    interior_grade: str,
    electrical_grade: str,
    vendor_type: dict,
    rc_transfer_cost: int = 3500,
    idv_value: float = 0,
) -> dict:
    current_month      = datetime.now().month
    seasonal_multiplier = get_seasonal_multiplier(current_month)
    disqualifier       = check_disqualifier(vehicle_age, odometer, owner_count, accident_history)
    recon              = get_recon_cost(engine_grade, tyre_grade, body_grade, interior_grade, electrical_grade, vendor_type, rc_transfer_cost)
    wheelr_risk        = get_wheelr_risk_deductions(owner_count, odometer, accident_history, registration_state, sale_state, loan_outstanding, seller_reason)
    # IDV gap analysis
    idv_analysis = None
    idv_extra_risk_deduction = 0
    idv_confidence_boost = 0
    if idv_value and idv_value > 0:
        idv_gap = market_value - idv_value
        idv_gap_pct = round((idv_gap / idv_value) * 100, 1)
        if idv_gap_pct > 20:
            flag = "ML price significantly above IDV — verify condition"
            flag_type = "warning"
            idv_extra_risk_deduction = 15000
        elif idv_gap_pct < -10:
            flag = "IDV above ML price — car may be undervalued, good buy"
            flag_type = "positive"
            idv_confidence_boost = 5
        else:
            flag = "IDV aligns with ML valuation"
            flag_type = "neutral"
        idv_analysis = {
            "idv_value": int(idv_value),
            "ml_value": int(market_value),
            "idv_gap": int(idv_gap),
            "idv_gap_pct": idv_gap_pct,
            "flag": flag,
            "flag_type": flag_type,
            "extra_risk_deduction": idv_extra_risk_deduction,
            "confidence_boost": idv_confidence_boost,
        }
    total_risk_deduction = wheelr_risk["total"] + idv_extra_risk_deduction
    enhanced_max_buy_price = max(0, int(recommended_buy_price - recon["total"] - total_risk_deduction))
    negotiation        = get_negotiation_trio(enhanced_max_buy_price, wheelr_risk["seller_reason_adj"])
    deal_health        = get_deal_health(market_value, recon["total"], profit_target, owner_count, odometer, accident_history)
    return {
        "disqualifier":        disqualifier,
        "seasonal_multiplier": seasonal_multiplier,
        "seasonal_month":      current_month,
        "recon": {
            "total":           recon["total"],
            "breakdown":       recon["breakdown"],
            "fixed_cost":      recon["fixed_cost"],
            "rc_transfer_cost": recon["rc_transfer_cost"],
        },
        "wheelr_risk":            wheelr_risk,
        "negotiation":            negotiation,
        "deal_health":            deal_health,
        "enhanced_max_buy_price": enhanced_max_buy_price,
        "idv_analysis":           idv_analysis,
    }
@app.post("/evaluate-enhanced")
def evaluate_enhanced(vehicle: EnhancedEvaluateRequest, model_variant: Optional[str] = None):
    base        = evaluate_vehicle(vehicle, model_variant=model_variant)
    vehicle_age = max(0, CURRENT_YEAR - int(vehicle.year))
    sale_state  = vehicle.sale_state or vehicle.registration_state or vehicle.city
    profit_target = int(
        base.get("expected_profit")
        or base.get("margin_amt")
        or base["market_value"] * (vehicle.target_margin_pct / 100)
    )
    enrichment = _wheelr_enrichment(
        market_value=base["market_value"],
        recommended_buy_price=base["recommended_buy_price"],
        profit_target=profit_target,
        vehicle_age=vehicle_age,
        odometer=int(vehicle.odometer_reading),
        owner_count=int(vehicle.owner_count),
        accident_history=vehicle.accident_history,
        registration_state=vehicle.registration_state,
        sale_state=sale_state,
        loan_outstanding=vehicle.loan_outstanding,
        seller_reason=vehicle.seller_reason,
        engine_grade=vehicle.engine_grade,
        tyre_grade=vehicle.tyre_grade,
        body_grade=vehicle.body_grade,
        interior_grade=vehicle.interior_grade,
        electrical_grade=vehicle.electrical_grade,
        vendor_type=vehicle.vendor_type,
        rc_transfer_cost=int(vehicle.rc_transfer_cost or 3500),
        idv_value=float(vehicle.idv_value or 0),
    )
    return {**base, **enrichment}
@app.post("/reverse-calculate")
def reverse_calculate(body: ReverseCalculateRequest):
    vehicle_age       = max(0, CURRENT_YEAR - int(body.year))
    odometer          = int(body.odometer)
    owner_count       = int(body.owner_count)
    expected_sell     = int(body.expected_sell_price)
    raw_pct           = float(body.target_margin_pct)
    target_margin_pct = raw_pct / 100.0 if raw_pct > 1.0 else raw_pct
    profit_target     = int(expected_sell * target_margin_pct)
    recon = get_recon_cost(body.engine_grade, body.tyre_grade, body.body_grade, body.interior_grade, body.electrical_grade, body.vendor_type)
    wheelr_risk = get_wheelr_risk_deductions(owner_count, odometer, body.accident_history, body.registration_state, body.sale_state, body.loan_outstanding, body.seller_reason)
    max_buy_price = max(0, expected_sell - recon["total"] - profit_target - wheelr_risk["total"])
    negotiation = get_negotiation_trio(max_buy_price, wheelr_risk["seller_reason_adj"])
    deal_health = get_deal_health(expected_sell, recon["total"], profit_target, owner_count, odometer, body.accident_history)
    disqualifier  = check_disqualifier(vehicle_age, odometer, owner_count, body.accident_history)
    current_month = datetime.now().month
    margin_label_pct = int(round(target_margin_pct * 100))
    return {
        "expected_sell_price": expected_sell,
        "recon":               {"total": recon["total"], "breakdown": recon["breakdown"]},
        "profit_target":       profit_target,
        "wheelr_risk":         wheelr_risk,
        "max_buy_price":       max_buy_price,
        "negotiation": {
            "opening":    negotiation["opening_offer"],
            "target":     negotiation["target_offer"],
            "walk_away":  negotiation["walk_away_price"],
        },
        "deal_health":         deal_health,
        "disqualifier":        disqualifier,
        "seasonal_multiplier": get_seasonal_multiplier(current_month),
        "price_breakdown": [
            {"label": "Expected selling price",      "value": expected_sell,          "sign": ""},
            {"label": "Reconditioning cost",         "value": recon["total"],         "sign": "-"},
            {"label": f"Profit target ({margin_label_pct}%)", "value": profit_target, "sign": "-"},
            {"label": "Risk deductions",             "value": wheelr_risk["total"],   "sign": "-"},
            {"label": "Max buy price",               "value": max_buy_price,          "sign": "="},
        ],
    }
@app.post("/bulk-evaluate")
def bulk_evaluate(vehicles: List[VehicleInput]):
    results = []
    for idx, vehicle in enumerate(vehicles, start=1):
        evaluation = evaluate_vehicle(vehicle)
        results.append({
            "row_number": idx,
            "vehicle":    f"{vehicle.year} {vehicle.brand} {vehicle.model}",
            "brand":      vehicle.brand,
            "model":      vehicle.model,
            "year":       vehicle.year,
            "city":       vehicle.city,
            "odometer":   vehicle.odometer_reading,
            **evaluation,
        })
    return {"count": len(results), "results": results}
