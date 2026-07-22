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
from fastapi import FastAPI
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
    apply_market_sanity_clamp,
    shap_explanation,
    generate_similar_cars,
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
    """
    Resolves the predictor, segment_models, metadata, dataset catalog, and active variant_id.
    Falls back to default variant or model_artifacts directory for backward compatibility.
    """
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
        except Exception:
            pass

    # Fallback to local model_artifacts if registry lookup fails
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

# ── Globals (populated during lifespan startup, NOT at import time) ──────────
predictor       = None
SEGMENT_MODELS  = {}
METADATA        = {}
DATASET_CATALOG = {}
ACTIVE_VARIANT_ID = "variant_2"
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
# ── Brand → segment_class — must match training TIER_TO_SEGMENT exactly ────
# Training: {0: "budget", 1: "economy", 2: "mid", 3: "premium", 4: "luxury"}
BRAND_SEGMENT_MAP: dict = {
    # Tier 0 → budget
    "datsun": "budget",
    # Tier 1 → economy
    "maruti suzuki": "economy", "maruti": "economy",
    "renault": "economy", "tata": "economy", "chevrolet": "economy",
    "fiat": "economy", "bajaj": "economy",
    "opel": "economy", "premier": "economy", "force": "economy",
    "ashok leyland": "economy", "ambassador": "economy", "dc": "economy",
    # Tier 2 → mid  ← this was the biggest bug (Hyundai/Honda/Kia sent as economy)
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

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Load ML models AFTER the port is bound (avoids OOM crash before port open)."""
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
    log.info("==> ML models loaded. Active variant: %s", ACTIVE_VARIANT_ID)
    yield  # Server is running
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
    variant: str = "unknown"        # trim level — sent by frontend, used by all 3 segment models
    year: int = 2021
    fuel_type: str = "Petrol"
    transmission: str = "Manual"
    odometer_reading: int = Field(28000, ge=0)
    fuel_efficiency: float = 17.5
    owner_count: int = Field(1, ge=1)
    engine_cc: int = Field(1497, ge=0)
    city: str = "Bangalore"
    color: str = "unknown"          # car colour — from dataset schema; improves accuracy
    inspected: bool = False         # inspection certificate present
    condition: str = "Good"
    seller_asking_price: float = 0
    target_margin_pct: float = 10
    repair_buffer: float = 25000
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
# When a user types a generic model name that maps to a more specific generation
# depending on the year, resolve it here BEFORE building features and before
# looking up the sanity-clamp band.
# Format: { clean_model: [ (min_year, max_year, canonical_model), ... ] }
MODEL_YEAR_ALIASES: dict[str, list[tuple[int, int, str]]] = {
    # Innova Crysta launched in 2016 — anything from 2014 onwards is considered Crysta
    "innova":        [(2014, 9999, "innova crysta")],
    # Creta gen-2 (2020+) has significantly higher resale vs gen-1
    "creta":         [(2020, 9999, "creta")],
    # Nexon EV (2020+) is a separate product from ICE Nexon
    "nexon":         [(2020, 9999, "nexon")],
    # Brezza (2022+ facelift has substantially higher value)
    "vitara brezza": [(2022, 9999, "brezza")],
    "brezza":        [(2022, 9999, "brezza")],
    # Scorpio N (2022+) is a new platform, separate from old Scorpio Classic
    "scorpio":       [(2022, 9999, "scorpio n")],
    # Safari (2021+ is monocoque premium crossover, pre-2021 is body-on-frame Storme/Dicor)
    "safari":        [(1900, 2020, "safari classic")],
    # Thar (2020+ is lifestyle SUV, pre-2020 is old rugged off-roader)
    "thar":          [(1900, 2019, "thar gen1")],
}


def _resolve_model_alias(model: str, year: int) -> str:
    """
    Return the canonical model name for a given year.
    Falls back to the original model string if no alias applies.
    """
    model_key = clean_text(model)
    rules = MODEL_YEAR_ALIASES.get(model_key)
    if rules:
        for min_yr, max_yr, canonical in rules:
            if min_yr <= year <= max_yr:
                return canonical
    return model_key


def normalize_model_name(brand: str, model_name: str, year: int = 0) -> str:
    brand_clean = clean_text(brand)
    # Resolve generation alias first (e.g. Innova 2019 -> innova crysta)
    model_clean = _resolve_model_alias(model_name, year)
    if brand_clean != "unknown" and brand_clean not in model_clean:
        return f"{brand_clean} {model_clean}"
    return model_clean


def condition_to_score(condition: str) -> int:
    return {"excellent": 90, "good": 75, "average": 58, "poor": 38}.get(clean_text(condition), 65)



# ── Lookup tables derived from training dataset (processed_widoutown-2.csv) ───
# These allow build_features() to supply the 7 features that were computed
# during data-cleaning but are unavailable at inference time.

# city → (locality_tier, locality_density_norm, representative_rto, locality)
_CITY_FEATURE_MAP: dict[str, tuple[int, float, str, str]] = {
    "bangalore":  (2, 1.000, "ka-03", "bellahalli"),   # Issue 8 fix: city-level default = tier-2 (mid)
    "bengaluru":  (2, 1.000, "ka-03", "bellahalli"),   # bellahalli locality is tier-1, but city default is tier-2
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

# Brand → mean popularity_score_log from training data
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

# km_per_year → usage_category_num bucket (0=very_low, 1=low, 2=moderate, 3=high)
def _usage_category_num(km_per_year: float) -> float:
    if km_per_year < 8_000:  return 0.0
    if km_per_year < 15_000: return 1.0
    if km_per_year < 25_000: return 2.0
    return 3.0


def build_features(vehicle: VehicleInput) -> pd.DataFrame:
    vehicle_age = max(0, CURRENT_YEAR - int(vehicle.year))
    km          = max(0, float(vehicle.odometer_reading or 0))
    owner       = max(1, int(vehicle.owner_count or 1))
    # Issue 6 fix: use 0.5 floor so a 6-month-old car gets ~2x correct km/yr
    km_per_year = min(km / max(vehicle_age, 0.5), 100_000)

    # Issue 1 fix: match training scale {1:100, 2:75, 3:50, 4:25, 5+:10}
    _TRUST_MAP = {1: 100, 2: 75, 3: 50, 4: 25, 5: 10, 6: 10}
    ownership_trust_score = float(_TRUST_MAP.get(owner, 10))

    # Issue 2 fix: match training formula — score 0-100 not 0-1
    vehicle_health_score = float(max(0.0, min(100.0,
        100.0 - (vehicle_age * 3) - (km / 10_000) - ((owner - 1) * 8)
    )))

    seg_class    = get_segment_class(vehicle.brand)
    color        = clean_text(getattr(vehicle, 'color', None) or 'unknown')
    high_mileage = 1 if km > 93_143 else 0
    luxury_brand = 1 if seg_class == "luxury" else 0
    inspected    = 1 if getattr(vehicle, 'inspected', False) else 0

    brand_clean        = clean_text(vehicle.brand)
    brand_tier         = _BRAND_TIER_MAP.get(brand_clean, 1)
    age_km_interaction = float(vehicle_age) * float(km)
    is_high_mileage    = 1 if km_per_year > 15_000 else 0

    # Luxury brand age/km penalty features (used by unified model)
    brand_age_penalty = float(brand_tier) * float(vehicle_age)
    brand_km_penalty  = float(brand_tier) * (km / 10_000)

    # ── 7 previously missing features — now properly estimated ──────────────
    city_clean = clean_text(vehicle.city)
    city_info  = _CITY_FEATURE_MAP.get(city_clean)

    if city_info:
        locality_tier_val, density_norm, rto_val, locality_val = city_info
    else:
        # Unknown city → Tier-2 defaults (conservative middle estimate)
        locality_tier_val, density_norm, rto_val, locality_val = 2, 0.20, "unknown", city_clean

    usage_cat_num   = _usage_category_num(km_per_year)
    pop_log         = _BRAND_POPULARITY_LOG.get(brand_clean, 3.50)

    row = {
        # Categorical — core model features
        "brand":         brand_clean,
        "model":         normalize_model_name(vehicle.brand, vehicle.model, int(vehicle.year)),
        "variant":       clean_text(vehicle.variant or "unknown"),
        "city":          city_clean,
        "locality":      locality_val,               # ← was missing
        "rto":           rto_val,                    # ← was missing
        "segment_class": seg_class,
        "fuel_type":     clean_text(vehicle.fuel_type),
        "transmission":  clean_text(vehicle.transmission),
        "seller_type":   "dealer",                   # ← was missing (dealer is majority class in training)
        "color":         color,
        # Legacy compat
        "rto_state":     rto_val,
        # Numeric — core model features
        "vehicle_age":           float(vehicle_age),
        "odometer_reading":      float(km),
        "km_per_year":           float(km_per_year),
        "owner_count":           float(owner),
        "brand_tier":            float(brand_tier),
        "age_km_interaction":    float(age_km_interaction),
        "ownership_trust_score": float(ownership_trust_score),
        "vehicle_health_score":  float(vehicle_health_score),
        "is_high_mileage":       float(is_high_mileage),
        "locality_tier":         float(locality_tier_val),       # ← was missing
        "usage_category_num":    float(usage_cat_num),           # ← was missing
        "locality_density_norm": float(density_norm),            # ← was missing
        "popularity_score_log":  float(pop_log),                 # ← was missing
        # Additional
        "brand_age_penalty":     brand_age_penalty,
        "brand_km_penalty":      brand_km_penalty,
        "inspected":      float(inspected),
        "high_mileage":   float(high_mileage),
        "luxury_brand":   float(luxury_brand),
        "has_list_price": 0.0,
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
    """Return the segment class (economy / premium / luxury) for a given brand.
    Brand is always known at inference time — O(1) dict lookup.
    Unknown brands default to 'economy' (safest prior).
    """
    return BRAND_SEGMENT_MAP.get(clean_text(brand), "economy")


# Median depreciation ratio from training data (used when list_price is unknown)
_MEDIAN_DEP_RATIO = 0.75


def _run_class_model(features: pd.DataFrame, artifact: dict) -> float:
    """Run the three sub-models for a brand class and return the blended log-price.
    Uses category_levels saved in the pkl to normalise unseen values to 'unknown'.
    Correctly encodes categoricals per model type:
      - CatBoost: string columns (handles internally)
      - LightGBM: pd.Categorical dtype (required by lgb.Booster.predict)
      - XGBoost:  integer-encoded columns
    """
    cb_f  = features.copy()
    lgb_f = features.copy()
    xgb_f = features.copy()
    for col in artifact.get("cat_features", []):
        if col not in features.columns:
            continue
        cat_levels = artifact.get("category_levels", {}).get(col, [])
        raw = features[col].astype(str)
        if cat_levels:
            # Normalise unseen values → "unknown" (must be in cat_levels)
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
        # Reindex to the model's own feature list to guard against extra/stale columns
        cb_model = artifact["catboost"]
        cb_cols   = cb_model.feature_names_
        # Fill any missing expected columns
        for _col in cb_cols:
            if _col not in cb_f.columns:
                cb_f[_col] = "unknown" if _col in artifact.get("cat_features", []) else 0.0
        cb_f_aligned = cb_f[cb_cols]
        # Ensure all cat feature columns are strings (CatBoost requirement)
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
        # Fallback: all weights zero — just use catboost (with same alignment guard)
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


# Core prediction
def predict_base_market_value(vehicle: VehicleInput, model_variant: Optional[str] = None) -> tuple[int, str]:
    """
    Returns (market_value_inr: int, routing_note: str).

    Segment model routing (Issue 4 fix):
    Training saves price-band models keyed as '6_12_lakh', '12_plus_lakh'.
    We pick the right one based on a rough price estimate, fall back to global.
    """
    var_id = model_variant or vehicle.model_variant
    pred_obj, seg_models, meta, _, active_id = resolve_variant_data(var_id)

    resolved_model = _resolve_model_alias(vehicle.model, int(vehicle.year))
    features  = build_features(vehicle)
    seg_class = get_segment_class(vehicle.brand)

    # Choose price-band segment model by brand segment as proxy
    # Routing: budget/economy/mid -> 0_6_lakh, premium -> 6_12_lakh, luxury -> 12_plus_lakh
    _SEG_TO_BAND = {
        "budget":  "0_6_lakh",
        "economy": "0_6_lakh",
        "mid":     "6_12_lakh",
        "premium": "6_12_lakh",
        "luxury":  "12_plus_lakh",
    }
    price_band = _SEG_TO_BAND.get(seg_class, "0_6_lakh")
    artifact   = seg_models.get(price_band)

    if artifact and isinstance(artifact, dict) and "model" in artifact:
        try:
            # Segment model is a raw CatBoost model, not an ensemble
            from catboost import Pool
            seg_m = artifact["model"]
            cat_levels = artifact.get("cat_levels", {})
            cb_f = features.copy()
            cat_cols = [c for c in CAT_FEATURES if c in cb_f.columns]
            # Normalise unseen categories
            for col in cat_cols:
                known = set(cat_levels.get(col, ["unknown"]))
                cb_f[col] = cb_f[col].astype(str).apply(lambda x: x if x in known else "unknown")
            # Align columns to model's feature names
            for col in seg_m.feature_names_:
                if col not in cb_f.columns:
                    cb_f[col] = "unknown" if col in cat_cols else 0.0
            log_price    = float(seg_m.predict(cb_f[seg_m.feature_names_])[0])
            routing_note = f"segment model '{price_band}' used [{active_id}]"
        except Exception as e:
            log.warning("Segment model failed (%s), falling back to global: %s", price_band, e)
            log_price    = pred_obj.predict_log_price(features)
            routing_note = f"segment model error — global fallback [{active_id}]"
    else:
        log_price    = pred_obj.predict_log_price(features)
        routing_note = f"global model ({seg_class}) [{active_id}]"

    market_value = float(np.expm1(log_price))
    if not math.isfinite(market_value):
        market_value = 0
    market_value = max(50_000, min(market_value, 20_000_000))
    return int(round(market_value / 500) * 500), routing_note


def predict_market_value(vehicle: VehicleInput, model_variant: Optional[str] = None) -> dict:
    """Return base ML value and final condition-calibrated, sanity-clamped market value."""
    resolved_model = _resolve_model_alias(vehicle.model, int(vehicle.year))
    base_value, routing_note = predict_base_market_value(vehicle, model_variant=model_variant)
    seg_class = get_segment_class(vehicle.brand)
    age       = max(0, CURRENT_YEAR - int(vehicle.year))

    mult     = condition_multiplier(vehicle.condition)
    adjusted = max(50_000, min(base_value * mult, 20_000_000))
    adjusted = int(round(adjusted / 500) * 500)

    # Issue 5 fix: pass fuel_type and odometer_km so the new depreciation modifiers apply
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        resolved_model, seg_class, age, float(adjusted),
        city=str(vehicle.city or ""),
        fuel_type=str(vehicle.fuel_type or "petrol"),
        odometer_km=float(vehicle.odometer_reading or 0),
    )
    final_value = int(round(clamped_value / 500) * 500)

    var_id = model_variant or vehicle.model_variant
    _, seg_models, meta, _, active_id = resolve_variant_data(var_id)

    return {
        "base_market_value":     int(base_value),
        "market_value":          final_value,
        "condition_multiplier":  round(mult, 3),
        "condition_adjustment":  int(adjusted - base_value),
        "condition_score":       condition_to_score(vehicle.condition),
        "segment_class":         seg_class,
        "segment_model_used":    seg_class in seg_models,
        "routing_note":          routing_note,
        "sanity_clamped":        sanity_clamped,
        "sanity_note":           sanity_note,
        "resolved_model":        resolved_model,
        "model_variant":         active_id,
    }


def shap_like_explanation(vehicle: VehicleInput, market_value: int) -> list[dict]:
    """Delegate to the monetary SHAP function in decision_engine."""
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


def warnings_for(vehicle: VehicleInput, decision: dict) -> list[str]:
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
    return warnings


def evaluate_vehicle(vehicle: VehicleInput, model_variant: Optional[str] = None) -> dict:
    prediction   = predict_market_value(vehicle, model_variant=model_variant)
    market_value = prediction["market_value"]
    decision     = calculate_decision(vehicle, market_value)
    seg_class    = prediction.get("segment_class", "economy")

    var_id = model_variant or vehicle.model_variant
    _, _, meta, _, active_id = resolve_variant_data(var_id)

    similar = generate_similar_cars(
        market_value = float(market_value),
        brand        = vehicle.brand,
        model        = vehicle.model,
        year         = int(vehicle.year),
        fuel         = str(vehicle.fuel_type),
        city         = str(vehicle.city),
        segment      = seg_class,
    )

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
        "warnings":           warnings_for(vehicle, decision),
        "similar_cars":       similar,
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
    # Reload global variables for active default variant
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
    # Normalise: if caller sends 15 (percent) convert to 0.15 (fraction)
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
