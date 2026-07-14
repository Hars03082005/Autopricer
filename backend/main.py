from __future__ import annotations

import json
import math
import os
import re
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

# ── Paths & startup ───────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model_artifacts"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES           = METADATA["features"]
CAT_FEATURES       = METADATA["categorical_features"]
CURRENT_YEAR       = METADATA.get("current_year_used_for_age", datetime.now().year)
CONDITION_MULTIPLIERS = {
    # v9.0 — Improvement #8: stronger condition impact for realistic valuation
    "excellent": 1.05,   # +5%  — near showroom condition
    "good":      1.00,   # baseline
    "average":   0.92,   # -8%  — noticeable wear, refurb needed
    "poor":      0.82,   # -18% — major reconditioning required
}

predictor    = EnsemblePredictor.from_artifact_dir(ARTIFACT_DIR)
BRAND_CATALOG = build_brand_catalog()

# ── Segment models (economy / premium / luxury) ──────────────────────────────
SEGMENT_MODELS: dict = {}
for _seg in ["economy", "premium", "luxury"]:
    _path = ARTIFACT_DIR / f"ensemble_{_seg}.pkl"
    if _path.exists():
        SEGMENT_MODELS[_seg] = joblib.load(_path)
# Backward-compat: also try old brand-class names and alias them
for _old, _new in [("budget", "economy"), ("mid", "economy")]:
    if _old not in SEGMENT_MODELS:
        _path = ARTIFACT_DIR / f"ensemble_{_old}.pkl"
        if _path.exists() and _new not in SEGMENT_MODELS:
            SEGMENT_MODELS[_new] = joblib.load(_path)

# Brand → segment map (loaded from metadata, fallback inline)
BRAND_SEGMENT_MAP: dict = METADATA.get("brand_segment_map", {
    # Economy
    "maruti": "economy", "maruti suzuki": "economy", "datsun": "economy",
    "bajaj": "economy", "chevrolet": "economy", "fiat": "economy",
    "opel": "economy", "premier": "economy", "force": "economy",
    "ashok leyland": "economy", "ambassador": "economy",
    "hyundai": "economy", "honda": "economy", "tata": "economy",
    "renault": "economy", "nissan": "economy", "ford": "economy",
    "mahindra": "economy", "mitsubishi": "economy",
    "isuzu": "economy", "citroen": "economy", "dc": "economy",
    # Premium
    "volkswagen": "premium", "skoda": "premium", "toyota": "premium",
    "mg": "premium", "jeep": "premium", "kia": "premium",
    "mini": "premium", "volvo": "premium", "lexus": "premium",
    # Luxury
    "bmw": "luxury", "mercedes-benz": "luxury", "audi": "luxury",
    "jaguar": "luxury", "land rover": "luxury", "porsche": "luxury",
    "maserati": "luxury", "aston martin": "luxury", "bentley": "luxury",
    "rolls-royce": "luxury", "ferrari": "luxury", "lamborghini": "luxury",
    "hummer": "luxury",
})

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="PriceRef ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
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
    city: str = "Mumbai"
    color: str = "unknown"          # car colour — from dataset schema; improves accuracy
    inspected: bool = False         # inspection certificate present
    condition: str = "Good"
    seller_asking_price: float = 0
    target_margin_pct: float = 15
    repair_buffer: float = 25000


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
    target_margin_pct: float = Field(0.15, ge=0, le=1)


# ── Helper functions ──────────────────────────────────────────────────────────
def clean_text(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text if text else default


def normalize_model_name(brand: str, model_name: str) -> str:
    brand_clean = clean_text(brand)
    model_clean = clean_text(model_name)
    if brand_clean != "unknown" and brand_clean not in model_clean:
        return f"{brand_clean} {model_clean}"
    return model_clean


def condition_to_score(condition: str) -> int:
    return {"excellent": 90, "good": 75, "average": 58, "poor": 38}.get(clean_text(condition), 65)


def build_features(vehicle: VehicleInput) -> pd.DataFrame:
    vehicle_age = max(0, CURRENT_YEAR - int(vehicle.year))
    km          = max(0, float(vehicle.odometer_reading or 0))
    owner       = max(1, int(vehicle.owner_count or 1))
    km_per_year = min(km / max(vehicle_age, 1), 100_000)

    # Scores — mirror train_ml_model.py v5.0 formulas exactly
    ownership_trust_score = (
        (1 / owner) * 0.5
        + (1 - min(vehicle_age / 35, 1.0)) * 0.3
        + (1 - min(km / 600_000, 1.0)) * 0.2
    )
    vehicle_health_score = (
        (1 - min(km / 600_000, 1.0)) * 0.5
        + (1 - min(vehicle_age / 35, 1.0)) * 0.3
        + (1 / owner) * 0.2
    )

    seg_class   = get_segment_class(vehicle.brand)   # economy/premium/luxury
    color       = clean_text(getattr(vehicle, 'color', None) or 'unknown')
    high_mileage = 1 if km > 93_143 else 0          # 75th percentile from training
    luxury_brand = 1 if seg_class == "luxury" else 0
    inspected    = 1 if getattr(vehicle, 'inspected', False) else 0

    row = {
        # Categorical
        "brand":         clean_text(vehicle.brand),
        "model":         normalize_model_name(vehicle.brand, vehicle.model),
        "variant":       clean_text(vehicle.variant or "unknown"),
        "city":          clean_text(vehicle.city),
        "rto_state":     "unknown",   # not collected at basic input
        "color":         color,
        "segment_class": seg_class,
        "fuel_type":     clean_text(vehicle.fuel_type),
        "transmission":  clean_text(vehicle.transmission),
        # Numeric
        "vehicle_age":           float(vehicle_age),
        "odometer_reading":      float(km),
        "km_per_year":           float(km_per_year),
        "owner_count":           float(owner),
        "ownership_trust_score": float(ownership_trust_score),
        "vehicle_health_score":  float(vehicle_health_score),
        # Binary
        "inspected":     float(inspected),
        "high_mileage":  float(high_mileage),
        "luxury_brand":  float(luxury_brand),
        "has_list_price": 0.0,   # not known at inference
    }
    df = pd.DataFrame([row], columns=FEATURES)
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)
    return df


def condition_multiplier(condition: str) -> float:
    return float(
        CONDITION_MULTIPLIERS.get(
            clean_text(condition, "good"),
            CONDITION_MULTIPLIERS.get("good", 1.0),
        )
    )


# ── Segment routing helpers ────────────────────────────────────────────────────
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
        preds["catboost"] = float(artifact["catboost"].predict(cb_f)[0])
    if weights.get("lightgbm", 0) > 0:
        preds["lightgbm"] = float(artifact["lightgbm"].predict(lgb_f)[0])
    if weights.get("xgboost", 0) > 0:
        preds["xgboost"]  = float(artifact["xgboost"].predict(xgb_f)[0])
    if not preds:
        # Fallback: all weights zero — just use catboost
        preds["catboost"] = float(artifact["catboost"].predict(cb_f)[0])
        weights = {"catboost": 1.0}
    return sum(weights[k] * preds[k] for k in preds)



# ── Core prediction ────────────────────────────────────────────────────────────
def predict_base_market_value(vehicle: VehicleInput) -> tuple[int, str]:
    """
    Returns (market_value_inr: int, routing_note: str).

    Routing: brand → segment_class (economy / premium / luxury).
    Brand is always known at input time — O(1) dict lookup.
    Falls back to global ensemble if segment model file is missing.
    """
    features  = build_features(vehicle)
    seg_class = get_segment_class(vehicle.brand)
    artifact  = SEGMENT_MODELS.get(seg_class)

    if artifact:
        try:
            log_price    = _run_class_model(features, artifact)
            routing_note = f"{seg_class} segment model used"
        except Exception:
            log_price    = predictor.predict_log_price(features)
            routing_note = f"{seg_class} segment model error — fell back to global"
    else:
        log_price    = predictor.predict_log_price(features)
        routing_note = "global model used (segment model not found)"

    market_value = float(np.expm1(log_price))
    if not math.isfinite(market_value):
        market_value = 0
    market_value = max(50_000, min(market_value, 20_000_000))
    return int(round(market_value / 500) * 500), routing_note


def predict_market_value(vehicle: VehicleInput) -> dict:
    """Return base ML value and final condition-calibrated, sanity-clamped market value."""
    base_value, routing_note = predict_base_market_value(vehicle)
    seg_class = get_segment_class(vehicle.brand)
    age       = max(0, CURRENT_YEAR - int(vehicle.year))

    mult     = condition_multiplier(vehicle.condition)
    adjusted = max(50_000, min(base_value * mult, 20_000_000))
    adjusted = int(round(adjusted / 500) * 500)

    # Apply market sanity clamp AFTER condition adjustment
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        vehicle.model, seg_class, age, float(adjusted),
        city=str(vehicle.city or ""),
    )
    final_value = int(round(clamped_value / 500) * 500)

    return {
        "base_market_value":     int(base_value),
        "market_value":          final_value,
        "condition_multiplier":  round(mult, 3),
        "condition_adjustment":  int(adjusted - base_value),
        "condition_score":       condition_to_score(vehicle.condition),
        "segment_class":         seg_class,
        "segment_model_used":    seg_class in SEGMENT_MODELS,
        "routing_note":          routing_note,
        "sanity_clamped":        sanity_clamped,
        "sanity_note":           sanity_note,
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
    # market_value is never touched here — no tuple unpacking needed. Task 5 confirmed correct.
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


def evaluate_vehicle(vehicle: VehicleInput) -> dict:
    prediction   = predict_market_value(vehicle)
    market_value = prediction["market_value"]   # int — sanity-clamped
    decision     = calculate_decision(vehicle, market_value)
    seg_class    = prediction.get("segment_class", "economy")

    # Similar cars
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
        "model_name":         METADATA["model_name"],
        "is_ml_powered":      True,
        "metrics":            METADATA.get("metrics", {}),
        "train_metrics":      METADATA.get("train_metrics", {}),
        "validation_metrics": METADATA.get("validation_metrics", {}),
        "test_metrics":       METADATA.get("test_metrics", {}),
        "overfitting_check":  METADATA.get("overfitting_check", {}),
        "shap":               shap_like_explanation(vehicle, market_value),
        "warnings":           warnings_for(vehicle, decision),
        "similar_cars":       similar,
        **decision,
    }


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":              "ok",
        "model_loaded":        (ARTIFACT_DIR / "vehicle_price_lightgbm.txt").exists(),
        "ensemble_enabled":    METADATA.get("ensemble", {}).get("enabled", False),
        "model_name":          METADATA["model_name"],
        "segmentation":        "segment_class",
        "segments_loaded":     list(SEGMENT_MODELS.keys()),
    }


@app.get("/metadata")
def metadata():
    return METADATA


@app.get("/api/brands")
def get_brands():
    return {"brands": BRAND_CATALOG}


@app.post("/predict")
def predict(vehicle: VehicleInput):
    prediction = predict_market_value(vehicle)
    return {
        **prediction,
        "model_name":    METADATA["model_name"],
        "is_ml_powered": True,
    }


@app.post("/evaluate")
def evaluate(vehicle: VehicleInput):
    return evaluate_vehicle(vehicle)


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

    # ── IDV gap analysis ────────────────────────────────────────────────────────
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
def evaluate_enhanced(vehicle: EnhancedEvaluateRequest):
    base        = evaluate_vehicle(vehicle)
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
    target_margin_pct = float(body.target_margin_pct)
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
