

from __future__ import annotations
import json
import math
import os
from datetime import datetime

# ENGINE CONFIG — loaded once at startup from model_artifacts/engine_config.json
#
# If the file does not exist the engine falls back to safe hardcoded defaults,
# so the backend always starts even before generate_engine_config.py is run.

def _load_engine_config() -> dict:
    """Load engine_config.json from model_artifacts/. Robust fallback on error."""
    _here     = os.path.dirname(os.path.abspath(__file__))
    _cfg_path = os.path.normpath(os.path.join(_here, "..", "model_artifacts", "engine_config.json"))
    try:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        _ts = _cfg.get("generated_at", "unknown")
        _ds = _cfg.get("dataset", "unknown")
        print(f"[decision_engine] engine_config.json loaded  dataset={_ds}  generated={_ts}")
        return _cfg
    except FileNotFoundError:
        print("[decision_engine] engine_config.json not found — using hardcoded defaults")
        return {}
    except Exception as _e:
        print(f"[decision_engine] WARNING: could not load engine_config.json: {_e}")
        return {}


_CFG: dict = _load_engine_config()


def _cfg_get(key: str, default):
    """Safely read a key from the loaded config, falling back to default."""
    return _CFG.get(key, default)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))

def _round500(v: float) -> int:
    return int(round(v / 500) * 500)


# MARKET REFERENCE — loaded from engine_config.json, with hardcoded fallbacks

# Market bands: {model: [p10, p90]}  ← computed from processed dataset
_MARKET_BANDS_CFG: dict[str, list[float]] = _cfg_get("market_bands", {})

# Segment bands: {economy/premium/luxury: [p5, p95]}
_SEGMENT_BANDS_CFG: dict[str, list[float]] = _cfg_get("segment_bands", {
    "economy": [200_000,  1_500_000],
    "premium": [600_000,  3_500_000],
    "luxury":  [2_500_000, 20_000_000],
})

# Locality demand: {locality: fraction_uplift}  — replaces city_demand
_LOCALITY_DEMAND: dict[str, float] = _cfg_get("locality_demand", {})

def get_locality_demand(locality: str, segment: str = "economy") -> float:
    """
    Return a realistic intracity price adjustment for a locality, segment-aware.

    - Economy (Maruti, Hyundai, Honda): ±5.0% max cap
    - Premium (Toyota, Kia, Mahindra): ±6.5% max cap
    - Luxury  (BMW, Mercedes, Audi):    ±8.0% max cap (luxury resale demand shows stronger geographic concentration)
    """
    if not locality:
        return 0.0
    key = str(locality).strip().lower()
    raw = float(_LOCALITY_DEMAND.get(key, 0.0))

    seg = str(segment or "economy").strip().lower()
    if seg == "luxury":
        dampened = raw * 0.30
        return max(-0.08, min(0.08, round(dampened, 4)))
    elif seg == "premium":
        dampened = raw * 0.25
        return max(-0.065, min(0.065, round(dampened, 4)))
    else:
        dampened = raw * 0.20
        return max(-0.05, min(0.05, round(dampened, 4)))

# RTO demand: {KA-01: fraction_uplift}
_RTO_DEMAND: dict[str, float] = _cfg_get("rto_demand", {})

# Annual km tiers — from percentiles of km_per_year
_ANNUAL_KM_TIERS: dict[str, float] = _cfg_get("annual_km_tiers", {
    "very_low":  8_000,
    "low":      12_000,
    "moderate": 18_000,
    "high":     25_000,
    "very_high": 35_000,
})

# Clamp tolerance: {economy/premium/luxury: [lo_ratio, hi_ratio]}
_CLAMP_TOLERANCE_CFG: dict[str, list[float]] = _cfg_get("clamp_tolerance", {
    "economy": [0.90, 1.10],
    "premium": [0.86, 1.14],
    "luxury":  [0.80, 1.20],
})

# Confidence baseline from training metrics
_CONF_BASELINE: dict = _cfg_get("confidence_baseline", {
    "mc_base": 88.0,
    "bc_base": 84.0,
    "mape_frac": 0.0616,
    "r2": 0.9777,
})

# Certified vehicle premium
_CERTIFIED_PREMIUM: float = float(_cfg_get("certified_vehicle_premium", 0.035))

# Owner statistics {owner_count_str: ratio}
_OWNER_STATS: dict[str, float] = _cfg_get("owner_statistics", {
    "1": 1.00, "2": 0.94, "3": 0.88, "4": 0.80,
})


# HARDCODED CONSTANTS — stay fixed (legal/operational costs, brand data)

# Brand repair multiplier (affects reconditioning cost, NOT ML prediction)
_BRAND_REPAIR_MULTIPLIER: dict[str, float] = {
    "maruti suzuki": 0.78, "maruti": 0.78,
    "toyota":        0.80, "honda": 0.82, "hyundai": 0.83,
    "suzuki":        0.80, "datsun": 0.85,
    "tata": 1.00, "mahindra": 1.02, "renault": 1.05,
    "ford": 1.05, "nissan": 1.00,  "kia": 0.95,
    "mg":   1.10, "citroen": 1.15,
    "volkswagen": 1.30, "skoda": 1.25,
    "mini":       1.45, "volvo": 1.40,
    "jeep":       1.35, "lexus": 1.20,
    "bmw":            1.65, "mercedes-benz": 1.70,
    "audi":           1.60, "porsche": 1.80,
    "jaguar":     2.00, "land rover": 2.20,
    "bentley":    2.50, "rolls-royce": 2.50,
    "aston martin": 2.30, "maserati": 2.10,
    "ferrari":    2.50, "lamborghini": 2.50,
}

# Brand popularity — inventory days factor
_BRAND_POPULARITY: dict[str, float] = {
    "maruti suzuki": 0.75, "maruti": 0.75,
    "hyundai": 0.80, "tata": 0.82, "honda": 0.85,
    "renault": 0.90, "kia": 0.88, "toyota": 0.85,
    "mahindra": 1.00, "volkswagen": 1.05,
    "skoda": 1.10, "ford": 1.00, "mg": 0.95,
    "nissan": 1.05, "datsun": 1.00,
    "jeep": 1.20, "mini": 1.30, "volvo": 1.25,
    "citroen": 1.35, "mitsubishi": 1.20,
    "bmw": 1.40, "mercedes-benz": 1.45, "audi": 1.40,
    "porsche": 1.60, "lexus": 1.50,
    "jaguar": 1.80, "land rover": 1.70, "bentley": 2.20,
    "rolls-royce": 2.50, "ferrari": 2.50, "lamborghini": 2.50,
}

# Dealer profit limits (min, max) per vehicle category
_PROFIT_LIMITS: dict[str, tuple[int, int]] = {
    "economy":       (4_000,    12_000),
    "premium_hatch": (6_000,    15_000),
    "compact_suv":   (8_000,    18_000),
    "mid_suv":       (10_000,   22_000),
    "luxury":        (20_000,   45_000),
}

# Holding cost parameters by segment
_HOLDING: dict[str, dict] = {
    "economy": {"rate_pct": 0.5, "days": 20},
    "premium": {"rate_pct": 0.6, "days": 25},
    "luxury":  {"rate_pct": 0.8, "days": 35},
}

# Reconditioning base cost by segment
_RECON_BASE: dict[str, int] = {
    "economy": 5_000,
    "premium": 8_000,
    "luxury":  15_000,
}

# Government / statutory charges — these are FIXED legal costs, never data-driven
_DOC = {
    "rc_transfer":    1_500,
    "noc":              500,
    "insurance":        500,
    "hypothecation":  1_000,
    "state_transfer": 3_000,
}

# Segment margin base rates (%)
_MARGIN_BASE: dict[str, float] = {
    "economy": 2.0,
    "premium": 2.5,
    "luxury":  3.0,
}

# Segment margin caps [min%, max%]
_MARGIN_CAPS: dict[str, tuple[float, float]] = {
    "economy": (1.5,  3.5),
    "premium": (2.0,  4.5),
    "luxury":  (2.5,  6.0),
}

# Condition multipliers (reference only — applied in main.py before engine)
_CONDITION_MULTIPLIERS_REF = {
    "excellent": 1.05, "good": 1.00, "average": 0.92, "poor": 0.82,
}


# VEHICLE CATEGORY CLASSIFIER

def classify_vehicle_category(brand: str, model: str) -> str:
    b = brand.lower().strip()
    m = model.lower().strip()
    luxury_brands = {"bmw", "mercedes-benz", "mercedes", "audi", "lexus", "volvo",
                     "land rover", "jaguar", "porsche", "bentley", "rolls-royce",
                     "ferrari", "lamborghini", "aston martin", "maserati"}
    if b in luxury_brands:
        return "luxury"
    if any(k in m for k in ("fortuner", "endeavour", "glc", "c class", "3 series",
                             "5 series", "x1", "x3", "q3", "q5", "a4", "a6",
                             "xuv700", "safari", "defender", "range rover")):
        return "luxury"
    # BUG-11 fix: added "xuv500" — was falling through to "economy" tier incorrectly
    if any(k in m for k in ("creta", "seltos", "grand vitara", "hector", "compass",
                             "harrier", "scorpio", "ertiga", "carens", "city",
                             "ciaz", "innova", "thar", "xuv300", "xuv500")):
        return "mid_suv"
    if any(k in m for k in ("venue", "nexon", "brezza", "sonet", "ecosport",
                             "duster", "kiger", "magnite", "punch")):
        return "compact_suv"
    if any(k in m for k in ("swift", "baleno", "i20", "altroz", "glanza",
                             "ignis", "polo", "i10", "grand i10")):
        return "premium_hatch"
    return "economy"


# ANNUAL MILEAGE

def _annual_km(km: float, age: int) -> float:
    return km / max(age, 0.5)


def _annual_km_risk_factor(km: float, age: int) -> float:
    """Return 0.0–1.0 risk from annual mileage. Uses dataset-derived tiers."""
    ann = _annual_km(km, age)
    if ann < _ANNUAL_KM_TIERS["very_low"]:   return 0.05
    if ann < _ANNUAL_KM_TIERS["low"]:        return 0.10
    if ann < _ANNUAL_KM_TIERS["moderate"]:   return 0.20
    if ann < _ANNUAL_KM_TIERS["high"]:       return 0.45
    if ann < _ANNUAL_KM_TIERS["very_high"]:  return 0.70
    return 0.90


# 1. MARKET SANITY CLAMP — data-driven, no manual depreciation tables
#
# The ML model already learned age/fuel/odometer relationships during training.
# The clamp only catches gross ML prediction errors (outliers), using
# dataset-derived P10/P90 market bands and MAPE-based tolerance.

def _normalise_model(model_name: str) -> str:
    return " ".join(model_name.lower().split())


def apply_market_sanity_clamp(
    model_name: str,
    segment: str,
    vehicle_age: int,
    raw_value: float,
    city: str = "",
    pre_clamp_confidence: float = 70.0,
    fuel_type: str = "petrol",
    odometer_km: float = 0.0,
    locality: str = "",
    rto: str = "",
) -> tuple[float, bool, str]:
    """
    Clamp ML prediction to dataset-derived market bands.

    - Market bands come from engine_config.json (P10-P90 per model).
    - Clamp tolerance comes from engine_config.json (MAPE-based).
    - No manual age/fuel/odometer adjustments — ML already learned these.
    - Locality/RTO demand used to widen upper band in high-demand areas.

    Returns (clamped_value, was_clamped, note).
    """
    model_key = _normalise_model(model_name)

    # Find market band — try longest match first
    band = None
    keys_to_try = [model_key] + [
        " ".join(model_key.split()[:i])
        for i in range(len(model_key.split()), 0, -1)
    ]
    for key in keys_to_try:
        if key in _MARKET_BANDS_CFG:
            band_data = _MARKET_BANDS_CFG[key]
            band = (float(band_data[0]), float(band_data[1]))
            break

    if band is None:
        seg_data = _SEGMENT_BANDS_CFG.get(segment, [100_000, 20_000_000])
        band     = (float(seg_data[0]), float(seg_data[1]))

    # Locality/RTO uplift on upper band only (genuine market demand, not ML correction)
    loc_key  = (locality or "").strip().lower()
    rto_key  = (rto or "").strip().upper()
    loc_uplt = _LOCALITY_DEMAND.get(loc_key, 0.0)
    rto_uplt = _RTO_DEMAND.get(rto_key, 0.0)
    # Use the stronger of the two signals; cap at ±20%
    geo_uplift = _clamp(max(loc_uplt, rto_uplt), -0.20, 0.20)
    upper_adj  = 1.0 + geo_uplift * 0.5   # apply 50% of uplift to upper band

    lower = band[0]
    upper = band[1] * upper_adj

    # Confidence-scaled tolerance from engine_config
    clamp_data  = _CLAMP_TOLERANCE_CFG.get(segment, [0.88, 1.12])
    base_lo, base_hi = float(clamp_data[0]), float(clamp_data[1])
    conf_factor = max(0.0, (100.0 - pre_clamp_confidence) / 100.0)
    lo_ratio    = base_lo - (1.0 - base_lo) * conf_factor * 0.5
    hi_ratio    = base_hi + (base_hi - 1.0) * conf_factor * 0.5

    clamped = False
    note    = "within expected market band"

    if raw_value > upper * hi_ratio:
        raw_value = upper
        clamped   = True
        note = f"clamped from above — ML overestimated vs {segment} market band"
    elif raw_value < lower * lo_ratio:
        raw_value = lower * 0.92
        clamped   = True
        note = f"clamped from below — ML underestimated vs {segment} market band"

    return float(raw_value), clamped, note


# 2. DYNAMIC RECONDITIONING COST

def compute_dynamic_recon_cost(
    segment: str,
    age: int,
    km: float,
    condition: str,
    inspected: bool,
    brand: str = "",
) -> int:
    base    = _RECON_BASE.get(segment, 18_000)
    age_add = (
        2_000 if 1 <= age <= 3 else
        5_000 if 4 <= age <= 6 else
        10_000 if 7 <= age <= 9 else
        18_000 if age >= 10 else 0
    )
    ann_km  = _annual_km(km, age)
    km_add  = (
        3_000  if _ANNUAL_KM_TIERS["low"]      <= ann_km < _ANNUAL_KM_TIERS["moderate"]  else
        8_000  if _ANNUAL_KM_TIERS["moderate"] <= ann_km < _ANNUAL_KM_TIERS["high"]      else
        14_000 if _ANNUAL_KM_TIERS["high"]     <= ann_km < _ANNUAL_KM_TIERS["very_high"] else
        20_000 if ann_km >= _ANNUAL_KM_TIERS["very_high"] else 0
    )
    subtotal    = base + age_add + km_add
    cond_mult   = {
        "excellent": 0.65, "good": 1.00, "average": 1.45, "poor": 2.20,
    }.get(condition.lower().strip(), 1.00)
    subtotal    = int(subtotal * cond_mult)
    if inspected:
        subtotal = int(subtotal * 0.85)
    brand_mult  = _BRAND_REPAIR_MULTIPLIER.get(brand.lower().strip(), 1.00)
    subtotal    = int(subtotal * brand_mult)
    caps        = {"economy": 70_000, "premium": 150_000, "luxury": 350_000}
    return min(subtotal, caps.get(segment, 100_000))


# 3. DYNAMIC HOLDING COST

def compute_holding_cost(
    segment: str,
    market_value: float,
    brand: str = "",
) -> tuple[int, int]:
    h          = _HOLDING.get(segment, {"rate_pct": 1.8, "days": 30})
    rate       = h["rate_pct"] / 100.0
    base_days  = h["days"]
    pop_factor = _BRAND_POPULARITY.get(brand.lower().strip(), 1.0)
    eff_days   = max(10, min(int(base_days * pop_factor), 120))
    cost       = int(market_value * rate * (eff_days / 30))
    return cost, eff_days


# 4. DOCUMENTATION COST  (statutory — always hardcoded)

def compute_doc_cost(
    registration_state: str = "",
    sale_state: str = "",
    loan_outstanding: bool = False,
) -> tuple[int, dict]:
    rc        = _DOC["rc_transfer"]
    noc       = _DOC["noc"]
    insurance = _DOC["insurance"]
    hypo      = _DOC["hypothecation"] if loan_outstanding else 0
    state     = (
        _DOC["state_transfer"]
        if registration_state and sale_state
           and registration_state.strip().lower() != sale_state.strip().lower()
        else 0
    )
    total = rc + noc + insurance + hypo + state
    return int(total), {
        "rc_transfer": rc, "noc": noc,
        "insurance_trans": insurance, "hypothecation": hypo,
        "state_transfer": state,
    }


# 5. DYNAMIC DEALER PROFIT MARGIN

def dynamic_target_margin(
    segment: str,
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    inspected: bool,
    fuel: str,
    user_target_pct: float = 3.0,  # BUG-07 fix: normalized to same scale as _MARGIN_BASE (was 15.0 → always clamped)
) -> float:
    base   = _MARGIN_BASE.get(segment, 11.0)
    ann_km = _annual_km(km, vehicle_age)

    if vehicle_age <= 2:             base += 0.8
    elif vehicle_age <= 4:           base += 0.5
    if ann_km < _ANNUAL_KM_TIERS["low"]:       base += 0.5
    elif ann_km < _ANNUAL_KM_TIERS["moderate"]: base += 0.2
    if owner_count == 1:             base += 0.4
    if inspected:                    base += 0.3
    if condition.lower() == "excellent": base += 0.4
    if fuel.lower() in {"petrol", "hybrid"}: base += 0.2

    if vehicle_age > 8:              base -= 1.0
    elif vehicle_age > 6:            base -= 0.6
    elif vehicle_age > 4:            base -= 0.2
    if ann_km > _ANNUAL_KM_TIERS["very_high"]:  base -= 0.8
    elif ann_km > _ANNUAL_KM_TIERS["high"]:     base -= 0.5
    elif ann_km > _ANNUAL_KM_TIERS["moderate"]: base -= 0.2
    if owner_count >= 4:             base -= 0.8
    elif owner_count == 3:           base -= 0.6
    elif owner_count == 2:           base -= 0.3
    if condition.lower() == "poor":    base -= 0.8
    elif condition.lower() == "average": base -= 0.4

    lo, hi   = _MARGIN_CAPS.get(segment, (8.0, 18.0))
    computed = _clamp(base, lo, hi)
    blended  = 0.55 * computed + 0.45 * float(user_target_pct)
    return round(_clamp(blended, lo, hi), 1)


# 6. RISK SCORE

def compute_risk_score(
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    fuel: str,
    inspected: bool,
    sanity_clamped: bool = False,
    variant_known: bool = True,
    color_known: bool   = True,
    accident_history: str = "none",
) -> tuple[int, str]:
    age_risk  = _clamp(vehicle_age * 8.0)
    km_risk   = _annual_km_risk_factor(km, vehicle_age) * 100.0
    owner_risk = {1: 10, 2: 32, 3: 58, 4: 75}.get(min(owner_count, 4), 85)
    cond_risk  = {"excellent": 6, "good": 20, "average": 52, "poor": 85}.get(
        condition.lower().strip(), 40
    )
    fuel_risk  = {
        "petrol": 14, "diesel": 24, "cng": 30,
        "electric": 28, "hybrid": 16,
    }.get(fuel.lower().strip(), 25)
    acc_penalty = {"none": 0, "minor": 15, "major": 35}.get(
        accident_history.lower().strip(), 0
    )
    raw = (
        0.22 * age_risk
        + 0.22 * km_risk
        + 0.18 * owner_risk
        + 0.18 * cond_risk
        + 0.07 * fuel_risk
        + 0.05 * (0 if inspected else 25)
        + 0.08 * acc_penalty
    )
    unknown_penalties = 0
    if not variant_known:     unknown_penalties += 6
    if not color_known:       unknown_penalties += 2
    if accident_history in {"unknown", ""}:
        unknown_penalties += 8
    if sanity_clamped:
        unknown_penalties += 12

    score = round(_clamp(raw + unknown_penalties, 5, 95))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return score, level


# 7. RISK BUFFER

def compute_risk_buffer(
    market_value: float,
    risk_score: int,
    segment: str,
    age: int,
    km: float,
    owner_count: int,
    condition: str,
    inspected: bool,
    variant_known: bool      = True,
    owner_known: bool        = True,
    service_hist_known: bool = True,
    accident_hist_known: bool = True,
    reg_state_known: bool    = True,
    color_known: bool        = True,
) -> int:
    seg_factor  = {"economy": 0.80, "premium": 1.00, "luxury": 1.40}.get(segment, 1.00)
    base_buffer = market_value * risk_score * 0.0001 * seg_factor

    age_add = (
        0       if age < 3 else
        1_500   if age < 6 else
        3_000   if age < 9 else
        6_000
    )
    ann_km = _annual_km(km, age)
    km_add = (
        0       if ann_km < _ANNUAL_KM_TIERS["moderate"] else
        1_500   if ann_km < _ANNUAL_KM_TIERS["high"]     else
        3_000   if ann_km < _ANNUAL_KM_TIERS["very_high"] else
        5_000
    )
    owner_add = {1: 0, 2: 1_000, 3: 3_000}.get(min(owner_count, 3), 5_000)
    insp_add  = 0 if inspected else 1_500
    cond_add  = {"poor": 5_000, "average": 2_000}.get(condition.lower(), 0)

    missing_penalties = 0
    if not variant_known:         missing_penalties += 1_500
    if not owner_known:           missing_penalties += 2_000
    if not service_hist_known:    missing_penalties += 1_500
    if not accident_hist_known:   missing_penalties += 3_000
    if not reg_state_known:       missing_penalties += 1_000
    if not color_known:           missing_penalties += 500

    total = int(base_buffer + age_add + km_add + owner_add + insp_add
                + cond_add + missing_penalties)
    # BUG-06 fix: segment-aware minimum floor so buffer is meaningful for cheap cars
    seg_min = {"economy": 3_000, "premium": 6_000, "luxury": 15_000}.get(segment, 3_000)
    cap     = max(seg_min * 4, int(market_value * 0.05))
    return int(_clamp(total, seg_min, cap))


# 8. TWO-COMPONENT CONFIDENCE SCORE  (uses dataset-derived baselines)

def compute_confidence_score(
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    fuel: str,
    variant: str,
    fuel_efficiency: float,
    risk_score: int,
    sanity_clamped: bool,
    city: str = "",
    inspected: bool = False,
    owner_known: bool   = True,
    accident_hist: str  = "none",
    locality: str = "",
) -> tuple[int, int, int]:
    """
    Returns (final_confidence, model_confidence, business_confidence).

    Baselines sourced from engine_config.json confidence_baseline:
      mc_base = 50 + R2 * 40  (88-90 for R2=0.97)
      bc_base = 50 + R2 * 35  (83-85 for R2=0.97)
    """
    mc = float(_CONF_BASELINE.get("mc_base", 88.0))
    bc = float(_CONF_BASELINE.get("bc_base", 84.0))

    # Model confidence adjustments
    if sanity_clamped:                                mc -= 18
    if not locality or locality.lower() in {"", "unknown"}:
        mc -= 3    # locality unknown (weaker signal vs city in old code)
    if variant.lower() in {"", "unknown", "base"}:    mc -= 7
    if fuel_efficiency <= 0:                           mc -= 4
    if km > 150_000:                                   mc -= 10
    elif km > 100_000:                                 mc -= 6
    elif km > 80_000:                                  mc -= 3
    if vehicle_age > 10:                               mc -= 10
    elif vehicle_age > 7:                              mc -= 5
    elif vehicle_age > 4:                              mc -= 2
    if km < 5_000 and vehicle_age > 3:                mc -= 8
    mc -= risk_score * 0.12

    # Business confidence adjustments
    if not owner_known:                                bc -= 12
    if accident_hist.lower() in {"unknown", ""}:       bc -= 10
    if owner_count > 3:                                bc -= (owner_count - 1) * 5
    elif owner_count == 3:                             bc -= 8
    elif owner_count == 2:                             bc -= 3
    if condition.lower() == "poor":                    bc -= 10
    elif condition.lower() == "average":               bc -= 5
    ann_km = _annual_km(km, vehicle_age)
    if ann_km > _ANNUAL_KM_TIERS["very_high"]:        bc -= 12
    elif ann_km > _ANNUAL_KM_TIERS["high"]:           bc -= 6
    if vehicle_age > 10:                               bc -= 8

    # Bonuses
    if inspected:                                      bc += 8
    if condition.lower() == "excellent":               bc += 6
    if owner_count == 1:                               bc += 5
    if ann_km < _ANNUAL_KM_TIERS["very_low"]:         bc += 4
    if fuel.lower() in {"petrol", "hybrid"}:           bc += 2
    if vehicle_age <= 2:                               mc += 3

    mc_clamped = _clamp(mc, 40, 98)
    bc_clamped = _clamp(bc, 38, 96)
    final      = int(round(math.sqrt(mc_clamped * bc_clamped)))
    return int(_clamp(final, 42, 95)), int(mc_clamped), int(bc_clamped)


# 9. MONETARY SHAP EXPLANATION
#    Locality/RTO demand replaces the old city_demand lookup.
#    Age/fuel/odometer impacts removed — ML already captured these.

def shap_explanation(
    market_value: float,
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    fuel: str,
    transmission: str,
    city: str,
    inspected: bool,
    fuel_efficiency: float,
    brand: str = "",
    segment: str = "economy",
    locality: str = "",
    rto: str = "",
) -> list[dict]:
    """
    Monetary impact of each business factor.
    The ML model handles age/fuel/odometer internally; we only surface
    business-layer signals here.
    """
    items: list[dict] = []
    ann_km = _annual_km(km, vehicle_age)

    # Annual mileage intensity (business interpretation of ML signal)
    ann_km_impact = 0
    if ann_km > _ANNUAL_KM_TIERS["very_high"]:
        ann_km_impact = -int(market_value * 0.07)
        lbl = f"Very high annual usage ({ann_km/1000:.0f}k km/yr) — heavy wear risk"
    elif ann_km > _ANNUAL_KM_TIERS["high"]:
        ann_km_impact = -int(market_value * 0.04)
        lbl = f"High annual mileage ({ann_km/1000:.0f}k km/yr) — above-average wear"
    elif ann_km < _ANNUAL_KM_TIERS["very_low"]:
        ann_km_impact = int(market_value * 0.02)
        lbl = f"Very low usage ({ann_km/1000:.0f}k km/yr) — lightly driven"
    else:
        lbl = f"Normal annual usage ({ann_km/1000:.0f}k km/yr)"
    if abs(ann_km_impact) > 1_000:
        items.append({"feature": "Annual Mileage", "value": f"{ann_km/1000:.0f}k km/yr",
                      "contribution": ann_km_impact, "label": lbl})

    # Condition
    cond_impact = {
        "excellent": int(market_value * 0.045),
        "good":      0,
        "average":   -int(market_value * 0.07),
        "poor":      -int(market_value * 0.16),
    }.get(condition.lower().strip(), 0)
    if abs(cond_impact) > 1_000:
        sign = "adds" if cond_impact > 0 else "reduces value by"
        items.append({
            "feature": "Condition", "value": condition.title(),
            "contribution": cond_impact,
            "label": f"{condition.title()} condition {sign} Rs.{abs(cond_impact)//1000:.0f},000",
        })

    # Ownership
    owner_impact = {1: 9_000, 2: -7_000, 3: -20_000, 4: -32_000}.get(
        min(owner_count, 4), -32_000
    )
    items.append({
        "feature": "Ownership", "value": f"{owner_count} owner(s)",
        "contribution": owner_impact,
        "label": (
            "First owner — highest buyer preference: +Rs.9,000"
            if owner_count == 1
            else f"{owner_count} previous owners reduce value by Rs.{abs(owner_impact)//1000:.0f},000"
        ),
    })

    # Locality demand (replaces city_demand — more granular for Bangalore)
    loc_key    = (locality or "").strip().lower()
    rto_key    = (rto or "").strip().upper()
    loc_prem   = _LOCALITY_DEMAND.get(loc_key, _RTO_DEMAND.get(rto_key, 0.0))
    loc_impact = int(market_value * loc_prem)
    if abs(loc_impact) > 1_000:
        area_name = locality.title() if locality and locality != "unknown" else rto.upper()
        sign      = "adds" if loc_impact > 0 else "reduces value by"
        items.append({
            "feature": "Area Demand", "value": area_name,
            "contribution": loc_impact,
            "label": f"{area_name} market {sign} Rs.{abs(loc_impact)//1000:.0f},000",
        })

    # Transmission
    if transmission.lower() in {"automatic", "cvt", "dct", "amt"}:
        at_impact = int(market_value * 0.025)
        items.append({
            "feature": "Transmission", "value": "Automatic",
            "contribution": at_impact,
            "label": f"Automatic transmission adds Rs.{at_impact//1000:.0f},000 (buyer preference)",
        })

    # Inspection
    if inspected:
        insp_impact = int(market_value * _CERTIFIED_PREMIUM * 0.5)
        items.append({
            "feature": "Inspection", "value": "Certified",
            "contribution": insp_impact,
            "label": f"Certified inspection adds Rs.{insp_impact//1000:.0f},000 to buyer confidence",
        })

    # Fuel efficiency
    if fuel_efficiency and fuel_efficiency > 0:
        fe_delta  = fuel_efficiency - 16.0
        fe_impact = int(fe_delta * market_value * 0.003)
        if abs(fe_impact) > 1_500:
            sign = "adds" if fe_impact > 0 else "reduces"
            items.append({
                "feature": "Fuel Efficiency", "value": f"{fuel_efficiency:.1f} km/l",
                "contribution": fe_impact,
                "label": f"{fuel_efficiency:.1f} km/l efficiency {sign} Rs.{abs(fe_impact)//1000:.0f},000",
            })

    # Brand tier signal
    _BRAND_TIER_MAP = {
        **{b: "luxury"  for b in {"bmw", "mercedes-benz", "audi", "jaguar",
                                   "land rover", "porsche", "ferrari", "bentley"}},
        # BUG-09 fix: added mahindra to premium tier (XUV500/700 are mid-SUV premium, not mid-tier)
        **{b: "premium" for b in {"volkswagen", "skoda", "toyota", "mg",
                                   "kia", "jeep", "mini", "volvo", "lexus", "mahindra"}},
        **{b: "budget"  for b in {"maruti", "maruti suzuki", "datsun", "chevrolet"}},
    }
    brand_tier = _BRAND_TIER_MAP.get(brand.lower().strip(), "mid")
    brand_impact = {
        "luxury":  int(market_value * 0.03),
        "premium": int(market_value * 0.015),
        "mid":     0,
        "budget":  -int(market_value * 0.01),
    }.get(brand_tier, 0)
    if abs(brand_impact) > 2_000:
        sign = "adds" if brand_impact > 0 else "reduces"
        items.append({
            "feature": "Brand Premium", "value": brand.title(),
            "contribution": brand_impact,
            "label": f"{brand.title()} brand {sign} Rs.{abs(brand_impact)//1000:.0f},000",
        })

    return sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:8]


# 10. NEGOTIATION TRIO  (uses locality_demand instead of city_demand)

def compute_negotiation_trio(
    recommended_buy_price: float,
    city: str,
    condition: str,
    risk_score: int,
    seller_reason: str = "upgrading",
    seller_asking_price: float = 0,
    locality: str = "",
    rto: str = "",
    market_sell_price: float = 0,   # BUG-05 fix: cap walk_away below market sell ceiling
) -> dict:
    loc_key  = (locality or "").strip().lower()
    rto_key  = (rto or "").strip().upper()
    demand   = _LOCALITY_DEMAND.get(loc_key, _RTO_DEMAND.get(rto_key, 0.015))
    nego_pct = max(0.04, 0.07 - demand * 0.8)
    nego_room = int(recommended_buy_price * nego_pct)

    seller_adj = {
        "financial":  -int(nego_room * 0.30),
        "relocating": -int(nego_room * 0.15),
        "upgrading":  0,
        "unused":      int(nego_room * 0.10),
        "problem":    -int(nego_room * 0.25),
    }.get(seller_reason.lower().strip(), 0)

    risk_adj  = int(risk_score * 80)
    opening   = _round500(max(0, recommended_buy_price - nego_room - risk_adj + seller_adj))
    ideal     = _round500(max(0, recommended_buy_price - int(nego_room * 0.35)))
    walk_raw  = recommended_buy_price * 1.015
    walk_away = _round500(_clamp(walk_raw,
                                  recommended_buy_price + 3_000,
                                  recommended_buy_price + 25_000))
    # BUG-05 fix: walk_away must never exceed the market sell ceiling
    if market_sell_price > 0:
        walk_away = min(walk_away, int(round(market_sell_price * 0.98 / 500) * 500))

    potential_savings = (
        _round500(max(0, seller_asking_price - ideal))
        if seller_asking_price > 0 else 0
    )
    return {
        "opening_offer":    opening,
        "target_offer":     ideal,
        "walk_away_price":  walk_away,
        "negotiation_room": nego_room,
        "potential_savings": potential_savings,
        "seller_adjustment": seller_adj,
    }


# 11. ADAPTIVE COMPARABLE VEHICLE SERVICE + CONFIDENCE ENGINE

import json as _json
import os as _os
import math as _math

# ── VALUATION CONFIG ──────────────────────────────────────────────────────────

def _load_valuation_config() -> dict:
    """Load valuation_config.json from the backend directory. Robust fallback."""
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_here, "valuation_config.json")
    try:
        with open(_path, encoding="utf-8-sig") as _f:
            return _json.load(_f)
    except Exception as _e:
        print(f"[decision_engine] valuation_config.json not found — using defaults: {_e}")
        return {}

_VCFG: dict = _load_valuation_config()

def _vcfg(key: str, default):
    return _VCFG.get(key, default)

# ── DATASET LOADER ─────────────────────────────────────────────────────────────

_DATASET_DF         = None
_DATASET_LOAD_TRIED = False
_DATASET_NORM_VER   = 0         # bump to force re-normalization on next load
_VARIANT_NORM_VERSION = 2       # current normalization version — bump when rules change

def _load_dataset_df():
    global _DATASET_DF, _DATASET_LOAD_TRIED, _DATASET_NORM_VER
    if _DATASET_LOAD_TRIED and _DATASET_NORM_VER == _VARIANT_NORM_VERSION:
        return _DATASET_DF
    _DATASET_LOAD_TRIED = True
    _DATASET_NORM_VER   = _VARIANT_NORM_VERSION
    try:
        import pandas as _pd
        _here     = _os.path.dirname(_os.path.abspath(__file__))
        _data_dir = _os.path.normpath(_os.path.join(_here, "..", "ml_training", "data"))

        _candidates = [
            "processed_overall.csv",
            "processed_s1_s4_owner_1.csv",
            "processed_s1_s4_owner.csv",
            "processed_pincode without owner-4.csv",
            "processed_with owner filled.csv",
        ]
        _csv = None
        for _name in _candidates:
            _p = _os.path.join(_data_dir, _name)
            if _os.path.exists(_p):
                _csv = _p
                break

        if _csv is None:
            return None

        _base_cols  = ["brand", "model", "variant", "fuel_type", "transmission",
                       "year", "odometer_reading", "selling_price"]
        _extra_cols = ["owner_count", "vehicle_age", "city", "locality"]
        _want = _base_cols + _extra_cols

        _df   = _pd.read_csv(_csv, low_memory=False)
        _keep = [c for c in _want if c in _df.columns]
        _df   = _df[_keep].copy()
        _df   = _df.dropna(subset=["brand", "model", "selling_price"])

        for c in ["brand", "model", "fuel_type", "transmission"]:
            if c in _df.columns:
                _df[c] = _df[c].astype(str).str.strip().str.lower().fillna("unknown")
        # Normalize variant strings so typographic variants of the same trim match:
        # "zxi+" == "zxi plus", "sx(o)" == "sx o", "ags" == "amt", etc.
        if "variant" in _df.columns:
            _df["variant"] = _df["variant"].astype(str).str.strip().str.lower() \
                                            .fillna("unknown").apply(_normalize_variant)

        _df["selling_price"]    = _pd.to_numeric(_df["selling_price"],    errors="coerce")
        _df["odometer_reading"] = _pd.to_numeric(_df["odometer_reading"], errors="coerce").fillna(0)

        if "year" in _df.columns:
            _df["year"] = _pd.to_numeric(_df["year"], errors="coerce")
        else:
            _df["year"] = float("nan")

        if "vehicle_age" not in _df.columns and "year" in _df.columns:
            _df["vehicle_age"] = (datetime.now().year - _df["year"]).clip(lower=0)
        elif "vehicle_age" in _df.columns:
            _df["vehicle_age"] = _pd.to_numeric(_df["vehicle_age"], errors="coerce").fillna(0)

        if "owner_count" in _df.columns:
            _df["owner_count"] = _pd.to_numeric(_df["owner_count"], errors="coerce").fillna(1)

        _df = _df.dropna(subset=["selling_price"])
        _df = _df[_df["selling_price"].between(50_000, 20_000_000)]
        _DATASET_DF = _df.reset_index(drop=True)
    except Exception:
        pass
    return _DATASET_DF


def _normalize_variant(v: str) -> str:
    """
    Canonicalize Indian car variant name strings so that typographic variants
    of the same trim are treated as identical during similarity scoring.

    Covers ALL major Indian brands:
      Maruti:   zxi+/zxi plus, vxi+/vxi plus, lxi+/lxi plus, zdi+/zdi plus, etc.
      Hyundai:  sx(o)/sxo, asta(o)/asta o, magna+/magna plus, era+/era plus
      Tata:     xz+/xz plus, xt+/xt plus, xza+/xza plus, creative+/creative plus
      Ford:     titanium+/titanium plus, trend+/trend plus
      Kia:      htk+/htk plus, htx+/htx plus, gtx+/gtx plus
      Mahindra: w8(o)/w8 o, w6(o)/w6 o
      Renault:  rxz+/rxz plus, rxt+/rxt plus, rxl(o)/rxl o, climber(o)/climber o
      VW:       comfortline(p)/comfortline p, comfortline(d)/comfortline d
      Others:   d-lite+/d-lite plus, k6+/k6 plus, fearless+/fearless plus

    Transmission synonyms:  amt == ags == at
    Drivetrain synonyms:    4wd == 4x4 == awd
    Optional suffix:        (opt) → opt
    Spacing:                dual tone == dualtone
    """
    import re as _re
    s = str(v or "").strip().lower()
    if not s or s in ("", "unknown", "nan", "none"):
        return "unknown"
    s = _re.sub(r"\s+", " ", s)          # collapse whitespace

    # ── Step 1: Parenthesized option letters → space-separated ────────────
    # Handles (o), (s), (p), (d), (l), (hs), (ps) for ALL brands
    # e.g. "sx(o)" → "sx o", "asta (o)" → "asta o", "rxl(o)" → "rxl o"
    #      "comfortline 1.5 (d)" → "comfortline 1.5 d"
    #      "xz plus (hs)" → "xz plus hs", "fearless plus (ps)" → "fearless plus ps"
    s = _re.sub(r"\s*\(o\)", " o", s)
    s = _re.sub(r"\s*\(s\)", " s", s)
    s = _re.sub(r"\s*\(p\)", " p", s)
    s = _re.sub(r"\s*\(d\)", " d", s)
    s = _re.sub(r"\s*\(l\)", " l", s)
    s = _re.sub(r"\s*\(hs\)", " hs", s)
    s = _re.sub(r"\s*\(ps\)", " ps", s)

    # (opt) → opt  (e.g. "titanium 1.0 ecoboost (opt)" → "titanium 1.0 ecoboost opt")
    s = _re.sub(r"\s*\(opt\)", " opt", s)

    # Generic: any remaining single-letter/short parenthesized tokens
    # e.g. "a(o)" → "a o", "std(o)" → "std o", "b6(o)" → "b6 o"
    s = _re.sub(r"\(([a-z]{1,3})\)", r" \1", s)

    # ── Step 2: + → plus (ALL brands, ALL positions) ─────────────────────
    # Handles trailing, mid-word, and spaced + for every brand:
    # "zxi+" → "zxi plus", "titanium + 1.5l" → "titanium plus 1.5l"
    # "creative+ amt" → "creative plus amt", "d-lite+" → "d-lite plus"
    # "vxi + (o) amt" → already (o) stripped → "vxi + o amt" → "vxi plus o amt"
    s = _re.sub(r"\s*\+\s*", " plus ", s)

    # ── Step 3: Transmission synonyms ─────────────────────────────────────
    # amt == ags (Maruti) == at (generic automatic) — all brands
    s = _re.sub(r"\bags\b", "amt", s)
    # Only standalone "at" (not inside words like "asta")
    # Use word boundary + negative lookbehind/ahead to avoid mangling
    s = _re.sub(r"(?<!\w)\bat\b(?!\w*a)", "amt", s)

    # ── Step 4: Drivetrain synonyms ───────────────────────────────────────
    s = _re.sub(r"\b(4x4|awd)\b", "4wd", s)

    # ── Step 5: Fuel type tokens in variant names ─────────────────────────
    # "diesel" → "d", "petrol" → "p" (standalone only, all brands)
    s = _re.sub(r"\bdiesel\b", "d", s)
    s = _re.sub(r"\bpetrol\b", "p", s)

    # ── Step 6: "dual tone" / "dualtone" / "dual-tone" normalization ──────
    s = _re.sub(r"\bdual[\s-]?tone\b", "dualtone", s)

    # ── Step 7: Abbreviation synonyms ─────────────────────────────────────
    # "pl" → "plus" (rare scraper abbreviation)
    s = _re.sub(r"\bpl\b", "plus", s)
    # "bs-iv" / "bsiv" / "bs iv" → "bs4"
    s = _re.sub(r"\bbs[\s-]?iv\b", "bs4", s)
    s = _re.sub(r"\bbs[\s-]?vi\b", "bs6", s)
    # "s-cng" / "scng" → "cng"
    s = _re.sub(r"\bs[\s-]?cng\b", "cng", s)
    # "i-vtec" / "ivtec" → "ivtec" (Honda)
    s = _re.sub(r"\bi[\s-]vtec\b", "ivtec", s)
    # "i-dtec" / "idtec" → "idtec" (Honda)
    s = _re.sub(r"\bi[\s-]dtec\b", "idtec", s)
    # "ti-vct" / "tivct" → "tivct" (Ford)
    s = _re.sub(r"\bti[\s-]vct\b", "tivct", s)
    # "ecoboost" stays as-is (already uniform)

    # ── Step 8: Remove "outside fitted" qualifier for CNG/LPG ─────────────
    s = _re.sub(r"\(outside fitted\)", "", s)
    s = _re.sub(r"\boutside fitted\b", "", s)

    return " ".join(s.split())   # final whitespace collapse


class AdaptiveComparableService:

    """
    Production-grade adaptive comparable vehicle search.

    Key differences from the old 4-stage hierarchical approach:
    - Every candidate receives a weighted Gaussian similarity score in one pass.
    - No progressive feature dropping — quality over quantity.
    - Only vehicles above a configurable min_similarity_threshold are kept.
    - Similarity weights and all thresholds are read from valuation_config.json.
    - Brand-aware: luxury brands use a lower threshold (fewer listings expected).
    """

    def __init__(self) -> None:
        # Load tuning params from config (already loaded at module level as _VCFG)
        self._weights: dict[str, float] = _vcfg("similarity_weights", {
            "brand": 0.18, "model": 0.18, "variant": 0.16,
            "vehicle_age": 0.12, "odometer_reading": 0.10,
            "fuel_type": 0.10, "transmission": 0.07,
            "owner_count": 0.05, "seller_type": 0.02, "locality": 0.02,
        })
        self._age_sigma: float      = float(_vcfg("age_sigma", 3.0))
        self._odo_sigma: float      = float(_vcfg("odometer_sigma", 25_000))
        self._min_sim: float        = float(_vcfg("min_similarity_threshold", 0.55))
        self._luxury_min_sim: float = float(_vcfg("luxury_min_similarity_threshold", 0.45))
        self._luxury_brands: set    = set(_vcfg("luxury_brands", [
            "bmw", "mercedes-benz", "audi", "jaguar", "land rover",
            "porsche", "volvo", "lexus", "mini", "maserati",
            "bentley", "rolls-royce", "ferrari", "lamborghini",
        ]))
        self._max_comps: int  = int(_vcfg("max_comps_used_for_range", 50))
        self._top_n: int      = int(_vcfg("top_n_display", 5))

    # ── public API ─────────────────────────────────────────────────────────────

    def search(self, *, brand, model, variant, fuel, transmission,
               year, odometer, owner_count=1, current_year=None,
               seller_type="unknown", locality="unknown") -> dict:
        """
        Returns a result dict with:
          comps         — list[dict] of comparable records (sorted by sim score)
          similar_cars  — list[dict] UI-ready cards (top_n_display)
          sim_scores    — list[float] parallel to comps
          stage         — always 1 (single-pass)
          stage_label   — "adaptive_similarity"
        """
        import pandas as _pd
        import numpy as _np

        df = _load_dataset_df()
        if df is None or df.empty:
            return self._empty_result()

        _cy         = current_year or datetime.now().year
        vehicle_age = max(0, _cy - int(year)) if year else 0
        bk  = str(brand or "").strip().lower()
        mk  = str(model or "").strip().lower()
        # Normalize the query variant so "zxi+" and "zxi plus" resolve to the same token
        vk  = _normalize_variant(str(variant or "").strip().lower())
        fk  = str(fuel or "").strip().lower()
        tk  = str(transmission or "").strip().lower()
        slk = str(seller_type or "").strip().lower()
        lok = str(locality or "").strip().lower()

        is_luxury   = bk in self._luxury_brands
        threshold   = self._luxury_min_sim if is_luxury else self._min_sim

        # ── compute per-attribute scores ──────────────────────────────────────
        W  = self._weights
        s  = _pd.Series(0.0, index=df.index)

        # Categorical exact matches
        s += df["brand"].eq(bk).astype(float)          * W.get("brand", 0.20)
        s += df["model"].eq(mk).astype(float)           * W.get("model", 0.20)
        s += df["fuel_type"].eq(fk).astype(float)       * W.get("fuel_type", 0.10)
        s += df["transmission"].eq(tk).astype(float)    * W.get("transmission", 0.08)

        # Variant: exact = 1.0, high overlap (≥80%) = 0.90, moderate = scaled, unknown = 0
        # This graduated scoring ensures "zxi plus" ≈ "zxi plus amt" scores high
        # instead of the old flat 0.5x penalty that killed match scores.
        if vk and vk not in ("", "unknown"):
            vk_tokens = set(vk.split())
            exact_mask = df["variant"].eq(vk)
            def _token_sim(cell_val):
                tokens = set(str(cell_val).split())
                if not tokens or not vk_tokens:
                    return 0.0
                intersection = len(vk_tokens & tokens)
                # Jaccard-style: overlap relative to the LARGER set
                union = len(vk_tokens | tokens)
                jaccard = intersection / max(union, 1)
                # Also check directional overlap (query tokens found in candidate)
                query_recall = intersection / max(len(vk_tokens), 1)
                # Use the better of the two signals
                overlap = max(jaccard, query_recall)
                # Graduated scoring: high overlap → near-exact, low → proportional
                if overlap >= 0.90:
                    return 0.95   # near-exact (e.g. "zxi plus" vs "zxi plus amt")
                elif overlap >= 0.75:
                    return 0.85   # strong match
                elif overlap >= 0.50:
                    return 0.65   # moderate match
                else:
                    return overlap * 0.50  # weak — proportional penalty
            token_scores = df["variant"].map(_token_sim)
            # Exact match overrides token overlap
            variant_scores = _np.where(exact_mask, 1.0, token_scores)
            s += _pd.Series(variant_scores, index=df.index) * W.get("variant", 0.16)

        # Seller type: exact = 1.0, missing/unknown = 0.5
        if "seller_type" in df.columns and slk:
            st_score = _np.where(
                df["seller_type"].eq(slk), 1.0,
                _np.where(df["seller_type"].isin(["", "unknown", "nan"]), 0.5, 0.0)
            )
            s += _pd.Series(st_score, index=df.index) * W.get("seller_type", 0.02)
        else:
            s += 0.5 * W.get("seller_type", 0.02)  # no data → neutral

        # Locality: exact = 1.0, else 0.0
        if "locality" in df.columns and lok and lok not in ("", "unknown"):
            s += df["locality"].eq(lok).astype(float) * W.get("locality", 0.01)

        # Owner count: exact = 1.0, off-by-1 = 0.5, else 0.0
        if "owner_count" in df.columns:
            oc = float(owner_count or 1)
            owner_sim = _np.where(
                df["owner_count"].eq(oc), 1.0,
                _np.where(df["owner_count"].sub(oc).abs().le(1), 0.5, 0.0)
            )
            s += _pd.Series(owner_sim, index=df.index) * W.get("owner_count", 0.05)

        # Vehicle age: Gaussian decay
        if "vehicle_age" in df.columns:
            age_diff = (df["vehicle_age"] - vehicle_age).abs()
            age_sim  = _np.exp(-0.5 * (age_diff / self._age_sigma) ** 2)
            s += _pd.Series(age_sim.values, index=df.index) * W.get("vehicle_age", 0.12)

        # Odometer: Gaussian decay
        odo_diff = (df["odometer_reading"] - float(odometer or 0)).abs()
        odo_sim  = _np.exp(-0.5 * (odo_diff / self._odo_sigma) ** 2)
        s += _pd.Series(odo_sim.values, index=df.index) * W.get("odometer_reading", 0.10)

        # ── threshold filter ──────────────────────────────────────────────────
        df = df.copy()
        df["_sim"] = s.values
        filtered = df[df["_sim"] >= threshold].sort_values("_sim", ascending=False)

        if filtered.empty:
            return self._empty_result()

        # Cap at max_comps for range calculation; keep top_n for display
        range_rows = filtered.head(self._max_comps)
        sim_scores = range_rows["_sim"].tolist()
        comps      = range_rows.drop(columns=["_sim"]).to_dict("records")
        ui_cards   = self._to_ui_cards(filtered.head(self._top_n))

        return {
            "comps":       comps,
            "sim_scores":  sim_scores,
            "stage":       1,
            "stage_label": "adaptive_similarity",
            "similar_cars": ui_cards,
        }

    # ── internal helpers ───────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "comps": [], "sim_scores": [], "stage": 0,
            "stage_label": "no_data", "similar_cars": [],
        }

    def _to_ui_cards(self, rows) -> list[dict]:
        import math as _m
        results, seen = [], set()
        for _, row in rows.iterrows():
            key = (str(row.get("model", "")), str(row.get("year", "")),
                   str(row.get("selling_price", "")))
            if key in seen:
                continue
            seen.add(key)
            yr  = row.get("year",  0)
            odo = row.get("odometer_reading", 0)
            sp  = row.get("selling_price", 0)
            oc  = row.get("owner_count", 1)
            sim = float(row.get("_sim", 0))
            try:
                yr  = int(float(yr))  if _m.isfinite(float(yr))  else 0
                odo = int(float(odo)) if _m.isfinite(float(odo)) else 0
                sp  = int(round(float(sp) / 500) * 500)
                oc  = int(float(oc))  if _m.isfinite(float(oc))  else 1
            except (TypeError, ValueError):
                pass
            results.append({
                "brand":        str(row.get("brand",        "")).title(),
                "model":        str(row.get("model",        "")).title(),
                "variant":      str(row.get("variant",      "")).title(),
                "year":         yr,
                "fuel":         str(row.get("fuel_type",    "")).title(),
                "transmission": str(row.get("transmission", "")).title(),
                "odometer":     odo,
                "owner_count":  oc,
                "market_value": sp,
                "city":         str(row.get("city", row.get("locality", ""))).title(),
                "condition":    "Good",
                "source":       "dataset",
                "similarity":   round(sim * 100, 1),
            })
        return results


# ── PHASE 2: ADAPTIVE RANGE ENGINE ────────────────────────────────────────────

class AdaptiveRangeEngine:
    """
    Builds market range with ML prediction as the anchor.

    Strategy (read from valuation_config.json):
    - High Confidence  (≥ high_confidence_min_comps, avg_sim ≥ high_confidence_avg_sim)
        range = prediction ± sim-weighted comp deviations
                blended 70% comp / 30% MAPE
    - Medium Confidence (≥ medium_confidence_min_comps, avg_sim ≥ medium_confidence_avg_sim)
        range = prediction ± blend(comp_deviation, MAPE, 50/50)
    - Low Confidence   (<medium_confidence_min_comps or avg_sim too low)
        range = prediction ± MAPE

    Hard cap: range width ≤ max_allowed_range_pct × prediction.
    Result is always centred on the ML prediction.
    """

    def __init__(self) -> None:
        self._hi_min_comps: int   = int(_vcfg("high_confidence_min_comps", 10))
        self._med_min_comps: int  = int(_vcfg("medium_confidence_min_comps", 4))
        self._hi_avg_sim: float   = float(_vcfg("high_confidence_avg_sim", 0.75))
        self._med_avg_sim: float  = float(_vcfg("medium_confidence_avg_sim", 0.60))
        self._cw_hi: float        = float(_vcfg("comp_weight_high", 0.70))
        self._mlw_hi: float       = float(_vcfg("ml_weight_high", 0.30))
        self._cw_med: float       = float(_vcfg("comp_weight_medium", 0.50))
        self._mlw_med: float      = float(_vcfg("ml_weight_medium", 0.50))
        self._max_range_pct: float = float(_vcfg("max_allowed_range_pct", 0.25))

    def build(self, *, prediction: float, comps: list[dict],
              sim_scores: list[float], mape: float,
              odometer: float = 0.0) -> dict:
        """
        Returns:
          price_min, price_max, price_median — all ₹ integers
          confidence_case   — "high" | "medium" | "low"
          avg_similarity    — 0–1 float
          n_comps           — int
        """
        import numpy as _np

        pred = float(prediction)
        n    = len(comps)
        mape = max(0.01, float(mape))  # guard against 0

        prices = []
        valid_sims = []
        for rec, sim in zip(comps, sim_scores):
            p = rec.get("selling_price")
            if p and _math.isfinite(float(p)) and float(p) > 0:
                prices.append(float(p))
                valid_sims.append(float(sim))

        n_valid   = len(prices)
        avg_sim   = float(_np.mean(valid_sims)) if valid_sims else 0.0

        # ── determine confidence case ─────────────────────────────────────────
        if n_valid >= self._hi_min_comps and avg_sim >= self._hi_avg_sim:
            case = "high"
        elif n_valid >= self._med_min_comps and avg_sim >= self._med_avg_sim:
            case = "medium"
        else:
            case = "low"

        # ── compute similarity-weighted deviations ────────────────────────────
        if n_valid > 0 and case in ("high", "medium"):
            devs = _np.array([(p - pred) / pred for p in prices])
            sims = _np.array(valid_sims)

            neg_mask = devs < 0
            pos_mask = devs >= 0
            sum_sims = sims.sum() or 1.0

            wlo = float((devs * sims * neg_mask).sum() / sum_sims)  # ≤ 0
            whi = float((devs * sims * pos_mask).sum() / sum_sims)  # ≥ 0

            comp_lo_frac = abs(wlo)
            comp_hi_frac = abs(whi)
        else:
            comp_lo_frac = comp_hi_frac = 0.0

        # ── build bands ───────────────────────────────────────────────────────
        if case == "high":
            lo_frac = self._cw_hi * comp_lo_frac + self._mlw_hi * mape
            hi_frac = self._cw_hi * comp_hi_frac + self._mlw_hi * mape
        elif case == "medium":
            lo_frac = self._cw_med * comp_lo_frac + self._mlw_med * mape
            hi_frac = self._cw_med * comp_hi_frac + self._mlw_med * mape
        else:
            lo_frac = hi_frac = mape

        # ── Anchor center: similarity-weighted top-comp anchor ───────────────────
        # Diagnostic finding: with top_k=10, comps 9–12 were 1-owner BUT 72k km
        # (vs user's 55k km), priced at ₹7.53L–₹7.62L — dragging the anchor from
        # ₹9.5L (top 94% match) down to ₹8.37L. Fix: top_k=5 + odometer proximity
        # penalty so high-mileage outliers carry proportionally less weight.
        if n_valid > 0 and case in ("high", "medium"):
            sims_arr   = _np.array(valid_sims)
            prices_arr = _np.array(prices)

            # Use only the top 5 highest-similarity comps as the price anchor.
            # This avoids dilution from lower-ranked comps with very different
            # odometer readings (e.g. 72k km when query is 55k km).
            top_k = min(5, n_valid)
            top_prices     = prices_arr[:top_k]
            top_sims       = sims_arr[:top_k]
            top_comps_list = comps[:top_k]

            # Query odometer for proximity penalty — use the real user input, NOT comps[0]'s
            # odometer (which is a dataset vehicle's reading, not the car being evaluated).
            query_odo = float(odometer) if odometer else 0.0

            combined_weights = []
            for c, s in zip(top_comps_list, top_sims):
                oc  = c.get("owner_count", 1)
                odo = float(c.get("odometer_reading", query_odo) or query_odo)

                # Owner alignment boost: same owner count preferred
                w_oc = 1.30 if str(oc) == "1" or oc == 1 else 0.80

                # Odometer proximity: Gaussian decay with sigma=20,000 km.
                # A comp at 72k km vs query 55k km → penalty ≈ 0.64×.
                # A comp at 51k km vs query 55k km → penalty ≈ 0.98× (near-perfect).
                odo_sigma = 20_000.0
                if query_odo > 0 and odo > 0:
                    w_odo = float(_np.exp(-0.5 * ((odo - query_odo) / odo_sigma) ** 2))
                else:
                    w_odo = 1.0

                combined_weights.append((s ** 6) * w_oc * w_odo)

            exp_weights = _np.array(combined_weights)
            comp_weighted_anchor = float(_np.average(top_prices, weights=exp_weights))

            comp_median = comp_weighted_anchor
            comp_p25    = float(_np.percentile(prices, 25))
            comp_p75    = float(_np.percentile(prices, 75))

            if case == "high":
                # High confidence: trust top weighted comp anchor heavily (70%)
                blended_center = 0.70 * comp_weighted_anchor + 0.30 * pred
            else:
                # Medium confidence: equal blend
                blended_center = 0.50 * comp_weighted_anchor + 0.50 * pred
        else:
            blended_center = pred
            comp_p25 = comp_p75 = pred

        # ── build final price range around the blended center ─────────────────
        price_median = int(round(blended_center / 500) * 500)
        price_min    = int(round(blended_center * (1 - lo_frac) / 500) * 500)
        price_max    = int(round(blended_center * (1 + hi_frac) / 500) * 500)

        # Sanity: min < median < max
        price_min    = min(price_min, price_median - 500)
        price_max    = max(price_max, price_median + 500)


        return {
            "price_min":       price_min,
            "price_max":       price_max,
            "price_median":    price_median,
            "confidence_case": case,
            "avg_similarity":  round(avg_sim, 4),
            "n_comps":         n_valid,
            # BUG-02 fix: real quartiles for IQR outlier detection in main.py
            "comp_p25":        int(round(comp_p25 / 500) * 500) if n_valid > 0 else price_min,
            "comp_p75":        int(round(comp_p75 / 500) * 500) if n_valid > 0 else price_max,
        }


# ── PHASE 3: CONFIDENCE ENGINE ────────────────────────────────────────────────

class ConfidenceEngine:
    """
    Produces a composite confidence score (0–100) and label.

    Inputs:
      n_comps          int    — number of quality comparables used
      avg_similarity   float  — mean similarity score (0–1) of those comps
      mape             float  — model MAPE as fraction (e.g. 0.062)
      ensemble_variance float — std of individual model log-predictions
      range_width_pct  float  — final range width as fraction of prediction

    Output:
      score  float  0–100
      label  str    Very High | High | Medium | Low | Very Low
      market_support  str   Strong | Good | Moderate | Weak
    """

    def __init__(self) -> None:
        self._hi_min = int(_vcfg("high_confidence_min_comps", 10))
        w = _vcfg("confidence_weights", {})
        self._w_comp  = float(w.get("comp_score",       0.30))
        self._w_sim   = float(w.get("similarity_score", 0.25))
        self._w_mape  = float(w.get("mape_score",       0.20))
        self._w_var   = float(w.get("variance_score",   0.15))
        self._w_width = float(w.get("width_score",      0.10))

        lbl = _vcfg("confidence_labels", {})
        self._vh_min  = int(lbl.get("very_high_min", 90))
        self._h_min   = int(lbl.get("high_min",      75))
        self._m_min   = int(lbl.get("medium_min",    55))
        self._l_min   = int(lbl.get("low_min",       35))

        ms = _vcfg("market_support_labels", {})
        self._ms_strong   = int(ms.get("strong_min",   90))
        self._ms_good     = int(ms.get("good_min",     70))
        self._ms_moderate = int(ms.get("moderate_min", 50))

    def score(self, *, n_comps: int, avg_similarity: float, mape: float,
              ensemble_variance: float, range_width_pct: float) -> dict:
        comp_score   = min(100.0, (n_comps / max(self._hi_min, 1)) * 100.0)
        sim_score    = float(avg_similarity) * 100.0
        mape_score   = max(0.0, 100.0 - float(mape) * 1000.0)
        var_score    = max(0.0, 100.0 - float(ensemble_variance) * 5000.0)
        width_score  = max(0.0, 100.0 - float(range_width_pct) * 200.0)

        total = (
            comp_score  * self._w_comp  +
            sim_score   * self._w_sim   +
            mape_score  * self._w_mape  +
            var_score   * self._w_var   +
            width_score * self._w_width
        )
        total = max(0.0, min(100.0, total))

        if total >= self._vh_min:
            label = "Very High"
        elif total >= self._h_min:
            label = "High"
        elif total >= self._m_min:
            label = "Medium"
        elif total >= self._l_min:
            label = "Low"
        else:
            label = "Very Low"

        if total >= self._ms_strong:
            market_support = "Strong"
        elif total >= self._ms_good:
            market_support = "Good"
        elif total >= self._ms_moderate:
            market_support = "Moderate"
        else:
            market_support = "Weak"

        return {
            "confidence_score":  round(total / 100.0, 4),
            "confidence":        label,
            "market_support":    market_support,
        }


# ── SINGLETONS ────────────────────────────────────────────────────────────────

_adaptive_comparable_service = AdaptiveComparableService()
_adaptive_range_engine       = AdaptiveRangeEngine()
_confidence_engine           = ConfidenceEngine()

# Backward-compatible alias so any existing direct references still work
_comparable_service = _adaptive_comparable_service


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def generate_similar_cars(market_value, brand, model, year, fuel, city, segment,
                           variant="unknown", transmission="manual", owner_count=1,
                           odometer=0) -> list[dict]:
    result = _adaptive_comparable_service.search(
        brand=brand, model=model, variant=variant,
        fuel=fuel, transmission=transmission,
        year=year, odometer=float(odometer or 0), owner_count=owner_count,
    )
    return result.get("similar_cars", [])


def get_market_range_result(*, brand, model, variant, fuel, transmission,
                             year, odometer, owner_count, prediction,
                             model_mape: float = 0.0647,
                             ensemble_variance: float = 0.0,
                             seller_type: str = "unknown",
                             locality: str = "unknown") -> dict:
    """
    Adaptive valuation engine entry point.

    Returns all original keys (backward-compatible) plus new enrichment fields:
      confidence, confidence_score, market_support,
      comparables_used, average_similarity, ensemble_variance, expected_model_error
    """
    _mape = float(_CONF_BASELINE.get("mape_frac", model_mape))

    # Phase 1 — weighted similarity search
    result = _adaptive_comparable_service.search(
        brand=brand, model=model, variant=variant,
        fuel=fuel, transmission=transmission,
        year=year, odometer=odometer, owner_count=owner_count,
        seller_type=seller_type, locality=locality,
    )
    comps      = result.get("comps", [])
    sim_scores = result.get("sim_scores", [])

    # Phase 2 — anchor-centric range
    rng = _adaptive_range_engine.build(
        prediction=prediction,
        comps=comps,
        sim_scores=sim_scores,
        mape=_mape,
        odometer=float(odometer or 0),
    )

    price_min    = rng["price_min"]
    price_max    = rng["price_max"]
    price_median = rng["price_median"]
    n_comps      = rng["n_comps"]
    avg_sim      = rng["avg_similarity"]
    conf_case    = rng["confidence_case"]

    range_width_pct = (price_max - price_min) / max(prediction, 1)

    # Phase 3 — confidence score
    conf = _confidence_engine.score(
        n_comps=n_comps,
        avg_similarity=avg_sim,
        mape=_mape,
        ensemble_variance=ensemble_variance,
        range_width_pct=range_width_pct,
    )

    source = "dataset" if n_comps > 0 else "mape_fallback"

    return {
        # ── Backward-compatible keys ─────────────────────────────────────────
        "price_min":                price_min,
        "price_max":                price_max,
        "price_median":             price_median,
        "market_range_comp_count":  n_comps,
        "market_range_stage":       result["stage"],
        "market_range_stage_label": result["stage_label"],
        "market_range_source":      source,
        "similar_cars":             result.get("similar_cars", []),
        # ── New enrichment fields ────────────────────────────────────────────
        "confidence":               conf["confidence"],
        "confidence_score":         conf["confidence_score"],
        "market_support":           conf["market_support"],
        "comparables_used":         n_comps,
        "average_similarity":       round(avg_sim * 100, 1),
        "ensemble_variance":        round(ensemble_variance, 6),
        "expected_model_error":     round(_mape * 100, 2),
        "confidence_case":          conf_case,
    }

# INLINE BRAND → SEGMENT MAP

_INLINE_BRAND_SEGMENT: dict[str, str] = {
    **{b: "economy"  for b in {
        "maruti", "maruti suzuki", "datsun", "bajaj", "chevrolet", "fiat",
        "opel", "premier", "force", "ashok leyland", "ambassador",
        "hyundai", "honda", "tata", "renault", "nissan", "ford",
        "mahindra", "mitsubishi", "isuzu", "citroen", "dc", "hindustan motors",
    }},
    **{b: "premium"  for b in {
        "volkswagen", "skoda", "toyota", "mg", "jeep", "kia",
        "mini", "volvo", "lexus",
    }},
    **{b: "luxury"   for b in {
        "bmw", "mercedes-benz", "audi", "jaguar", "land rover", "porsche",
        "maserati", "aston martin", "bentley", "rolls-royce",
        "ferrari", "lamborghini", "hummer",
    }},
}


# MAIN DECISION FUNCTION

def calculate_decision(vehicle, market_value: float) -> dict:
    """
    Convert ML market value -> complete dealer valuation package.

    Waterfall:
        Market Value (ML prediction)
          - Reconditioning Cost
          - Holding Cost
          - Documentation Cost
          - Risk Buffer
          - Target Dealer Profit
          = Recommended Buy Price
    """
    def _g(attr, default):
        return getattr(vehicle, attr, default) or default

    target_margin_pct  = float(_g("target_margin_pct", 10))
    repair_buffer      = float(_g("repair_buffer", 0))
    seller_asking      = float(_g("seller_asking_price", 0))
    age                = max(0, datetime.now().year - int(_g("year", datetime.now().year - 3)))
    km                 = max(0, float(_g("odometer_reading", 0)))
    owner_count        = max(1, int(_g("owner_count", 1)))
    condition          = str(_g("condition", "Good")).strip().lower()
    fuel               = str(_g("fuel_type", "Petrol")).strip().lower()
    transmission       = str(_g("transmission", "Manual")).strip().lower()
    city               = str(_g("city", "")).strip().lower()
    locality           = str(_g("locality", "")).strip().lower()
    rto                = str(_g("rto", "")).strip()
    inspected          = bool(_g("inspected", False))
    brand              = str(_g("brand", ""))
    model_name         = str(_g("model", ""))
    fuel_eff           = float(_g("fuel_efficiency", 0))
    variant            = str(_g("variant", ""))
    seller_reason      = str(_g("seller_reason", "upgrading"))
    reg_state          = str(_g("registration_state", ""))
    sale_state         = str(_g("sale_state", "") or city)
    loan_out           = bool(_g("loan_outstanding", False))
    accident_hist      = str(_g("accident_history", "none")).lower().strip()
    color              = str(_g("color", ""))

    variant_known       = variant.lower() not in {"", "unknown", "base"}
    color_known         = color.lower() not in {"", "unknown"}
    owner_known         = owner_count > 0
    service_hist_known  = inspected
    accident_hist_known = accident_hist not in {"unknown", ""}
    reg_state_known     = bool(reg_state)

    segment        = _INLINE_BRAND_SEGMENT.get(brand.lower().strip(), "economy")
    sanity_clamped = False
    sanity_note    = "ML-first: no sanity clamp applied"

    risk_score, risk_level = compute_risk_score(
        age, km, owner_count, condition, fuel, inspected, sanity_clamped,
        variant_known=variant_known, color_known=color_known,
        accident_history=accident_hist,
    )

    confidence_score, model_conf, business_conf = compute_confidence_score(
        age, km, owner_count, condition, fuel, variant, fuel_eff,
        risk_score, sanity_clamped, city, inspected,
        owner_known=owner_known, accident_hist=accident_hist,
        locality=locality,
    )

    eff_margin_pct = dynamic_target_margin(
        segment, age, km, owner_count, condition, inspected, fuel, target_margin_pct
    )

    # Reconditioning
    if repair_buffer > 5_000:
        recon_cost = int(repair_buffer)
        recon_note = "Dealer-entered repair estimate"
    else:
        recon_cost = compute_dynamic_recon_cost(segment, age, km, condition, inspected, brand)
        recon_note = (
            f"Dynamic: {brand or 'unknown'} brand mult "
            f"x{_BRAND_REPAIR_MULTIPLIER.get(brand.lower().strip(), 1.0):.2f}, "
            f"{condition} condition, {age}yr, {_annual_km(km,age)/1000:.0f}k km/yr"
        )

    holding_cost, eff_days = compute_holding_cost(segment, market_value, brand)
    holding_note = (
        f"{segment.title()}: {_HOLDING.get(segment, {}).get('rate_pct', 1.8):.1f}%/mo "
        f"x {eff_days}d inventory (brand popularity "
        f"x{_BRAND_POPULARITY.get(brand.lower().strip(), 1.0):.2f})"
    )

    doc_cost, doc_breakdown = compute_doc_cost(reg_state, sale_state, loan_out)

    risk_buffer = compute_risk_buffer(
        market_value, risk_score, segment, age, km, owner_count, condition, inspected,
        variant_known=variant_known, owner_known=owner_known,
        service_hist_known=service_hist_known,
        accident_hist_known=accident_hist_known,
        reg_state_known=reg_state_known, color_known=color_known,
    )

    veh_category     = classify_vehicle_category(brand, model_name)
    p_min, p_max     = _PROFIT_LIMITS.get(veh_category, (25_000, 100_000))
    raw_profit       = market_value * (eff_margin_pct / 100)
    target_profit    = int(_clamp(raw_profit, p_min, p_max))

    total_deductions      = recon_cost + holding_cost + doc_cost + risk_buffer + target_profit
    recommended_buy_price = market_value - total_deductions
    recommended_buy_price = min(recommended_buy_price, market_value * 0.95)
    recommended_buy_price = _round500(recommended_buy_price)

    # Sell price — uses locality/RTO demand instead of old city_demand
    loc_key    = locality.strip().lower()
    rto_key    = rto.strip().upper()
    geo_prem   = _LOCALITY_DEMAND.get(loc_key, _RTO_DEMAND.get(rto_key, 0.015))
    geo_prem   = float(geo_prem)
    recon_uplift_pct       = min((recon_cost / max(market_value, 1)) * 0.60, 0.08)
    recommended_sell_price = _round500(market_value * (1 + recon_uplift_pct + geo_prem * 0.5))
    min_sell = recommended_buy_price + recon_cost + holding_cost + doc_cost + target_profit
    recommended_sell_price = max(recommended_sell_price, _round500(min_sell))
    expected_profit        = int(recommended_sell_price - recommended_buy_price
                                 - recon_cost - holding_cost - doc_cost)
    expected_profit        = max(expected_profit, target_profit)
    expected_margin_pct    = (expected_profit / max(recommended_buy_price, 1)) * 100

    inv_duration_label = (
        "Fast"   if eff_days <= 20 else
        "Normal" if eff_days <= 45 else
        "Slow"   if eff_days <= 70 else
        "Very Slow"
    )

    roi = (expected_profit / max(recommended_buy_price, 1)) * 100
    flexible_reasons = {"financial", "relocating", "problem"}

    if confidence_score < 52 or sanity_clamped:
        action = "MANUAL REVIEW"
    elif roi >= 4.5 and risk_score <= 30 and confidence_score >= 72 and eff_days <= 45:
        action = "BUY"
    elif roi >= 3.5 and risk_score <= 45 and not inspected:
        action = "BUY AFTER INSPECTION"
    elif roi >= 3.5 and risk_score <= 40 and confidence_score >= 65:
        action = "BUY"
    elif roi >= 2.5 and risk_score <= 55:
        action = "NEGOTIATE"
    elif roi >= 1.5 and risk_score <= 70 and seller_reason.lower().strip() in flexible_reasons:
        action = "NEGOTIATE AGGRESSIVELY"
    elif roi >= 1.5 and risk_score <= 65:
        action = "NEGOTIATE"
    else:
        action = "REJECT"

    # BUG-05 fix: pass market sell price so walk_away is capped below the sell ceiling
    nego = compute_negotiation_trio(
        recommended_buy_price, city, condition, risk_score,
        seller_reason, seller_asking, locality=locality, rto=rto,
        market_sell_price=float(recommended_sell_price),
    )

    demand_score          = round(_clamp(88 - age * 2.5 - (km / 200_000) * 35))
    brand_retention_score = round(_clamp(80 - age * 1.2 + (5 if fuel in {"petrol", "hybrid"} else 0)))
    vehicle_health_score  = round(_clamp(100 - age * 3 - km / 10_000 - (owner_count - 1) * 8))
    resale_liquidity_score = round(_clamp(
        (demand_score + brand_retention_score + vehicle_health_score) / 3
    ))
    deal_quality_score = round(_clamp(
        0.35 * _clamp(roi * 5) + 0.30 * confidence_score + 0.35 * (100 - risk_score)
    ))
    urgency_score = round(_clamp(
        65 + (deal_quality_score - 65) * 0.5 + (100 - risk_score) * 0.15
    ))
    urgency_label = "High" if urgency_score >= 75 else "Medium" if urgency_score >= 55 else "Low"

    positive_factors, negative_factors = [], []

    if age <= 3:
        positive_factors.append(f"Recent vehicle ({age} yr{'s' if age!=1 else ''}) — strong resale demand")
    elif age >= 8:
        negative_factors.append(f"Vehicle age ({age} yrs) — significantly elevated depreciation risk")

    ann_km = _annual_km(km, age)
    if ann_km < _ANNUAL_KM_TIERS["low"]:
        positive_factors.append(f"Low annual usage ({ann_km/1000:.0f}k km/yr) — lightly driven")
    elif ann_km > _ANNUAL_KM_TIERS["very_high"]:
        negative_factors.append(f"Very high annual mileage ({ann_km/1000:.0f}k km/yr) — heavy wear expected")

    if owner_count == 1:
        positive_factors.append("First-owner vehicle — strongest buyer preference in Indian market")
    elif owner_count >= 3:
        negative_factors.append(f"{owner_count} previous owners — extensive due-diligence required")

    if condition in {"excellent", "good"}:
        positive_factors.append(f"{condition.title()} condition — ready for immediate resale")
    else:
        negative_factors.append(f"{condition.title()} condition — reconditioning investment required")

    if inspected:
        positive_factors.append("Inspection certificate present — reduces buyer uncertainty premium")
    else:
        negative_factors.append("No inspection report — consider pre-purchase inspection before acquisition")

    if eff_days <= 25:
        positive_factors.append(f"Fast-moving inventory ({eff_days}d expected) — quick capital recovery")
    elif eff_days >= 65:
        negative_factors.append(f"Slow-moving inventory ({eff_days}d expected) — capital tied up longer")

    if not variant_known:
        negative_factors.append("Variant/trim unknown — may affect resale pricing accuracy")
    if not accident_hist_known:
        negative_factors.append("Accident history unknown — hidden damage risk factored into buffer")
    if sanity_clamped:
        negative_factors.append(f"ML prediction adjusted by market sanity band — {sanity_note}")

    r2_str = f"R2={_CONF_BASELINE.get('r2', 0.97):.2f}"
    positive_factors.append(f"Market value predicted by CatBoost+LightGBM+XGBoost ensemble ({r2_str})")

    mape_frac    = float(_CONF_BASELINE.get("mape_frac", 0.065))
    price_spread = int(round(market_value * mape_frac))
    seller_gap   = _round500(seller_asking - recommended_buy_price) if seller_asking > 0 else 0

    waterfall = [
        {"label": "ML Market Value",           "value": int(market_value),       "sign": "",  "note": "CatBoost ensemble prediction"},
        {"label": f"Reconditioning ({brand or 'standard'})", "value": int(recon_cost),  "sign": "-", "note": recon_note},
        {"label": f"Holding ({eff_days}d)",    "value": int(holding_cost),       "sign": "-", "note": holding_note},
        {"label": "RC + Documentation",        "value": int(doc_cost),           "sign": "-", "note": "RC transfer + NOC + insurance" + (" + hypo" if loan_out else "") + (" + state" if doc_breakdown.get("state_transfer") else "")},
        {"label": "Risk Buffer",               "value": int(risk_buffer),        "sign": "-", "note": f"Risk {risk_score}/100 + unknown-field penalties"},
        {"label": f"Target Profit ({eff_margin_pct:.1f}%)", "value": int(target_profit), "sign": "-", "note": f"Dynamic margin, capped [{p_min//1000}k-{p_max//1000}k]"},
        {"label": "Recommended Buy Price",     "value": int(recommended_buy_price), "sign": "=", "note": "Maximum acquisition price"},
    ]

    return {
        "market_value":             int(market_value),
        "ci":                       int(price_spread),
        "recommended_buy_price":    int(recommended_buy_price),
        "recommended_sell_price":   int(recommended_sell_price),
        "expected_profit":          int(expected_profit),
        "expected_margin_pct":      round(expected_margin_pct, 1),
        "dealer_acq_price":         int(recommended_buy_price),
        "suggested_sell_price":     int(recommended_sell_price),
        "margin_pct":               round(expected_margin_pct, 1),
        "margin_amt":               int(expected_profit),
        "recon_cost":               int(recon_cost),
        "holding_cost":             int(holding_cost),
        "doc_cost":                 int(doc_cost),
        "doc_breakdown":            doc_breakdown,
        "risk_buffer":              int(risk_buffer),
        "target_profit":            int(target_profit),
        "repair_buffer":            int(recon_cost),
        "waterfall":                waterfall,
        "opening_offer":            nego["opening_offer"],
        "max_offer":                nego["walk_away_price"],
        "target_offer":             nego["target_offer"],
        "negotiation_room":         nego["negotiation_room"],
        "potential_savings":        nego["potential_savings"],
        "seller_gap":               int(seller_gap),
        "risk_score":               int(risk_score),
        "risk_level":               risk_level,
        "confidence_score":         int(confidence_score),
        "model_confidence":         int(model_conf),
        "business_confidence":      int(business_conf),
        "demand_score":             int(demand_score),
        "brand_retention_score":    int(brand_retention_score),
        "vehicle_health_score":     int(vehicle_health_score),
        "resale_liquidity_score":   int(resale_liquidity_score),
        "deal_quality_score":       int(deal_quality_score),
        "urgency_score":            int(urgency_score),
        "urgency_label":            urgency_label,
        "action":                   action,
        "effective_margin_pct":     float(eff_margin_pct),
        "target_margin_pct":        float(target_margin_pct),
        "inventory_days":           int(eff_days),
        "inventory_label":          inv_duration_label,
        "positive_factors":         positive_factors[:5],
        "negative_factors":         negative_factors[:5],
        "sanity_clamped":           sanity_clamped,
        "sanity_note":              sanity_note,
        "vehicle_category":         veh_category,       # used by evaluate_vehicle profit cap
        "quote_message": (
            f"Based on ML ensemble valuation ({r2_str}), {brand or 'vehicle'} condition, "
            f"and {'(' + locality.title() + ')' if locality and locality != 'unknown' else 'Bangalore'} "
            f"market demand, recommended acquisition is "
            f"Rs.{recommended_buy_price/100_000:.2f}L. "
            f"Seller gap: Rs.{abs(seller_gap)//1000:.0f}k "
            f"{'above' if seller_gap > 0 else 'below'} target."
            if seller_asking else
            f"Recommended acquisition: Rs.{recommended_buy_price/100_000:.2f}L -> "
            f"target sell Rs.{recommended_sell_price/100_000:.2f}L -> "
            f"expected dealer profit Rs.{target_profit//1000:.0f}k."
        ),
    }


# LEGACY FUNCTIONS — unchanged interface

def check_disqualifier(vehicle_age: int, odometer: int,
                        owner_count: int, accident_history: str) -> dict:
    acc = (accident_history or "none").lower().strip()
    if vehicle_age > 12:
        return {"disqualified": True, "reason": "Vehicle age exceeds 12 years"}
    if odometer > 150_000:
        return {"disqualified": True, "reason": "Odometer reading exceeds 150,000 km"}
    if owner_count >= 4 and acc in {"minor", "major"}:
        return {"disqualified": True,
                "reason": f"Multiple owners ({owner_count}) + accident history detected"}
    return {"disqualified": False, "reason": "Passes pre-screening"}


def get_seasonal_multiplier(month: int) -> float:
    return {
        1: 0.97, 2: 0.97, 3: 1.04, 4: 0.98, 5: 0.98,
        6: 1.06, 7: 1.06, 8: 0.99, 9: 0.99,
        10: 1.05, 11: 1.05, 12: 0.96,
    }.get(month, 1.0)


def get_wheelr_risk_deductions(owner_count: int, odometer: int,
                                accident_history: str = "none",
                                registration_state: str = "",
                                sale_state: str = "",
                                loan_outstanding: bool = False,
                                seller_reason: str = "upgrading") -> dict:
    acc        = (accident_history or "none").lower().strip()
    sr         = (seller_reason or "upgrading").lower().strip()
    owner_ded  = {1: 0, 2: 8_000, 3: 18_000}.get(owner_count, 30_000)
    km_ded     = (0 if odometer < 40_000 else 5_000 if odometer < 80_000
                  else 12_000 if odometer < 120_000 else 25_000)
    acc_ded    = {"none": 0, "minor": 10_000, "major": 35_000}.get(acc, 0)
    state_ded  = (0 if not registration_state or not sale_state
                  or registration_state.lower() == sale_state.lower() else 8_000)
    loan_ded   = 5_000 if loan_outstanding else 0
    sr_adj     = {"upgrading": 0, "relocating": -5_000, "financial": -12_000,
                  "unused": 5_000, "problem": -8_000}.get(sr, 0)
    total = owner_ded + km_ded + acc_ded + state_ded + loan_ded
    return {
        "total": int(total),
        "breakdown": {
            "owner_deduction": int(owner_ded), "km_deduction": int(km_ded),
            "accident_deduction": int(acc_ded), "state_deduction": int(state_ded),
            "loan_deduction": int(loan_ded),
        },
        "seller_reason_adj": int(sr_adj),
    }


def get_recon_cost(engine_grade: str = "good", tyre_grade: str = "good",
                   body_grade: str = "clean", interior_grade: str = "clean",
                   electrical_grade: str = "all_good",
                   vendor_type: dict = None, rc_transfer_cost: int = 3_500) -> dict:
    if vendor_type is None:
        vendor_type = {k: "vendor" for k in
                       ["engine", "tyre", "body", "interior", "electrical"]}
    eg  = (engine_grade or "good").lower().strip()
    tg  = (tyre_grade or "good").lower().strip()
    bg  = (body_grade or "clean").lower().strip()
    ig  = (interior_grade or "clean").lower().strip()
    elg = (electrical_grade or "all_good").lower().strip()

    ec  = {"good": 0, "average": {"inhouse":4_000,"vendor":8_000},
           "poor": {"inhouse":18_000,"vendor":35_000},
           "critical": {"inhouse":45_000,"vendor":80_000}}.get(eg, 0)
    ec  = ec if isinstance(ec, int) else ec.get(vendor_type.get("engine","vendor"), 0)
    tc  = {"good": 0, "two_bad": {"inhouse":4_000,"vendor":6_000},
           "all_bad": {"inhouse":8_000,"vendor":12_000}}.get(tg, 0)
    tc  = tc if isinstance(tc, int) else tc.get(vendor_type.get("tyre","vendor"), 0)
    bc  = {"clean": 0, "minor": {"inhouse":3_000,"vendor":5_000},
           "major": {"inhouse":10_000,"vendor":18_000},
           "accident": {"inhouse":22_000,"vendor":40_000}}.get(bg, 0)
    bc  = bc if isinstance(bc, int) else bc.get(vendor_type.get("body","vendor"), 0)
    ic  = {"clean": 0, "needs_cleaning": {"inhouse":1_500,"vendor":3_000},
           "full_refurb": {"inhouse":6_000,"vendor":10_000}}.get(ig, 0)
    ic  = ic if isinstance(ic, int) else ic.get(vendor_type.get("interior","vendor"), 0)
    elc = {"all_good": 0, "ac_fault": {"inhouse":4_500,"vendor":8_000},
           "multi_fault": {"inhouse":8_000,"vendor":15_000}}.get(elg, 0)
    elc = elc if isinstance(elc, int) else elc.get(vendor_type.get("electrical","vendor"), 0)

    rc_transfer_cost = max(0, int(rc_transfer_cost or 3_500))
    fixed  = rc_transfer_cost + 2_500 + 2_000
    total  = ec + tc + bc + ic + elc + fixed
    return {
        "engine_cost": int(ec), "tyre_cost": int(tc), "body_cost": int(bc),
        "interior_cost": int(ic), "electrical_cost": int(elc),
        "fixed_cost": int(fixed), "rc_transfer_cost": int(rc_transfer_cost),
        "total": int(total),
        "breakdown": {"engine": int(ec), "tyres": int(tc), "body_paint": int(bc),
                      "interior": int(ic), "electricals": int(elc), "fixed": int(fixed)},
    }


def get_negotiation_trio(max_buy_price: int, seller_reason_adj: int = 0) -> dict:
    return {
        "opening_offer":   int(max(0, max_buy_price - 15_000 + seller_reason_adj)),
        "target_offer":    int(max_buy_price - 8_000),
        "walk_away_price": int(max_buy_price),
    }


def get_deal_health(ml_market_value: int, recon_total: int, profit_target: int,
                    owner_count: int, odometer: int, accident_history: str = "none") -> str:
    if ml_market_value <= 0:
        return "red"
    margin_pct = profit_target / ml_market_value
    recon_pct  = recon_total / ml_market_value
    acc        = (accident_history or "none").lower().strip()
    if margin_pct >= 0.12 and recon_pct <= 0.20 and owner_count <= 2 and acc == "none":
        return "green"
    if margin_pct < 0.08 or recon_pct > 0.35:
        return "red"
    return "yellow"
