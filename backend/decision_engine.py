"""
PriceRef — Dealer Decision Engine  v8.0
==========================================
Business logic layer on top of the ML prediction.

Key principles
--------------
1. Market Sanity Clamp    — segment-aware bands with per-segment tolerance
2. Dynamic Reconditioning — age + km + condition + inspection + segment
3. Dynamic Holding Cost   — segment-specific rates × inventory days
4. Dynamic Documentation  — RC + hypothecation + NOC + insurance + state transfer
5. Dynamic Margin         — risk-adjusted, segment-capped, never flat 15%
6. Rupee-Based Risk Buffer — additive factors, not a percentage guess
7. Refined Confidence     — 88 baseline, granular deductions, inspection bonus
8. Six-Action Decision    — BUY / BUY AFTER INSPECTION / NEGOTIATE /
                             NEGOTIATE AGGRESSIVELY / MANUAL REVIEW / REJECT
9. Demand-Driven Negos    — percentage-based opening/ideal/walkaway
10. Market-Band Comps     — median/low/high/avg from real price bands
11. Monetary SHAP         — "Vehicle age reduces value by ₹58,000"
"""

from __future__ import annotations
import math
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _round500(v: float) -> int:
    return int(round(v / 500) * 500)


# ──────────────────────────────────────────────────────────────────────────────
# MARKET REFERENCE BANDS
# Indian used-car transaction prices (NOT listing prices).
# Format: (lower_2021_price, upper_2021_price)  — age-adjusted at runtime.
# ──────────────────────────────────────────────────────────────────────────────
_MARKET_BANDS: dict[str, tuple[float, float]] = {
    # Economy Hatchbacks
    "alto":          (250_000,   500_000),
    "alto k10":      (280_000,   540_000),
    "s-presso":      (320_000,   560_000),
    "kwid":          (280_000,   490_000),
    "celerio":       (330_000,   590_000),
    "wagonr":        (360_000,   680_000),
    "wagon r":       (360_000,   680_000),
    "eon":           (200_000,   380_000),
    "santro":        (350_000,   600_000),
    "tiago":         (420_000,   720_000),
    "magnite":       (560_000,   870_000),
    "punch":         (600_000,   960_000),
    # Premium Hatchbacks
    "swift":         (550_000,   900_000),
    "baleno":        (600_000,   950_000),
    "i20":           (680_000, 1_100_000),
    "i10":           (380_000,   680_000),
    "grand i10":     (400_000,   700_000),
    "altroz":        (650_000, 1_000_000),
    "glanza":        (600_000,   920_000),
    "ignis":         (540_000,   820_000),
    "polo":          (600_000,   960_000),
    # Compact SUVs
    "nexon":         (850_000, 1_600_000),
    "brezza":        (820_000, 1_550_000),
    "vitara brezza": (700_000, 1_300_000),
    "venue":         (780_000, 1_450_000),
    "sonet":         (800_000, 1_480_000),
    "ecosport":      (700_000, 1_200_000),
    "duster":        (650_000, 1_100_000),
    "kiger":         (680_000, 1_050_000),
    # Mid SUVs / Sedans
    "creta":         (1_100_000, 2_200_000),
    "seltos":        (1_200_000, 2_300_000),
    "grand vitara":  (1_350_000, 2_450_000),
    "hector":        (1_200_000, 2_100_000),
    "compass":       (1_400_000, 2_500_000),
    "harrier":       (1_300_000, 2_400_000),
    "safari":        (1_500_000, 2_800_000),
    "ertiga":        (850_000,  1_500_000),
    "carens":        (1_000_000, 1_900_000),
    "city":          (850_000,  1_700_000),
    "ciaz":          (700_000,  1_300_000),
    "vento":         (750_000,  1_250_000),
    "rapid":         (700_000,  1_200_000),
    "innova crysta": (1_500_000, 2_800_000),
    "innova":        (1_400_000, 2_600_000),
    "scorpio":       (1_000_000, 2_000_000),
    "scorpio n":     (1_400_000, 2_600_000),
    "thar":          (1_500_000, 2_800_000),
    "xuv700":        (1_800_000, 3_500_000),
    "xuv300":        (800_000,  1_400_000),
    # Premium / Luxury
    "octavia":       (1_800_000, 3_200_000),
    "superb":        (2_500_000, 4_500_000),
    "fortuner":      (2_800_000, 5_500_000),
    "endeavour":     (2_500_000, 4_800_000),
    "tucson":        (2_000_000, 3_500_000),
    "defender":      (6_000_000, 15_000_000),
    "range rover":   (5_000_000, 18_000_000),
    "glc":           (4_500_000,  8_000_000),
    "c class":       (3_500_000,  7_000_000),
    "3 series":      (3_200_000,  7_500_000),
    "5 series":      (5_000_000, 10_000_000),
    "x1":            (2_800_000,  5_500_000),
    "x3":            (4_000_000,  7_500_000),
    "q3":            (3_000_000,  6_000_000),
    "q5":            (4_500_000,  8_500_000),
    "a4":            (3_000_000,  6_500_000),
    "a6":            (5_000_000,  9_500_000),
}

# ── Segment-level fallback bands ────────────────────────────────────────────
_SEGMENT_BANDS: dict[str, tuple[float, float]] = {
    "economy": (200_000,  1_500_000),
    "premium": (600_000,  3_500_000),
    "luxury":  (2_500_000, 20_000_000),
}

# ── Per-segment sanity clamp tolerances (lower_ratio, upper_ratio) ────────────
_CLAMP_TOLERANCE: dict[str, tuple[float, float]] = {
    "economy": (0.88, 1.12),
    "premium": (0.85, 1.15),
    "luxury":  (0.80, 1.20),
}

# ── Age depreciation schedule (fraction of 2021 price retained per year) ────
_AGE_DEPRECIATION: dict[int, float] = {
    0: 1.00, 1: 0.86, 2: 0.78, 3: 0.71, 4: 0.65,
    5: 0.59, 6: 0.54, 7: 0.50, 8: 0.46, 9: 0.42,
    10: 0.38, 11: 0.35, 12: 0.32,
}

# ── Segment profit limits (min, max) ────────────────────────────────────────
_PROFIT_LIMITS: dict[str, tuple[int, int]] = {
    "economy":       (25_000,   60_000),
    "premium_hatch": (40_000,   80_000),
    "compact_suv":   (60_000,  100_000),
    "mid_suv":       (80_000,  150_000),
    "luxury":        (150_000, 300_000),
}

# ── City demand premium (fraction of market value) ───────────────────────────
_CITY_DEMAND: dict[str, float] = {
    "mumbai": 0.045, "pune": 0.030, "delhi": 0.040, "ncr": 0.038,
    "bangalore": 0.042, "bengaluru": 0.042, "hyderabad": 0.035,
    "chennai": 0.032, "kolkata": 0.025, "ahmedabad": 0.028,
    "surat": 0.020, "jaipur": 0.018, "lucknow": 0.015,
    "chandigarh": 0.022, "kochi": 0.020, "bhubaneswar": 0.012,
}

# ── Holding cost parameters by segment ──────────────────────────────────────
_HOLDING: dict[str, dict] = {
    "economy": {"rate_pct": 1.5, "days": 25},
    "premium": {"rate_pct": 2.0, "days": 40},
    "luxury":  {"rate_pct": 2.8, "days": 65},
}

# ── Base reconditioning cost by segment ──────────────────────────────────────
_RECON_BASE: dict[str, int] = {
    "economy": 12_000,
    "premium": 20_000,
    "luxury":  40_000,
}

# ── Documentation cost components ────────────────────────────────────────────
_DOC_RC_TRANSFER    = 3_500
_DOC_NOC            =   500
_DOC_INSURANCE      = 1_200
_DOC_HYPOTHECATION  = 2_000   # charged only when loan_outstanding
_DOC_STATE_TRANSFER = 8_000   # charged only for out-of-state


# ──────────────────────────────────────────────────────────────────────────────
# VEHICLE CATEGORY CLASSIFIER (for profit limit lookup)
# ──────────────────────────────────────────────────────────────────────────────
def classify_vehicle_category(brand: str, model: str) -> str:
    b = brand.lower().strip()
    m = model.lower().strip()

    if b in {"bmw", "mercedes-benz", "mercedes", "audi", "lexus", "volvo",
             "land rover", "jaguar", "porsche"}:
        return "luxury"
    if any(k in m for k in ("fortuner", "endeavour", "glc", "c class", "3 series",
                             "5 series", "x1", "x3", "q3", "q5", "a4", "a6",
                             "xuv700", "safari", "defender", "range rover")):
        return "luxury"
    if any(k in m for k in ("creta", "seltos", "grand vitara", "hector", "compass",
                             "harrier", "scorpio", "ertiga", "carens", "city",
                             "ciaz", "innova", "thar", "xuv300")):
        return "mid_suv"
    if any(k in m for k in ("venue", "nexon", "brezza", "sonet", "ecosport",
                             "duster", "kiger", "magnite", "punch")):
        return "compact_suv"
    if any(k in m for k in ("swift", "baleno", "i20", "altroz", "glanza",
                             "ignis", "polo", "i10", "grand i10")):
        return "premium_hatch"
    return "economy"


# ──────────────────────────────────────────────────────────────────────────────
# 1. MARKET SANITY CLAMP  (segment-aware tolerances)
# ──────────────────────────────────────────────────────────────────────────────
def _normalise_model(model_name: str) -> str:
    return " ".join(model_name.lower().split())


def apply_market_sanity_clamp(
    model_name: str,
    segment: str,
    vehicle_age: int,
    raw_value: float,
    city: str = "",
) -> tuple[float, bool, str]:
    """
    Clamp the ML raw value to a realistic age-adjusted, segment-aware market band.
    Returns (clamped_value, was_clamped, note).
    """
    model_key = _normalise_model(model_name)
    band = None
    for key in [model_key] + [" ".join(model_key.split()[:i])
                               for i in range(len(model_key.split()), 0, -1)]:
        if key in _MARKET_BANDS:
            band = _MARKET_BANDS[key]
            break
    if band is None:
        band = _SEGMENT_BANDS.get(segment, (100_000, 20_000_000))

    age_factor = _AGE_DEPRECIATION.get(min(vehicle_age, 12), 0.30)

    # City-demand adjustment to the upper band (high-demand cities allow higher prices)
    city_adj = _CITY_DEMAND.get(city.lower().strip(), 0.0)
    upper_adj = 1.0 + (city_adj * 0.5)   # max ~+2.25% on upper

    lower = band[0] * age_factor
    upper = band[1] * age_factor * upper_adj

    lo_ratio, hi_ratio = _CLAMP_TOLERANCE.get(segment, (0.88, 1.12))

    clamped = False
    note = "within expected market band"

    if raw_value > upper * hi_ratio:
        raw_value = upper
        clamped   = True
        note = "clamped from above — ML overestimated vs market band"
    elif raw_value < lower * lo_ratio:
        raw_value = lower * 0.92
        clamped   = True
        note = "clamped from below — ML underestimated vs market band"

    return float(raw_value), clamped, note


# ──────────────────────────────────────────────────────────────────────────────
# 2. DYNAMIC RECONDITIONING COST
# ──────────────────────────────────────────────────────────────────────────────
def compute_dynamic_recon_cost(
    segment: str,
    age: int,
    km: float,
    condition: str,
    inspected: bool,
) -> int:
    """
    Dynamic reconditioning estimate based on segment, age, mileage,
    condition, and inspection status.
    """
    base = _RECON_BASE.get(segment, 18_000)

    # Age additions
    age_add = 0
    if 1 <= age <= 3:   age_add = 2_000
    elif 4 <= age <= 6: age_add = 5_000
    elif 7 <= age <= 9: age_add = 10_000
    elif age >= 10:     age_add = 18_000

    # Mileage additions
    km_add = 0
    if 30_000 <= km < 70_000:  km_add = 3_000
    elif 70_000 <= km < 120_000: km_add = 8_000
    elif km >= 120_000:          km_add = 18_000

    subtotal = base + age_add + km_add

    # Condition multiplier
    cond_mult = {
        "excellent": 0.70,
        "good":      1.00,
        "average":   1.40,
        "poor":      2.10,
    }.get(condition.lower().strip(), 1.00)

    subtotal = int(subtotal * cond_mult)

    # Inspection discount
    if inspected:
        subtotal = int(subtotal * 0.85)

    # Segment caps
    caps = {"economy": 60_000, "premium": 120_000, "luxury": 250_000}
    return min(subtotal, caps.get(segment, 80_000))


# ──────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC HOLDING COST
# ──────────────────────────────────────────────────────────────────────────────
def compute_holding_cost(segment: str, market_value: float) -> int:
    """
    Holding cost = market_value × segment_rate × (inventory_days / 30).
    Luxury cars sit longer in inventory than economy cars.
    """
    h = _HOLDING.get(segment, {"rate_pct": 1.8, "days": 30})
    rate   = h["rate_pct"] / 100.0
    days   = h["days"]
    return int(market_value * rate * (days / 30))


# ──────────────────────────────────────────────────────────────────────────────
# 4. DYNAMIC DOCUMENTATION COST
# ──────────────────────────────────────────────────────────────────────────────
def compute_doc_cost(
    registration_state: str = "",
    sale_state: str = "",
    loan_outstanding: bool = False,
) -> tuple[int, dict]:
    """
    Calculate documentation cost from actual components.
    Returns (total, breakdown_dict).
    """
    rc         = _DOC_RC_TRANSFER
    noc        = _DOC_NOC
    insurance  = _DOC_INSURANCE
    hypo       = _DOC_HYPOTHECATION if loan_outstanding else 0
    state      = 0
    if registration_state and sale_state:
        if registration_state.strip().lower() != sale_state.strip().lower():
            state = _DOC_STATE_TRANSFER

    total = rc + noc + insurance + hypo + state
    breakdown = {
        "rc_transfer":     rc,
        "noc":             noc,
        "insurance_trans": insurance,
        "hypothecation":   hypo,
        "state_transfer":  state,
    }
    return int(total), breakdown


# ──────────────────────────────────────────────────────────────────────────────
# 5. DYNAMIC MARGIN CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────
def dynamic_target_margin(
    segment: str,
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    inspected: bool,
    fuel: str,
    user_target_pct: float = 15.0,
) -> float:
    """
    Compute realistic dealer target margin (%) based on risk factors.
    """
    segment_base = {"economy": 11.0, "premium": 14.0, "luxury": 17.0}
    base = segment_base.get(segment, 11.0)

    # Positive adjustments
    if vehicle_age <= 2:   base += 2.0
    elif vehicle_age <= 4: base += 1.0
    if km < 30_000:        base += 1.0
    if owner_count == 1:   base += 1.0
    if inspected:          base += 1.5
    if condition.lower() == "excellent": base += 1.0
    if fuel.lower() in {"petrol", "hybrid"}: base += 0.5

    # Negative adjustments
    if vehicle_age > 7:    base -= 2.0
    elif vehicle_age > 5:  base -= 1.0
    if km > 80_000:        base -= 1.5
    elif km > 50_000:      base -= 0.8
    if owner_count >= 3:   base -= 1.5
    elif owner_count == 2: base -= 0.8
    if condition.lower() == "poor":    base -= 2.0
    elif condition.lower() == "average": base -= 1.0

    # Segment ranges
    segment_ranges = {
        "economy": (8.0,  16.0),
        "premium": (10.0, 19.0),
        "luxury":  (13.0, 23.0),
    }
    lo, hi = segment_ranges.get(segment, (8.0, 18.0))
    computed = _clamp(base, lo, hi)

    # 55% computed (risk-adjusted) + 45% user input (preference)
    blended = 0.55 * computed + 0.45 * float(user_target_pct)
    return round(_clamp(blended, lo, hi), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 6. RUPEE-BASED DYNAMIC RISK BUFFER
# ──────────────────────────────────────────────────────────────────────────────
def compute_risk_buffer(
    market_value: float,
    risk_score: int,
    segment: str,
    age: int,
    km: float,
    owner_count: int,
    condition: str,
    inspected: bool,
) -> int:
    """
    Rupee-based risk buffer = base (risk-scaled) + additive factors.
    """
    # Base: market value × risk score × segment scaling factor
    seg_factor = {"economy": 0.8, "premium": 1.0, "luxury": 1.4}.get(segment, 1.0)
    base_buffer = market_value * risk_score * 0.0004 * seg_factor

    # Additive components (rupees)
    age_add = 0
    if 3 <= age <= 6:  age_add = 5_000
    elif 7 <= age <= 9: age_add = 12_000
    elif age >= 10:     age_add = 22_000

    km_add = 0
    if 50_000 <= km < 100_000: km_add = 5_000
    elif km >= 100_000:         km_add = 15_000

    owner_add = {1: 0, 2: 3_000, 3: 10_000}.get(min(owner_count, 3), 20_000)
    insp_add  = 0 if inspected else 5_000
    cond_add  = {"poor": 15_000, "average": 6_000}.get(condition.lower(), 0)

    total = int(base_buffer + age_add + km_add + owner_add + insp_add + cond_add)

    # Floor 5,000 — cap at 12% of market value
    return int(_clamp(total, 5_000, market_value * 0.12))


# ──────────────────────────────────────────────────────────────────────────────
# 7. FACTOR-BASED RISK SCORE
# ──────────────────────────────────────────────────────────────────────────────
def compute_risk_score(
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    fuel: str,
    inspected: bool,
    sanity_clamped: bool = False,
) -> tuple[int, str]:
    age_risk   = _clamp(vehicle_age * 8.0)
    km_risk    = _clamp((km / 150_000) * 100)
    owner_risk = {1: 10, 2: 32, 3: 58, 4: 78}.get(min(owner_count, 4), 85)
    cond_risk  = {
        "excellent": 8, "good": 22, "average": 55, "poor": 85,
    }.get(condition.lower().strip(), 40)
    fuel_risk  = {
        "petrol": 15, "diesel": 24, "cng": 30, "electric": 28, "hybrid": 18,
    }.get(fuel.lower().strip(), 25)

    raw = (
        0.25 * age_risk
        + 0.25 * km_risk
        + 0.20 * owner_risk
        + 0.18 * cond_risk
        + 0.07 * fuel_risk
        + 0.05 * (0 if inspected else 25)
    )

    if sanity_clamped:
        raw += 12

    score = round(_clamp(raw, 5, 95))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return score, level


# ──────────────────────────────────────────────────────────────────────────────
# 8. REFINED CONFIDENCE SCORE  (starts at 88, granular deductions)
# ──────────────────────────────────────────────────────────────────────────────
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
) -> int:
    score = 88.0

    # Deductions
    if variant.lower() in {"", "unknown", "base"}:    score -= 8
    if not city or city.lower() in {"", "unknown"}:   score -= 5
    if fuel_efficiency <= 0:                           score -= 4
    if sanity_clamped:                                 score -= 15
    if km > 150_000:                                   score -= 12
    elif km > 100_000:                                 score -= 7
    elif km > 80_000:                                  score -= 3
    if km < 3_000 and vehicle_age > 2:                 score -= 8   # suspiciously low km
    if vehicle_age > 10:                               score -= 10
    elif vehicle_age > 7:                              score -= 5
    elif vehicle_age > 4:                              score -= 2
    if owner_count > 3:                                score -= (owner_count - 1) * 5
    elif owner_count == 3:                             score -= 8
    elif owner_count == 2:                             score -= 3
    score -= risk_score * 0.15

    # Bonuses
    if condition.lower() == "excellent":               score += 4
    if inspected:                                      score += 5
    if fuel.lower() in {"petrol", "hybrid"}:           score += 2
    if vehicle_age <= 2:                               score += 3

    return int(round(_clamp(score, 42, 95)))


# ──────────────────────────────────────────────────────────────────────────────
# 9. MONETARY SHAP EXPLANATION  (enhanced with brand_tier + transmission)
# ──────────────────────────────────────────────────────────────────────────────
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
) -> list[dict]:
    items: list[dict] = []

    # Age impact (2.8% per year)
    age_impact = -int(vehicle_age * market_value * 0.028)
    items.append({
        "feature": "Vehicle Age",
        "value": f"{vehicle_age} yr{'s' if vehicle_age != 1 else ''}",
        "contribution": age_impact,
        "label": (
            f"Vehicle age ({vehicle_age} yrs) reduces value by ₹{abs(age_impact)//1000:.0f},000"
            if age_impact < 0 else "Recent vehicle adds resale value"
        ),
    })

    # Mileage impact (₹1.10 per km above 25k baseline)
    km_above_base = max(0, km - 25_000)
    km_impact = -int(km_above_base * 1.1)
    if abs(km_impact) > 2_000:
        items.append({
            "feature": "Odometer Reading",
            "value": f"{km/1000:.0f}k km",
            "contribution": km_impact,
            "label": f"Mileage ({km/1000:.0f}k km) reduces value by ₹{abs(km_impact)//1000:.0f},000",
        })
    else:
        items.append({
            "feature": "Odometer Reading",
            "value": f"{km/1000:.0f}k km",
            "contribution": 3_000,
            "label": f"Low mileage ({km/1000:.0f}k km) adds ₹3,000 to value",
        })

    # Condition
    cond_impact = {
        "excellent": int(market_value * 0.035),
        "good":      0,
        "average":   -int(market_value * 0.060),
        "poor":      -int(market_value * 0.140),
    }.get(condition.lower().strip(), 0)
    if abs(cond_impact) > 1_000:
        sign = "adds" if cond_impact > 0 else "reduces value by"
        lbl  = f"{condition.title()} condition {sign} ₹{abs(cond_impact)//1000:.0f},000"
        items.append({"feature": "Condition", "value": condition.title(),
                      "contribution": cond_impact, "label": lbl})

    # Ownership
    owner_impact = {1: 8_000, 2: -6_000, 3: -18_000, 4: -30_000}.get(
        min(owner_count, 4), -30_000
    )
    owner_lbl = (
        "First owner increases buyer confidence: +₹8,000"
        if owner_count == 1
        else f"{owner_count} previous owners reduce value by ₹{abs(owner_impact)//1000:.0f},000"
    )
    items.append({"feature": "Ownership", "value": f"{owner_count} owner(s)",
                  "contribution": owner_impact, "label": owner_lbl})

    # City demand
    city_prem_pct = _CITY_DEMAND.get(city.lower().strip(), 0.010)
    city_impact   = int(market_value * city_prem_pct)
    if city_impact > 1_000:
        items.append({
            "feature": "City Demand",
            "value": city.title(),
            "contribution": city_impact,
            "label": f"{city.title()} market demand adds ₹{city_impact//1000:.0f},000 to value",
        })

    # Fuel type
    fuel_impact = {
        "petrol": 6_000, "hybrid": 10_000, "electric": 8_000,
        "diesel": -2_000, "cng": -4_000,
    }.get(fuel.lower(), 0)
    if fuel_impact != 0:
        sign = "adds" if fuel_impact > 0 else "reduces"
        items.append({
            "feature": "Fuel Type",
            "value": fuel.title(),
            "contribution": fuel_impact,
            "label": f"{fuel.title()} fuel type {sign} ₹{abs(fuel_impact)//1000:.0f},000 to demand",
        })

    # Transmission
    if transmission.lower() in {"automatic", "cvt", "dct", "amt"}:
        at_impact = int(market_value * 0.025)
        items.append({
            "feature": "Transmission",
            "value": transmission.title(),
            "contribution": at_impact,
            "label": f"Automatic transmission adds ₹{at_impact//1000:.0f},000 (buyer preference)",
        })

    # Inspection
    if inspected:
        items.append({
            "feature": "Inspection",
            "value": "Certified",
            "contribution": 5_000,
            "label": "Certified inspection adds ₹5,000 to buyer confidence value",
        })

    # Fuel efficiency
    if fuel_efficiency and fuel_efficiency > 0:
        fe_delta  = fuel_efficiency - 16.0
        fe_impact = int(fe_delta * market_value * 0.003)
        if abs(fe_impact) > 1_500:
            sign = "adds" if fe_impact > 0 else "reduces"
            items.append({
                "feature": "Fuel Efficiency",
                "value": f"{fuel_efficiency:.1f} km/l",
                "contribution": fe_impact,
                "label": f"Fuel efficiency ({fuel_efficiency:.1f} km/l) {sign} ₹{abs(fe_impact)//1000:.0f},000",
            })

    # Brand tier signal
    _BRAND_TIER_IMPACT = {
        "luxury": int(market_value * 0.03),
        "premium": int(market_value * 0.015),
        "mid": 0,
        "budget": -int(market_value * 0.01),
    }
    _BRAND_SEG_MAP = {
        **{b: "luxury"  for b in {"bmw", "mercedes-benz", "audi", "jaguar",
                                   "land rover", "porsche", "ferrari", "bentley"}},
        **{b: "premium" for b in {"volkswagen", "skoda", "toyota", "mg", "kia",
                                   "jeep", "mini", "volvo", "lexus"}},
        **{b: "budget"  for b in {"maruti", "maruti suzuki", "datsun", "chevrolet",
                                   "fiat", "bajaj"}},
    }
    brand_tier_key = _BRAND_SEG_MAP.get(brand.lower().strip(), "mid")
    brand_impact   = _BRAND_TIER_IMPACT.get(brand_tier_key, 0)
    if abs(brand_impact) > 2_000:
        sign = "adds" if brand_impact > 0 else "reduces"
        items.append({
            "feature": "Brand Tier",
            "value": brand.title(),
            "contribution": brand_impact,
            "label": f"{brand.title()} brand premium {sign} ₹{abs(brand_impact)//1000:.0f},000",
        })

    return sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:8]


# ──────────────────────────────────────────────────────────────────────────────
# 10. NEGOTIATION TRIO — DEMAND-DRIVEN, PERCENTAGE-BASED
# ──────────────────────────────────────────────────────────────────────────────
def compute_negotiation_trio(
    recommended_buy_price: float,
    city: str,
    condition: str,
    risk_score: int,
    seller_reason: str = "upgrading",
) -> dict:
    """
    Opening / Ideal / Walk-away  —  percentage-based, demand-driven.
    Walk-away is no longer a fixed add-on; it's a percentage of buy price.
    """
    demand = _CITY_DEMAND.get(city.lower().strip(), 0.015)

    # Higher demand → less negotiation room
    nego_pct  = max(0.04, 0.07 - demand * 0.8)
    nego_room = int(recommended_buy_price * nego_pct)

    # Seller reason adjustment
    seller_adj = {
        "financial":  -int(nego_room * 0.30),
        "relocating": -int(nego_room * 0.15),
        "upgrading":  0,
        "unused":      int(nego_room * 0.10),
        "problem":    -int(nego_room * 0.25),
    }.get(seller_reason.lower().strip(), 0)

    risk_adj = int(risk_score * 80)

    opening   = _round500(max(0, recommended_buy_price - nego_room - risk_adj + seller_adj))
    ideal     = _round500(max(0, recommended_buy_price - int(nego_room * 0.35)))
    # Walk-away: 1.5% above buy price, floor +3k, cap +25k
    walk_raw  = recommended_buy_price * 1.015
    walk_away = _round500(_clamp(walk_raw,
                                  recommended_buy_price + 3_000,
                                  recommended_buy_price + 25_000))

    return {
        "opening_offer":   opening,
        "target_offer":    ideal,
        "walk_away_price": walk_away,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 11. MARKET-BAND SIMILAR VEHICLES  (median/low/high/avg from real bands)
# ──────────────────────────────────────────────────────────────────────────────
def generate_similar_cars(
    market_value: float,
    brand: str,
    model: str,
    year: int,
    fuel: str,
    city: str,
    segment: str,
) -> list[dict]:
    """
    Generate realistic comp prices using market bands.
    Returns up to 3 comps with median/low/high pricing context.
    """
    current_year = datetime.now().year
    age          = max(0, current_year - year)
    age_factor   = _AGE_DEPRECIATION.get(min(age, 12), 0.30)

    model_key = _normalise_model(model)
    band = None
    for key in [model_key] + [" ".join(model_key.split()[:i])
                               for i in range(len(model_key.split()), 0, -1)]:
        if key in _MARKET_BANDS:
            band = _MARKET_BANDS[key]
            break
    if band is None:
        band = _SEGMENT_BANDS.get(segment, (200_000, 5_000_000))

    band_low  = band[0] * age_factor
    band_high = band[1] * age_factor
    band_med  = (band_low + band_high) / 2

    comps = []
    variations = [
        (brand, model, year,      fuel, band_med),
        (brand, model, year - 1,  fuel, band_low * 1.05),
        (brand, model, year + 1,  fuel, band_high * 0.95),
    ]
    for b, m, y, f, val in variations:
        if y < 2010 or y > current_year:
            continue
        comp_km = max(5_000, (current_year - y) * 12_000 + 8_000)
        comp_km = (comp_km // 1_000) * 1_000
        comps.append({
            "brand":         b,
            "model":         m,
            "year":          y,
            "fuel":          f,
            "city":          city,
            "market_value":  _round500(val),
            "lowest_price":  _round500(band_low),
            "median_price":  _round500(band_med),
            "highest_price": _round500(band_high),
            "odometer":      comp_km,
            "condition":     "Good",
            "segment":       segment,
        })

    return comps[:3]


# ── Inline brand→segment map (mirrors main.py BRAND_SEGMENT_MAP) ─────────────
# Kept here to avoid circular import. Keep in sync with main.py.
_INLINE_BRAND_SEGMENT: dict[str, str] = {
    # Economy
    "maruti": "economy", "maruti suzuki": "economy", "datsun": "economy",
    "bajaj": "economy", "chevrolet": "economy", "fiat": "economy",
    "opel": "economy", "premier": "economy", "force": "economy",
    "ashok leyland": "economy", "ambassador": "economy",
    "hyundai": "economy", "honda": "economy", "tata": "economy",
    "renault": "economy", "nissan": "economy", "ford": "economy",
    "mahindra": "economy", "mitsubishi": "economy", "isuzu": "economy",
    "citroen": "economy", "dc": "economy", "hindustan motors": "economy",
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
}

def calculate_decision(vehicle, market_value: float) -> dict:
    """
    Convert ML market value into a full dealer decision package.

    Waterfall:
        Buy Price = Market Value
                  − Reconditioning Cost  (dynamic: age/km/condition/inspection/segment)
                  − Holding Cost         (segment-specific rate × inventory days)
                  − Documentation Cost   (RC + NOC + insurance + optional hypo/state)
                  − Risk Buffer          (rupee-based, additive factors)
                  − Target Dealer Profit (dynamic, segment-capped)
    """
    # ── Extract inputs ────────────────────────────────────────────────────────
    target_margin_pct = float(getattr(vehicle, "target_margin_pct", 15) or 15)
    repair_buffer     = float(getattr(vehicle, "repair_buffer", 0) or 0)
    seller_asking     = float(getattr(vehicle, "seller_asking_price", 0) or 0)
    age               = max(0, 2026 - int(getattr(vehicle, "year", 2021) or 2021))
    km                = max(0, float(getattr(vehicle, "odometer_reading", 0) or 0))
    owner_count       = max(1, int(getattr(vehicle, "owner_count", 1) or 1))
    condition         = str(getattr(vehicle, "condition", "Good") or "Good").strip().lower()
    fuel              = str(getattr(vehicle, "fuel_type", "Petrol") or "Petrol").strip().lower()
    transmission      = str(getattr(vehicle, "transmission", "Manual") or "Manual").strip().lower()
    city              = str(getattr(vehicle, "city", "") or "").strip().lower()
    inspected         = bool(getattr(vehicle, "inspected", False))
    brand             = str(getattr(vehicle, "brand", "") or "")
    model_name        = str(getattr(vehicle, "model", "") or "")
    fuel_eff          = float(getattr(vehicle, "fuel_efficiency", 0) or 0)
    variant           = str(getattr(vehicle, "variant", "") or "")
    seller_reason     = str(getattr(vehicle, "seller_reason", "upgrading") or "upgrading")
    reg_state         = str(getattr(vehicle, "registration_state", "") or "")
    sale_state        = str(getattr(vehicle, "sale_state", "") or city)
    loan_out          = bool(getattr(vehicle, "loan_outstanding", False))

    # Segment — resolved inline to avoid circular import with main.py
    segment = _INLINE_BRAND_SEGMENT.get(brand.lower().strip(), "economy")

    # ── Apply market sanity clamp ─────────────────────────────────────────────
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        model_name, segment, age, float(market_value), city
    )
    market_value = clamped_value

    # ── Risk & confidence ─────────────────────────────────────────────────────
    risk_score, risk_level = compute_risk_score(
        age, km, owner_count, condition, fuel, inspected, sanity_clamped
    )
    confidence_score = compute_confidence_score(
        age, km, owner_count, condition, fuel, variant, fuel_eff,
        risk_score, sanity_clamped, city, inspected
    )

    # ── Dynamic margin ────────────────────────────────────────────────────────
    eff_margin_pct = dynamic_target_margin(
        segment, age, km, owner_count, condition, inspected, fuel, target_margin_pct
    )

    # ── Waterfall cost components ─────────────────────────────────────────────
    # 1. Reconditioning — dynamic; user override if large enough
    if repair_buffer > 1_000:
        recon_cost = int(repair_buffer)
    else:
        recon_cost = compute_dynamic_recon_cost(segment, age, km, condition, inspected)

    # 2. Holding cost — segment-specific
    holding_cost = compute_holding_cost(segment, market_value)

    # 3. Documentation — dynamic
    doc_cost, doc_breakdown = compute_doc_cost(reg_state, sale_state, loan_out)

    # 4. Risk buffer — rupee-based
    risk_buffer = compute_risk_buffer(
        market_value, risk_score, segment, age, km, owner_count, condition, inspected
    )

    # 5. Target profit — segment-capped
    veh_category     = classify_vehicle_category(brand, model_name)
    p_min, p_max     = _PROFIT_LIMITS.get(veh_category, (25_000, 100_000))
    raw_profit       = market_value * (eff_margin_pct / 100)
    target_profit    = int(_clamp(raw_profit, p_min, p_max))

    # ── Waterfall → Buy Price ─────────────────────────────────────────────────
    total_deductions      = recon_cost + holding_cost + doc_cost + risk_buffer + target_profit
    recommended_buy_price = market_value - total_deductions

    # Floor: never below 45% of market value
    recommended_buy_price = max(market_value * 0.45, recommended_buy_price)
    recommended_buy_price = _round500(recommended_buy_price)

    # ── Sell price & expected profit ──────────────────────────────────────────
    city_premium          = _CITY_DEMAND.get(city, 0.015)
    recommended_sell_price = _round500(market_value * (1 + city_premium * 0.5))
    expected_profit        = target_profit
    expected_margin_pct    = (expected_profit / max(recommended_buy_price, 1)) * 100

    # ── ROI-based 6-action dealer recommendation ──────────────────────────────
    roi = (expected_profit / max(recommended_buy_price, 1)) * 100

    if confidence_score < 52 or sanity_clamped:
        action = "MANUAL REVIEW"
    elif roi >= 14 and risk_score <= 30 and confidence_score >= 70:
        action = "BUY"
    elif roi >= 12 and risk_score <= 45 and not inspected:
        action = "BUY AFTER INSPECTION"
    elif roi >= 9 and risk_score <= 55:
        action = "NEGOTIATE"
    elif roi >= 6 and risk_score <= 70:
        # Check if seller is flexible
        flexible_reasons = {"financial", "relocating", "problem"}
        if seller_reason.lower().strip() in flexible_reasons:
            action = "NEGOTIATE AGGRESSIVELY"
        else:
            action = "NEGOTIATE"
    else:
        action = "REJECT"

    # ── Negotiation trio ──────────────────────────────────────────────────────
    nego = compute_negotiation_trio(
        recommended_buy_price, city, condition, risk_score, seller_reason
    )

    # ── Composite scores ──────────────────────────────────────────────────────
    demand_score          = round(_clamp(85 - age * 2.5 - (km / 200_000) * 35))
    brand_retention_score = round(_clamp(78 - age * 1.2 + (5 if fuel in {"petrol", "hybrid"} else 0)))
    vehicle_health_score  = round(_clamp(100 - age * 3 - km / 10_000 - (owner_count - 1) * 8))
    resale_liquidity_score = round(_clamp(
        (demand_score + brand_retention_score + vehicle_health_score) / 3
    ))

    deal_quality_score = round(_clamp(
        0.35 * _clamp(roi * 5)
        + 0.30 * confidence_score
        + 0.35 * (100 - risk_score)
    ))
    urgency_score = round(_clamp(65 + (deal_quality_score - 65) * 0.5 + (100 - risk_score) * 0.15))
    urgency_label = "High" if urgency_score >= 75 else "Medium" if urgency_score >= 55 else "Low"

    # ── Positive / negative factors ───────────────────────────────────────────
    positive_factors, negative_factors = [], []

    if age <= 3:
        positive_factors.append(f"Recent vehicle ({age} yr{'s' if age != 1 else ''}) supports strong resale demand")
    elif age >= 8:
        negative_factors.append(f"Vehicle age ({age} yrs) significantly increases depreciation risk")

    if km < 30_000:
        positive_factors.append(f"Low odometer ({km/1000:.0f}k km) improves acquisition confidence")
    elif km >= 100_000:
        negative_factors.append(f"High odometer ({km/1000:.0f}k km) will reduce buyer demand")

    if owner_count == 1:
        positive_factors.append("First-owner vehicle — strong buyer preference in Indian market")
    elif owner_count >= 3:
        negative_factors.append(f"{owner_count} previous owners — extensive due-diligence required")

    if condition in {"excellent", "good"}:
        positive_factors.append(f"{condition.title()} condition — ready for immediate resale")
    else:
        negative_factors.append(f"{condition.title()} condition — reconditioning investment required")

    if inspected:
        positive_factors.append("Inspection certificate present — reduces buyer uncertainty")
    else:
        negative_factors.append("No inspection report — consider pre-purchase inspection")

    if expected_margin_pct >= eff_margin_pct:
        positive_factors.append(f"Expected margin ({expected_margin_pct:.1f}%) meets dealer target ({eff_margin_pct:.1f}%)")
    else:
        negative_factors.append(f"Expected margin ({expected_margin_pct:.1f}%) below target ({eff_margin_pct:.1f}%)")

    if sanity_clamped:
        negative_factors.append(f"ML prediction adjusted by market sanity band — {sanity_note}")

    positive_factors.append("Market value predicted by CatBoost + LightGBM + XGBoost ensemble")
    if not negative_factors:
        negative_factors.append("No major risk signals detected — standard due-diligence applies")

    # ── Confidence band ───────────────────────────────────────────────────────
    price_spread = market_value * (0.06 + risk_score * 0.0005)
    price_min    = _round500(market_value - price_spread)
    price_max    = _round500(market_value + price_spread)

    # ── Seller gap ────────────────────────────────────────────────────────────
    seller_gap = _round500(seller_asking - recommended_buy_price) if seller_asking > 0 else 0

    return {
        # Prices
        "market_value":             int(market_value),
        "price_min":                int(price_min),
        "price_max":                int(price_max),
        "ci":                       int(price_spread),
        "recommended_buy_price":    int(recommended_buy_price),
        "recommended_sell_price":   int(recommended_sell_price),
        "expected_profit":          int(expected_profit),
        "expected_margin_pct":      round(expected_margin_pct, 1),
        "dealer_acq_price":         int(recommended_buy_price),
        "suggested_sell_price":     int(recommended_sell_price),
        "margin_pct":               round(expected_margin_pct, 1),
        "margin_amt":               int(expected_profit),
        # Cost waterfall
        "recon_cost":               int(recon_cost),
        "holding_cost":             int(holding_cost),
        "doc_cost":                 int(doc_cost),
        "doc_breakdown":            doc_breakdown,
        "risk_buffer":              int(risk_buffer),
        "target_profit":            int(target_profit),
        "repair_buffer":            int(recon_cost),
        # Waterfall for frontend display
        "waterfall": [
            {"label": "ML Market Value",         "value": int(market_value),          "sign": "",
             "note": "CatBoost ensemble prediction, condition-adjusted"},
            {"label": "Reconditioning Cost",      "value": int(recon_cost),            "sign": "-",
             "note": f"Dynamic estimate — {condition} condition, {age}yr, {km/1000:.0f}k km"},
            {"label": f"Holding Cost ({_HOLDING.get(segment, {}).get('days', 30)} days)",
                                                  "value": int(holding_cost),           "sign": "-",
             "note": f"{segment.title()} segment: {_HOLDING.get(segment, {}).get('rate_pct', 1.8):.1f}%/month × inventory days"},
            {"label": "RC + Documentation",       "value": int(doc_cost),              "sign": "-",
             "note": "RC transfer, NOC, insurance transfer" + (", hypothecation removal" if loan_out else "") + (", state transfer" if reg_state and sale_state and reg_state.lower() != sale_state.lower() else "")},
            {"label": "Risk Buffer",              "value": int(risk_buffer),           "sign": "-",
             "note": f"Rupee-based risk at score {risk_score}/100"},
            {"label": "Target Dealer Profit",     "value": int(target_profit),         "sign": "-",
             "note": f"Dynamic margin {eff_margin_pct:.1f}%, capped for {veh_category}"},
            {"label": "Recommended Buy Price",    "value": int(recommended_buy_price), "sign": "=",
             "note": "Maximum acquisition price"},
        ],
        # Negotiation
        "opening_offer":            nego["opening_offer"],
        "max_offer":                nego["walk_away_price"],
        "target_offer":             nego["target_offer"],
        "seller_gap":               int(seller_gap),
        # Risk & Confidence
        "risk_score":               int(risk_score),
        "risk_level":               risk_level,
        "confidence_score":         int(confidence_score),
        "demand_score":             int(demand_score),
        "brand_retention_score":    int(brand_retention_score),
        "vehicle_health_score":     int(vehicle_health_score),
        "resale_liquidity_score":   int(resale_liquidity_score),
        "deal_quality_score":       int(deal_quality_score),
        "urgency_score":            int(urgency_score),
        "urgency_label":            urgency_label,
        # Decision
        "action":                   action,
        "effective_margin_pct":     float(eff_margin_pct),
        "target_margin_pct":        float(target_margin_pct),
        # Factors
        "positive_factors":         positive_factors[:5],
        "negative_factors":         negative_factors[:5],
        # Sanity
        "sanity_clamped":           sanity_clamped,
        "sanity_note":              sanity_note,
        # Quote
        "quote_message": (
            f"Based on ML ensemble valuation, vehicle age, mileage, condition, and "
            f"{city.title() or 'local'} market demand, our recommended acquisition price is "
            f"₹{recommended_buy_price/100_000:.2f}L. "
            f"Seller gap: ₹{abs(seller_gap)//1000:.0f}k {'above' if seller_gap > 0 else 'below'} target."
            if seller_asking else
            f"Based on ML ensemble valuation, our recommended acquisition price is "
            f"₹{recommended_buy_price/100_000:.2f}L with target sell at ₹{recommended_sell_price/100_000:.2f}L."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY / WHEELR FUNCTIONS  (unchanged interface — called by existing routes)
# ──────────────────────────────────────────────────────────────────────────────
def check_disqualifier(vehicle_age: int, odometer: int,
                        owner_count: int, accident_history: str) -> dict:
    accident_hist_clean = (accident_history or "none").lower().strip()
    if vehicle_age > 12:
        return {"disqualified": True, "reason": "Vehicle age exceeds 12 years"}
    if odometer > 150_000:
        return {"disqualified": True, "reason": "Odometer reading exceeds 150,000 km"}
    if owner_count >= 4 and accident_hist_clean in ["minor", "major"]:
        return {"disqualified": True,
                "reason": f"Multiple owners ({owner_count}) + accident history detected"}
    return {"disqualified": False, "reason": "Passes pre-screening"}


def get_seasonal_multiplier(month: int) -> float:
    multipliers = {
        1: 0.97, 2: 0.97, 3: 1.04, 4: 0.98, 5: 0.98,
        6: 1.06, 7: 1.06, 8: 0.99, 9: 0.99,
        10: 1.05, 11: 1.05, 12: 0.96,
    }
    return multipliers.get(month, 1.0)


def get_wheelr_risk_deductions(owner_count: int, odometer: int,
                                accident_history: str = "none",
                                registration_state: str = "",
                                sale_state: str = "",
                                loan_outstanding: bool = False,
                                seller_reason: str = "upgrading") -> dict:
    accident_hist       = (accident_history or "none").lower().strip()
    seller_reason_clean = (seller_reason or "upgrading").lower().strip()
    owner_deduction     = {1: 0, 2: 8_000, 3: 18_000}.get(owner_count, 30_000)
    if odometer < 40_000:      km_deduction = 0
    elif odometer < 80_000:    km_deduction = 5_000
    elif odometer < 120_000:   km_deduction = 12_000
    else:                      km_deduction = 25_000
    accident_deduction  = {"none": 0, "minor": 10_000, "major": 35_000}.get(accident_hist, 0)
    state_deduction     = 0
    if registration_state and sale_state:
        state_deduction = 0 if registration_state.lower() == sale_state.lower() else 8_000
    loan_deduction      = 5_000 if loan_outstanding else 0
    seller_reason_adj   = {
        "upgrading": 0, "relocating": -5_000, "financial": -12_000,
        "unused": 5_000, "problem": -8_000,
    }.get(seller_reason_clean, 0)
    total = owner_deduction + km_deduction + accident_deduction + state_deduction + loan_deduction
    return {
        "total": int(total),
        "breakdown": {
            "owner_deduction":    int(owner_deduction),
            "km_deduction":       int(km_deduction),
            "accident_deduction": int(accident_deduction),
            "state_deduction":    int(state_deduction),
            "loan_deduction":     int(loan_deduction),
        },
        "seller_reason_adj": int(seller_reason_adj),
    }


def get_recon_cost(engine_grade: str = "good", tyre_grade: str = "good",
                   body_grade: str = "clean", interior_grade: str = "clean",
                   electrical_grade: str = "all_good",
                   vendor_type: dict = None,
                   rc_transfer_cost: int = 3_500) -> dict:
    if vendor_type is None:
        vendor_type = {k: "vendor" for k in
                       ["engine", "tyre", "body", "interior", "electrical"]}
    engine_grade     = (engine_grade or "good").lower().strip()
    tyre_grade       = (tyre_grade or "good").lower().strip()
    body_grade       = (body_grade or "clean").lower().strip()
    interior_grade   = (interior_grade or "clean").lower().strip()
    electrical_grade = (electrical_grade or "all_good").lower().strip()

    engine_costs  = {
        "good":     {"inhouse": 0,      "vendor": 0},
        "average":  {"inhouse": 4_000,  "vendor": 8_000},
        "poor":     {"inhouse": 18_000, "vendor": 35_000},
        "critical": {"inhouse": 45_000, "vendor": 80_000},
    }
    tyre_costs    = {
        "good":    {"inhouse": 0,     "vendor": 0},
        "two_bad": {"inhouse": 4_000, "vendor": 6_000},
        "all_bad": {"inhouse": 8_000, "vendor": 12_000},
    }
    body_costs    = {
        "clean":    {"inhouse": 0,      "vendor": 0},
        "minor":    {"inhouse": 3_000,  "vendor": 5_000},
        "major":    {"inhouse": 10_000, "vendor": 18_000},
        "accident": {"inhouse": 22_000, "vendor": 40_000},
    }
    interior_costs = {
        "clean":         {"inhouse": 0,     "vendor": 0},
        "needs_cleaning": {"inhouse": 1_500, "vendor": 3_000},
        "full_refurb":   {"inhouse": 6_000, "vendor": 10_000},
    }
    electrical_costs = {
        "all_good":   {"inhouse": 0,     "vendor": 0},
        "ac_fault":   {"inhouse": 4_500, "vendor": 8_000},
        "multi_fault": {"inhouse": 8_000, "vendor": 15_000},
    }

    ec  = engine_costs.get(engine_grade, engine_costs["good"])[vendor_type.get("engine", "vendor")]
    tc  = tyre_costs.get(tyre_grade, tyre_costs["good"])[vendor_type.get("tyre", "vendor")]
    bc  = body_costs.get(body_grade, body_costs["clean"])[vendor_type.get("body", "vendor")]
    ic  = interior_costs.get(interior_grade, interior_costs["clean"])[vendor_type.get("interior", "vendor")]
    elc = electrical_costs.get(electrical_grade, electrical_costs["all_good"])[vendor_type.get("electrical", "vendor")]

    rc_transfer_cost = max(0, int(rc_transfer_cost or 3_500))
    fixed_cost = rc_transfer_cost + 2_500 + 2_000
    total      = ec + tc + bc + ic + elc + fixed_cost
    return {
        "engine_cost": int(ec), "tyre_cost": int(tc), "body_cost": int(bc),
        "interior_cost": int(ic), "electrical_cost": int(elc),
        "fixed_cost": int(fixed_cost), "rc_transfer_cost": int(rc_transfer_cost),
        "total": int(total),
        "breakdown": {
            "engine": int(ec), "tyres": int(tc), "body_paint": int(bc),
            "interior": int(ic), "electricals": int(elc), "fixed": int(fixed_cost),
        },
    }


def get_negotiation_trio(max_buy_price: int, seller_reason_adj: int = 0) -> dict:
    """Legacy function — used by /evaluate-enhanced and /reverse-calculate."""
    opening_offer   = max(0, max_buy_price - 15_000 + seller_reason_adj)
    target_offer    = max_buy_price - 8_000
    walk_away_price = max_buy_price
    return {
        "opening_offer":   int(opening_offer),
        "target_offer":    int(target_offer),
        "walk_away_price": int(walk_away_price),
    }


def get_deal_health(ml_market_value: int, recon_total: int,
                    profit_target: int, owner_count: int,
                    odometer: int, accident_history: str = "none") -> str:
    if ml_market_value <= 0:
        return "red"
    margin_pct = profit_target / ml_market_value if ml_market_value > 0 else 0
    recon_pct  = recon_total / ml_market_value if ml_market_value > 0 else 0
    accident_h = (accident_history or "none").lower().strip()
    if margin_pct >= 0.12 and recon_pct <= 0.20 and owner_count <= 2 and accident_h == "none":
        return "green"
    if margin_pct < 0.08 or recon_pct > 0.35:
        return "red"
    return "yellow"
