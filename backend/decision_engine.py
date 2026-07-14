"""
PricerPoint — Dealer Decision Engine  v7.0
==========================================
Business logic layer on top of the ML prediction.

Key principles
--------------
1. Market Sanity Clamp   — segment-aware price bands prevent impossible outputs
2. Realistic Waterfall   — buy price = market − recon − holding − docs − risk_buf − profit
3. Segment Profit Caps   — economy ₹25k–₹60k, luxury ₹1.5L–₹3L
4. Dynamic Margins       — risk-adjusted, never a flat 15%
5. Factor-based Risk     — age/km/owner/condition/fuel/inspection
6. ROI-based Action      — BUY / NEGOTIATE / ONLY IF BELOW / PASS
7. Monetary SHAP         — "Vehicle age reduces value by ₹58,000"
8. Demand-driven Negos   — opening/ideal/walkaway from city demand + vehicle factors
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
# Indian used-car transaction prices (NOT listing prices) based on OLX Autos /
# Cars24 / Spinny data patterns.  Key: lowercase model name.
# Format: (lower_2021_price, upper_2021_price)  — age-adjusted at runtime.
# ──────────────────────────────────────────────────────────────────────────────
_MARKET_BANDS: dict[str, tuple[float, float]] = {
    # Economy Hatchbacks (Maruti / Hyundai base)
    "alto":         (250_000, 500_000),
    "alto k10":     (280_000, 540_000),
    "s-presso":     (320_000, 560_000),
    "kwid":         (280_000, 490_000),
    "celerio":      (330_000, 590_000),
    "wagonr":       (360_000, 680_000),
    "wagon r":      (360_000, 680_000),
    "eon":          (200_000, 380_000),
    "santro":       (350_000, 600_000),
    "tiago":        (420_000, 720_000),
    "magnite":      (560_000, 870_000),
    "punch":        (600_000, 960_000),
    # Premium Hatchbacks
    "swift":        (550_000, 900_000),
    "baleno":       (600_000, 950_000),
    "i20":          (680_000, 1_100_000),
    "i10":          (380_000, 680_000),
    "grand i10":    (400_000, 700_000),
    "altroz":       (650_000, 1_000_000),
    "glanza":       (600_000, 920_000),
    "ignis":        (540_000, 820_000),
    "polo":         (600_000, 960_000),
    # Compact SUVs
    "nexon":        (850_000, 1_600_000),
    "brezza":       (820_000, 1_550_000),
    "vitara brezza": (700_000, 1_300_000),
    "venue":        (780_000, 1_450_000),
    "sonet":        (800_000, 1_480_000),
    "ecosport":     (700_000, 1_200_000),
    "duster":       (650_000, 1_100_000),
    "kiger":        (680_000, 1_050_000),
    # Mid SUVs
    "creta":        (1_100_000, 2_200_000),
    "seltos":       (1_200_000, 2_300_000),
    "grand vitara": (1_350_000, 2_450_000),
    "hector":       (1_200_000, 2_100_000),
    "compass":      (1_400_000, 2_500_000),
    "harrier":      (1_300_000, 2_400_000),
    "safari":       (1_500_000, 2_800_000),
    "ertiga":       (850_000, 1_500_000),
    "carens":       (1_000_000, 1_900_000),
    # Premium / Luxury
    "city":         (850_000, 1_700_000),
    "ciaz":         (700_000, 1_300_000),
    "vento":        (750_000, 1_250_000),
    "rapid":        (700_000, 1_200_000),
    "octavia":      (1_800_000, 3_200_000),
    "superb":       (2_500_000, 4_500_000),
    "innova crysta":(1_500_000, 2_800_000),
    "fortuner":     (2_800_000, 5_500_000),
    "endeavour":    (2_500_000, 4_800_000),
    "tucson":       (2_000_000, 3_500_000),
    "thar":         (1_500_000, 2_800_000),
    "xuv700":       (1_800_000, 3_500_000),
    "scorpio":      (1_000_000, 2_000_000),
    "scorpio n":    (1_400_000, 2_600_000),
    "defender":     (6_000_000, 15_000_000),
    "range rover":  (5_000_000, 18_000_000),
    "glc":          (4_500_000, 8_000_000),
    "c class":      (3_500_000, 7_000_000),
    "3 series":     (3_200_000, 7_500_000),
    "x1":           (2_800_000, 5_500_000),
    "q3":           (3_000_000, 6_000_000),
    "a4":           (3_000_000, 6_500_000),
}

# ── Segment-level bands (fallback when model not in dict) ────────────────────
_SEGMENT_BANDS: dict[str, tuple[float, float]] = {
    "economy": (200_000,  1_500_000),
    "premium": (600_000,  3_500_000),
    "luxury":  (2_500_000, 20_000_000),
}

# ── Age depreciation schedule (% of new price retained per year) ────────────
_AGE_DEPRECIATION = {
    0: 1.00, 1: 0.86, 2: 0.78, 3: 0.71, 4: 0.65,
    5: 0.59, 6: 0.54, 7: 0.50, 8: 0.46, 9: 0.42,
    10: 0.38, 11: 0.35, 12: 0.32,
}

# ── Segment profit limits (min, max) ────────────────────────────────────────
_PROFIT_LIMITS: dict[str, tuple[int, int]] = {
    "economy":       (25_000,  60_000),
    "premium_hatch": (40_000,  80_000),
    "compact_suv":   (60_000, 100_000),
    "mid_suv":       (80_000, 150_000),
    "luxury":        (150_000, 300_000),
}

def classify_vehicle_category(brand: str, model: str) -> str:
    b = brand.lower().strip()
    m = model.lower().strip()

    # 1. Luxury & Premium Large
    if b in {"bmw", "mercedes-benz", "mercedes", "audi", "lexus", "volvo", "land rover", "jaguar", "porsche", "defender"}:
        return "luxury"
    if "fortuner" in m or "endeavour" in m or "glc" in m or "c class" in m or "3 series" in m or "x1" in m or "q3" in m or "a4" in m or "xuv700" in m or "safari" in m:
        return "luxury"

    # 2. Mid SUVs / Sedans
    if "creta" in m or "seltos" in m or "grand vitara" in m or "hector" in m or "compass" in m or "harrier" in m or "scorpio" in m or "ertiga" in m or "carens" in m or "city" in m or "ciaz" in m or "innova" in m:
        return "mid_suv"

    # 3. Compact SUVs
    if "venue" in m or "nexon" in m or "brezza" in m or "sonet" in m or "ecosport" in m or "duster" in m or "kiger" in m or "magnite" in m or "punch" in m or "thar" in m:
        return "compact_suv"

    # 4. Premium Hatchbacks
    if "swift" in m or "baleno" in m or "i20" in m or "altroz" in m or "glanza" in m or "ignis" in m or "polo" in m or "i10" in m or "grand i10" in m:
        return "premium_hatch"

    # 5. Economy Hatchbacks (Default fallback)
    return "economy"


# ── City demand premium (additive to market value) ───────────────────────────
_CITY_DEMAND: dict[str, float] = {
    "mumbai": 0.045, "pune": 0.030, "delhi": 0.040, "ncr": 0.038,
    "bangalore": 0.042, "bengaluru": 0.042, "hyderabad": 0.035,
    "chennai": 0.032, "kolkata": 0.025, "ahmedabad": 0.028,
    "surat": 0.020, "jaipur": 0.018, "lucknow": 0.015,
    "chandigarh": 0.022, "kochi": 0.020, "bhubaneswar": 0.012,
}

# ── Reconditioning estimates by segment (default, no inspection data) ────────
_DEFAULT_RECON: dict[str, int] = {
    "economy": 15_000,
    "premium": 22_000,
    "luxury":  45_000,
}

# ── Holding cost basis (monthly, as % of market value) ───────────────────────
_HOLDING_COST_MONTHLY_PCT = 0.018   # 1.8% per month (finance + lot + overhead)
_AVG_HOLDING_DAYS         = 30      # assumed days to sell

# ── Documentation & registration ─────────────────────────────────────────────
_DOC_COST = 4_500   # RC transfer + hypothecation NOC + misc


# ──────────────────────────────────────────────────────────────────────────────
# 1. MARKET SANITY CLAMP
# ──────────────────────────────────────────────────────────────────────────────
def _normalise_model(model_name: str) -> str:
    """Strip brand prefix duplicates, lowercase, single space."""
    return " ".join(model_name.lower().split())


def apply_market_sanity_clamp(
    model_name: str,
    segment: str,
    vehicle_age: int,
    raw_value: float,
) -> tuple[float, bool, str]:
    """
    Clamp the ML raw value to a realistic age-adjusted market band.

    Returns (clamped_value, was_clamped, note).
    """
    model_key = _normalise_model(model_name)
    # Find band — try progressively shorter model keys
    band = None
    for key in [model_key] + [" ".join(model_key.split()[:i]) for i in range(len(model_key.split()), 0, -1)]:
        if key in _MARKET_BANDS:
            band = _MARKET_BANDS[key]
            break
    if band is None:
        band = _SEGMENT_BANDS.get(segment, (100_000, 20_000_000))

    age_factor = _AGE_DEPRECIATION.get(min(vehicle_age, 12), 0.30)
    lower = band[0] * age_factor
    upper = band[1] * age_factor

    clamped  = False
    note     = "within expected market band"

    if raw_value > upper * 1.10:
        raw_value = upper
        clamped   = True
        note      = f"clamped from above — ML overestimated vs market band"
    elif raw_value < lower * 0.85:
        raw_value = lower * 0.90
        clamped   = True
        note      = f"clamped from below — ML underestimated vs market band"

    return float(raw_value), clamped, note


# ──────────────────────────────────────────────────────────────────────────────
# 2. DYNAMIC MARGIN CALCULATOR
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
    Overrides the user's fixed input when vehicle risk warrants adjustment.
    """
    base = 12.0

    # Positive factors
    if vehicle_age <= 2:  base += 2.5
    elif vehicle_age <= 4: base += 1.5
    if km < 30_000:       base += 1.0
    if owner_count == 1:  base += 1.0
    if inspected:         base += 1.5
    if condition.lower() == "excellent": base += 1.0
    if fuel.lower() in {"petrol", "hybrid"}: base += 0.5

    # Negative factors (risk deductions)
    if vehicle_age > 7:   base -= 2.0
    elif vehicle_age > 5: base -= 1.0
    if km > 80_000:       base -= 1.5
    elif km > 50_000:     base -= 0.8
    if owner_count >= 3:  base -= 1.5
    elif owner_count == 2: base -= 0.8
    if condition.lower() == "poor":    base -= 2.0
    elif condition.lower() == "average": base -= 1.0

    # Segment cap ranges
    segment_ranges = {
        "economy": (8.0, 16.0),
        "premium": (10.0, 18.0),
        "luxury":  (12.0, 22.0),
    }
    lo, hi = segment_ranges.get(segment, (8.0, 18.0))
    computed = _clamp(base, lo, hi)

    # Blend: 60% computed (risk-adjusted), 40% user target (preference)
    blended = 0.60 * computed + 0.40 * float(user_target_pct)
    return round(_clamp(blended, lo, hi), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. FACTOR-BASED RISK SCORE
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
        raw += 12   # ML was unreliable → raise risk

    score = round(_clamp(raw, 5, 95))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return score, level


# ──────────────────────────────────────────────────────────────────────────────
# 4. REALISTIC CONFIDENCE SCORE
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
) -> int:
    score = 92.0

    # Deductions
    if variant.lower() in {"", "unknown", "base"}:
        score -= 8
    if fuel_efficiency <= 0:
        score -= 4
    if sanity_clamped:
        score -= 14
    if km > 120_000:
        score -= 10
    elif km > 80_000:
        score -= 5
    if km < 3_000 and vehicle_age > 2:
        score -= 8   # suspiciously low mileage
    score -= (owner_count - 1) * 4
    score -= max(0, vehicle_age - 4) * 1.5
    score -= risk_score * 0.18

    # Bonuses
    if condition.lower() == "excellent":
        score += 3
    if fuel.lower() in {"petrol", "hybrid"}:
        score += 2

    return int(round(_clamp(score, 42, 95)))


# ──────────────────────────────────────────────────────────────────────────────
# 5. MONETARY SHAP EXPLANATION
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

    # Mileage impact (₹1.2 per km above 25k baseline)
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
        lbl = f"{condition.title()} condition {sign} ₹{abs(cond_impact)//1000:.0f},000"
        items.append({"feature": "Condition", "value": condition.title(), "contribution": cond_impact, "label": lbl})

    # Ownership
    owner_impact = {1: 8_000, 2: -6_000, 3: -18_000, 4: -30_000}.get(min(owner_count, 4), -30_000)
    owner_lbl = (
        "First owner increases buyer confidence: +₹8,000"
        if owner_count == 1
        else f"{owner_count} previous owners reduce value by ₹{abs(owner_impact)//1000:.0f},000"
    )
    items.append({"feature": "Ownership", "value": f"{owner_count} owner(s)", "contribution": owner_impact, "label": owner_lbl})

    # City demand
    city_premium_pct = _CITY_DEMAND.get(city.lower().strip(), 0.010)
    city_impact = int(market_value * city_premium_pct)
    if city_impact > 1_000:
        items.append({
            "feature": "City Demand",
            "value": city.title(),
            "contribution": city_impact,
            "label": f"{city.title()} market demand adds ₹{city_impact//1000:.0f},000 to value",
        })

    # Fuel type
    fuel_impact = {"petrol": 6_000, "hybrid": 10_000, "electric": 8_000, "diesel": -2_000, "cng": -4_000}.get(fuel.lower(), 0)
    if fuel_impact != 0:
        sign = "adds" if fuel_impact > 0 else "reduces"
        items.append({
            "feature": "Fuel Type",
            "value": fuel.title(),
            "contribution": fuel_impact,
            "label": f"{fuel.title()} fuel type {sign} ₹{abs(fuel_impact)//1000:.0f},000 to demand",
        })

    # Transmission
    if transmission.lower() in {"automatic", "cvt", "dct"}:
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
        fe_delta = fuel_efficiency - 16.0
        fe_impact = int(fe_delta * market_value * 0.003)
        if abs(fe_impact) > 1_500:
            sign = "adds" if fe_impact > 0 else "reduces"
            items.append({
                "feature": "Fuel Efficiency",
                "value": f"{fuel_efficiency:.1f} km/l",
                "contribution": fe_impact,
                "label": f"Fuel efficiency ({fuel_efficiency:.1f} km/l) {sign} ₹{abs(fe_impact)//1000:.0f},000",
            })

    return sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:8]


# ──────────────────────────────────────────────────────────────────────────────
# 6. NEGOTIATION TRIO — DEMAND-DRIVEN
# ──────────────────────────────────────────────────────────────────────────────
def compute_negotiation_trio(
    recommended_buy_price: float,
    city: str,
    condition: str,
    risk_score: int,
    seller_reason: str = "upgrading",
) -> dict:
    """Opening / Ideal / Walk-away with demand and condition adjustments."""
    demand = _CITY_DEMAND.get(city.lower().strip(), 0.015)

    # Higher demand city → less negotiation room (seller has more options)
    # Lower demand → more room to negotiate
    nego_room = max(5_000, int(recommended_buy_price * (0.05 - demand * 0.6)))

    # Seller reason adjustment
    seller_adj = {
        "financial": -nego_room * 0.3,
        "relocating": -nego_room * 0.15,
        "upgrading": 0,
        "unused": nego_room * 0.1,
        "problem": -nego_room * 0.25,
    }.get(seller_reason.lower().strip(), 0)

    # Risk: higher risk → negotiate harder (lower opening)
    risk_adj = int(risk_score * 80)

    opening   = _round500(max(0, recommended_buy_price - nego_room - risk_adj + seller_adj))
    ideal     = _round500(max(0, recommended_buy_price - nego_room * 0.3))
    walk_away = _round500(recommended_buy_price + 2_500)

    return {
        "opening_offer":    opening,
        "target_offer":     ideal,
        "walk_away_price":  walk_away,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. SIMILAR CARS — REALISTIC COMPS
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
    """Generate 3 realistic comparable vehicles at market-consistent prices."""
    age = max(0, datetime.now().year - year)
    comps = []

    variations = [
        (brand, model, year, fuel, 0.96),
        (brand, model, year - 1, fuel, 0.91),
        (brand, model, year + 1, fuel, 1.05),
    ]
    for b, m, y, f, factor in variations:
        if y < 2010 or y > 2025:
            continue
        comp_val = _round500(market_value * factor)
        comp_km  = max(5_000, int((datetime.now().year - y) * 12_000 + 8_000))
        comp_km  = (comp_km // 1000) * 1000
        comps.append({
            "brand":        b,
            "model":        m,
            "year":         y,
            "fuel":         f,
            "city":         city,
            "market_value": comp_val,
            "odometer":     comp_km,
            "condition":    "Good",
            "segment":      segment,
        })

    return comps[:3]


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN DECISION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def calculate_decision(vehicle, market_value: float) -> dict:
    """
    Convert ML market value into a full dealer decision package.

    Waterfall:
        Buy Price = Market Value
                  − Reconditioning Cost
                  − Holding Cost
                  − Documentation
                  − Risk Buffer
                  − Dealer Target Profit
    """
    # ── Extract inputs ───────────────────────────────────────────────────────
    target_margin_pct = float(getattr(vehicle, "target_margin_pct", 15) or 15)
    repair_buffer     = float(getattr(vehicle, "repair_buffer", 0) or 0)  # user override
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

    # Segment
    from backend.main import get_segment_class
    segment = get_segment_class(brand)

    # ── Apply market sanity clamp ────────────────────────────────────────────
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        model_name, segment, age, float(market_value)
    )
    market_value = clamped_value

    # ── Risk & confidence ────────────────────────────────────────────────────
    risk_score, risk_level = compute_risk_score(
        age, km, owner_count, condition, fuel, inspected, sanity_clamped
    )
    confidence_score = compute_confidence_score(
        age, km, owner_count, condition, fuel, variant, fuel_eff,
        risk_score, sanity_clamped
    )

    # ── Dynamic margin ───────────────────────────────────────────────────────
    eff_margin_pct = dynamic_target_margin(
        segment, age, km, owner_count, condition, inspected, fuel, target_margin_pct
    )

    # ── Waterfall cost components ────────────────────────────────────────────
    # Reconditioning — use user override if provided, else segment default
    recon_cost = int(repair_buffer) if repair_buffer > 1_000 else _DEFAULT_RECON.get(segment, 18_000)

    # Holding cost (1.8% / month × 30 days = 1.8% of value)
    holding_cost = int(market_value * _HOLDING_COST_MONTHLY_PCT)

    # Documentation (RC transfer + misc)
    doc_cost = _DOC_COST

    # Risk buffer: dynamic, based on risk score
    risk_buffer = int(market_value * (risk_score / 100) * 0.045)

    # Target profit — segment-capped dynamically
    veh_category = classify_vehicle_category(brand, model_name)
    p_min, p_max = _PROFIT_LIMITS.get(veh_category, (25_000, 100_000))
    raw_profit_target = market_value * (eff_margin_pct / 100)
    target_profit = int(_clamp(raw_profit_target, p_min, p_max))

    # ── Waterfall → Buy Price ────────────────────────────────────────────────
    total_deductions = recon_cost + holding_cost + doc_cost + risk_buffer + target_profit
    recommended_buy_price = market_value - total_deductions

    # Floor: never below 45% of market value (prevents absurd negatives)
    recommended_buy_price = max(market_value * 0.45, recommended_buy_price)
    recommended_buy_price = _round500(recommended_buy_price)

    # ── Sell price & actual profit ───────────────────────────────────────────
    # Sell price = market value + city demand premium (dealer can achieve this)
    city_premium = _CITY_DEMAND.get(city, 0.015)
    recommended_sell_price = _round500(market_value * (1 + city_premium * 0.5))

    # Set expected profit to target profit (so it scales dynamically with car value, respecting the caps)
    expected_profit   = target_profit
    expected_margin_pct = (expected_profit / max(recommended_buy_price, 1)) * 100

    # ── ROI-based dealer action ───────────────────────────────────────────────
    roi = (expected_profit / max(recommended_buy_price, 1)) * 100
    if confidence_score < 50 or sanity_clamped:
        action = "MANUAL REVIEW"
    elif roi >= 12 and risk_score <= 35:
        action = "BUY"
    elif roi >= 8 and risk_score <= 50:
        action = "NEGOTIATE"
    elif roi >= 4 and risk_score <= 65:
        action = "NEGOTIATE"   # with a price target note
    else:
        action = "REJECT"

    # ── Negotiation trio ─────────────────────────────────────────────────────
    nego = compute_negotiation_trio(
        recommended_buy_price, city, condition, risk_score, seller_reason
    )

    # ── Factor scores ────────────────────────────────────────────────────────
    demand_score          = round(_clamp(85 - age * 2.5 - (km / 200_000) * 35))
    brand_retention_score = round(_clamp(78 - age * 1.2 + (5 if fuel in {"petrol", "hybrid"} else 0)))
    vehicle_health_score  = round(_clamp(100 - age * 3 - km / 10_000 - (owner_count - 1) * 8))
    resale_liquidity_score = round(_clamp((demand_score + brand_retention_score + vehicle_health_score) / 3))

    deal_quality_score = round(_clamp(
        0.35 * _clamp(roi * 5)
        + 0.30 * confidence_score
        + 0.35 * (100 - risk_score)
    ))
    urgency_score = round(_clamp(65 + (deal_quality_score - 65) * 0.5 + (100 - risk_score) * 0.15))
    urgency_label = "High" if urgency_score >= 75 else "Medium" if urgency_score >= 55 else "Low"

    # ── Positive / negative factors ──────────────────────────────────────────
    positive_factors = []
    negative_factors = []

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
        negative_factors.append(f"{condition.title()} condition — reconditioning investment required before sale")

    if inspected:
        positive_factors.append("Inspection certificate present — reduces buyer uncertainty")

    if expected_margin_pct >= eff_margin_pct:
        positive_factors.append(f"Expected margin ({expected_margin_pct:.1f}%) meets dealer target ({eff_margin_pct:.1f}%)")
    else:
        negative_factors.append(f"Expected margin ({expected_margin_pct:.1f}%) below target ({eff_margin_pct:.1f}%)")

    if sanity_clamped:
        negative_factors.append(f"ML prediction adjusted by market sanity band — {sanity_note}")

    positive_factors.append("Market value predicted by CatBoost + LightGBM + XGBoost ensemble")
    if not negative_factors:
        negative_factors.append("No major risk signals detected — standard due-diligence applies")

    # ── Confidence band ──────────────────────────────────────────────────────
    price_spread   = market_value * (0.06 + risk_score * 0.0005)
    price_min      = _round500(market_value - price_spread)
    price_max      = _round500(market_value + price_spread)

    # ── Seller gap ───────────────────────────────────────────────────────────
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
        "risk_buffer":              int(risk_buffer),
        "target_profit":            int(target_profit),
        "repair_buffer":            int(recon_cost),
        # Waterfall for frontend display
        "waterfall": [
            {"label": "ML Market Value",       "value": int(market_value),       "sign": ""},
            {"label": "Reconditioning Cost",   "value": int(recon_cost),         "sign": "-"},
            {"label": "Holding Cost (30 days)","value": int(holding_cost),       "sign": "-"},
            {"label": "RC + Documentation",    "value": int(doc_cost),           "sign": "-"},
            {"label": "Risk Buffer",           "value": int(risk_buffer),        "sign": "-"},
            {"label": "Target Dealer Profit",  "value": int(target_profit),      "sign": "-"},
            {"label": "Recommended Buy Price", "value": int(recommended_buy_price), "sign": "="},
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
# LEGACY / WHEELR FUNCTIONS  (unchanged — called by existing routes)
# ──────────────────────────────────────────────────────────────────────────────
def check_disqualifier(vehicle_age: int, odometer: int,
                       owner_count: int, accident_history: str) -> dict:
    accident_hist_clean = (accident_history or "none").lower().strip()
    if vehicle_age > 12:
        return {"disqualified": True, "reason": "Vehicle age exceeds 12 years"}
    if odometer > 150000:
        return {"disqualified": True, "reason": "Odometer reading exceeds 150,000 km"}
    if owner_count >= 4 and accident_hist_clean in ["minor", "major"]:
        return {"disqualified": True, "reason": f"Multiple owners ({owner_count}) + accident history detected"}
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
    accident_hist      = (accident_history or "none").lower().strip()
    seller_reason_clean = (seller_reason or "upgrading").lower().strip()
    owner_deduction    = {1: 0, 2: 8000, 3: 18000}.get(owner_count, 30000)
    if odometer < 40000:      km_deduction = 0
    elif odometer < 80000:    km_deduction = 5000
    elif odometer < 120000:   km_deduction = 12000
    else:                     km_deduction = 25000
    accident_deduction = {"none": 0, "minor": 10000, "major": 35000}.get(accident_hist, 0)
    state_deduction    = 0
    if registration_state and sale_state:
        state_deduction = 0 if registration_state.lower() == sale_state.lower() else 8000
    loan_deduction     = 5000 if loan_outstanding else 0
    seller_reason_adj  = {
        "upgrading": 0, "relocating": -5000, "financial": -12000,
        "unused": 5000, "problem": -8000,
    }.get(seller_reason_clean, 0)
    total = owner_deduction + km_deduction + accident_deduction + state_deduction + loan_deduction
    return {
        "total": int(total),
        "breakdown": {
            "owner_deduction": int(owner_deduction),
            "km_deduction": int(km_deduction),
            "accident_deduction": int(accident_deduction),
            "state_deduction": int(state_deduction),
            "loan_deduction": int(loan_deduction),
        },
        "seller_reason_adj": int(seller_reason_adj),
    }


def get_recon_cost(engine_grade: str = "good", tyre_grade: str = "good",
                   body_grade: str = "clean", interior_grade: str = "clean",
                   electrical_grade: str = "all_good",
                   vendor_type: dict = None,
                   rc_transfer_cost: int = 3500) -> dict:
    if vendor_type is None:
        vendor_type = {"engine": "vendor", "tyre": "vendor", "body": "vendor",
                       "interior": "vendor", "electrical": "vendor"}
    engine_grade     = (engine_grade or "good").lower().strip()
    tyre_grade       = (tyre_grade or "good").lower().strip()
    body_grade       = (body_grade or "clean").lower().strip()
    interior_grade   = (interior_grade or "clean").lower().strip()
    electrical_grade = (electrical_grade or "all_good").lower().strip()
    engine_costs  = {"good": {"inhouse": 0, "vendor": 0}, "average": {"inhouse": 4000, "vendor": 8000},
                     "poor": {"inhouse": 18000, "vendor": 35000}, "critical": {"inhouse": 45000, "vendor": 80000}}
    tyre_costs    = {"good": {"inhouse": 0, "vendor": 0}, "two_bad": {"inhouse": 4000, "vendor": 6000},
                     "all_bad": {"inhouse": 8000, "vendor": 12000}}
    body_costs    = {"clean": {"inhouse": 0, "vendor": 0}, "minor": {"inhouse": 3000, "vendor": 5000},
                     "major": {"inhouse": 10000, "vendor": 18000}, "accident": {"inhouse": 22000, "vendor": 40000}}
    interior_costs = {"clean": {"inhouse": 0, "vendor": 0}, "needs_cleaning": {"inhouse": 1500, "vendor": 3000},
                      "full_refurb": {"inhouse": 6000, "vendor": 10000}}
    electrical_costs = {"all_good": {"inhouse": 0, "vendor": 0}, "ac_fault": {"inhouse": 4500, "vendor": 8000},
                        "multi_fault": {"inhouse": 8000, "vendor": 15000}}
    ec = engine_costs.get(engine_grade, engine_costs["good"])[vendor_type.get("engine", "vendor")]
    tc = tyre_costs.get(tyre_grade, tyre_costs["good"])[vendor_type.get("tyre", "vendor")]
    bc = body_costs.get(body_grade, body_costs["clean"])[vendor_type.get("body", "vendor")]
    ic = interior_costs.get(interior_grade, interior_costs["clean"])[vendor_type.get("interior", "vendor")]
    elc = electrical_costs.get(electrical_grade, electrical_costs["all_good"])[vendor_type.get("electrical", "vendor")]
    rc_transfer_cost = max(0, int(rc_transfer_cost or 3500))
    fixed_cost = rc_transfer_cost + 2500 + 2000
    total      = ec + tc + bc + ic + elc + fixed_cost
    return {
        "engine_cost": int(ec), "tyre_cost": int(tc), "body_cost": int(bc),
        "interior_cost": int(ic), "electrical_cost": int(elc),
        "fixed_cost": int(fixed_cost), "rc_transfer_cost": int(rc_transfer_cost),
        "total": int(total),
        "breakdown": {"engine": int(ec), "tyres": int(tc), "body_paint": int(bc),
                      "interior": int(ic), "electricals": int(elc), "fixed": int(fixed_cost)},
    }


def get_negotiation_trio(max_buy_price: int, seller_reason_adj: int = 0) -> dict:
    opening_offer    = max(0, max_buy_price - 15000 + seller_reason_adj)
    target_offer     = max_buy_price - 8000
    walk_away_price  = max_buy_price
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
    margin_pct  = profit_target / ml_market_value if ml_market_value > 0 else 0
    recon_pct   = recon_total / ml_market_value if ml_market_value > 0 else 0
    accident_h  = (accident_history or "none").lower().strip()
    if margin_pct >= 0.12 and recon_pct <= 0.20 and owner_count <= 2 and accident_h == "none":
        return "green"
    if margin_pct < 0.08 or recon_pct > 0.35:
        return "red"
    return "yellow"
