

from __future__ import annotations
import math
from datetime import datetime

# HELPERS
def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))

def _round500(v: float) -> int:
    return int(round(v / 500) * 500)


# CONFIG — All business rules live here as named constants.
# Update these dicts when market conditions change; never hard-code in logic.

# ── Market Reference Bands (Indian used-car transaction prices, baseline 2021)
# Format: (lower_₹, upper_₹) — age-adjusted at runtime via _AGE_DEPRECIATION
_MARKET_BANDS: dict[str, tuple[float, float]] = {
    # Economy hatchbacks
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
    # Premium hatchbacks
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
    "safari classic": (800_000,  1_500_000),  # pre-2021 old Tata Safari/Storme
    "ertiga":        (850_000,  1_500_000),
    "carens":        (1_000_000, 1_900_000),
    "city":          (850_000,  1_700_000),
    "verna":         (850_000,  1_600_000),  # mid-size sedan
    "ciaz":          (700_000,  1_300_000),
    "vento":         (750_000,  1_250_000),
    "rapid":         (700_000,  1_200_000),
    "slavia":        (1_100_000, 1_850_000), # modern Skoda sedan
    "virtus":        (1_150_000, 1_900_000), # modern VW sedan
    "innova crysta": (900_000, 2_400_000),   # 2014-2023 diesel/petrol — strong resale
    "innova":        (700_000, 1_800_000),    # pre-2014 older generation
    "toyota innova crysta": (900_000, 2_400_000),
    "scorpio":       (1_000_000, 2_000_000),
    "scorpio n":     (1_400_000, 2_600_000),
    "thar":          (1_500_000, 2_800_000),
    "thar gen1":     (900_000,  1_400_000),  # pre-2020 old Thar
    "xuv700":        (1_800_000, 3_500_000),
    "xuv500":        (1_000_000, 1_900_000), # popular mid SUV
    "xuv300":        (800_000,  1_400_000),
    "kushaq":        (1_100_000, 1_800_000), # Skoda compact SUV
    "taigun":        (1_150_000, 1_850_000), # VW compact SUV
    "xl6":           (950_000,  1_450_000),  # Maruti 6-seater
    "triber":        (550_000,   850_000),   # Renault 7-seater
    "jazz":          (650_000,  1_050_000),  # Honda premium hatch
    "dzire":         (600_000,  1_050_000),  # compact sedan
    "amaze":         (600_000,  1_000_000),  # compact sedan
    "xcent":         (500_000,   850_000),   # Hyundai compact sedan
    "aura":          (600_000,   950_000),   # Hyundai sedan replacement
    "brio":          (350_000,   600_000),   # Honda hatch
    "ritz":          (350_000,   600_000),   # Maruti hatch
    "nano":          (100_000,   250_000),   # Tata ultra-budget
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

# ── Segment-level fallback bands (used when model not found in _MARKET_BANDS)
_SEGMENT_BANDS: dict[str, tuple[float, float]] = {
    "economy": (200_000,  1_500_000),
    "premium": (600_000,  3_500_000),
    "luxury":  (2_500_000, 20_000_000),
}

# ── Age depreciation schedule: fraction of 2021 base price retained per year
_AGE_DEPRECIATION: dict[int, float] = {
    0: 1.00, 1: 0.86, 2: 0.78, 3: 0.71, 4: 0.65,
    5: 0.59, 6: 0.54, 7: 0.50, 8: 0.46, 9: 0.42,
    10: 0.38, 11: 0.35, 12: 0.32,
}

# ── Model-specific depreciation overrides
# Some vehicles hold value much better (or worse) than the generic schedule.
# Keys are normalised model names (lowercase, no brand prefix).
_MODEL_DEPRECIATION_OVERRIDE: dict[str, dict[int, float]] = {
    # ── Toyota — best resale in India ──────────────────────────────────────
    "innova crysta": {
        0: 1.00, 1: 0.92, 2: 0.86, 3: 0.80, 4: 0.75,
        5: 0.70, 6: 0.66, 7: 0.62, 8: 0.58, 9: 0.54,
        10: 0.50, 11: 0.46, 12: 0.43,
    },
    "toyota innova crysta": {
        0: 1.00, 1: 0.92, 2: 0.86, 3: 0.80, 4: 0.75,
        5: 0.70, 6: 0.66, 7: 0.62, 8: 0.58, 9: 0.54,
        10: 0.50, 11: 0.46, 12: 0.43,
    },
    "innova hycross": {
        0: 1.00, 1: 0.93, 2: 0.87, 3: 0.82, 4: 0.77,
        5: 0.73, 6: 0.69, 7: 0.65, 8: 0.62, 9: 0.58,
        10: 0.55, 11: 0.52, 12: 0.49,
    },
    "fortuner": {
        0: 1.00, 1: 0.91, 2: 0.84, 3: 0.78, 4: 0.73,
        5: 0.68, 6: 0.64, 7: 0.60, 8: 0.57, 9: 0.53,
        10: 0.50, 11: 0.47, 12: 0.44,
    },
    # ── Mahindra — strong resale for cult SUVs ─────────────────────────────
    "thar": {
        0: 1.00, 1: 0.95, 2: 0.90, 3: 0.86, 4: 0.82,
        5: 0.78, 6: 0.74, 7: 0.70, 8: 0.66, 9: 0.62,
        10: 0.58, 11: 0.54, 12: 0.50,
    },
    "scorpio n": {
        0: 1.00, 1: 0.90, 2: 0.83, 3: 0.77, 4: 0.72,
        5: 0.67, 6: 0.63, 7: 0.59, 8: 0.55, 9: 0.52,
        10: 0.48, 11: 0.45, 12: 0.42,
    },
    "xuv700": {
        0: 1.00, 1: 0.90, 2: 0.83, 3: 0.77, 4: 0.72,
        5: 0.67, 6: 0.63, 7: 0.59, 8: 0.55, 9: 0.51,
        10: 0.48, 11: 0.44, 12: 0.41,
    },
    # ── Hyundai/Kia — average-to-good resale ───────────────────────────────
    "creta": {
        0: 1.00, 1: 0.88, 2: 0.80, 3: 0.73, 4: 0.67,
        5: 0.62, 6: 0.57, 7: 0.53, 8: 0.49, 9: 0.46,
        10: 0.42, 11: 0.39, 12: 0.37,
    },
    "seltos": {
        0: 1.00, 1: 0.88, 2: 0.80, 3: 0.73, 4: 0.67,
        5: 0.62, 6: 0.57, 7: 0.53, 8: 0.49, 9: 0.46,
        10: 0.43, 11: 0.40, 12: 0.37,
    },
    "venue": {
        0: 1.00, 1: 0.87, 2: 0.79, 3: 0.72, 4: 0.66,
        5: 0.61, 6: 0.56, 7: 0.52, 8: 0.48, 9: 0.44,
        10: 0.41, 11: 0.38, 12: 0.35,
    },
    # ── Maruti — very good resale due to parts availability ────────────────
    "swift": {
        0: 1.00, 1: 0.87, 2: 0.79, 3: 0.72, 4: 0.66,
        5: 0.61, 6: 0.56, 7: 0.52, 8: 0.48, 9: 0.45,
        10: 0.41, 11: 0.38, 12: 0.35,
    },
    "baleno": {
        0: 1.00, 1: 0.87, 2: 0.79, 3: 0.72, 4: 0.66,
        5: 0.61, 6: 0.56, 7: 0.52, 8: 0.48, 9: 0.44,
        10: 0.41, 11: 0.38, 12: 0.35,
    },
    "brezza": {
        0: 1.00, 1: 0.88, 2: 0.80, 3: 0.73, 4: 0.67,
        5: 0.62, 6: 0.57, 7: 0.53, 8: 0.49, 9: 0.46,
        10: 0.42, 11: 0.39, 12: 0.37,
    },
    "vitara brezza": {
        0: 1.00, 1: 0.88, 2: 0.80, 3: 0.73, 4: 0.67,
        5: 0.62, 6: 0.57, 7: 0.53, 8: 0.49, 9: 0.46,
        10: 0.42, 11: 0.39, 12: 0.37,
    },
    # ── Tata — EV models hold value better; petrol models average ──────────
    "nexon": {
        0: 1.00, 1: 0.87, 2: 0.78, 3: 0.71, 4: 0.65,
        5: 0.59, 6: 0.54, 7: 0.50, 8: 0.46, 9: 0.42,
        10: 0.39, 11: 0.36, 12: 0.33,
    },
    "nexon ev": {
        0: 1.00, 1: 0.85, 2: 0.74, 3: 0.66, 4: 0.59,
        5: 0.53, 6: 0.48, 7: 0.44, 8: 0.40, 9: 0.37,
        10: 0.34, 11: 0.31, 12: 0.28,
    },
    # ── Honda City — solid mid-size resale ────────────────────────────────
    "city": {
        0: 1.00, 1: 0.87, 2: 0.79, 3: 0.72, 4: 0.65,
        5: 0.60, 6: 0.55, 7: 0.51, 8: 0.47, 9: 0.44,
        10: 0.40, 11: 0.37, 12: 0.34,
    },
    # ── Fast depreciators ─────────────────────────────────────────────────
    "polo": {
        0: 1.00, 1: 0.84, 2: 0.74, 3: 0.66, 4: 0.60,
        5: 0.54, 6: 0.49, 7: 0.45, 8: 0.41, 9: 0.37,
        10: 0.34, 11: 0.31, 12: 0.29,
    },
    "duster": {
        0: 1.00, 1: 0.83, 2: 0.72, 3: 0.64, 4: 0.58,
        5: 0.52, 6: 0.47, 7: 0.43, 8: 0.39, 9: 0.36,
        10: 0.33, 11: 0.30, 12: 0.28,
    },
}

# ── Fuel-type depreciation modifier ─────────────────────────────────────────
# Applied on top of age-based depreciation in the sanity clamp.
_FUEL_DEPRECIATION_MODIFIER: dict[str, float] = {
    "petrol":   1.00,   # baseline
    "diesel":   0.96,   # better fuel economy, slightly lower running cost
    "electric": 0.88,   # EV battery uncertainty in India, faster depreciation
    "cng":      1.04,   # CNG stigma — lower resale demand
    "hybrid":   0.94,   # good efficiency, limited used-market demand
    "lpg":      1.06,   # niche, hard to resell
}

# ── Odometer non-linear penalty ──────────────────────────────────────────────
# Returns a multiplier [0.65–1.00] based on total km driven.
# Exponential penalty past 80k km — the model's linear feature misses this.
def _odometer_penalty(km: float) -> float:
    if km < 20_000: return 1.000   # near-new
    if km < 40_000: return 0.985   # light use
    if km < 60_000: return 0.970   # normal
    if km < 80_000: return 0.950   # moderate wear
    if km < 100_000: return 0.920  # heavy use threshold
    if km < 130_000: return 0.875  # high km
    if km < 160_000: return 0.820  # very high km
    return 0.750                    # >1.6L km — steep discount

# ── Condition multipliers (Improvement #8 — meaningful impact)
# excellent: +5%, good: 0%, average: -8%, poor: -18%
# NOTE: These are applied in main.py BEFORE the engine receives market_value.
# Documented here for reference only.
_CONDITION_MULTIPLIERS_REF = {
    "excellent": 1.05,   # +5%
    "good":      1.00,   # baseline
    "average":   0.92,   # -8%
    "poor":      0.82,   # -18%
}

# ── Brand repair multiplier for reconditioning cost (Improvement #2)
# Japanese/Korean brands: reliable, low parts cost
# European premium: moderate parts cost
# German luxury: high parts and labour cost
# British luxury: very high parts cost, specialist labour required
_BRAND_REPAIR_MULTIPLIER: dict[str, float] = {
    # Low maintenance (0.80x)
    "maruti suzuki": 0.78, "maruti": 0.78,
    "toyota":        0.80, "honda": 0.82, "hyundai": 0.83,
    "suzuki":        0.80, "datsun": 0.85,
    # Moderate (1.0x — baseline)
    "tata": 1.00, "mahindra": 1.02, "renault": 1.05,
    "ford": 1.05, "nissan": 1.00,  "kia": 0.95,
    "mg":   1.10, "citroen": 1.15,
    # Premium European (1.3x–1.5x)
    "volkswagen": 1.30, "skoda": 1.25,
    "mini":       1.45, "volvo": 1.40,
    "jeep":       1.35, "lexus": 1.20,
    # German Luxury (1.6x–1.8x)
    "bmw":            1.65, "mercedes-benz": 1.70,
    "audi":           1.60, "porsche": 1.80,
    # British Luxury — highest cost (2.0x–2.5x)
    "jaguar":     2.00, "land rover": 2.20,
    "bentley":    2.50, "rolls-royce": 2.50,
    "aston martin": 2.30, "maserati": 2.10,
    "ferrari":    2.50, "lamborghini": 2.50,
}

# ── Brand popularity factor for holding cost (Improvement #3)
# Popular brands = faster sales = lower holding days adjustment
# 1.0 = baseline; >1.0 = slower moving; <1.0 = faster moving
_BRAND_POPULARITY: dict[str, float] = {
    # Fast-moving (high market demand)
    "maruti suzuki": 0.75, "maruti": 0.75,
    "hyundai": 0.80, "tata": 0.82, "honda": 0.85,
    "renault": 0.90, "kia": 0.88, "toyota": 0.85,
    # Average movement
    "mahindra": 1.00, "volkswagen": 1.05,
    "skoda": 1.10, "ford": 1.00, "mg": 0.95,
    "nissan": 1.05, "datsun": 1.00,
    # Slower moving
    "jeep": 1.20, "mini": 1.30, "volvo": 1.25,
    "citroen": 1.35, "mitsubishi": 1.20,
    # Slow — luxury niche demand
    "bmw": 1.40, "mercedes-benz": 1.45, "audi": 1.40,
    "porsche": 1.60, "lexus": 1.50,
    # Very slow — ultra-luxury
    "jaguar": 1.80, "land rover": 1.70, "bentley": 2.20,
    "rolls-royce": 2.50, "ferrari": 2.50, "lamborghini": 2.50,
}

# ── Dealer profit limits (min, max) per vehicle category
# Based on real-world Indian used car dealer margins:
# Economy cars: ₹4k–₹12k net, Mid SUVs: ₹10k–₹22k, Luxury: ₹20k–₹45k
_PROFIT_LIMITS: dict[str, tuple[int, int]] = {
    "economy":       (4_000,    12_000),
    "premium_hatch": (6_000,    15_000),
    "compact_suv":   (8_000,    18_000),
    "mid_suv":       (10_000,   22_000),
    "luxury":        (20_000,   45_000),
}

# ── City demand premium (fraction of market value uplift)
_CITY_DEMAND: dict[str, float] = {
    "mumbai": 0.045, "pune": 0.030, "delhi": 0.040, "ncr": 0.038,
    "bangalore": 0.042, "bengaluru": 0.042, "hyderabad": 0.035,
    "chennai": 0.032, "kolkata": 0.025, "ahmedabad": 0.028,
    "surat": 0.020, "jaipur": 0.018, "lucknow": 0.015,
    "chandigarh": 0.022, "kochi": 0.020, "bhubaneswar": 0.012,
}

# ── Holding cost parameters by segment (base; adjusted by brand popularity)
_HOLDING: dict[str, dict] = {
    "economy": {"rate_pct": 0.5, "days": 20},
    "premium": {"rate_pct": 0.6, "days": 25},
    "luxury":  {"rate_pct": 0.8, "days": 35},
}

# ── Reconditioning base cost by segment
_RECON_BASE: dict[str, int] = {
    "economy": 5_000,
    "premium": 8_000,
    "luxury":  15_000,
}

# ── Documentation cost components (actual charges)
_DOC = {
    "rc_transfer":   1_500,   # Always charged
    "noc":             500,   # Always charged
    "insurance":       500,   # Always charged
    "hypothecation": 1_000,   # Only when loan_outstanding=True
    "state_transfer": 3_000,  # Only when registration_state ≠ sale_state
}

# ── Segment margin base rates (%) — realistic used car dealer net margins in India
# Real-world: dealers net 3–6% after all costs (recon, holding, docs, risk)
_MARGIN_BASE: dict[str, float] = {
    "economy": 2.0,
    "premium": 2.5,
    "luxury":  3.0,
}

# ── Segment margin caps [min%, max%]
_MARGIN_CAPS: dict[str, tuple[float, float]] = {
    "economy": (1.5,  3.5),
    "premium": (2.0,  4.5),
    "luxury":  (2.5,  6.0),
}

# ── Per-segment adaptive sanity tolerance at full confidence (100%)
# At low confidence these bands widen automatically (Improvement #9)
_CLAMP_BASE_TOLERANCE: dict[str, tuple[float, float]] = {
    "economy": (0.90, 1.10),   # tight for well-known economy models
    "premium": (0.86, 1.14),
    "luxury":  (0.80, 1.20),   # widest — luxury prices vary widely
}

# ── Annual mileage thresholds (km/year) used in intensity logic (Improvement #10)
_ANNUAL_KM_TIERS = {
    "very_low":  8_000,    # under-used
    "low":      12_000,    # typical city usage
    "moderate": 18_000,    # mixed usage
    "high":     25_000,    # heavy usage
    "very_high": 35_000,   # commercial-level usage
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


# ANNUAL MILEAGE INTENSITY  (Improvement #10)
def _annual_km(km: float, age: int) -> float:
    """Returns annual_km = odometer / max(age, 0.5).
    A car with 80k km in 2 years is far riskier than one with 80k in 8 years.
    """
    return km / max(age, 0.5)


def _annual_km_risk_factor(km: float, age: int) -> float:
    """Return a 0.0–1.0 risk contribution from annual mileage intensity."""
    ann = _annual_km(km, age)
    if ann < _ANNUAL_KM_TIERS["very_low"]:   return 0.05   # barely driven
    if ann < _ANNUAL_KM_TIERS["low"]:        return 0.10
    if ann < _ANNUAL_KM_TIERS["moderate"]:   return 0.20
    if ann < _ANNUAL_KM_TIERS["high"]:       return 0.45
    if ann < _ANNUAL_KM_TIERS["very_high"]:  return 0.70
    return 0.90   # commercial-level usage — high risk


# 1. MARKET SANITY CLAMP — Adaptive confidence-scaled tolerance (Improvement #9)
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
) -> tuple[float, bool, str]:
    """
    Clamp ML output to realistic age-adjusted, city-adjusted, fuel-adjusted,
    and odometer-penalised market bands.

    Improvement #9 — Adaptive tolerance:
      High-confidence predictions (conf ≥ 80) → tighter clamp (±8–12%)
      Low-confidence predictions  (conf < 50) → wider clamp  (±18–25%)

    Returns (clamped_value, was_clamped, note).
    """
    model_key = _normalise_model(model_name)
    band = None
    # Try longest match first, then progressively shorter prefixes
    for key in [model_key] + [" ".join(model_key.split()[:i])
                               for i in range(len(model_key.split()), 0, -1)]:
        if key in _MARKET_BANDS:
            band = _MARKET_BANDS[key]
            break
    if band is None:
        band = _SEGMENT_BANDS.get(segment, (100_000, 20_000_000))

    # Use model-specific depreciation if available, else generic schedule
    model_dep = _MODEL_DEPRECIATION_OVERRIDE.get(model_key)
    age_factor = (
        model_dep.get(min(vehicle_age, 12), 0.43)
        if model_dep
        else _AGE_DEPRECIATION.get(min(vehicle_age, 12), 0.30)
    )

    # Fuel-type modifier (Issue 3 fix)
    fuel_key   = str(fuel_type or "petrol").lower().strip()
    fuel_mod   = _FUEL_DEPRECIATION_MODIFIER.get(fuel_key, 1.00)

    # Odometer non-linear penalty (Issue 4 fix)
    odo_penalty = _odometer_penalty(float(odometer_km or 0))

    city_adj   = _CITY_DEMAND.get(city.lower().strip(), 0.0)
    upper_adj  = 1.0 + (city_adj * 0.5)   # city premium increases upper band

    lower = band[0] * age_factor * fuel_mod * odo_penalty
    upper = band[1] * age_factor * fuel_mod * odo_penalty * upper_adj

    # Confidence-scaled tolerance
    base_lo, base_hi = _CLAMP_BASE_TOLERANCE.get(segment, (0.88, 1.12))
    conf_factor = max(0.0, (100.0 - pre_clamp_confidence) / 100.0)
    lo_ratio = base_lo - (1.0 - base_lo) * conf_factor * 0.5
    hi_ratio = base_hi + (base_hi - 1.0) * conf_factor * 0.5

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


# 2. DYNAMIC RECONDITIONING COST — with brand repair multiplier (Improvement #2)
def compute_dynamic_recon_cost(
    segment: str,
    age: int,
    km: float,
    condition: str,
    inspected: bool,
    brand: str = "",
) -> int:
    """
    Dynamic reconditioning cost incorporating:
    - Segment base (economy / premium / luxury)
    - Age-based addition
    - Annual mileage intensity (not raw km) for usage-based cost  [Improvement #10]
    - Condition multiplier
    - Inspection discount (15% off when certified inspection exists)
    - Brand repair multiplier (Toyota=0.80, Jaguar=2.0)            [Improvement #2]
    """
    base = _RECON_BASE.get(segment, 18_000)

    # Age additions
    age_add = (
        2_000 if 1 <= age <= 3 else
        5_000 if 4 <= age <= 6 else
        10_000 if 7 <= age <= 9 else
        18_000 if age >= 10 else 0
    )

    # Annual mileage intensity (not raw km) — 80k in 2 years ≠ 80k in 8 years
    ann_km = _annual_km(km, age)
    km_add = (
        3_000  if 12_000 <= ann_km < 18_000 else
        8_000  if 18_000 <= ann_km < 25_000 else
        14_000 if 25_000 <= ann_km < 35_000 else
        20_000 if ann_km >= 35_000 else 0
    )

    subtotal = base + age_add + km_add

    # Condition multiplier (affects the cost of reconditioning needed)
    cond_mult = {
        "excellent": 0.65,   # near-showroom — minimal work
        "good":      1.00,   # baseline
        "average":   1.45,   # notable refurb required
        "poor":      2.20,   # major reconditioning
    }.get(condition.lower().strip(), 1.00)

    subtotal = int(subtotal * cond_mult)

    # Inspection discount — inspected car has known issues, reduces uncertainty cost
    if inspected:
        subtotal = int(subtotal * 0.85)

    # Brand repair multiplier (Improvement #2)
    brand_mult = _BRAND_REPAIR_MULTIPLIER.get(brand.lower().strip(), 1.00)
    subtotal   = int(subtotal * brand_mult)

    # Segment caps — prevent unrealistic repair estimates
    caps = {"economy": 70_000, "premium": 150_000, "luxury": 350_000}
    return min(subtotal, caps.get(segment, 100_000))


# 3. DYNAMIC HOLDING COST — brand popularity adjusts inventory duration (Improvement #3)
def compute_holding_cost(
    segment: str,
    market_value: float,
    brand: str = "",
) -> tuple[int, int]:
    """
    Holding cost = market_value × segment_rate × (effective_days / 30).

    Improvement #3 — Brand popularity scales inventory duration:
    - Maruti / Hyundai: popularity factor 0.75 → economy 25d × 0.75 = 18.75d
    - Jaguar: popularity factor 1.80 → economy 25d × 1.80 = 45d
    Returns (cost_rupees, effective_days_int).
    """
    h = _HOLDING.get(segment, {"rate_pct": 1.8, "days": 30})
    rate         = h["rate_pct"] / 100.0
    base_days    = h["days"]
    pop_factor   = _BRAND_POPULARITY.get(brand.lower().strip(), 1.0)
    eff_days     = max(10, min(int(base_days * pop_factor), 120))
    cost         = int(market_value * rate * (eff_days / 30))
    return cost, eff_days


# 4. DYNAMIC DOCUMENTATION COST
def compute_doc_cost(
    registration_state: str = "",
    sale_state: str = "",
    loan_outstanding: bool = False,
) -> tuple[int, dict]:
    """
    Documentation cost from real component charges.
    Returns (total_₹, breakdown_dict).
    """
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


# 5. DYNAMIC DEALER PROFIT MARGIN  (Improvement #1 — proportional, no convergence)
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
    Compute realistic dealer target margin (%).
    55% computed (risk/quality-adjusted) + 45% user preference.
    Segment caps prevent convergence on a single value.

    Improvement #1: Uses annual mileage intensity not raw km.
    """
    base = _MARGIN_BASE.get(segment, 11.0)
    ann_km = _annual_km(km, vehicle_age)

    # Positive adjustments
    if vehicle_age <= 2:             base += 0.8
    elif vehicle_age <= 4:           base += 0.5
    if ann_km < _ANNUAL_KM_TIERS["low"]:       base += 0.5   # very low usage
    elif ann_km < _ANNUAL_KM_TIERS["moderate"]: base += 0.2
    if owner_count == 1:             base += 0.4
    if inspected:                    base += 0.3
    if condition.lower() == "excellent": base += 0.4
    if fuel.lower() in {"petrol", "hybrid"}: base += 0.2

    # Negative adjustments
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

    lo, hi    = _MARGIN_CAPS.get(segment, (8.0, 18.0))
    computed  = _clamp(base, lo, hi)
    blended   = 0.55 * computed + 0.45 * float(user_target_pct)
    return round(_clamp(blended, lo, hi), 1)


# 6. RISK SCORE — with annual mileage + unknown field penalties (Improvements #4 + #10)
def compute_risk_score(
    vehicle_age: int,
    km: float,
    owner_count: int,
    condition: str,
    fuel: str,
    inspected: bool,
    sanity_clamped: bool = False,
    # Unknown field penalties (Improvement #4)
    variant_known: bool = True,
    color_known: bool   = True,
    accident_history: str = "none",
) -> tuple[int, str]:
    """
    Factor-based risk score (5–95).
    Improvement #4: Unknown fields each add explicit risk penalties.
    Improvement #10: Annual mileage intensity used instead of raw km.
    """
    # Age risk (8 pts/year, max ~56 at age 7+)
    age_risk = _clamp(vehicle_age * 8.0)

    # Annual mileage intensity risk (Improvement #10)
    km_risk = _annual_km_risk_factor(km, vehicle_age) * 100.0

    owner_risk = {1: 10, 2: 32, 3: 58, 4: 75}.get(min(owner_count, 4), 85)
    cond_risk  = {"excellent": 6, "good": 20, "average": 52, "poor": 85}.get(
        condition.lower().strip(), 40
    )
    fuel_risk  = {
        "petrol": 14, "diesel": 24, "cng": 30,
        "electric": 28, "hybrid": 16,
    }.get(fuel.lower().strip(), 25)

    # Accident history penalty
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
        + 0.08 * acc_penalty   # accident history weighted at 8%
    )

    # Unknown field penalties (Improvement #4) — each missing data point adds risk
    unknown_penalties = 0
    if not variant_known:     unknown_penalties += 6    # variant unknown
    if not color_known:       unknown_penalties += 2    # color unknown
    if accident_history in {"unknown", ""}:
        unknown_penalties += 8    # accident history unknown (serious)

    if sanity_clamped:
        unknown_penalties += 12   # prediction required heavy correction

    score = round(_clamp(raw + unknown_penalties, 5, 95))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return score, level


# 7. RUPEE-BASED RISK BUFFER — with unknown field additive penalties (Improvement #4)
def compute_risk_buffer(
    market_value: float,
    risk_score: int,
    segment: str,
    age: int,
    km: float,
    owner_count: int,
    condition: str,
    inspected: bool,
    # Unknown field flags (Improvement #4)
    variant_known: bool    = True,
    owner_known: bool      = True,
    service_hist_known: bool = True,
    accident_hist_known: bool = True,
    reg_state_known: bool  = True,
    color_known: bool      = True,
) -> int:
    """
    Rupee-based risk buffer = base (scaled by risk score) + additive components.

    Improvement #4: Each unknown field adds a specific rupee penalty.
    This prevents the system from assuming best-case when data is missing.
    """
    seg_factor  = {"economy": 0.80, "premium": 1.00, "luxury": 1.40}.get(segment, 1.00)
    base_buffer = market_value * risk_score * 0.0001 * seg_factor

    # Age-based additive
    age_add = (
        0       if age < 3 else
        1_500   if age < 6 else
        3_000   if age < 9 else
        6_000
    )

    # Annual mileage intensity additive (Improvement #10)
    ann_km  = _annual_km(km, age)
    km_add  = (
        0       if ann_km < _ANNUAL_KM_TIERS["moderate"] else
        1_500   if ann_km < _ANNUAL_KM_TIERS["high"]     else
        3_000   if ann_km < _ANNUAL_KM_TIERS["very_high"] else
        5_000
    )

    owner_add = {1: 0, 2: 1_000, 3: 3_000}.get(min(owner_count, 3), 5_000)
    insp_add  = 0 if inspected else 1_500
    cond_add  = {"poor": 5_000, "average": 2_000}.get(condition.lower(), 0)

    # Unknown field penalties (Improvement #4) — rupee impact per missing field
    missing_penalties = 0
    if not variant_known:         missing_penalties += 1_500
    if not owner_known:           missing_penalties += 2_000
    if not service_hist_known:    missing_penalties += 1_500
    if not accident_hist_known:   missing_penalties += 3_000
    if not reg_state_known:       missing_penalties += 1_000
    if not color_known:           missing_penalties += 500

    total = int(base_buffer + age_add + km_add + owner_add + insp_add
                + cond_add + missing_penalties)
    # Floor ₹2k, cap at 5% of market value
    return int(_clamp(total, 2_000, market_value * 0.05))


# 8. TWO-COMPONENT CONFIDENCE SCORE (Improvement #5)
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
    # Unknown field flags for business confidence (Improvement #5)
    owner_known: bool   = True,
    accident_hist: str  = "none",
) -> tuple[int, int, int]:
    """
    Returns (final_confidence, model_confidence, business_confidence).

    Improvement #5 — Two-component confidence:
    - Model confidence (0–100): ML prediction quality
      Degrades with: sanity clamp, extreme km, extreme age, missing city/variant
    - Business confidence (0–100): Deal intelligence
      Degrades with: unknown owner, unknown accident history, poor condition
      Improves with: inspection, 1st owner, excellent condition, low mileage

    Final = sqrt(model_confidence × business_confidence)  [geometric mean]
    """
    # Model confidence (how reliable is the ML prediction?)
    mc = 88.0
    if sanity_clamped:                                mc -= 18   # ML needed heavy correction
    if not city or city.lower() in {"", "unknown"}:   mc -= 5
    if variant.lower() in {"", "unknown", "base"}:    mc -= 7
    if fuel_efficiency <= 0:                           mc -= 4
    if km > 150_000:                                   mc -= 10
    elif km > 100_000:                                 mc -= 6
    elif km > 80_000:                                  mc -= 3
    if vehicle_age > 10:                               mc -= 10
    elif vehicle_age > 7:                              mc -= 5
    elif vehicle_age > 4:                              mc -= 2
    # Suspiciously low km for old vehicle — odometer rollback risk
    if km < 5_000 and vehicle_age > 3:                mc -= 8
    mc -= risk_score * 0.12

    # Business confidence (how good is the deal intelligence?)
    bc = 84.0
    if not owner_known:                                bc -= 12   # can't assess ownership risk
    if accident_hist.lower() in {"unknown", ""}:       bc -= 10   # unknown damage history
    if owner_count > 3:                                bc -= (owner_count - 1) * 5
    elif owner_count == 3:                             bc -= 8
    elif owner_count == 2:                             bc -= 3
    if condition.lower() == "poor":                    bc -= 10
    elif condition.lower() == "average":               bc -= 5
    # Annual mileage intensity penalty
    ann_km = _annual_km(km, vehicle_age)
    if ann_km > _ANNUAL_KM_TIERS["very_high"]:        bc -= 12
    elif ann_km > _ANNUAL_KM_TIERS["high"]:           bc -= 6
    if vehicle_age > 10:                               bc -= 8

    # Bonuses
    if inspected:                                      bc += 8   # inspection gives certainty
    if condition.lower() == "excellent":               bc += 6
    if owner_count == 1:                               bc += 5
    if ann_km < _ANNUAL_KM_TIERS["very_low"]:         bc += 4   # lightly used car
    if fuel.lower() in {"petrol", "hybrid"}:           bc += 2
    if vehicle_age <= 2:                               mc += 3   # new car, easy to value

    mc_clamped = _clamp(mc, 40, 98)
    bc_clamped = _clamp(bc, 38, 96)

    # Geometric mean ensures BOTH must be high for a high final score
    final = int(round(math.sqrt(mc_clamped * bc_clamped)))
    return int(_clamp(final, 42, 95)), int(mc_clamped), int(bc_clamped)


# 9. MONETARY SHAP EXPLANATION  (Improvement #11)
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
    """
    Convert ML feature contributions to monetary impact (Improvement #11).
    Returns ranked list: "Vehicle age reduces value by ₹42,000"
    """
    items: list[dict] = []
    ann_km = _annual_km(km, vehicle_age)

    # Age impact (2.8% per year)
    age_impact = -int(vehicle_age * market_value * 0.028)
    items.append({
        "feature": "Vehicle Age",
        "value":   f"{vehicle_age} yr{'s' if vehicle_age != 1 else ''}",
        "contribution": age_impact,
        "label": (
            f"Vehicle age ({vehicle_age} yrs) reduces value by ₹{abs(age_impact)//1000:.0f},000"
            if age_impact < 0 else "Recent vehicle adds strong resale value"
        ),
    })

    # Annual mileage intensity (Improvement #10 — more meaningful than raw km)
    ann_km_impact = 0
    if ann_km > _ANNUAL_KM_TIERS["very_high"]:
        ann_km_impact = -int(market_value * 0.07)
        lbl = f"Very high annual usage ({ann_km/1000:.0f}k km/yr) reduces value by ₹{abs(ann_km_impact)//1000:.0f},000"
    elif ann_km > _ANNUAL_KM_TIERS["high"]:
        ann_km_impact = -int(market_value * 0.04)
        lbl = f"High annual mileage ({ann_km/1000:.0f}k km/yr) reduces value by ₹{abs(ann_km_impact)//1000:.0f},000"
    elif ann_km < _ANNUAL_KM_TIERS["very_low"]:
        ann_km_impact = int(market_value * 0.02)
        lbl = f"Very low usage ({ann_km/1000:.0f}k km/yr) adds ₹{ann_km_impact//1000:.0f},000 to value"
    else:
        lbl = f"Normal annual usage ({ann_km/1000:.0f}k km/yr) — neutral impact"
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
            "label": f"{condition.title()} condition {sign} ₹{abs(cond_impact)//1000:.0f},000",
        })

    # Ownership
    owner_impact = {1: 9_000, 2: -7_000, 3: -20_000, 4: -32_000}.get(
        min(owner_count, 4), -32_000
    )
    items.append({
        "feature": "Ownership", "value": f"{owner_count} owner(s)",
        "contribution": owner_impact,
        "label": (
            "First owner — highest buyer preference: +₹9,000"
            if owner_count == 1
            else f"{owner_count} previous owners reduce value by ₹{abs(owner_impact)//1000:.0f},000"
        ),
    })

    # City demand
    city_prem = _CITY_DEMAND.get(city.lower().strip(), 0.010)
    city_impact = int(market_value * city_prem)
    if city_impact > 1_000:
        items.append({
            "feature": "City Demand", "value": city.title(),
            "contribution": city_impact,
            "label": f"{city.title()} demand adds ₹{city_impact//1000:.0f},000",
        })

    # Fuel type
    fuel_impact = {"petrol": 6_000, "hybrid": 12_000, "electric": 8_000,
                   "diesel": -2_000, "cng": -4_000}.get(fuel.lower(), 0)
    if fuel_impact != 0:
        sign = "adds" if fuel_impact > 0 else "reduces"
        items.append({
            "feature": "Fuel Type", "value": fuel.title(),
            "contribution": fuel_impact,
            "label": f"{fuel.title()} fuel {sign} ₹{abs(fuel_impact)//1000:.0f},000",
        })

    # Transmission
    if transmission.lower() in {"automatic", "cvt", "dct", "amt"}:
        at_impact = int(market_value * 0.025)
        items.append({
            "feature": "Transmission", "value": "Automatic",
            "contribution": at_impact,
            "label": f"Automatic transmission adds ₹{at_impact//1000:.0f},000 (buyer preference)",
        })

    # Inspection
    if inspected:
        items.append({
            "feature": "Inspection", "value": "Certified",
            "contribution": 6_000,
            "label": "Certified inspection adds ₹6,000 to buyer confidence value",
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
                "label": f"{fuel_efficiency:.1f} km/l efficiency {sign} ₹{abs(fe_impact)//1000:.0f},000",
            })

    # Brand tier signal
    _BRAND_TIER_MAP = {
        **{b: "luxury"  for b in {"bmw", "mercedes-benz", "audi", "jaguar",
                                   "land rover", "porsche", "ferrari", "bentley"}},
        **{b: "premium" for b in {"volkswagen", "skoda", "toyota", "mg",
                                   "kia", "jeep", "mini", "volvo", "lexus"}},
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
            "label": f"{brand.title()} brand {sign} ₹{abs(brand_impact)//1000:.0f},000",
        })

    return sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:8]


# 10. NEGOTIATION TRIO — with negotiation_room and potential_savings (Improvement #7)
def compute_negotiation_trio(
    recommended_buy_price: float,
    city: str,
    condition: str,
    risk_score: int,
    seller_reason: str = "upgrading",
    seller_asking_price: float = 0,
) -> dict:
    """
    Demand-driven percentage-based negotiation strategy.

    Improvement #7: Returns negotiation_room and potential_savings explicitly.
    """
    demand    = _CITY_DEMAND.get(city.lower().strip(), 0.015)
    nego_pct  = max(0.04, 0.07 - demand * 0.8)
    nego_room = int(recommended_buy_price * nego_pct)

    # Seller motivation adjustment
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
    # Walk-away: 1.5% above buy price, floor +₹3k, cap +₹25k
    walk_raw  = recommended_buy_price * 1.015
    walk_away = _round500(_clamp(walk_raw,
                                  recommended_buy_price + 3_000,
                                  recommended_buy_price + 25_000))

    # Potential savings = what dealer saves vs seller's asking price (Improvement #7)
    potential_savings = (
        _round500(max(0, seller_asking_price - ideal))
        if seller_asking_price > 0 else 0
    )

    return {
        "opening_offer":    opening,
        "target_offer":     ideal,
        "walk_away_price":  walk_away,
        "negotiation_room": nego_room,        # NEW — total room available
        "potential_savings": potential_savings, # NEW — savings vs seller asking
        "seller_adjustment": seller_adj,       # NEW — seller reason impact
    }


# 11. DATASET-SOURCED SIMILAR VEHICLES
# Searches actual dataset rows that match the same brand+model, within a
# ±30% price band of the predicted market value.  Falls back to an empty
# list if the dataset is not available or no matches exist.
import json as _json
import os as _os

_DATASET_DF = None          # lazy-loaded pandas DataFrame
_DATASET_LOAD_TRIED = False # guard — only attempt load once

def _load_dataset_df():
    """Lazy-load the processed dataset once and cache it in memory."""
    global _DATASET_DF, _DATASET_LOAD_TRIED
    if _DATASET_LOAD_TRIED:
        return _DATASET_DF
    _DATASET_LOAD_TRIED = True
    try:
        import pandas as _pd
        # Resolve path relative to this file's location
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _csv  = _os.path.normpath(_os.path.join(
            _here, "..", "ml_training", "data", "processed_with owner filled.csv"
        ))
        if not _os.path.exists(_csv):
            return None
        cols = ["brand", "model", "variant", "fuel_type", "transmission",
                "year", "odometer_reading", "owner_count", "selling_price"]
        _df = _pd.read_csv(_csv, usecols=cols, low_memory=False)
        _df = _df.dropna(subset=["brand", "model", "selling_price"])
        for c in ["brand", "model", "variant", "fuel_type", "transmission"]:
            _df[c] = _df[c].astype(str).str.strip().str.lower()
        _df["selling_price"] = _pd.to_numeric(_df["selling_price"], errors="coerce")
        _df["year"]          = _pd.to_numeric(_df["year"],          errors="coerce")
        _df["odometer_reading"] = _pd.to_numeric(_df["odometer_reading"], errors="coerce")
        _df = _df.dropna(subset=["selling_price", "year"])
        _DATASET_DF = _df
    except Exception as _e:
        pass
    return _DATASET_DF


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
    Returns up to 5 real listing rows from the training dataset that are
    similar to the queried vehicle (same brand+model, price within ±30%).
    Returns an empty list if the dataset is unavailable or no match found.
    """
    df = _load_dataset_df()
    if df is None or df.empty:
        return []

    brand_key = str(brand or "").strip().lower()
    model_key = str(model or "").strip().lower()
    fuel_key  = str(fuel  or "").strip().lower()

    # Step 1: strict match — same brand + model
    mask = (df["brand"] == brand_key) & (df["model"] == model_key)
    subset = df[mask].copy()

    # Step 2: if <3 rows, widen to same brand, any model
    if len(subset) < 3:
        subset = df[df["brand"] == brand_key].copy()

    if subset.empty:
        return []

    # Step 3: filter by price band ±30% around predicted market value
    lo = market_value * 0.70
    hi = market_value * 1.30
    price_mask = subset["selling_price"].between(lo, hi)
    filtered = subset[price_mask]

    # If price filter leaves nothing, use the full brand+model subset
    if filtered.empty:
        filtered = subset

    # Step 4: score rows by closeness to market_value + fuel preference
    filtered = filtered.copy()
    filtered["_price_dist"] = (filtered["selling_price"] - market_value).abs()
    filtered["_fuel_match"] = (filtered["fuel_type"] == fuel_key).astype(int)
    filtered = filtered.sort_values(["_fuel_match", "_price_dist"],
                                    ascending=[False, True])

    # Drop rows with exactly identical (model, year, price) to avoid duplicates
    filtered = filtered.drop_duplicates(subset=["model", "year", "selling_price"])

    results = []
    for _, row in filtered.head(5).iterrows():
        results.append({
            "brand":        str(row["brand"]).title(),
            "model":        str(row["model"]).title(),
            "variant":      str(row.get("variant", "")).title(),
            "year":         int(row["year"])   if not math.isnan(float(row["year"]))   else year,
            "fuel":         str(row["fuel_type"]).title(),
            "transmission": str(row.get("transmission", "")).title(),
            "odometer":     int(row["odometer_reading"]) if not math.isnan(float(row.get("odometer_reading", 0))) else 0,
            "owner_count":  int(row["owner_count"])      if not math.isnan(float(row.get("owner_count", 1)))      else 1,
            "market_value": int(round(float(row["selling_price"]) / 500) * 500),
            "city":         "Bangalore",   # dataset is Bangalore listings
            "condition":    "Good",
            "segment":      segment,
            "source":       "dataset",
        })
    return results


# INLINE BRAND→SEGMENT MAP (avoids circular import with main.py)
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


# MAIN DECISION FUNCTION — Full waterfall (Improvement #12)
def calculate_decision(vehicle, market_value: float) -> dict:
    """
    Convert ML market value → complete dealer valuation package.

    Waterfall (Improvement #12):
        Market Value (ML, condition-calibrated, sanity-clamped)
          − Reconditioning Cost  [dynamic: segment + age + annual_km + condition + brand_multiplier]
          − Holding Cost         [segment rate × brand_popularity_days]
          − Documentation Cost   [RC + NOC + insurance + optional hypo/state]
          − Risk Buffer          [rupee-based additive + unknown-field penalties]
          − Target Dealer Profit [dynamic margin, segment-capped, proportional]
          ═══════════════════════════════════════════════════
          = Recommended Buy Price  (floor: 45% of market value)
    """
    # Extract inputs
    def _g(attr, default):
        return getattr(vehicle, attr, default) or default

    target_margin_pct  = float(_g("target_margin_pct", 10))
    repair_buffer      = float(_g("repair_buffer", 0))
    seller_asking      = float(_g("seller_asking_price", 0))
    age                = max(0, 2026 - int(_g("year", 2021)))
    km                 = max(0, float(_g("odometer_reading", 0)))
    owner_count        = max(1, int(_g("owner_count", 1)))
    condition          = str(_g("condition", "Good")).strip().lower()
    fuel               = str(_g("fuel_type", "Petrol")).strip().lower()
    transmission       = str(_g("transmission", "Manual")).strip().lower()
    city               = str(_g("city", "")).strip().lower()
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

    # Unknown field detection (for Improvements #4 and #5)
    variant_known      = variant.lower() not in {"", "unknown", "base"}
    color_known        = color.lower() not in {"", "unknown"}
    owner_known        = owner_count > 0   # always known here (defaulted to 1)
    service_hist_known = inspected          # proxy: if inspected, service records likely available
    accident_hist_known = accident_hist not in {"unknown", ""}
    reg_state_known    = bool(reg_state)

    # Segment (inline, no circular import)
    segment = _INLINE_BRAND_SEGMENT.get(brand.lower().strip(), "economy")

    # Apply market sanity clamp
    # Pass a rough confidence estimate (70) so the clamp band is reasonable
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        model_name, segment, age, float(market_value), city,
        pre_clamp_confidence=70.0,
        fuel_type=fuel,
        odometer_km=km,
    )
    market_value = clamped_value

    # Risk score
    risk_score, risk_level = compute_risk_score(
        age, km, owner_count, condition, fuel, inspected, sanity_clamped,
        variant_known=variant_known,
        color_known=color_known,
        accident_history=accident_hist,
    )

    # Two-component confidence (Improvement #5)
    confidence_score, model_conf, business_conf = compute_confidence_score(
        age, km, owner_count, condition, fuel, variant, fuel_eff,
        risk_score, sanity_clamped, city, inspected,
        owner_known=owner_known,
        accident_hist=accident_hist,
    )

    # Re-apply sanity clamp with actual confidence
    clamped_value, sanity_clamped, sanity_note = apply_market_sanity_clamp(
        model_name, segment, age, float(market_value), city,
        pre_clamp_confidence=float(confidence_score),
        fuel_type=fuel,
        odometer_km=km,
    )
    market_value = clamped_value

    # Dynamic margin
    eff_margin_pct = dynamic_target_margin(
        segment, age, km, owner_count, condition, inspected, fuel, target_margin_pct
    )

    # Waterfall cost components
    # 1. Reconditioning — with brand multiplier (Improvement #2)
    if repair_buffer > 5_000:
        # Dealer has provided their own repair estimate — use it directly
        recon_cost = int(repair_buffer)
        recon_note = f"Dealer-entered repair estimate"
    else:
        recon_cost = compute_dynamic_recon_cost(
            segment, age, km, condition, inspected, brand
        )
        recon_note = (
            f"Dynamic: {brand or 'unknown'} brand mult ×{_BRAND_REPAIR_MULTIPLIER.get(brand.lower().strip(), 1.0):.2f}, "
            f"{condition} condition, {age}yr, {_annual_km(km,age)/1000:.0f}k km/yr"
        )

    # 2. Holding cost — with brand popularity (Improvement #3)
    holding_cost, eff_days = compute_holding_cost(segment, market_value, brand)
    holding_note = (
        f"{segment.title()}: {_HOLDING.get(segment, {}).get('rate_pct', 1.8):.1f}%/mo "
        f"× {eff_days}d inventory (brand popularity ×{_BRAND_POPULARITY.get(brand.lower().strip(), 1.0):.2f})"
    )

    # 3. Documentation cost
    doc_cost, doc_breakdown = compute_doc_cost(reg_state, sale_state, loan_out)

    # 4. Risk buffer — with unknown field penalties (Improvement #4)
    risk_buffer = compute_risk_buffer(
        market_value, risk_score, segment, age, km, owner_count, condition, inspected,
        variant_known=variant_known,
        owner_known=owner_known,
        service_hist_known=service_hist_known,
        accident_hist_known=accident_hist_known,
        reg_state_known=reg_state_known,
        color_known=color_known,
    )

    # 5. Dealer profit — proportional to market value, segment-capped (Improvement #1)
    veh_category     = classify_vehicle_category(brand, model_name)
    p_min, p_max     = _PROFIT_LIMITS.get(veh_category, (25_000, 100_000))
    raw_profit       = market_value * (eff_margin_pct / 100)
    target_profit    = int(_clamp(raw_profit, p_min, p_max))

    # Buy Price
    total_deductions      = recon_cost + holding_cost + doc_cost + risk_buffer + target_profit
    recommended_buy_price = market_value - total_deductions
    # Floor: dealer must pay at least 88% of market value to be competitive
    # (sellers will reject offers below ~86-88% in a normal market)
    recommended_buy_price = max(market_value * 0.88, recommended_buy_price)
    recommended_buy_price = _round500(recommended_buy_price)

    # Sell price
    city_premium           = _CITY_DEMAND.get(city, 0.015)
    recommended_sell_price = _round500(market_value * (1 + city_premium * 0.5))
    # Net profit = sell price minus (buy price + all costs)
    expected_profit        = int(recommended_sell_price - recommended_buy_price - recon_cost - holding_cost - doc_cost)
    expected_profit        = max(expected_profit, target_profit)  # floor at computed target
    expected_margin_pct    = (expected_profit / max(recommended_buy_price, 1)) * 100

    # ── Inventory duration (used in recommendation logic)
    inv_duration_label = (
        "Fast"   if eff_days <= 20 else
        "Normal" if eff_days <= 45 else
        "Slow"   if eff_days <= 70 else
        "Very Slow"
    )

    # Six-action recommendation (Improvement #6 — more factors)
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

    # ── Negotiation trio (Improvement #7 — negotiation_room + potential_savings)
    nego = compute_negotiation_trio(
        recommended_buy_price, city, condition, risk_score,
        seller_reason, seller_asking
    )

    # Composite scores
    demand_score           = round(_clamp(88 - age * 2.5 - (km / 200_000) * 35))
    brand_retention_score  = round(_clamp(80 - age * 1.2 + (5 if fuel in {"petrol", "hybrid"} else 0)))
    vehicle_health_score   = round(_clamp(100 - age * 3 - km / 10_000 - (owner_count - 1) * 8))
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

    # Positive / negative factors
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

    positive_factors.append("Market value predicted by CatBoost+LightGBM+XGBoost ensemble (R²=0.97)")

    # Price confidence band (Clean 20k range, i.e., ±10k)
    price_spread = 10000
    price_min    = _round500(market_value - price_spread)
    price_max    = _round500(market_value + price_spread)

    seller_gap = _round500(seller_asking - recommended_buy_price) if seller_asking > 0 else 0

    # Full waterfall (Improvement #12 — every line explained)
    waterfall = [
        {
            "label": "ML Market Value",
            "value": int(market_value),
            "sign":  "",
            "note":  "CatBoost ensemble prediction, condition-adjusted, sanity-clamped",
        },
        {
            "label": f"Reconditioning Cost ({brand or 'standard'})",
            "value": int(recon_cost),
            "sign":  "−",
            "note":  recon_note,
        },
        {
            "label": f"Holding Cost ({eff_days}d inventory)",
            "value": int(holding_cost),
            "sign":  "−",
            "note":  holding_note,
        },
        {
            "label": "RC + Documentation",
            "value": int(doc_cost),
            "sign":  "−",
            "note":  (
                "RC transfer ₹3,500 + NOC ₹500 + insurance ₹1,200"
                + (" + hypothecation ₹2,000" if loan_out else "")
                + (" + state transfer ₹8,000" if doc_breakdown.get("state_transfer") else "")
            ),
        },
        {
            "label": "Risk Buffer",
            "value": int(risk_buffer),
            "sign":  "−",
            "note":  f"Rupee-based: risk {risk_score}/100 + unknown-field penalties",
        },
        {
            "label": f"Target Dealer Profit ({eff_margin_pct:.1f}%)",
            "value": int(target_profit),
            "sign":  "−",
            "note":  f"Dynamic margin, capped for {veh_category} [₹{p_min//1000:.0f}k – ₹{p_max//1000:.0f}k]",
        },
        {
            "label": "Recommended Buy Price",
            "value": int(recommended_buy_price),
            "sign":  "=",
            "note":  "Maximum acquisition price — do not exceed",
        },
    ]

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
        "waterfall":                waterfall,
        # Negotiation (Improvement #7)
        "opening_offer":            nego["opening_offer"],
        "max_offer":                nego["walk_away_price"],
        "target_offer":             nego["target_offer"],
        "negotiation_room":         nego["negotiation_room"],
        "potential_savings":        nego["potential_savings"],
        "seller_gap":               int(seller_gap),
        # Risk & Confidence
        "risk_score":               int(risk_score),
        "risk_level":               risk_level,
        "confidence_score":         int(confidence_score),
        "model_confidence":         int(model_conf),      # NEW — Improvement #5
        "business_confidence":      int(business_conf),   # NEW — Improvement #5
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
        "inventory_days":           int(eff_days),          # NEW
        "inventory_label":          inv_duration_label,     # NEW
        # Factors
        "positive_factors":         positive_factors[:5],
        "negative_factors":         negative_factors[:5],
        # Sanity
        "sanity_clamped":           sanity_clamped,
        "sanity_note":              sanity_note,
        # Quote
        "quote_message": (
            f"Based on ML ensemble valuation (R²=0.97), {brand or 'vehicle'} condition, "
            f"and {city.title() or 'local'} market demand, recommended acquisition is "
            f"₹{recommended_buy_price/100_000:.2f}L. "
            f"Seller gap: ₹{abs(seller_gap)//1000:.0f}k "
            f"{'above' if seller_gap > 0 else 'below'} target."
            if seller_asking else
            f"Recommended acquisition: ₹{recommended_buy_price/100_000:.2f}L → "
            f"target sell ₹{recommended_sell_price/100_000:.2f}L → "
            f"expected dealer profit ₹{target_profit//1000:.0f}k."
        ),
    }


# LEGACY FUNCTIONS — unchanged interface, used by /evaluate-enhanced
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
    acc             = (accident_history or "none").lower().strip()
    sr              = (seller_reason or "upgrading").lower().strip()
    owner_ded       = {1: 0, 2: 8_000, 3: 18_000}.get(owner_count, 30_000)
    km_ded          = (0 if odometer < 40_000 else 5_000 if odometer < 80_000
                       else 12_000 if odometer < 120_000 else 25_000)
    acc_ded         = {"none": 0, "minor": 10_000, "major": 35_000}.get(acc, 0)
    state_ded       = (0 if not registration_state or not sale_state
                       or registration_state.lower() == sale_state.lower() else 8_000)
    loan_ded        = 5_000 if loan_outstanding else 0
    sr_adj          = {"upgrading": 0, "relocating": -5_000, "financial": -12_000,
                       "unused": 5_000, "problem": -8_000}.get(sr, 0)
    total = owner_ded + km_ded + acc_ded + state_ded + loan_ded
    return {
        "total": int(total),
        "breakdown": {
            "owner_deduction":    int(owner_ded), "km_deduction": int(km_ded),
            "accident_deduction": int(acc_ded),   "state_deduction": int(state_ded),
            "loan_deduction":     int(loan_ded),
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
    fixed   = rc_transfer_cost + 2_500 + 2_000
    total   = ec + tc + bc + ic + elc + fixed
    return {
        "engine_cost": int(ec), "tyre_cost": int(tc), "body_cost": int(bc),
        "interior_cost": int(ic), "electrical_cost": int(elc),
        "fixed_cost": int(fixed), "rc_transfer_cost": int(rc_transfer_cost),
        "total": int(total),
        "breakdown": {"engine": int(ec), "tyres": int(tc), "body_paint": int(bc),
                      "interior": int(ic), "electricals": int(elc), "fixed": int(fixed)},
    }


def get_negotiation_trio(max_buy_price: int, seller_reason_adj: int = 0) -> dict:
    """Legacy function — used by /evaluate-enhanced and /reverse-calculate."""
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
