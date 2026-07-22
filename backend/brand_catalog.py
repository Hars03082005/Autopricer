from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "ml_training" / "data" / "processed_widoutown-2.csv"

# Comprehensive Indian used-car catalog (minimum required set + common variants).
INDIAN_BRAND_CATALOG: Dict[str, List[str]] = {
    "Maruti": [
        "Swift", "Dzire", "Baleno", "Alto", "WagonR", "Vitara Brezza",
        "Ertiga", "Ciaz", "S-Cross", "Ignis", "Celerio", "S-Presso",
        "Alto K10", "XL6", "Grand Vitara", "Brezza", "Fronx", "Jimny",
        "Eeco", "Omni", "Gypsy", "Ritz",
    ],
    "Hyundai": [
        "i10", "Grand i10", "Grand i10 Nios",
        "i20", "Elite i20", "i20 Active",
        "Creta", "Verna", "Tucson", "Elantra", "Venue",
        "Santro", "Aura", "Alcazar", "Ioniq 5", "Kona",
        "Xcent", "Eon",
    ],
    "Tata": [
        "Nexon", "Nexon EV", "Harrier", "Safari", "Tiago", "Tiago EV",
        "Tigor", "Tigor EV", "Altroz", "Punch", "Punch EV",
        "Hexa", "Bolt", "Zest", "Indica", "Indigo", "Nano", "Curvv",
    ],
    "Honda": [
        "City", "Amaze", "Jazz", "WR-V", "CR-V", "Civic", "Accord",
        "BR-V", "HR-V",
    ],
    "Toyota": [
        "Innova", "Innova Crysta", "Innova Hycross",
        "Fortuner", "Camry", "Corolla", "Glanza",
        "Urban Cruiser", "Hilux", "Vellfire", "Yaris",
        "Etios", "Platinum Etios", "Liva", "Hyryder",
    ],
    "Mahindra": [
        "Scorpio", "Scorpio N", "Scorpio Classic",
        "XUV500", "XUV300", "XUV400", "XUV700",
        "Thar", "Thar Roxx", "Bolero", "Bolero Neo",
        "KUV100", "Marazzo", "Alturas G4", "BE6", "XEV9e",
    ],
    "Kia": ["Seltos", "Sonet", "Carnival", "Carens", "EV6", "EV9"],
    "Renault": ["Kwid", "Duster", "Triber", "Kiger", "Captur", "Fluence", "Scala"],
    "Nissan": ["Magnite", "Kicks", "Terrano", "Sunny", "Micra", "GT-R"],
    "Volkswagen": ["Polo", "Vento", "Taigun", "Virtus", "Tiguan", "T-Roc", "Passat"],
    "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia", "Kodiaq", "Karoq", "Fabia"],
    "Ford": ["EcoSport", "Endeavour", "Figo", "Figo Aspire", "Freestyle", "Aspire", "Mustang"],
    "Jeep": ["Compass", "Meridian", "Wrangler", "Grand Cherokee"],
    "MG": ["Hector", "Astor", "Gloster", "ZS EV", "Comet EV", "Windsor EV"],
    "Mitsubishi": ["Pajero Sport", "Outlander", "Eclipse Cross", "Cedia"],
    "Citroen": ["C3", "C5 Aircross", "C3 Aircross"],
    "Datsun": ["Go", "Go+", "Redi-Go"],
    "Isuzu": ["D-Max", "MU-X", "V-Cross"],
    "Chevrolet": ["Beat", "Sail", "Spark", "Cruze", "Aveo", "Captiva", "Trailblazer"],
    "Fiat": ["Punto", "Linea", "Avventura", "Urban Cross"],
    "BMW": [
        "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X7",
        "M3", "M5", "i4", "iX", "2 Series", "4 Series", "6 Series GT",
    ],
    "Mercedes-Benz": [
        "A-Class", "C-Class", "E-Class", "S-Class",
        "GLA", "GLC", "GLE", "GLS", "AMG GT", "EQS", "EQB",
        "CLA", "CLS", "G-Class",
    ],
    "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron", "RS5", "TT"],
    "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60"],
    "Porsche": ["Cayenne", "Macan", "Panamera", "911", "Taycan", "Boxster"],
    "Land Rover": [
        "Defender", "Discovery", "Discovery Sport", "Range Rover",
        "Range Rover Sport", "Range Rover Evoque", "Freelander",
    ],
    "Jaguar": ["XE", "XF", "XJ", "F-Pace", "E-Pace", "I-Pace", "F-Type"],
    "Lamborghini": ["Urus", "Huracan", "Aventador"],
    "Ferrari": ["Roma", "Portofino", "SF90", "812 Superfast", "F8"],
    "Rolls-Royce": ["Ghost", "Phantom", "Cullinan", "Wraith", "Dawn"],
    "Bentley": ["Bentayga", "Continental GT", "Flying Spur", "Mulsanne"],
    "Lexus": ["ES", "LS", "NX", "RX", "UX", "LC", "LX", "IS"],
    "Infiniti": ["Q50", "Q60", "QX50", "QX60", "QX80"],
    "Maserati": ["Ghibli", "Quattroporte", "Levante", "Grecale", "MC20"],
    "Bugatti": ["Chiron", "Veyron", "Divo"],
    "McLaren": ["720S", "570S", "GT", "Artura", "Senna"],
    "Aston Martin": ["DB11", "Vantage", "DBS", "DBX"],
}

BRAND_ALIASES = {
    "maruti suzuki": "Maruti",
    "maruti": "Maruti",
    "mercedes benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "land rover": "Land Rover",
    "range rover": "Land Rover",
    "rolls royce": "Rolls-Royce",
    "aston martin": "Aston Martin",
}


def _title_words(value: str) -> str:
    parts = str(value or "").strip().split()
    return " ".join(p[:1].upper() + p[1:] if p else "" for p in parts)


def normalize_brand_name(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        return ""
    if key in BRAND_ALIASES:
        return BRAND_ALIASES[key]
    for brand in INDIAN_BRAND_CATALOG:
        if brand.lower() == key:
            return brand
    return _title_words(raw)


def _merge_models(catalog: Dict[str, List[str]], brand: str, models: List[str]) -> None:
    existing = {m.lower(): m for m in catalog.get(brand, [])}
    for model in models:
        cleaned = str(model or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in existing:
            existing[key] = _title_words(cleaned)
    catalog[brand] = sorted(existing.values(), key=str.casefold)


def _load_dataset_brands() -> Dict[str, List[str]]:
    if not DATASET_PATH.exists():
        return {}

    try:
        frame = pd.read_csv(DATASET_PATH, usecols=["brand_name", "model_name"], low_memory=False)
    except ValueError:
        frame = pd.read_csv(DATASET_PATH, low_memory=False)
        if "brand_name" not in frame.columns or "model_name" not in frame.columns:
            return {}

    grouped: Dict[str, List[str]] = {}
    for brand_raw, model_raw in frame[["brand_name", "model_name"]].dropna().itertuples(index=False):
        brand = normalize_brand_name(brand_raw)
        model = str(model_raw).strip()
        if not brand or not model:
            continue
        grouped.setdefault(brand, []).append(_title_words(model))
    return grouped


def build_brand_catalog() -> Dict[str, List[str]]:
    """
    Build the brand → [models] catalog.
    PRIMARY source: model_artifacts/dataset_catalog.json (only real dataset models).
    FALLBACK: hardcoded INDIAN_BRAND_CATALOG if the JSON doesn't exist yet.
    """
    import json as _json

    catalog_json = ROOT / "model_artifacts" / "dataset_catalog.json"
    if catalog_json.exists():
        with open(catalog_json, "r", encoding="utf-8") as f:
            raw: Dict = _json.load(f)
        catalog: Dict[str, List[str]] = {}
        for brand_lower, models_dict in raw.items():
            brand = normalize_brand_name(brand_lower)
            if not brand:
                continue
            # models_dict values are lists of variant strings; keys are model names
            model_names = sorted(
                {_title_words(m) for m in models_dict.keys() if str(m).strip()},
                key=str.casefold,
            )
            if model_names:
                catalog[brand] = model_names
        return dict(sorted(catalog.items(), key=lambda item: item[0].casefold()))

    # Fallback: dataset CSV or hardcoded catalog
    catalog = {brand: list(models) for brand, models in INDIAN_BRAND_CATALOG.items()}
    for brand, models in _load_dataset_brands().items():
        if brand in catalog:
            _merge_models(catalog, brand, models)
        else:
            unique = sorted({_title_words(m) for m in models if str(m).strip()}, key=str.casefold)
            if unique:
                catalog[brand] = unique
    return dict(sorted(catalog.items(), key=lambda item: item[0].casefold()))

