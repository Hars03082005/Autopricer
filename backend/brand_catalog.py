from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "ml_training" / "data" / "processed_widoutown-2.csv"

# Canonical brand aliases → lowercase dataset key (matches dataset_catalog.json keys exactly)
BRAND_ALIASES: dict[str, str] = {
    "maruti":               "maruti suzuki",
    "marutisuzuki":         "maruti suzuki",
    "maruti-suzuki":        "maruti suzuki",
    "maruti suzuki":        "maruti suzuki",
    "suzuki":               "maruti suzuki",
    "mercedes":             "mercedes-benz",
    "mercedes benz":        "mercedes-benz",
    "mercedesbenz":         "mercedes-benz",
    "mercedes-benz":        "mercedes-benz",
    "merc":                 "mercedes-benz",
    "land-rover":           "land rover",
    "landrover":            "land rover",
    "range rover":          "land rover",
    "vw":                   "volkswagen",
    "volkswagon":           "volkswagen",
    "hyundai motor":        "hyundai",
    "tata motors":          "tata",
    "honda cars":           "honda",
    "general motors":       "chevrolet",
    "chevy":                "chevrolet",
    "bajaj auto":           "bajaj",
    "fiat chrysler":        "fiat",
    "mg motor":             "mg",
}


def normalize_brand_name(raw: str) -> str:
    """Return canonical lowercase brand key matching dataset_catalog.json."""
    key = str(raw or "").strip().lower()
    key = " ".join(key.split())  # collapse whitespace
    if not key:
        return ""
    return BRAND_ALIASES.get(key, key)


def build_brand_catalog() -> dict[str, list[str]]:
    """
    Build the brand → [models] catalog.
    SOURCE OF TRUTH: model_artifacts/dataset_catalog.json.
    Returns only dataset-backed brands and models.
    """
    import json as _json

    catalog_json = ROOT / "model_artifacts" / "dataset_catalog.json"
    if catalog_json.exists():
        with open(catalog_json, encoding="utf-8") as f:
            raw: dict = _json.load(f)
        catalog: dict[str, list[str]] = {}
        for brand_key, models_dict in raw.items():
            # brand_key is already lowercase (e.g. "maruti suzuki")
            canonical = normalize_brand_name(brand_key)
            if not canonical:
                continue
            model_names = sorted(
                {str(m).strip() for m in models_dict if str(m).strip()},
                key=str.casefold,
            )
            if model_names:
                catalog[canonical] = model_names
        return dict(sorted(catalog.items(), key=lambda item: item[0].casefold()))

    # Fallback: derive from dataset CSV (no hardcoded brands)
    if not DATASET_PATH.exists():
        return {}
    try:
        frame = pd.read_csv(DATASET_PATH, usecols=["brand_name", "model_name"], low_memory=False)
    except (ValueError, KeyError):
        return {}

    catalog: dict[str, list[str]] = {}
    for brand_raw, model_raw in frame[["brand_name", "model_name"]].dropna().itertuples(index=False):
        brand = normalize_brand_name(brand_raw)
        model = str(model_raw).strip()
        if not brand or not model:
            continue
        catalog.setdefault(brand, []).append(model)

    return {
        b: sorted({m for m in models if m}, key=str.casefold)
        for b, models in sorted(catalog.items(), key=lambda item: item[0].casefold())
        if models
    }


def get_catalog_variants(brand_raw: str, model_raw: str) -> list[str] | None:
    """
    Return dataset-backed variants for a brand+model pair.
    Returns None when model is not in the catalog.
    Returns [] when model is in catalog but has no variant data.
    """
    import json as _json

    catalog_json = ROOT / "model_artifacts" / "dataset_catalog.json"
    if not catalog_json.exists():
        return None

    with open(catalog_json, encoding="utf-8") as f:
        raw: dict = _json.load(f)

    brand_key = normalize_brand_name(brand_raw)
    brand_data = raw.get(brand_key)
    if brand_data is None:
        return None

    model_key = str(model_raw or "").strip().lower()
    if model_key in brand_data:
        return list(brand_data[model_key])

    # Case-insensitive fallback
    for m in brand_data:
        if m.lower() == model_key:
            return list(brand_data[m])

    return None
