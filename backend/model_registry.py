"""
backend/model_registry.py
Model Registry — loads, caches, and manages versioned model variants.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib

from backend.ensemble_predictor import EnsemblePredictor

log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[1]
REGISTRY_DIR  = ROOT / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"

# In-memory cache: variant_id → {predictor, segment_models, metadata}
_CACHE: dict[str, dict] = {}


def _read_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"default": None, "variants": {}}


def _write_registry(data: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_variants() -> list[dict]:
    """Return all registered variants sorted by MAPE asc."""
    reg = _read_registry()
    variants = []
    for vid, info in reg.get("variants", {}).items():
        entry = dict(info)
        entry["variant_id"] = vid
        entry["is_default"] = (vid == reg.get("default"))
        variants.append(entry)
    variants.sort(key=lambda v: v.get("metrics", {}).get("mape", 9999))
    return variants


def get_default_variant_id() -> Optional[str]:
    # Allow the environment to pin a specific variant (e.g. for memory-constrained deploys)
    env_pin = os.environ.get("ACTIVE_VARIANT_ID", "").strip()
    if env_pin:
        return env_pin
    return _read_registry().get("default")


def get_variant_path(variant_id: str) -> Optional[Path]:
    reg = _read_registry()
    info = reg.get("variants", {}).get(variant_id)
    if not info:
        return None
    p = ROOT / info["artifact_path"]
    return p if p.exists() else None


def _best_variant_id(reg: dict) -> Optional[str]:
    """Pick variant with lowest test MAPE, break ties with RMSE then R²."""
    best_id, best_mape, best_rmse, best_r2 = None, float("inf"), float("inf"), -float("inf")
    for vid, info in reg.get("variants", {}).items():
        m = info.get("metrics", {})
        mape = m.get("mape", 9999)
        rmse = m.get("rmse", 9999999)
        r2   = m.get("r2",   -9999)
        if (mape < best_mape or
           (mape == best_mape and rmse < best_rmse) or
           (mape == best_mape and rmse == best_rmse and r2 > best_r2)):
            best_id, best_mape, best_rmse, best_r2 = vid, mape, rmse, r2
    return best_id


def register_variant(
    variant_id: str,
    artifact_path: Path,
    dataset: str,
    trained_at: str,
    metrics: dict,
    auto_promote: bool = True,
) -> None:
    """Add or update a variant in registry.json."""
    reg = _read_registry()
    rel_path = str(artifact_path.relative_to(ROOT)).replace("\\", "/")
    reg.setdefault("variants", {})[variant_id] = {
        "dataset":       dataset,
        "trained_at":    trained_at,
        "metrics":       metrics,
        "artifact_path": rel_path,
        "status":        "candidate",
    }
    if auto_promote:
        best = _best_variant_id(reg)
        if best:
            for vid in reg["variants"]:
                reg["variants"][vid]["status"] = "archived"
            reg["variants"][best]["status"] = "active"
            reg["default"] = best
    _write_registry(reg)
    log.info("Registered variant %s (default=%s)", variant_id, reg.get("default"))


def activate_variant(variant_id: str) -> bool:
    """Manually promote a variant to default."""
    reg = _read_registry()
    if variant_id not in reg.get("variants", {}):
        return False
    for vid in reg["variants"]:
        reg["variants"][vid]["status"] = "archived"
    reg["variants"][variant_id]["status"] = "active"
    reg["default"] = variant_id
    _write_registry(reg)
    _CACHE.clear()
    log.info("Activated variant %s", variant_id)
    return True


def _load_variant_data(variant_id: str) -> dict:
    """Load predictor + segment models for a variant (uncached)."""
    path = get_variant_path(variant_id)
    if path is None:
        raise FileNotFoundError(f"Variant '{variant_id}' artifact path not found in registry.")

    predictor = EnsemblePredictor.from_artifact_dir(path)

    segment_models: dict = {}
    for seg in ["economy", "premium", "luxury"]:
        pkl = path / f"ensemble_{seg}.pkl"
        if pkl.exists():
            segment_models[seg] = joblib.load(pkl)
    for old, new in [("budget", "economy"), ("mid", "economy")]:
        if old not in segment_models:
            pkl = path / f"ensemble_{old}.pkl"
            if pkl.exists() and new not in segment_models:
                segment_models[new] = joblib.load(pkl)

    with open(path / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    catalog: dict = {}
    cat_path = path / "dataset_catalog.json"
    if cat_path.exists():
        with open(cat_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    return {
        "predictor":      predictor,
        "segment_models": segment_models,
        "metadata":       metadata,
        "catalog":        catalog,
        "artifact_dir":   path,
    }


def get_variant(variant_id: str) -> dict:
    """Return cached variant data, loading from disk on first access."""
    if variant_id not in _CACHE:
        log.info("Loading model variant '%s' from disk …", variant_id)
        _CACHE.clear()  # Keep only 1 active variant in RAM to stay well under 512MB
        import gc
        gc.collect()
        _CACHE[variant_id] = _load_variant_data(variant_id)
        gc.collect()
        log.info("Variant '%s' loaded and cached.", variant_id)
    return _CACHE[variant_id]


def get_default_variant() -> Optional[dict]:
    """Return the default variant data, or None if registry is empty."""
    vid = get_default_variant_id()
    if vid is None:
        return None
    return get_variant(vid)


def next_variant_id() -> str:
    """Return the next available variant ID (variant_1, variant_2, …)."""
    reg = _read_registry()
    existing = reg.get("variants", {})
    n = 1
    while f"variant_{n}" in existing:
        n += 1
    return f"variant_{n}"
