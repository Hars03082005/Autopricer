"""Model Registry — loads, caches, and manages versioned model variants."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None

try:
    from backend.ensemble_predictor import EnsemblePredictor
except ImportError:
    EnsemblePredictor = None

log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[1]
REGISTRY_DIR  = ROOT / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"

_CACHE: dict[str, dict] = {}


def _read_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, encoding="utf-8") as f:
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


def get_default_variant_id() -> str | None:
    env_pin = os.environ.get("ACTIVE_VARIANT_ID", "").strip()
    if env_pin:
        return env_pin
    return _read_registry().get("default")


def get_variant_path(variant_id: str) -> Path | None:
    reg = _read_registry()
    info = reg.get("variants", {}).get(variant_id)
    if not info:
        return None
    p = ROOT / info["artifact_path"]
    return p if p.exists() else None


def _best_variant_id(reg: dict) -> str | None:
    """Pick variant with lowest test MAPE, break ties with RMSE then R² then newest trained_at."""
    best_id, best_mape, best_rmse, best_r2, best_ts = None, float("inf"), float("inf"), -float("inf"), ""
    for vid, info in reg.get("variants", {}).items():
        m = info.get("metrics", {})
        mape    = m.get("mape", 9999)
        rmse    = m.get("rmse", 9999999)
        r2      = m.get("r2",   -9999)
        trained = info.get("trained_at", "")
        is_better = (
            mape < best_mape or
            (mape == best_mape and rmse < best_rmse) or
            (mape == best_mape and rmse == best_rmse and r2 > best_r2) or
            (mape == best_mape and rmse == best_rmse and r2 == best_r2 and trained > best_ts)
        )
        if is_better:
            best_id, best_mape, best_rmse, best_r2, best_ts = vid, mape, rmse, r2, trained
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
    routing_path = path / "routing_table.json"
    if routing_path.exists():
        from catboost import CatBoostRegressor
        with open(routing_path, encoding="utf-8") as f:
            routing = json.load(f)
        for seg_name, info in routing.items():
            if info.get("active") and info.get("model_file"):
                model_file  = path / info["model_file"]
                levels_file = path / info.get("levels_file", "__nonexistent__")
                if model_file.exists():
                    try:
                        m = CatBoostRegressor()
                        m.load_model(str(model_file))
                        levels = joblib.load(levels_file) if levels_file.exists() else {}
                        segment_models[seg_name] = {
                            "model":      m,
                            "cat_levels": levels,
                            "price_range": info.get("price_range", [0, 20_000_000]),
                            "mape":        info.get("mape"),
                        }
                        log.info("Loaded segment model '%s' (MAPE %.2f%%)",
                                 seg_name, info.get("mape", 0))
                    except Exception as e:
                        log.warning("Failed to load segment model '%s': %s", seg_name, e)
    else:
        for seg in ["economy", "premium", "luxury"]:
            pkl = path / f"ensemble_{seg}.pkl"
            if pkl.exists():
                segment_models[seg] = joblib.load(pkl)

    with open(path / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    catalog: dict = {}
    cat_path = path / "dataset_catalog.json"
    if cat_path.exists():
        with open(cat_path, encoding="utf-8") as f:
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
        _CACHE.clear()
        import gc
        gc.collect()
        _CACHE[variant_id] = _load_variant_data(variant_id)
        gc.collect()
        log.info("Variant '%s' loaded and cached.", variant_id)
    return _CACHE[variant_id]


def get_default_variant() -> dict | None:
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
