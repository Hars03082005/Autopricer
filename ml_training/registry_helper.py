"""
ml_training/registry_helper.py
Shared utility used by all train-*.py scripts.
Saves artifacts to model_registry/variant_N/ and updates registry.json.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "model_registry"


def _read_registry() -> dict:
    f = REGISTRY_DIR / "registry.json"
    if f.exists():
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"default": None, "variants": {}}


def _write_registry(data: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_DIR / "registry.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def next_variant_id() -> str:
    """Return the next unused variant_N id."""
    reg = _read_registry()
    n = 1
    while f"variant_{n}" in reg.get("variants", {}):
        n += 1
    return f"variant_{n}"


def get_variant_dir(variant_id: str) -> Path:
    """Create and return the artifact directory for this variant."""
    path = REGISTRY_DIR / variant_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _best_variant_id(reg: dict) -> str | None:
    best_id, best_mape, best_rmse, best_r2 = None, float("inf"), float("inf"), -float("inf")
    for vid, info in reg.get("variants", {}).items():
        m = info.get("metrics", {})
        mape = m.get("mape", 9999)
        rmse = m.get("rmse", 9999999)
        r2   = m.get("r2",   -9999)
        if (mape < best_mape
                or (mape == best_mape and rmse < best_rmse)
                or (mape == best_mape and rmse == best_rmse and r2 > best_r2)):
            best_id, best_mape, best_rmse, best_r2 = vid, mape, rmse, r2
    return best_id


def register_variant(
    variant_id: str,
    artifact_dir: Path,
    dataset_name: str,
    metrics: dict,
) -> None:
    """
    Register a newly trained variant in registry.json.
    Automatically promotes to default if it beats all existing variants by MAPE.
    """
    reg = _read_registry()
    rel = str(artifact_dir.relative_to(ROOT)).replace("\\", "/")
    reg.setdefault("variants", {})[variant_id] = {
        "dataset":       dataset_name,
        "trained_at":    datetime.now().isoformat(),
        "metrics":       metrics,
        "artifact_path": rel,
        "status":        "candidate",
    }

    # Auto-promote: find overall best variant
    best = _best_variant_id(reg)
    if best:
        for vid in reg["variants"]:
            reg["variants"][vid]["status"] = "archived"
        reg["variants"][best]["status"] = "active"
        reg["default"] = best

    _write_registry(reg)
    print(f"\nRegistry updated — variant: {variant_id}  default: {reg['default']}")


def copy_to_model_artifacts(variant_dir: Path) -> None:
    """
    Copy the active variant back to model_artifacts/ so the backend
    continues to work without a restart (backward compat).
    """
    dst = ROOT / "model_artifacts"
    dst.mkdir(exist_ok=True)
    for src_file in variant_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dst / src_file.name)
    print(f"Copied {variant_dir.name} → model_artifacts/")
