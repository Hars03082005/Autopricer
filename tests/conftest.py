"""Shared fixtures.

Note the ordering constraint that shapes this file: backend.main sets
OMP_NUM_THREADS and friends at import time and reads configuration at module
scope, so environment variables must be in place *before* the first import of
anything under `backend`. Fixtures that need different configuration therefore
patch `backend.config` rather than re-importing the module.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Baseline environment for every test. Set before backend imports happen.
os.environ.setdefault("APP_ENVIRONMENT", "development")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("ACTIVE_VARIANT_ID", "variant_1")


def _artifacts_available() -> bool:
    """True when the model artifacts needed for real inference are present."""
    variant = REPO_ROOT / "model_registry" / "variant_1"
    return (variant / "model_metadata.json").exists() and (
        variant / "vehicle_price_catboost.cbm"
    ).exists()


def _ml_stack_importable() -> bool:
    """True when the inference libraries are installed.

    Checked separately from artifact presence so that a developer with a light
    virtualenv (fastapi + pyjwt only, no ~1 GB of CatBoost/LightGBM/XGBoost) gets
    a clean skip instead of a collection error. CI runs these tests inside the
    built backend image, where both conditions hold.
    """
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("catboost", "lightgbm", "xgboost", "pandas"))


requires_models = pytest.mark.skipif(
    not (_artifacts_available() and _ml_stack_importable()),
    reason="requires model_registry/variant_1 artifacts and the ML inference stack",
)


@pytest.fixture
def clean_settings() -> Iterator[None]:
    """Reset the cached Settings before and after a test that changes the env."""
    from backend import config

    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, clean_settings: None):
    """Set environment variables and invalidate the settings cache.

    Usage:
        def test_x(env):
            settings = env(APP_ENVIRONMENT="production", CORS_ALLOWED_ORIGINS="https://a.example")
    """
    from backend import config

    def _apply(**values: str):
        for key, value in values.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        config.reset_settings_cache()
        return config.get_settings()

    return _apply
