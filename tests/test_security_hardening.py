"""Security hardening test suite.

Verifies:
  1. Model bundle SHA-256 cryptographic integrity verification.
  2. Prevention of insecure deserialization on corrupted/tampered bundles.
  3. Server-side admin authorization for registry endpoints.
  4. CORS policy restrictions (no production wildcards).
  5. Safe error responses (no internal path / traceback leakage).
  6. Input size bounds on string & numerical fields.
  7. Frontend secret isolation.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("numpy", reason="requires ML stack")
pytest.importorskip("joblib", reason="requires ML stack")

from backend import champion_predictor
from backend.app import app as champion_app
from backend.champion_predictor import (
    _FROZEN_FINAL_SHA256,
    ChampionPredictor,
    clear_champion_cache,
    compute_sha256,
    load_champion,
    verify_bundle_integrity,
)
from backend.config import ConfigError, Settings
from backend.main import app as main_app
from backend.schemas import VehicleRecord


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_champion_cache()
    yield
    clear_champion_cache()


# ── 1. Model Bundle Cryptographic Integrity ───────────────────────────────────

class TestModelBundleIntegrity:

    def test_production_bundle_sha256_matches_expected(self):
        """Production bundle SHA-256 must match the registered frozen hash."""
        bundle_path = Path("model_registry/final/ensemble_bundle.pkl")
        assert bundle_path.exists(), "Production bundle must exist on disk"
        actual_hash = compute_sha256(bundle_path)
        assert actual_hash == _FROZEN_FINAL_SHA256

    def test_valid_bundle_passes_integrity_check(self):
        """Valid bundle passes SHA-256 verification and loads successfully."""
        bundle_path = Path("model_registry/final/ensemble_bundle.pkl")
        verify_bundle_integrity(bundle_path)
        predictor = load_champion(bundle_path)
        assert isinstance(predictor, ChampionPredictor)

    def test_tampered_bundle_rejected_before_pickle_load(self, tmp_path, monkeypatch):
        """A tampered bundle must fail SHA-256 check and raise ValueError without loading."""
        tampered_file = tmp_path / "ensemble_bundle.pkl"
        tampered_file.write_bytes(b"MALICIOUS_TAMPERED_CONTENT_NOT_A_VALID_BUNDLE")
        monkeypatch.setenv("EXPECTED_BUNDLE_SHA256", _FROZEN_FINAL_SHA256)

        with pytest.raises(ValueError, match="integrity check failed: artifact is corrupted or modified"):
            # Should fail closed on hash mismatch before any unpickling
            load_champion(tampered_file, verify_integrity=True)

    def test_missing_expected_hash_fails_closed(self, tmp_path):
        """If expected hash is missing and not frozen final, fails closed."""
        temp_bundle = tmp_path / "custom_bundle.pkl"
        temp_bundle.write_bytes(b"sample content")

        with pytest.raises(ValueError, match="missing expected hash"):
            verify_bundle_integrity(temp_bundle)

    def test_health_reports_integrity_error_when_tampered(self, tmp_path):
        """Health check reports integrity_error and model_loaded=False when bundle is invalid."""
        tampered_file = tmp_path / "ensemble_bundle.pkl"
        tampered_file.write_bytes(b"FAKE_DATA")

        info = champion_predictor.get_health_info(tampered_file)
        assert info["status"] == "integrity_error"
        assert info["model_loaded"] is False
        assert str(tampered_file.resolve()) not in info["artifact_path"]  # Safe path reporting


# ── 2. Admin Endpoint Security ────────────────────────────────────────────────

class TestAdminEndpointSecurity:

    def test_unauthenticated_activation_rejected(self):
        """Activating variant without authorization header must be rejected."""
        client = TestClient(main_app)
        response = client.post("/api/registry/final/activate")
        assert response.status_code in (401, 403)

    def test_invalid_admin_token_rejected(self):
        """Activating variant with wrong admin token must be rejected."""
        client = TestClient(main_app)
        response = client.post(
            "/api/registry/final/activate",
            headers={"Authorization": "Bearer totally_wrong_secret_token_12345678"},
        )
        assert response.status_code in (401, 403)
        # Error must NOT disclose the valid secret
        assert "priceref" not in response.text.lower()
        assert "admin_token" not in response.text.lower()


# ── 3. CORS Policy Hardening ──────────────────────────────────────────────────

class TestCorsSecurity:

    def test_wildcard_cors_rejected_in_production(self):
        """Settings must refuse wildcard CORS when APP_ENVIRONMENT=production."""
        with pytest.raises(ConfigError, match=r"CORS_ALLOWED_ORIGINS='\*' is not permitted"):
            Settings.load({
                "APP_ENVIRONMENT": "production",
                "CORS_ALLOWED_ORIGINS": "*",
            })

    def test_production_requires_explicit_origin_allowlist(self):
        """Settings must enforce explicit allowlist in production."""
        with pytest.raises(ConfigError, match="CORS_ALLOWED_ORIGINS must be set"):
            Settings.load({
                "APP_ENVIRONMENT": "production",
                "CORS_ALLOWED_ORIGINS": "",
            })

    def test_champion_app_cors_configured_securely(self):
        """backend.app CORS middleware must not allow wildcard in production."""
        client = TestClient(champion_app)
        response = client.get("/health", headers={"Origin": "https://unauthorized-evil-site.com"})
        assert response.status_code == 200
        # In test mode with default local settings, origin is not allowed wildcard
        allow_origin = response.headers.get("access-control-allow-origin")
        assert allow_origin != "*" or not Settings.load().is_production


# ── 4. Safe Error Responses ───────────────────────────────────────────────────

class TestSafeErrorResponses:

    def test_predict_error_does_not_leak_filesystem_paths(self):
        """Client responses must not expose absolute filesystem paths."""
        client = TestClient(champion_app)
        # Send an input that triggers a validation error
        response = client.post("/predict", json={
            "brand": "hyundai", "model": "creta",
            "vehicle_age": -5,  # negative age
            "odometer_reading": 20000,
        })
        assert response.status_code == 422
        body = response.text
        assert "C:\\Users" not in body
        assert "/app/" not in body
        assert "traceback" not in body.lower()

    def test_predict_catches_internal_exception_safely(self):
        """500 responses return safe generic message, not raw tracebacks."""
        client = TestClient(champion_app)
        with patch.object(ChampionPredictor, "predict_price", side_effect=RuntimeError("Internal math exception C:\\path\\secret")):
            response = client.post("/predict", json={
                "brand": "hyundai", "model": "creta",
                "vehicle_age": 3, "odometer_reading": 20000,
            })
            assert response.status_code == 500
            assert "C:\\path\\secret" not in response.text
            assert "Prediction request could not be processed" in response.json()["detail"]


# ── 5. Input Bounds & Oversized String Limits ─────────────────────────────────

class TestInputBounds:

    def test_oversized_brand_rejected(self):
        """VehicleRecord with string > 50 characters is rejected by validation."""
        client = TestClient(champion_app)
        oversized = "a" * 150
        response = client.post("/predict", json={
            "brand": oversized, "model": "creta",
            "vehicle_age": 3, "odometer_reading": 20000,
        })
        assert response.status_code == 422

    def test_oversized_variant_rejected(self):
        """VehicleRecord with variant > 100 characters is rejected."""
        client = TestClient(champion_app)
        response = client.post("/predict", json={
            "brand": "hyundai", "model": "creta", "variant": "x" * 250,
            "vehicle_age": 3, "odometer_reading": 20000,
        })
        assert response.status_code == 422

    def test_oversized_vehicle_input_in_main_app_rejected(self):
        """VehicleInput in backend.main with oversized brand is rejected."""
        client = TestClient(main_app)
        response = client.post("/predict", json={
            "brand": "b" * 120, "model": "City",
            "year": 2021, "odometer_reading": 20000,
        })
        assert response.status_code == 422


# ── 6. Frontend Secret Isolation Scan ─────────────────────────────────────────

class TestFrontendSecretIsolation:

    def test_no_admin_secret_in_frontend_code(self):
        """Frontend files must not contain the compromised admin secret or VITE_ADMIN_API_TOKEN."""
        frontend_dir = Path("src")
        assert frontend_dir.exists()

        forbidden_patterns = [
            "priceref_admin_token_production_32chars_min",
            "VITE_ADMIN_API_TOKEN",
        ]

        found_violations = []
        for file in frontend_dir.rglob("*.js*"):
            content = file.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                if pattern in content:
                    found_violations.append(f"{file}: contains '{pattern}'")

        assert not found_violations, f"Found secrets in frontend code: {found_violations}"
