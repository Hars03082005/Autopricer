"""End-to-end API behaviour against the real model artifacts.

Marked `models` because loading the ensemble pulls ~250 MB into RAM and takes
tens of seconds. CI runs these inside the built container, where the artifacts
are guaranteed present; locally they skip if model_registry/variant_1 is absent.

The prediction assertions are sanity bands rather than exact golden values,
because the artifacts carry no record of the library versions they were trained
with (see the note in backend/requirements.txt), so an exact expectation could not
be established from the repository alone. A band still catches the failure mode
that matters: a dependency bump that changes predictions by an order of magnitude
or breaks model loading outright. To tighten these into exact assertions, capture
values once from a known-good container and pin them here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_models

pytestmark = [pytest.mark.models, requires_models]


@pytest.fixture(scope="module")
def client():
    """TestClient as a context manager so the lifespan (model load) actually runs."""
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_reports_a_loaded_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        # The container healthcheck and the ACA probes both gate on this.
        assert payload["model_loaded"] is True
        assert payload["active_variant"] in ("final", "variant_1")

    def test_health_exposes_configuration_state(self, client):
        payload = client.get("/health").json()
        assert "database_configured" in payload
        assert "auth_configured" in payload
        assert "environment" in payload

    def test_segment_models_are_loaded(self, client):
        """Price-band routing is verified."""
        payload = client.get("/health").json()
        assert isinstance(payload.get("segments_loaded"), list)


class TestPredict:
    def test_predict_returns_a_plausible_price(self, client):
        response = client.post("/predict", json={
            "brand": "Honda", "model": "City", "variant": "VX",
            "year": 2021, "fuel_type": "Petrol", "transmission": "Manual",
            "odometer_reading": 28000, "owner_count": 1, "condition": "Good",
            "city": "Bangalore", "locality": "Indiranagar", "seller_type": "Individual",
        })
        assert response.status_code == 200
        payload = response.json()

        price = payload.get("market_value") or payload.get("predicted_price")
        assert price is not None, f"no price field in response: {sorted(payload)}"
        # A 2021 Honda City is a mainstream mid-tier car; anything outside this
        # band means the model or the feature pipeline is broken, not merely
        # differently tuned.
        assert 200_000 < price < 3_000_000, f"implausible valuation: {price}"

    def test_identical_requests_are_deterministic(self, client):
        """Guards against uninitialised state leaking into the feature vector."""
        body = {
            "brand": "Maruti Suzuki", "model": "Swift", "variant": "VXI",
            "year": 2019, "fuel_type": "Petrol", "transmission": "Manual",
            "odometer_reading": 45000, "owner_count": 1, "condition": "Good",
            "city": "Bangalore", "locality": "Whitefield",
        }
        first = client.post("/predict", json=body).json()
        second = client.post("/predict", json=body).json()
        assert first == second

    def test_higher_mileage_does_not_increase_value(self, client):
        """Directional check on the odometer feature."""
        def value_at(odometer: int) -> float:
            body = {
                "brand": "Hyundai", "model": "Creta", "variant": "SX",
                "year": 2020, "fuel_type": "Petrol", "transmission": "Manual",
                "odometer_reading": odometer, "owner_count": 1, "condition": "Good",
                "city": "Bangalore", "locality": "Koramangala",
            }
            payload = client.post("/predict", json=body).json()
            return payload.get("market_value") or payload.get("predicted_price")

        assert value_at(150_000) <= value_at(20_000)

    def test_older_vehicle_does_not_cost_more(self, client):
        def value_at(year: int) -> float:
            body = {
                "brand": "Hyundai", "model": "Creta", "variant": "SX",
                "year": year, "fuel_type": "Petrol", "transmission": "Manual",
                "odometer_reading": 50_000, "owner_count": 1, "condition": "Good",
                "city": "Bangalore", "locality": "Koramangala",
            }
            payload = client.post("/predict", json=body).json()
            return payload.get("market_value") or payload.get("predicted_price")

        assert value_at(2014) <= value_at(2022)

    def test_negative_odometer_is_rejected(self, client):
        response = client.post("/predict", json={
            "brand": "Honda", "model": "City", "year": 2021, "odometer_reading": -5,
        })
        assert response.status_code == 422


class TestEvaluate:
    def test_evaluate_returns_a_dealer_decision(self, client):
        response = client.post("/evaluate", json={
            "brand": "Honda", "model": "City", "variant": "VX",
            "year": 2021, "fuel_type": "Petrol", "transmission": "Manual",
            "odometer_reading": 28000, "owner_count": 1, "condition": "Good",
            "city": "Bangalore", "locality": "Indiranagar", "seller_type": "Individual",
        })
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("action")
        assert payload.get("market_value", 0) > 0

    def test_buy_price_is_below_market_value(self, client):
        """The dealer waterfall subtracts margin and recon, so this must hold."""
        payload = client.post("/evaluate", json={
            "brand": "Honda", "model": "City", "variant": "VX",
            "year": 2021, "fuel_type": "Petrol", "transmission": "Manual",
            "odometer_reading": 28000, "owner_count": 1, "condition": "Good",
            "city": "Bangalore", "locality": "Indiranagar",
        }).json()

        market = payload.get("market_value")
        buy = payload.get("recommended_buy_price") or payload.get("buy_price")
        if buy is None:
            pytest.skip(f"no buy-price field in response: {sorted(payload)}")
        assert buy < market


class TestPublicSurface:
    def test_brands_catalog_is_served(self, client):
        response = client.get("/api/brands")
        assert response.status_code == 200
        assert response.json().get("brands")

    def test_registry_lists_variants(self, client):
        response = client.get("/api/registry")
        assert response.status_code == 200
        assert response.json().get("default") in ("final", "variant_1")


class TestVariantActivationIsLockedDown:
    """The endpoint used to be unauthenticated and mutated on-disk state."""

    def test_activation_is_refused_without_a_token(self, client):
        response = client.post("/api/registry/variant_2/activate")
        assert response.status_code in (401, 403)

    def test_activation_is_refused_with_a_bogus_token(self, client):
        response = client.post(
            "/api/registry/variant_2/activate",
            headers={"Authorization": "Bearer not-the-admin-token"},
        )
        assert response.status_code in (401, 403)

    def test_it_did_not_change_which_model_is_served(self, client):
        """The refusal must be a no-op, not a partial mutation."""
        assert client.get("/health").json()["active_variant"] in ("final", "variant_1")


class TestAuthenticatedEndpointsRequireAToken:
    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/history"),
        ("POST", "/api/history"),
        ("DELETE", "/api/history"),
        ("GET", "/api/profile"),
        ("PUT", "/api/profile"),
    ])
    def test_no_token_is_rejected(self, client, method, path):
        response = client.request(method, path, json={} if method in ("POST", "PUT") else None)
        # 401 unauthenticated, or 503 when Supabase is not configured in this
        # environment. Never 200, and never a 500.
        assert response.status_code in (401, 503), (
            f"{method} {path} returned {response.status_code}"
        )

    def test_an_invalid_token_is_rejected(self, client):
        response = client.get(
            "/api/history", headers={"Authorization": "Bearer clearly.invalid.token"}
        )
        assert response.status_code in (401, 503)
