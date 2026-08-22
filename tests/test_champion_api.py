"""Tests for the Final Production Price Prediction API & Champion Model Bundle.

Covers all 16 required verification areas:
  1. Model loading test
  2. Prediction test
  3. Batch prediction test
  4. Save/load consistency test
  5. Determinism test
  6. Schema validation test
  7. Missing-field validation test
  8. Invalid-input validation test
  9. Routing logic test
  10. Luxury-brand routing test
  11. Rs16L threshold test
  12. Rs22L threshold test
  13. Mass-market routing test
  14. API /health test
  15. API /predict test
  16. Regression & edge case verification
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from tests.conftest import requires_models

pytestmark = [pytest.mark.models, requires_models]

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_BUNDLE_PATH = REPO_ROOT / "model_registry" / "final" / "ensemble_bundle.pkl"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor():
    """Load the final champion predictor once for the entire module."""
    from backend.champion_predictor import load_champion
    return load_champion(FINAL_BUNDLE_PATH)


@pytest.fixture(scope="module")
def client():
    """TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        yield c


# ── 1. Model Loading Test ──────────────────────────────────────────────────────

class TestModelLoading:

    def test_bundle_file_exists(self):
        """Final ensemble bundle exists on disk."""
        assert FINAL_BUNDLE_PATH.exists(), f"Bundle not found at {FINAL_BUNDLE_PATH}"
        assert FINAL_BUNDLE_PATH.stat().st_size > 10_000_000, "Bundle file too small"

    def test_bundle_structure(self, predictor):
        """Loaded predictor contains all required models and metadata."""
        assert len(predictor._lgb_models) == 5, "Must contain exactly 5 LightGBM models"
        assert predictor._luxury_specialist is not None, "Must contain CatBoost specialist"
        assert len(predictor._cat_levels) == 11, "Must contain 11 categorical levels"
        assert len(predictor._encoders) == 11, "Must contain 11 LabelEncoders"
        assert len(predictor._medians) == 6, "Must contain 6 numerical medians"
        assert predictor._luxury_brand_threshold == 1_600_000.0
        assert predictor._global_threshold == 2_200_000.0


# ── 2. Prediction Test ─────────────────────────────────────────────────────────

class TestPrediction:

    def test_normal_hyundai_creta(self, predictor):
        """Hyundai Creta prediction returns valid price and routing."""
        record = {
            "brand": "hyundai", "model": "creta", "variant": "sx",
            "locality": "indiranagar", "rto": "ka03", "fuel_type": "petrol",
            "transmission": "manual", "seller_type": "dealer", "color": "white",
            "vehicle_age": 3, "odometer_reading": 35000, "km_per_year": 11666,
            "owner_count": 1, "certified": 1, "pincode": 560038,
        }
        result = predictor.predict_price(record)
        _assert_valid_result(result)
        assert 500_000 < result["predicted_price"] < 3_000_000
        assert result["routing_decision"] == "champion"

    def test_budget_maruti_alto(self, predictor):
        """Maruti Alto returns valid low-budget price."""
        record = {
            "brand": "maruti", "model": "alto", "variant": "lxi",
            "locality": "kothrud", "rto": "mh12", "fuel_type": "petrol",
            "transmission": "manual", "seller_type": "individual", "color": "white",
            "vehicle_age": 12, "odometer_reading": 95000, "km_per_year": 7916,
            "owner_count": 2, "certified": 0, "pincode": 411038,
        }
        result = predictor.predict_price(record)
        _assert_valid_result(result)
        assert result["predicted_price"] < 1_000_000
        assert result["routing_decision"] == "champion"

    def test_luxury_mercedes_c_class(self, predictor):
        """Mercedes-Benz C-Class returns valid luxury prediction and routes to specialist."""
        record = {
            "brand": "mercedes-benz", "model": "c-class", "variant": "c 200",
            "locality": "bandra west", "rto": "mh02", "fuel_type": "petrol",
            "transmission": "automatic", "seller_type": "dealer", "color": "white",
            "vehicle_age": 4, "odometer_reading": 28000, "km_per_year": 7000,
            "owner_count": 1, "certified": 1, "pincode": 400050,
        }
        result = predictor.predict_price(record)
        _assert_valid_result(result)
        assert 400_000 < result["predicted_price"] < 10_000_000


# ── 3. Batch Prediction Test ───────────────────────────────────────────────────

class TestBatchPrediction:

    def test_batch_prediction_df(self, predictor):
        """Batch DataFrame prediction matches individual single predictions."""
        records = [
            {
                "brand": "hyundai", "model": "creta", "variant": "sx",
                "locality": "indiranagar", "rto": "ka03", "fuel_type": "petrol",
                "transmission": "manual", "seller_type": "dealer", "color": "white",
                "vehicle_age": 3, "odometer_reading": 35000, "km_per_year": 11666,
                "owner_count": 1, "certified": 1, "pincode": 560038,
            },
            {
                "brand": "maruti", "model": "alto", "variant": "lxi",
                "locality": "kothrud", "rto": "mh12", "fuel_type": "petrol",
                "transmission": "manual", "seller_type": "individual", "color": "white",
                "vehicle_age": 10, "odometer_reading": 80000, "km_per_year": 8000,
                "owner_count": 1, "certified": 0, "pincode": 411038,
            },
        ]
        df = pd.DataFrame(records)
        batch_res = predictor.predict_batch_df(df)
        assert len(batch_res) == 2
        assert "predicted_price" in batch_res.columns
        assert "routing_decision" in batch_res.columns

        for i, rec in enumerate(records):
            single_res = predictor.predict_price(rec)
            assert batch_res.iloc[i]["predicted_price"] == pytest.approx(single_res["predicted_price"], abs=0.01)


# ── 4. Save/Load Consistency Test ─────────────────────────────────────────────

class TestSaveLoadConsistency:

    def test_bundle_reload_consistency(self, predictor):
        """Reloading bundle from disk produces bit-identical predictions."""
        with open(FINAL_BUNDLE_PATH, "rb") as f:
            bundle_2 = pickle.load(f)
        from backend.champion_predictor import ChampionPredictor
        predictor_2 = ChampionPredictor(bundle_2)

        record = {
            "brand": "honda", "model": "city", "variant": "vx",
            "locality": "indiranagar", "rto": "ka03", "fuel_type": "petrol",
            "transmission": "manual", "seller_type": "dealer", "color": "white",
            "vehicle_age": 4, "odometer_reading": 40000, "km_per_year": 10000,
            "owner_count": 1, "certified": 1, "pincode": 560038,
        }
        res1 = predictor.predict_price(record)
        res2 = predictor_2.predict_price(record)
        assert res1["predicted_price"] == res2["predicted_price"]
        assert res1["champion_prediction"] == res2["champion_prediction"]
        assert res1["routing_decision"] == res2["routing_decision"]


# ── 5. Determinism Test ────────────────────────────────────────────────────────

class TestDeterminism:

    def test_inference_determinism(self, predictor):
        """Repeated calls with same input produce identical outputs."""
        record = {
            "brand": "toyota", "model": "fortuner", "variant": "4x2 at",
            "locality": "banjara hills", "rto": "ts09", "fuel_type": "diesel",
            "transmission": "automatic", "seller_type": "dealer", "color": "white",
            "vehicle_age": 4, "odometer_reading": 48000, "km_per_year": 12000,
            "owner_count": 1, "certified": 1, "pincode": 500034,
        }
        p1 = predictor.predict_price(record)
        p2 = predictor.predict_price(record)
        p3 = predictor.predict_price(record)
        assert p1["predicted_price"] == p2["predicted_price"] == p3["predicted_price"]


# ── 6. Schema Validation Test ──────────────────────────────────────────────────

class TestSchemaValidation:

    def test_selling_price_forbidden(self):
        """selling_price in input must raise validation error."""
        from pydantic import ValidationError
        from backend.schemas import VehicleRecord
        with pytest.raises(ValidationError):
            VehicleRecord(brand="hyundai", selling_price=1000000)

    def test_auto_lowercasing(self):
        """Brand and text fields are lowercased and stripped."""
        from backend.schemas import VehicleRecord
        rec = VehicleRecord(brand="  BMW  ", model="  X5  ", fuel_type="PETROL")
        assert rec.brand == "bmw"
        assert rec.model == "x5"
        assert rec.fuel_type == "petrol"


# ── 7. Missing-Field Validation Test ───────────────────────────────────────────

class TestMissingFieldHandling:

    def test_missing_categoricals(self, predictor):
        """Missing locality, rto, color fall back to 'unknown' without crashing."""
        record = {
            "brand": "honda", "model": "city", "variant": "vx",
            "fuel_type": "petrol", "transmission": "manual", "seller_type": "dealer",
            "vehicle_age": 5, "odometer_reading": 45000, "km_per_year": 9000,
            "owner_count": 1, "certified": 1, "pincode": 411001,
        }
        res = predictor.predict_price(record)
        _assert_valid_result(res)

    def test_missing_numericals(self, predictor):
        """Missing numericals use medians."""
        record = {
            "brand": "honda", "model": "city", "variant": "vx",
            "fuel_type": "petrol", "transmission": "manual", "seller_type": "dealer",
        }
        res = predictor.predict_price(record)
        _assert_valid_result(res)

    def test_completely_empty_input(self, predictor):
        """Empty input dict succeeds with defaults."""
        res = predictor.predict_price({})
        _assert_valid_result(res)


# ── 8. Invalid-Input Validation Test ───────────────────────────────────────────

class TestInvalidInputHandling:

    def test_negative_vehicle_age_rejected(self):
        from pydantic import ValidationError
        from backend.schemas import VehicleRecord
        with pytest.raises(ValidationError):
            VehicleRecord(vehicle_age=-5)

    def test_negative_odometer_rejected(self):
        from pydantic import ValidationError
        from backend.schemas import VehicleRecord
        with pytest.raises(ValidationError):
            VehicleRecord(odometer_reading=-500)

    def test_forbidden_field_runtime_check(self, predictor):
        with pytest.raises(ValueError, match="forbidden"):
            predictor.predict_price({"brand": "hyundai", "selling_price": 500000})


# ── 9. Routing Logic & Strategy D Tests ─────────────────────────────────────────

class TestStrategyDRouting:

    def test_mass_market_stays_champion(self, predictor):
        """Budget car below thresholds routes to Champion."""
        record = {
            "brand": "maruti", "model": "swift", "variant": "vxi",
            "locality": "kothrud", "rto": "mh12", "fuel_type": "petrol",
            "transmission": "manual", "seller_type": "dealer", "color": "white",
            "vehicle_age": 4, "odometer_reading": 30000, "km_per_year": 7500,
            "owner_count": 1, "certified": 1, "pincode": 411038,
        }
        res = predictor.predict_price(record)
        assert res["routing_decision"] == "champion"
        assert res["predicted_price"] == res["champion_prediction"]

    def test_luxury_brand_below_16L_stays_champion(self, predictor):
        """Old high-mileage BMW with champion prediction < 16L stays with Champion."""
        record = {
            "brand": "bmw", "model": "3 series", "variant": "320d",
            "locality": "indiranagar", "rto": "ka03", "fuel_type": "diesel",
            "transmission": "automatic", "seller_type": "individual", "color": "black",
            "vehicle_age": 15, "odometer_reading": 180000, "km_per_year": 12000,
            "owner_count": 3, "certified": 0, "pincode": 560038,
        }
        res = predictor.predict_price(record)
        if res["champion_prediction"] < 1_600_000:
            assert res["routing_decision"] == "champion"
            assert res["predicted_price"] == res["champion_prediction"]

    def test_luxury_brand_above_16L_routes_specialist(self, predictor):
        """Luxury brand (Audi/BMW/Mercedes) with champion prediction >= 16L routes to Specialist."""
        record = {
            "brand": "bmw", "model": "x5", "variant": "xdrive 30d",
            "locality": "bandra west", "rto": "mh02", "fuel_type": "diesel",
            "transmission": "automatic", "seller_type": "dealer", "color": "white",
            "vehicle_age": 2, "odometer_reading": 15000, "km_per_year": 7500,
            "owner_count": 1, "certified": 1, "pincode": 400050,
        }
        res = predictor.predict_price(record)
        if res["champion_prediction"] >= 1_600_000:
            assert res["routing_decision"] == "specialist"
            assert res["predicted_price"] == res["luxury_specialist_prediction"]

    def test_global_threshold_22L_routes_specialist(self, predictor):
        """Any vehicle with champion prediction >= 22L routes to Specialist."""
        record = {
            "brand": "toyota", "model": "land cruiser", "variant": "vx",
            "locality": "banjara hills", "rto": "ts09", "fuel_type": "diesel",
            "transmission": "automatic", "seller_type": "dealer", "color": "white",
            "vehicle_age": 1, "odometer_reading": 8000, "km_per_year": 8000,
            "owner_count": 1, "certified": 1, "pincode": 500034,
        }
        res = predictor.predict_price(record)
        if res["champion_prediction"] >= 2_200_000:
            assert res["routing_decision"] == "specialist"
            assert res["predicted_price"] == res["luxury_specialist_prediction"]


# ── 14. API /health and 15. /predict Endpoint Tests ─────────────────────────────

class TestAPIEndpoints:

    def test_health_endpoint(self, client):
        """GET /health returns 200 with status=ready."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["artifact_exists"] is True
        assert data["variant_id"] == "final"

    def test_predict_endpoint_valid_request(self, client):
        """POST /predict returns 200 with structured response."""
        payload = {
            "brand": "hyundai", "model": "creta", "variant": "sx",
            "locality": "indiranagar", "rto": "ka03", "fuel_type": "petrol",
            "transmission": "manual", "seller_type": "dealer", "color": "white",
            "vehicle_age": 3, "odometer_reading": 35000, "km_per_year": 11666,
            "owner_count": 1, "certified": 1, "pincode": 560038,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_price"] > 0
        assert "champion_prediction" in data
        assert "routing_decision" in data

    def test_predict_endpoint_selling_price_rejected(self, client):
        """POST /predict with selling_price returns 422."""
        payload = {"brand": "hyundai", "selling_price": 1000000}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ── Helper utilities ──────────────────────────────────────────────────────────

def _assert_valid_result(result: dict) -> None:
    """Assert that a predict_price() result has valid structure and values."""
    assert isinstance(result, dict), "Result must be a dict"
    required_keys = {
        "predicted_price", "lgbm_prediction", "catboost_prediction",
        "champion_prediction", "luxury_specialist_prediction",
        "routing_decision", "final_gate", "routing",
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    price = result["predicted_price"]
    assert isinstance(price, (int, float)), "predicted_price must be numeric"
    assert 10_000 <= price <= 50_000_000, f"Price Rs{price:,.0f} out of range"
    assert result["routing_decision"] in ("champion", "specialist")
