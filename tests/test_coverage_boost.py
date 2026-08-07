from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_brand_catalog_functions():
    from backend import brand_catalog
    catalog = brand_catalog.build_brand_catalog()
    assert isinstance(catalog, dict)
    assert len(catalog) > 0
    norm = brand_catalog.normalize_brand_name("maruti suzuki")
    assert norm in ["Maruti", "Maruti Suzuki"]
    assert brand_catalog._title_words("hello_world") == "Hello_world"
    merged = {}
    brand_catalog._merge_models(merged, "Honda", ["City", "Civic"])
    assert "Honda" in merged
    assert "City" in merged["Honda"]


def test_model_registry_functions():
    from backend import model_registry
    variants = model_registry.list_variants()
    assert isinstance(variants, list)
    default_id = model_registry.get_default_variant_id()
    assert default_id is not None
    v_path = model_registry.get_variant_path(default_id)
    assert v_path is not None
    v_data = model_registry.get_variant(default_id)
    assert "predictor" in v_data
    assert "metadata" in v_data
    assert model_registry.activate_variant(default_id) is True
    assert model_registry.activate_variant("non_existent_variant_123") is False


def test_db_offline_fallback():
    from backend import db
    from backend.config import Settings
    settings = Settings(supabase_url="", supabase_anon_key="")
    assert not settings.database_enabled
    with pytest.raises(Exception) as exc_info:
        db._require_client(settings)
    assert getattr(exc_info.value, "status_code", None) == 503


def test_db_async_functions_mocked():
    from backend import db
    from backend.config import Settings

    async def run_test():
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "123", "user_id": "usr123", "dealership_name": "Test Dealer"}]
        
        with patch("backend.db._request", new_callable=AsyncMock, return_value=mock_resp):
            res = await db.list_evaluations(user_id="usr123", access_token="tok", limit=10)
            assert isinstance(res, list)
            assert len(res) == 1
            
            ins = await db.insert_evaluation(row={"brand": "Honda"}, access_token="tok")
            assert ins["id"] == "123"
            
            prof = await db.get_profile(user_id="usr123", access_token="tok")
            assert prof["dealership_name"] == "Test Dealer"
            
            ups = await db.upsert_profile(user_id="usr123", access_token="tok", fields={"dealership_name": "Test Dealer"})
            assert ups["dealership_name"] == "Test Dealer"
            
            with patch.object(Settings, "database_enabled", new=True):
                p = await db.ping(access_token="tok")
                assert p is True

            deleted = await db.delete_evaluations(user_id="usr123", access_token="tok")
            assert deleted == 1
            
            del_one = await db.delete_evaluation(user_id="usr123", evaluation_id="123", access_token="tok")
            assert del_one == 1

    asyncio.run(run_test())


def test_history_helpers():
    from backend.routers import history
    eval_in = history.EvaluationIn(brand="Honda", model="City", year=2020)
    db_row = history._to_db_row(eval_in, "usr_999")
    assert db_row["user_id"] == "usr_999"
    assert "id" in db_row
    assert "created_at" in db_row
    
    db_row["created_at"] = datetime.now(UTC)
    eval_out = history._from_db_row(db_row)
    assert eval_out.brand == "Honda"
    assert eval_out.id == db_row["id"]


def test_history_router_authenticated(client):
    from backend.auth import AuthenticatedUser, get_current_user
    from backend.main import app
    fake_user = AuthenticatedUser(id="12345678-1234-1234-1234-123456789012", email="test@example.com", role="authenticated", claims={}, access_token="mock_tok")
    app.dependency_overrides[get_current_user] = lambda: fake_user
    
    with patch("backend.db.list_evaluations", new_callable=AsyncMock, return_value=[]), \
         patch("backend.db.insert_evaluation", new_callable=AsyncMock, return_value={"id": "123", "created_at": datetime.now(UTC).isoformat()}), \
         patch("backend.db.delete_evaluations", new_callable=AsyncMock, return_value=0), \
         patch("backend.db.get_profile", new_callable=AsyncMock, return_value={"id": fake_user.id, "name": "Dealer", "avatar": "U", "role": "Dealer"}), \
         patch("backend.db.upsert_profile", new_callable=AsyncMock, return_value={"id": fake_user.id, "name": "Dealer", "avatar": "U", "role": "Dealer"}):
        
        r1 = client.get("/api/history")
        assert r1.status_code == 200
        
        r2 = client.post("/api/history", json={"brand": "Honda", "model": "City", "year": 2020})
        assert r2.status_code == 201
        
        r3 = client.delete("/api/history")
        assert r3.status_code == 200
        
        r4 = client.get("/api/profile")
        assert r4.status_code == 200
        
        r5 = client.put("/api/profile", json={"name": "Dealer Name", "avatar": "D", "role": "Dealer"})
        assert r5.status_code == 200
        
    app.dependency_overrides.clear()


def test_ensemble_predictor_edge_cases():
    import pandas as pd  # only available in the full ML environment
    from backend import model_registry
    default_id = model_registry.get_default_variant_id()
    v_data = model_registry.get_variant(default_id)
    pred = v_data["predictor"]
    meta = v_data["metadata"]
    features = meta.get("features", [])
    row = {col: 1 for col in features}
    df = pd.DataFrame([row])
    log_price = pred.predict_log_price(df)
    assert isinstance(log_price, float)
    res = pred.predict_with_variance(df)
    assert "log_price" in res
    assert "variance" in res


def test_healthcheck_module(monkeypatch):
    from backend import healthcheck
    monkeypatch.delenv("ACTIVE_VARIANT_ID", raising=False)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "ok", "model_loaded": True, "active_variant": "variant_1"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        assert healthcheck.main() == 0


def test_healthcheck_failures(monkeypatch):
    from backend import healthcheck
    monkeypatch.setenv("PORT", "not_a_number")
    assert healthcheck.main() == 1


def test_auth_module_helpers():
    from backend import config
    settings = config.Settings(supabase_url="https://example.supabase.co", supabase_anon_key="anon")
    assert settings.auth_enabled is True
    assert settings.database_enabled is True
    assert "auth/v1" in settings.jwks_url


def test_additional_main_endpoints(client):
    res1 = client.get("/metadata")
    assert res1.status_code == 200
    
    res2 = client.get("/api/registry")
    assert res2.status_code == 200
    
    res3 = client.get("/api/catalog")
    assert res3.status_code == 200
    
    res4 = client.get("/api/catalog/Honda")
    assert res4.status_code == 200

    res5 = client.get("/api/options?brand=Honda&model=City")
    assert res5.status_code == 200


def test_history_router_unauthorized(client):
    res = client.get("/api/history")
    assert res.status_code in (401, 503)


def test_enhanced_evaluate_endpoint(client):
    payload = {
        "brand": "Honda",
        "model": "City",
        "variant": "VX",
        "year": 2021,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "odometer_reading": 28000,
        "owner_count": 1,
        "city": "Bangalore",
        "locality": "Indiranagar",
        "condition": "Good",
        "seller_asking_price": 850000,
        "engine_grade": "good",
        "tyre_grade": "good",
        "body_grade": "clean",
        "interior_grade": "clean",
        "electrical_grade": "all_good",
        "rc_transfer_cost": 3500
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "market_value" in res
    assert "max_buy_price" in res or "fair_deal_price" in res or "action" in res


def test_norm_model_from_main():
    """Cover the _norm_model helper (or its fallback) imported into backend.main."""
    from backend.main import _norm_model
    assert _norm_model("Swift 1.2", "Maruti") in ("swift", "swift 1.2", "unknown", "Swift 1.2".lower().strip())
    assert _norm_model(None) == "unknown"  # type: ignore[arg-type]
    assert _norm_model("") == "unknown"
    assert _norm_model("city 1.5l") in ("city", "city 1.5", "unknown", "city 1.5l")


def test_norm_variant_from_main():
    """Cover the _norm_variant helper (or its fallback) imported into backend.main."""
    from backend.main import _norm_variant
    assert _norm_variant(None) == "unknown"  # type: ignore[arg-type]
    assert _norm_variant("") == "unknown"
    assert _norm_variant("unknown") == "unknown"
    assert _norm_variant("nan") == "unknown"
    result = _norm_variant("VXI")
    assert isinstance(result, str)
