import pytest
from fastapi.testclient import TestClient
from backend.main import app, CURRENT_YEAR

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data.get('status') == 'ok'
    assert 'active_variant' in data

def test_api_brands():
    response = client.get('/api/brands')
    assert response.status_code == 200
    data = response.json()
    assert 'brands' in data
    assert isinstance(data['brands'], dict)

def test_api_options():
    response = client.get('/api/options?brand=Mahindra&model=XUV500')
    assert response.status_code == 200
    data = response.json()
    assert 'years' in data
    assert 'fuel_types' in data
    assert 'transmissions' in data
    assert str(CURRENT_YEAR) in data['years']

def test_evaluate_endpoint():
    payload = {
        'brand': 'Mahindra',
        'model': 'XUV500',
        'variant': 'W8',
        'year': 2014,
        'odometer_reading': 120000,
        'fuel_type': 'Diesel',
        'transmission': 'Manual',
        'owner_count': 1,
        'city': 'Bangalore',
        'condition': 'Good'
    }
    response = client.post('/evaluate', json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 'market_value' in res
    assert 'price_min' in res
    assert 'price_max' in res
    assert res['price_min'] <= res['price_max']
    assert res['market_value'] > 0

def test_predict_endpoint():
    payload = {
        'brand': 'Maruti Suzuki',
        'model': 'Swift',
        'variant': 'VXI',
        'year': 2018,
        'odometer_reading': 50000,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'owner_count': 1,
        'city': 'Bangalore',
        'condition': 'Good'
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 'predicted_price' in res or 'market_value' in res

def test_reverse_calculate():
    payload = {
        'expected_sell_price': 500000,
        'year': 2020,
        'odometer': 45000,
        'owner_count': 1,
        'target_margin_pct': 10.0
    }
    response = client.post('/reverse-calculate', json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 'max_buy_price' in res
    assert res['max_buy_price'] >= 0
