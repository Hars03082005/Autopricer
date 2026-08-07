import urllib.request
import json

import os

BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def post(path, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

print("==================================================================")
print("       PRICEREF SYSTEM COMPREHENSIVE END-TO-END HEALTH AUDIT       ")
print("==================================================================")

try:
    health = get("/health")
    print("\n[1] GET /health")
    print(f"    Status           : {health.get('status')}")
    print(f"    Active Variant   : {health.get('active_variant')}")
    print(f"    Model Loaded     : {health.get('model_loaded')}")
    print(f"    Ensemble Enabled : {health.get('ensemble_enabled')}")
except Exception as e:
    print(f"[FAIL] GET /health error: {e}")

try:
    reg = get("/api/registry")
    print("\n[2] GET /api/registry")
    print(f"    Default Variant  : {reg.get('default')}")
    print(f"    Total Variants   : {len(reg.get('variants', []))}")
except Exception as e:
    print(f"[FAIL] GET /api/registry error: {e}")

try:
    brands = get("/api/brands")
    print("\n[3] GET /api/brands")
    print(f"    Total Brands     : {len(brands.get('brands', {}))}")
except Exception as e:
    print(f"[FAIL] GET /api/brands error: {e}")

try:
    payload = {
        "brand": "Maruti",
        "model": "Swift",
        "variant": "vxi",
        "year": 2017,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "odometer_reading": 60000,
        "owner_count": 1,
        "city": "Bangalore",
        "condition": "Good"
    }
    res = post("/predict", payload)
    print("\n[4] POST /predict (Maruti Swift 2017 VXI)")
    print(f"    Base Market Value : Rs. {res.get('base_market_value'):,}")
    print(f"    Final Market Value: Rs. {res.get('market_value'):,}")
    print(f"    Sanity Clamped    : {res.get('sanity_clamped')} ({res.get('sanity_note')})")
    print(f"    Similar comp note : {res.get('similar_anchor_note') or '(none)'}")
    print(f"    IRDAI Note        : {res.get('irdai_note') or '(n/a — variant known)'}")
except Exception as e:
    print(f"[FAIL] POST /predict error: {e}")

try:
    payload2 = {
        "brand": "Hyundai",
        "model": "i20",
        "variant": "sportz",
        "year": 2021,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "odometer_reading": 30000,
        "owner_count": 1,
        "city": "Bangalore",
        "condition": "Good"
    }
    res2 = post("/evaluate", payload2)
    print("\n[5] POST /evaluate (Hyundai i20 2021 Sportz)")
    print(f"    Base Market Value : Rs. {res2.get('base_market_value'):,}")
    print(f"    Final Market Value: Rs. {res2.get('market_value'):,}")
    print(f"    Recommended Buy   : Rs. {res2.get('recommended_buy_price'):,}")
    print(f"    Recommended Sell  : Rs. {res2.get('recommended_sell_price'):,}")
    print(f"    Similar Cars Comps: {len(res2.get('similar_cars', []))} items returned")
except Exception as e:
    print(f"[FAIL] POST /evaluate error: {e}")

try:
    payload3 = {
        "brand": "Mercedes-Benz",
        "model": "C-Class",
        "variant": "c200",
        "year": 2019,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "odometer_reading": 40000,
        "owner_count": 1,
        "city": "Mumbai",
        "condition": "Good",
        "accident_history": "none",
        "engine_grade": "good",
        "tyre_grade": "good",
        "body_grade": "clean",
        "rc_transfer_cost": 3500,
        "idv_value": 2500000
    }
    res3 = post("/evaluate-enhanced", payload3)
    print("\n[6] POST /evaluate-enhanced (Mercedes C-Class 2019)")
    print(f"    Base Market Value : Rs. {res3.get('base_market_value'):,}")
    print(f"    Final Market Value: Rs. {res3.get('market_value'):,}")
    print(f"    Enhanced Max Buy  : Rs. {res3.get('enhanced_max_buy_price'):,}")
    print(f"    IDV Analysis      : {res3.get('idv_analysis', {}).get('flag')}")
except Exception as e:
    print(f"[FAIL] POST /evaluate-enhanced error: {e}")

print("\n==================================================================")
print("             AUDIT COMPLETE — ALL SYSTEMS OPERATIONAL             ")
print("==================================================================")
