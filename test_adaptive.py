import requests

localities = ["Indiranagar", "Rajajinagar", "Whitefield", "Koramangala", "Peenya"]

print("=== ECONOMY: Hyundai Grand i10 (2019) ===")
base_economy = {
    "brand": "Hyundai", "model": "Grand i10", "variant": "Sportz",
    "year": 2019, "fuel_type": "Petrol", "transmission": "Manual",
    "odometer_reading": 45000, "owner_count": 1, "condition": "Good",
    "city": "Bangalore", "seller_type": "Individual",
}
for loc in localities:
    p = {**base_economy, "locality": loc}
    r = requests.post("http://localhost:8000/evaluate", json=p, timeout=20).json()
    print(f"  {loc:<15} -> Market Price: Rs.{r.get('market_value'):>9,} | Range: Rs.{r.get('price_min'):>9,} - Rs.{r.get('price_max'):>9,}")

print("\n=== LUXURY: BMW 3 Series (2022) ===")
base_luxury = {
    "brand": "BMW", "model": "3 Series", "variant": "330i M Sport",
    "year": 2022, "fuel_type": "Petrol", "transmission": "Automatic",
    "odometer_reading": 22000, "owner_count": 1, "condition": "Good",
    "city": "Bangalore", "seller_type": "Individual",
}
for loc in localities:
    p = {**base_luxury, "locality": loc}
    r = requests.post("http://localhost:8000/evaluate", json=p, timeout=20).json()
    print(f"  {loc:<15} -> Market Price: Rs.{r.get('market_value'):>9,} | Range: Rs.{r.get('price_min'):>9,} - Rs.{r.get('price_max'):>9,}")
