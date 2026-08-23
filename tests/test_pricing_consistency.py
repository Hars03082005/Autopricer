import pytest
from pathlib import Path

pytest.importorskip("joblib", reason="requires ML stack")
pytest.importorskip("pandas", reason="requires ML stack")

pytestmark = pytest.mark.models

from backend.main import evaluate_vehicle, VehicleInput  # noqa: E402


@pytest.mark.parametrize("brand,model,variant,year,fuel,transmission,odo,condition,owner", [
    ("Maruti", "Alto 800", "LXI", 2015, "Petrol", "Manual", 45000, "Good", 1),         # Low price
    ("Hyundai", "i10", "Sportz", 2016, "Petrol", "Manual", 75000, "Good", 1),          # User case
    ("Honda", "City", "VX", 2021, "Petrol", "Manual", 28000, "Good", 1),               # Mid price
    ("Hyundai", "Creta", "SX", 2020, "Diesel", "Manual", 55000, "Good", 1),             # Compact/Mid SUV
    ("BMW", "3 Series", "330i M Sport", 2022, "Petrol", "Automatic", 22000, "Good", 1), # Luxury
    ("Toyota", "Innova", "2.5 GX", 2014, "Diesel", "Manual", 145000, "Average", 2),    # High mileage (>1L km)
    ("Maruti", "Swift", "VXI", 2018, "Petrol", "Manual", 92000, "Good", 1),            # High-vol hatchback
])
def test_selling_range_contains_market_value(brand, model, variant, year, fuel, transmission, odo, condition, owner):
    """Enforces: selling_range_lower <= estimated_market_value <= selling_range_upper across all segments."""
    v = VehicleInput(
        brand=brand,
        model=model,
        variant=variant,
        year=year,
        fuel_type=fuel,
        transmission=transmission,
        odometer_reading=odo,
        owner_count=owner,
        condition=condition,
        city="Bangalore",
        locality="Indiranagar",
    )
    result = evaluate_vehicle(v)

    market_val = result["market_value"]
    price_min = result["price_min"]
    price_max = result["price_max"]

    assert price_min <= market_val, f"price_min ({price_min}) must be <= market_value ({market_val}) for {brand} {model}"
    assert market_val <= price_max, f"market_value ({market_val}) must be <= price_max ({price_max}) for {brand} {model}"
    assert price_min < price_max, f"price_min ({price_min}) must be strictly < price_max ({price_max})"


def test_buy_decision_contains_range_and_exact_target():
    """Buying side must provide both negotiation range and concrete target acquisition price."""
    v = VehicleInput(
        brand="Hyundai",
        model="i10",
        variant="Sportz",
        year=2016,
        fuel_type="Petrol",
        transmission="Manual",
        odometer_reading=75000,
        owner_count=1,
        condition="Good",
        city="Bangalore",
    )
    result = evaluate_vehicle(v)

    assert "recommended_buy_price" in result
    assert result["recommended_buy_price"] > 0
    assert "opening_offer" in result
    assert "max_offer" in result
    assert result["opening_offer"] <= result["recommended_buy_price"] <= result["max_offer"]


def test_profit_and_roi_mathematical_consistency():
    """Net profit and ROI must remain mathematically consistent with cost waterfall."""
    v = VehicleInput(
        brand="Honda",
        model="City",
        variant="VX",
        year=2021,
        fuel_type="Petrol",
        transmission="Manual",
        odometer_reading=28000,
        owner_count=1,
        condition="Good",
        city="Bangalore",
    )
    result = evaluate_vehicle(v)

    buy_price = result["recommended_buy_price"]
    sell_price = result["recommended_sell_price"]
    profit = result["expected_profit"]
    margin_pct = result["expected_margin_pct"]
    recon = result.get("recon_cost", 0)
    holding = result.get("holding_cost", 0)
    doc = result.get("doc_cost", 0)

    # Net profit = Sell - Buy - Operating costs
    expected_profit_calc = max(0, sell_price - buy_price - recon - holding - doc)
    assert abs(profit - expected_profit_calc) <= 1000, f"Profit {profit} != calculated {expected_profit_calc}"

    # Margin pct = (Profit / Buy) * 100
    expected_roi = round((profit / max(buy_price, 1)) * 100, 1)
    assert abs(margin_pct - expected_roi) <= 0.5, f"ROI {margin_pct}% != calculated {expected_roi}%"


def test_no_old_labels_in_result_screen():
    """Ensure ResultScreen does not contain removed misleading labels."""
    result_screen_path = Path(__file__).resolve().parents[1] / "src" / "screens" / "ResultScreen.jsx"
    if not result_screen_path.exists():
        result_screen_path = Path("src/screens/ResultScreen.jsx")
    if not result_screen_path.exists():
        pytest.skip("src/screens/ResultScreen.jsx not found")
    content = result_screen_path.read_text(encoding="utf-8")

    assert "Projected Retail Realization" not in content
    assert "Target Listing Price" not in content
    assert "Target listing price" not in content
    assert "Expected selling realization" not in content
    assert "EXPECTED SELLING RANGE" in content
    assert "RECOMMENDED BUY RANGE" in content
    assert "RECOMMENDED BUY PRICE" in content
    assert "EXPECTED NET PROFIT" in content
