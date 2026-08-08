import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import VehicleInput, predict_market_value  # pyrefly: ignore [missing-import]

base = dict(brand='Maruti', model='Swift', variant='VXI', year=2017,
            fuel_type='Petrol', transmission='Manual', owner_count=1,
            city='Bangalore', condition='Good')

def pred(**overrides):
    v = VehicleInput(**{**base, **overrides})
    return predict_market_value(v)['market_value']

ref = pred(odometer_reading=60000)
print('=== Feature Sensitivity Test (2017 Maruti Swift VXI, Bangalore) ===')
print('Reference (60k km, 1 owner, Good, Petrol): Rs.', ref)
print()

print('--- ODOMETER ---')
for km in [20000, 40000, 60000, 100000, 150000]:
    v = pred(odometer_reading=km)
    print(f'  {km//1000:>3}k km:  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- OWNER COUNT ---')
for o in [1, 2, 3, 4]:
    v = pred(odometer_reading=60000, owner_count=o)
    print(f'  {o} owner(s):  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- CONDITION ---')
for cond in ['Excellent', 'Good', 'Average', 'Poor']:
    v = pred(odometer_reading=60000, condition=cond)
    print(f'  {cond:<10}:  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- FUEL TYPE ---')
for fuel in ['Petrol', 'Diesel', 'CNG']:
    v = pred(odometer_reading=60000, fuel_type=fuel)
    print(f'  {fuel:<8}:  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- VEHICLE YEAR (AGE) ---')
for yr in [2022, 2020, 2018, 2017, 2015, 2013]:
    v = pred(odometer_reading=60000, year=yr)
    age = 2026 - yr
    print(f'  {yr} ({age} yrs):  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- CITY ---')
for city in ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Jaipur', 'Bhopal']:
    v = pred(odometer_reading=60000, city=city)
    print(f'  {city:<15}:  Rs. {v:>9,}  diff: {v - ref:+,}')

print()
print('--- BRAND / MODEL ---')
for brand, model in [('Maruti', 'Swift'), ('Hyundai', 'i20'), ('Honda', 'City'), ('Toyota', 'Innova')]:
    v = pred(odometer_reading=60000, brand=brand, model=model)
    print(f'  {brand} {model:<15}:  Rs. {v:>9,}')
