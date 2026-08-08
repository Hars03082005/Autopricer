import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import VehicleInput, predict_market_value, _normalize_brand, normalize_model_name  # pyrefly: ignore [missing-import]

print('=== Brand Normalization Test ===')
brands = [
    ('Maruti',          'maruti suzuki'),
    ('Mercedes',        'mercedes-benz'),
    ('VW',              'volkswagen'),
    ('maruti suzuki',   'maruti suzuki'),
    ('Land Rover',      'land rover'),
    ('mercedes benz',   'mercedes-benz'),
    ('Suzuki',          'maruti suzuki'),
]
for inp, expected in brands:
    got = _normalize_brand(inp)
    status = 'OK' if got == expected else 'FAIL'
    print(f'  [{status}] {inp!r:<22} -> {got!r}  (expected {expected!r})')

print()
print('=== Model Normalization Test (variant stripping) ===')
cases = [
    ('Maruti',  'Swift VXI',    2017, 'swift'),
    ('Hyundai', 'i20 Sportz',   2021, 'i20'),
    ('Maruti',  'Swift',        2017, 'swift'),
    ('Toyota',  'Innova',       2020, 'innova crysta'),
    ('Honda',   'City SV',      2021, 'city'),
    ('Tata',    'Nexon XZ+',    2022, 'nexon'),
]
for brand, model, year, expected in cases:
    got = normalize_model_name(brand, model, year)
    status = 'OK' if got == expected else 'INFO'
    print(f'  [{status}] {brand} {model!r} ({year}) -> {got!r}  (expected {expected!r})')

print()
print('=== Prediction Test ===')
v = VehicleInput(brand='Maruti', model='Swift', variant='VXI', year=2017,
                 fuel_type='Petrol', transmission='Manual', odometer_reading=60000,
                 owner_count=1, city='Bangalore', condition='Good')
r = predict_market_value(v)
print('  Market value:        Rs.', r['market_value'])
print('  Similar anchor note:', r['similar_anchor_note'] or '(none)')
print('  IRDAI note:         ', r['irdai_note'] or '(n/a - variant known)')
print('  Variant is known:   ', r['variant_is_known'])

print()
print('=== Unknown Variant -> IRDAI Note Test ===')
v2 = VehicleInput(brand='Maruti', model='Swift', variant='unknown', year=2017,
                  fuel_type='Petrol', transmission='Manual', odometer_reading=60000,
                  owner_count=1, city='Bangalore', condition='Good')
r2 = predict_market_value(v2)
print('  Market value:        Rs.', r2['market_value'])
print('  IRDAI note:         ', r2['irdai_note'])
print('  Variant is known:   ', r2['variant_is_known'])
