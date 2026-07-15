// ============================================================
// Mock Data – Vehicle Valuation Platform
// ============================================================

/** Car images mapped by brand + model key */
export const CAR_IMAGES = {
  'Honda City':          '/cars/honda_city.png',
  'Honda Amaze':         '/cars/honda_city.png',
  'Honda Jazz':          '/cars/honda_city.png',
  'Honda WR-V':          '/cars/honda_city.png',
  'Honda CR-V':          '/cars/honda_city.png',
  'Hyundai Creta':       '/cars/hyundai_creta.png',
  'Hyundai i20':         '/cars/hyundai_creta.png',
  'Hyundai Venue':       '/cars/hyundai_creta.png',
  'Hyundai Tucson':      '/cars/hyundai_creta.png',
  'Hyundai Alcazar':     '/cars/hyundai_creta.png',
  'Maruti Swift':        '/cars/maruti_swift.png',
  'Maruti Baleno':       '/cars/maruti_swift.png',
  'Maruti Vitara Brezza':'/cars/maruti_swift.png',
  'Maruti Ertiga':       '/cars/maruti_swift.png',
  'Maruti Alto':         '/cars/maruti_swift.png',
  'Toyota Fortuner':     '/cars/toyota_fortuner.png',
  'Toyota Innova':       '/cars/toyota_fortuner.png',
  'Toyota Camry':        '/cars/toyota_fortuner.png',
  'Toyota Yaris':        '/cars/toyota_fortuner.png',
  'Toyota Urban Cruiser':'/cars/toyota_fortuner.png',
  'BMW 3 Series':        '/cars/bmw_3series.png',
  'BMW 5 Series':        '/cars/bmw_3series.png',
  'BMW X1':              '/cars/bmw_3series.png',
  'BMW X3':              '/cars/bmw_3series.png',
  'BMW X5':              '/cars/bmw_3series.png',
  'Tesla Model 3':       '/cars/tesla_model3.png',
  'Tesla Model S':       '/cars/tesla_model3.png',
  'Tesla Model X':       '/cars/tesla_model3.png',
  'Tesla Model Y':       '/cars/tesla_model3.png',
  'Tata Nexon':          '/cars/tata_nexon.png',
  'Tata Harrier':        '/cars/tata_nexon.png',
  'Tata Safari':         '/cars/tata_nexon.png',
  'Tata Altroz':         '/cars/tata_nexon.png',
  'Tata Punch':          '/cars/tata_nexon.png',
  'Kia Seltos':          '/cars/kia_seltos.png',
  'Kia Sonet':           '/cars/kia_seltos.png',
  'Kia Carnival':        '/cars/kia_seltos.png',
  'Kia EV6':             '/cars/kia_seltos.png',
};

/** VIN Database: pre-loaded registrations for auto-fill demo */
export const VIN_DATABASE = {
  'VIN-HONDA-2021': {
    brand: 'Honda', model: 'City', year: 2021, fuel: 'Petrol',
    transmission: 'Manual', mileage: 28000, fuelEfficiency: 17.5, city: 'Mumbai', ownerCount: 1, engineCc: 1498, hoursRun: 950, sellerAskingPrice: 840000,
  },
  'VIN-TESLA-2023': {
    brand: 'Tesla', model: 'Model 3', year: 2023, fuel: 'Electric',
    transmission: 'Automatic', mileage: 12000, fuelEfficiency: 0, city: 'Bangalore', ownerCount: 1, engineCc: 0, hoursRun: 420, sellerAskingPrice: 4100000,
  },
  'VIN-HYUNDAI-2020': {
    brand: 'Hyundai', model: 'Creta', year: 2020, fuel: 'Diesel',
    transmission: 'Automatic', mileage: 45000, fuelEfficiency: 18.0, city: 'Delhi', ownerCount: 2, engineCc: 1493, hoursRun: 1850, sellerAskingPrice: 1050000,
  },
  'VIN-MARUTI-2019': {
    brand: 'Maruti', model: 'Swift', year: 2019, fuel: 'Petrol',
    transmission: 'Manual', mileage: 62000, fuelEfficiency: 20.4, city: 'Pune', ownerCount: 2, engineCc: 1197, hoursRun: 2300, sellerAskingPrice: 520000,
  },
  'VIN-BMW-2022': {
    brand: 'BMW', model: '3 Series', year: 2022, fuel: 'Petrol',
    transmission: 'Automatic', mileage: 18000, fuelEfficiency: 14.8, city: 'Hyderabad', ownerCount: 1, engineCc: 1998, hoursRun: 700, sellerAskingPrice: 4300000,
  },
};

/** Available brands and their models */
export const BRANDS = {
  Honda:   ['City', 'Amaze', 'Jazz', 'WR-V', 'CR-V'],
  Hyundai: ['Creta', 'i20', 'Venue', 'Tucson', 'Alcazar'],
  Maruti:  ['Swift', 'Baleno', 'Vitara Brezza', 'Ertiga', 'Alto'],
  Toyota:  ['Innova', 'Fortuner', 'Camry', 'Yaris', 'Urban Cruiser'],
  BMW:     ['3 Series', '5 Series', 'X1', 'X3', 'X5'],
  Tesla:   ['Model 3', 'Model S', 'Model X', 'Model Y'],
  Tata:    ['Nexon', 'Harrier', 'Safari', 'Altroz', 'Punch'],
  Kia:     ['Seltos', 'Sonet', 'Carnival', 'EV6'],
};

/** City demand scores and multipliers */
export const CITY_DEMAND = {
  Mumbai:    { score: 92, multiplier: 1.08 },
  Delhi:     { score: 88, multiplier: 1.05 },
  Bangalore: { score: 95, multiplier: 1.12 },
  Pune:      { score: 82, multiplier: 1.02 },
  Hyderabad: { score: 78, multiplier: 0.98 },
  Chennai:   { score: 75, multiplier: 0.96 },
  Kolkata:   { score: 68, multiplier: 0.91 },
  Ahmedabad: { score: 71, multiplier: 0.93 },
  Jaipur:    { score: 60, multiplier: 0.87 },
  Surat:     { score: 58, multiplier: 0.85 },
};

/** Comparable market transactions */

export const MODEL_COMPARISON = [
  { model: 'Linear Regression', mae: 98000, rmse: 142000, mape: 14.8, r2: 0.78, speed: 'Fast', selected: false },
  { model: 'Ridge Regression', mae: 91000, rmse: 133000, mape: 13.6, r2: 0.81, speed: 'Fast', selected: false },
  { model: 'Random Forest', mae: 64000, rmse: 93000, mape: 9.4, r2: 0.91, speed: 'Medium', selected: false },
  { model: 'XGBoost', mae: 58000, rmse: 84000, mape: 8.7, r2: 0.93, speed: 'Fast', selected: false },
  { model: 'CatBoost', mae: 52000, rmse: 79000, mape: 7.9, r2: 0.94, speed: 'Fast', selected: true },
];

export const BRAND_RETENTION = [
  { brand:'Toyota', retention:92, demand:88, margin:18.9 },
  { brand:'Hyundai', retention:84, demand:86, margin:18.1 },
  { brand:'Honda', retention:82, demand:81, margin:18.6 },
  { brand:'Kia', retention:78, demand:74, margin:18.1 },
  { brand:'Maruti', retention:76, demand:91, margin:18.6 },
  { brand:'Tata', retention:72, demand:77, margin:17.3 },
  { brand:'BMW', retention:69, demand:55, margin:16.9 },
  { brand:'Tesla', retention:67, demand:58, margin:18.9 },
];

export const FUEL_ANALYTICS = [
  { fuel:'Petrol', avgPrice:650000, avgRisk:39, avgMargin:18.4, demand:86 },
  { fuel:'Diesel', avgPrice:1500000, avgRisk:45, avgMargin:18.2, demand:78 },
  { fuel:'Electric', avgPrice:2500000, avgRisk:43, avgMargin:18.1, demand:72 },
  { fuel:'CNG', avgPrice:520000, avgRisk:51, avgMargin:16.8, demand:64 },
];

export const AGE_PRICE_CURVE = [
  { age:'0–2 yrs', price:1850000, risk:18, margin:13.5 },
  { age:'3–4 yrs', price:1320000, risk:29, margin:16.8 },
  { age:'5–6 yrs', price:940000, risk:38, margin:18.4 },
  { age:'7–8 yrs', price:690000, risk:55, margin:19.1 },
  { age:'9+ yrs', price:410000, risk:72, margin:20.3 },
];

export function formatINR(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return '₹0';
  if (Math.abs(n) >= 10000000) return `₹${(n / 10000000).toFixed(2)}Cr`;
  if (Math.abs(n) >= 100000) return `₹${(n / 100000).toFixed(2)}L`;
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n);
}

export function getConditionLabel(score) {
  if (score >= 85) return { label: 'Excellent', color: '#00a651' };
  if (score >= 70) return { label: 'Good', color: '#007be5' };
  if (score >= 50) return { label: 'Average', color: '#f7941d' };
  return { label: 'Poor', color: '#e02020' };
}
