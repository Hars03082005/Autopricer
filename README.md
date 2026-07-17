# PriceRef — Used Car Valuation for Dealerships

A dealership-internal tool that tells you in under a second whether to buy a car, at what price, and why.

The dealer fills in vehicle details, the ML backend predicts market value using a trained ensemble of CatBoost + LightGBM + XGBoost, applies a condition adjustment, then a rule-based decision engine computes buy price, sell price, expected profit, risk score, and a BUY / NEGOTIATE / REJECT call.

---

## How it works

```
Dealer enters vehicle details
        ↓
FastAPI backend receives the request
        ↓
Brand → segment class (economy / premium / luxury) via O(1) dict lookup
        ↓
Segment-specific ensemble predicts base market value (log1p → expm1)
        ↓
Condition multiplier applied (Excellent +3.5% / Good ±0% / Average −6% / Poor −14%)
        ↓
Decision engine outputs:
    buy price · sell price · expected profit · risk score
    confidence score · deal quality · negotiation trio
    BUY / NEGOTIATE / REJECT
        ↓
React frontend renders the result
```

---

## Stack

**Frontend** — React 19 + Vite, plain CSS, Recharts for charts, React Context for state.

**Backend** — Python 3.13, FastAPI, Uvicorn, Pydantic, Joblib.

**ML** — CatBoost (dominant model), LightGBM, XGBoost, scikit-learn for splits and metrics, SciPy SLSQP for ensemble weight optimisation.

**Mobile** — Flutter shell wrapping the React build in a WebView.

---

## Project layout

```
Price-Prediction/
├── backend/
│   ├── main.py                  # FastAPI routes and segment routing
│   ├── decision_engine.py       # Buy/sell/profit/risk formulas
│   ├── ensemble_predictor.py    # Model loader
│   └── brand_catalog.py         # Brand → segment map
│
├── ml_training/
│   ├── train_ml_model.py        # Training pipeline
│   ├── clean_data.py            # Data cleaning (handles multiple raw schemas)
│   └── data/                    # Raw + cleaned CSVs (gitignored)
│
├── model_artifacts/             # Trained model binaries (gitignored except metadata)
│   ├── vehicle_price_catboost.cbm
│   ├── vehicle_price_lightgbm.txt
│   ├── vehicle_price_xgboost.json
│   ├── ensemble_*.pkl           # Per-segment and per-band ensemble bundles
│   ├── segment_*.cbm            # Price-band CatBoost models (₹6–12L, ₹12L+)
│   ├── routing_table.json       # Which model to use for which price range
│   ├── model_metadata.json      # Full training metadata + metrics
│   └── training_report.json     # Last training run report
│
├── src/
│   ├── screens/                 # React screens (Auth, Home, Input, Result, Dashboard...)
│   ├── components/              # Shared components
│   ├── context/AppContext.jsx
│   └── utils/apiValuation.js
│
├── mobile/                      # Flutter WebView shell
├── scripts/                     # Build helpers
├── index.html
├── vite.config.js
└── docker-compose.yml
```

---

## ML details


**Features:** 19 total — 9 categorical (`brand`, `model`, `variant`, `city`, `rto_state`, `color`, `segment_class`, `fuel_type`, `transmission`) and 10 numeric (`vehicle_age`, `odometer_reading`, `km_per_year`, `owner_count`, `ownership_trust_score`, `vehicle_health_score`, `inspected`, `high_mileage`, `luxury_brand`, `has_list_price`).

**Training split:** 70% train / 15% validation / 15% test.

**Ensemble weights (current):** XGBoost 98%, CatBoost 2%, LightGBM 0% — SLSQP optimised on validation R².

**Target transform:** `log1p(selling_price)` during training, `expm1()` at prediction time.

### Model performance (test set)

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Ensemble | 0.9904 | ₹18,266 | 3.13% |
| XGBoost | 0.9904 | ₹18,009 | 3.09% |
| LightGBM | 0.9881 | ₹23,469 | 3.99% |
| CatBoost | 0.9725 | ₹45,949 | 7.72% |

### Segment model performance (v5.0 global model)

| Segment | Rows | R² | MAE | MAPE |
|---|---|---|---|---|
| Economy | 207,135 | 0.9872 | ₹32,503 | 5.26% |
| Premium | 3,301 | 0.9056 | ₹80,438 | 15.29% |
| Luxury | 3,384 | 0.9976 | ₹13,709 | 1.08% |

---

## API

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server + model status |
| GET | `/metadata` | Training metadata and metrics |
| GET | `/api/brands` | Brand list for dropdowns |
| POST | `/predict` | Market value prediction |
| POST | `/evaluate` | Full dealer evaluation |
| POST | `/evaluate-enhanced` | Evaluation with enriched recon/risk breakdown |
| POST | `/reverse-calculate` | Target sell price → max buy price |
| POST | `/bulk-evaluate` | Batch evaluation |

**Sample `/evaluate` request:**
```json
{
  "brand": "Honda",
  "model": "City",
  "year": 2021,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "odometer_reading": 28000,
  "owner_count": 1,
  "city": "Bangalore",
  "condition": "Good",
  "seller_asking_price": 750000,
  "target_margin_pct": 15,
  "repair_buffer": 25000
}
```

**Sample response:**
```json
{
  "market_value": 735000,
  "condition_multiplier": 1.0,
  "segment_class": "economy",
  "recommended_buy_price": 580000,
  "recommended_sell_price": 771750,
  "expected_profit": 91500,
  "risk_score": 22,
  "confidence_score": 78,
  "action": "BUY"
}
```

---

## Decision engine formula

```
Final Value   = ML Prediction × Condition Multiplier
Buy Price     = Final Value − Target Profit − Repair Buffer − Holding Cost − Risk Buffer
Sell Price    = Final Value × 1.05
Profit        = Sell Price − Buy Price − Repair Buffer − Holding Cost

Holding Cost  = Final Value × 2.5%
Risk Buffer   = Final Value × (risk_score / 100) × 8%

Action:
  BUY        → confidence ≥ 65 and risk < 55
  NEGOTIATE  → confidence 50–64 or risk 55–74
  REJECT     → confidence < 50 or risk ≥ 75
```

---

## Setup

**Prerequisites:** Python 3.10+, Node.js 18+

```bash
# 1. Train models (skip if model_artifacts/ already populated)
pip install -r ml_training/requirements.txt
python ml_training/clean_data.py
python ml_training/train_ml_model.py

# 2. Start backend
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# → http://localhost:8000/docs

# 3. Start frontend
npm install
npm run dev
# → http://localhost:5173
```

**Docker:**
```bash
docker-compose up --build
```

**Mobile (Flutter):**
```bash
npm run build:mobile
cd mobile && flutter pub get && flutter run
```

---

## Demo login

```
Email:    dealer@PriceRef.ai
Password: dealer123
```

---

## Version history

| Version | Rows | Global MAPE | Notes |
|---|---|---|---|
| v1.0–v2.1 | 36,956 | 14→12% | Initial cars.csv baseline |
| v3.0 | 36,956 | 11.93% | Brand-class routing |
| v4.0 | 214,825 | 4.82% | Combined 2026 dataset, luxury R² 0.9987 |
| v5.0–v6.0 | 213,820 | 5.36% | UI overhaul, enterprise portal |
| v8.0 | 34,266 | 7.43% | Phase 7 feature engineering, dynamic engine |
| **v9.0** | **34,266** | **7.43%** | Dynamic engine specs, monetary SHAP, adaptive confidence |


