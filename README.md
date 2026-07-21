# PriceRef — Used Car Valuation for Dealerships

A dealership-internal tool that tells you in under a second whether to buy a car, at what price, and why. Covers **Bangalore used car market**, powered by a trained ensemble of CatBoost + LightGBM + XGBoost.

---

## How it works

```
Dealer enters vehicle details
        ↓
FastAPI backend receives the request
        ↓
Brand → segment class (economy / premium / luxury)
        ↓
Segment-specific ensemble predicts base market value (log1p → expm1)
        ↓
Condition multiplier applied
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

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 + Vite, plain CSS, Recharts |
| **Backend** | Python 3.13, FastAPI, Uvicorn, Pydantic |
| **ML** | CatBoost, LightGBM, XGBoost, scikit-learn, SciPy SLSQP |
| **State** | React Context API |
| **Mobile** | Flutter WebView shell |

---

## Best Model — variant_2 ⭐ (Active)

**Dataset:** `processed_widoutown-2.csv` — 34,425 Bangalore rows (owner-agnostic)

**Trained:** 2026-07-21

### Features (21 total)

**Categorical (10):** `brand`, `model`, `variant`, `city`, `locality`, `rto`, `segment_class`, `fuel_type`, `transmission`, `seller_type`

**Numeric (11):** `vehicle_age`, `odometer_reading`, `km_per_year`, `brand_tier`, `age_km_interaction`, `vehicle_health_score`, `is_high_mileage`, `locality_tier`, `usage_category_num`, `locality_density_norm`, `popularity_score_log`

> `owner_count` removed — trained on owner-agnostic data for broader applicability.

### Ensemble Weights (SLSQP optimised on validation R²)

| Model | Weight |
|-------|--------|
| XGBoost | **55.5%** |
| LightGBM | 44.5% |
| CatBoost | 0% |

### Validation Metrics

| Metric | Value |
|--------|-------|
| **MAPE** | **3.09%** |
| **R²** | **0.9918** |
| **MAE** | ₹19,006 |
| **RMSE** | ₹51,391 |

### Price-band Segment Routing

| Segment | Rows | Mode | MAPE | R² |
|---------|------|------|------|-----|
| ₹0–6L | 20,342 | Global fallback | 5.96% | 0.964 |
| ₹6–12L | 11,095 | ✅ Segment model | 4.47% | 0.913 |
| ₹12L+ | 3,251 | ✅ Segment model | 3.45% | 0.946 |

---

## API Reference

**Base URL:** `http://localhost:9000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server + model status |
| `GET` | `/metadata` | Training metadata and metrics |
| `GET` | `/api/brands` | Brand list for dropdowns |
| `GET` | `/api/registry` | List all trained variants |
| `POST` | `/evaluate` | Full dealer evaluation |
| `POST` | `/evaluate-enhanced` | Evaluation with enriched recon/risk |
| `POST` | `/reverse-calculate` | Target sell price → max buy price |
| `POST` | `/bulk-evaluate` | Batch evaluation (array) |

### Sample `/evaluate` Request

```json
{
  "brand": "Honda",
  "model": "City",
  "year": 2021,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "odometer_reading": 28000,
  "city": "Bangalore",
  "condition": "Good",
  "target_margin_pct": 10,
  "repair_buffer": 25000
}
```

### Sample Response

```json
{
  "market_value": 735000,
  "price_min": 725000,
  "price_max": 745000,
  "recommended_buy_price": 580000,
  "recommended_sell_price": 771750,
  "expected_profit": 91500,
  "risk_score": 22,
  "confidence_score": 78,
  "action": "BUY"
}
```

---

## Project Layout

```
Price-Prediction/
├── backend/
│   ├── main.py                  # FastAPI routes and segment routing
│   ├── decision_engine.py       # Buy/sell/profit/risk formulas
│   ├── ensemble_predictor.py    # Model loader (CatBoost + LightGBM + XGBoost)
│   └── brand_catalog.py         # Brand → segment map
│
├── ml_training/
│   ├── train-1.py … train-6.py  # 6 training pipeline variants
│   ├── clean_data.py            # Data cleaning
│   ├── registry_helper.py       # Model registry management
│   └── data/                    # Raw + cleaned CSVs (gitignored)
│
├── model_registry/              # All 6 trained variant artifacts
│   ├── variant_1/ … variant_6/
│   └── registry.json            # Active variant pointer + all metrics
│
├── model_artifacts/             # Active model symlinked from default variant
│
├── src/
│   ├── screens/                 # React screens
│   ├── context/AppContext.jsx   # Global state
│   └── utils/apiValuation.js   # API calls + result normalisation
│
├── run_all_training.py          # Batch runner — trains all 6 variants sequentially
├── docker-compose.yml
├── vite.config.js
└── index.html
```

---

## Deployment

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- Git

---

### Option 1 — Local Development

#### 1. Clone the Repository

```bash
git clone https://github.com/UmaDamotharan/Price-Prediction.git
cd Price-Prediction
```

#### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000 --reload
```

Backend runs at: **http://127.0.0.1:9000**
API docs: **http://127.0.0.1:9000/docs**

#### 3. Frontend Setup

```bash
# In a new terminal (from project root)
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

### Option 2 — Docker (Full Stack)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop all services
docker-compose down
```

Services started:
- Backend → **http://localhost:9000**
- Frontend → **http://localhost:5173**

---

### Option 3 — Retrain All 6 Model Variants

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows

# Install ML training dependencies
pip install -r ml_training/requirements.txt

# Run all 6 training scripts sequentially (clears old registry first)
python run_all_training.py
```

This will:
1. **Wipe** existing `model_registry/` variants and `registry.json`
2. Train `train-1.py` → `train-6.py` sequentially (each ~8–12 min)
3. Auto-promote the best-MAPE variant as the active default
4. Save all 6 variants to `model_registry/variant_1/` … `variant_6/`

---

### Option 4 — Mobile (Flutter)

```bash
# Build the React app first
npm run build

# Copy to Flutter assets
npm run build:mobile

# Run Flutter app
cd mobile
flutter pub get
flutter run
```

---

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
VITE_API_URL=http://127.0.0.1:9000
```

---

## Demo Login

```
Email:    dealer@PriceRef.ai
Password: dealer123
```

---

## Version History

| Version | Dataset Rows | Global MAPE | Highlights |
|---------|-------------|-------------|------------|
| v1.0–v2.1 | 36,956 | 14–12% | Baseline |
| v3.0 | 36,956 | 11.93% | Brand-class routing |
| v4.0 | 214,825 | 4.82% | Combined 2026 dataset |
| v5.0–v6.0 | 213,820 | 5.36% | UI overhaul |
| v8.0–v9.0 | 34,266 | 7.43% | Dynamic engine, monetary SHAP |
| **v10.0** | **34,425** | **3.09%** | Owner-agnostic, price-band routing, 6 variants |
