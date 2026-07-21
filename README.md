# PriceRef — AI-Powered Used Car Valuation for Dealerships

> **Stop guessing. Start buying with confidence.**

PriceRef is a dealership-internal tool that uses machine learning to instantly value any used vehicle, compute a recommended buy price, estimate the expected profit, assess acquisition risk, and deliver a clear **BUY / NEGOTIATE / REJECT** decision — all in under a second.

It is designed for used-car dealers who need to make fast, data-driven acquisition decisions on the lot. Instead of relying on experience and gut feel, dealers enter the vehicle details and get a full financial breakdown: fair market value, acquisition cost, resale target, expected profit, and a confidence-weighted recommendation.

### What PriceRef Outputs for Every Valuation

| Output | Description |
|--------|-------------|
| **Market Value** | ML-predicted fair market price for the vehicle |
| **Price Band** | Tight ±₹10,000 range around the market value |
| **Recommended Buy Price** | Maximum price the dealer should pay |
| **Expected Sell Price** | Recommended retail listing price |
| **Expected Profit** | Net dealer profit after all costs |
| **Risk Score** | 0–100 acquisition risk score |
| **Confidence Score** | Model confidence in the prediction |
| **Decision** | BUY / BUY AFTER INSPECTION / NEGOTIATE / NEGOTIATE AGGRESSIVELY / REJECT / MANUAL REVIEW |

---

## Features

- **AI-Powered Vehicle Valuation** — Ensemble of CatBoost, LightGBM, and XGBoost trained on 34,425 Bangalore listings
- **Dealer Decision Engine** — Waterfall cost model computing buy price from market value minus all costs
- **Dynamic Buy/Sell Pricing** — Margin, reconditioning, holding, documentation, and risk costs all computed dynamically
- **Profit Estimation** — Per-deal net profit and ROI percentage
- **Risk Scoring** — 0–100 risk score based on age, mileage, owners, condition, and inspection status
- **Confidence Scoring** — Two-component confidence (model + business) combined into a single score
- **Negotiation Recommendations** — Opening offer, target offer, and walk-away price
- **Model Registry** — 6 trained variant models stored with full metrics and artifacts
- **Automatic Best-Model Selection** — Registry auto-promotes the lowest-MAPE model as the active default
- **Manual Model Variant Selection** — Dealers can switch between any of the 6 trained variants from the UI
- **Price-Band Segment Routing** — Sub-models for ₹6–12L and ₹12L+ bands improve accuracy in those ranges
- **REST API** — Full FastAPI backend with Swagger docs at `/docs`
- **Responsive React Dashboard** — Works on desktop, tablet, and mobile
- **Bulk Evaluation** — Evaluate multiple vehicles in a single API call

---

## Quick Start

Get PriceRef running locally in under 3 minutes.

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- Git

### 1 — Clone the Repository

```bash
git clone https://github.com/UmaDamotharan/Price-Prediction.git
cd Price-Prediction
```

### 2 — Start the Backend

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r backend/requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000 --reload
```

Backend → **http://127.0.0.1:9000**  
API Docs → **http://127.0.0.1:9000/docs**

### 3 — Start the Frontend

```bash
# In a new terminal, from the project root
npm install
npm run dev
```

Frontend → **http://localhost:5173**

### 4 — Log In

```
Email:    dealer@PriceRef.ai
Password: dealer123
```

---

## Architecture

```
Dealer Input (React UI)
         │
         ▼
  React Frontend (Vite)
  ─ Vehicle form
  ─ Model variant selector
  ─ Result dashboard
         │  REST API (JSON)
         ▼
  FastAPI Backend (port 9000)
  ─ Input validation (Pydantic)
  ─ Brand → segment routing
  ─ Price-band routing
         │
         ▼
  Model Registry
  ─ variant_1 … variant_6
  ─ registry.json (active pointer)
  ─ Auto-promotes best MAPE
         │
         ▼
  Active Ensemble Model
  ─ XGBoost 55.5%
  ─ LightGBM 44.5%
  ─ Price-band sub-models (₹6–12L, ₹12L+)
  ─ log1p → expm1 transform
         │
         ▼
  Decision Engine
  ─ Condition adjustment
  ─ Waterfall cost deductions
  ─ Risk + confidence scoring
         │
         ▼
  Buy Price │ Sell Price │ Profit │ Risk │ Recommendation
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 19 + Vite | SPA, fast HMR in development |
| Styling | Plain CSS | No framework dependency |
| Charts | Recharts | Waterfall and score charts |
| State | React Context API | Global input/result state |
| Backend | FastAPI + Uvicorn | Async REST API |
| Validation | Pydantic v2 | Request schema validation |
| ML — Primary | CatBoost | Handles categoricals natively |
| ML — Ensemble | LightGBM + XGBoost | SLSQP-optimised blending |
| Optimisation | SciPy SLSQP | Ensemble weight optimisation |
| Serialisation | Joblib + Pickle | Model artifact storage |
| Mobile | Flutter | WebView shell wrapping the React build |

---

## Project Structure

```
Price-Prediction/
│
├── backend/
│   ├── main.py                  # FastAPI routes, segment routing, Pydantic models
│   ├── decision_engine.py       # Full waterfall: recon → holding → doc → risk → profit
│   ├── ensemble_predictor.py    # CatBoost + LightGBM + XGBoost loader and blender
│   ├── model_registry.py        # Registry read/write, variant resolution
│   └── brand_catalog.py         # Brand → segment class (economy / premium / luxury)
│
├── ml_training/
│   ├── train-1.py … train-6.py  # 6 training pipelines (different dataset variants)
│   ├── registry_helper.py       # Auto-assign variant IDs and update registry.json
│   ├── clean_data.py            # Data cleaning (handles multiple raw CSV schemas)
│   └── data/                    # Raw + cleaned CSVs (gitignored — 100 MB+ datasets)
│
├── model_registry/
│   ├── registry.json            # Active variant + all variant metrics
│   └── variant_1/ … variant_6/  # Full artifact set per trained model
│       ├── vehicle_price_catboost.cbm
│       ├── vehicle_price_lightgbm.txt
│       ├── vehicle_price_xgboost.json
│       ├── ensemble_bundle.pkl
│       ├── segment_6_12_lakh.cbm
│       ├── segment_12_plus_lakh.cbm
│       ├── routing_table.json
│       ├── model_metadata.json
│       └── training_report.json
│
├── model_artifacts/             # Symlinked artifacts from the active default variant
│
├── src/
│   ├── screens/
│   │   ├── InputScreen.jsx      # Vehicle valuation form
│   │   ├── ResultScreen.jsx     # Prediction + decision result
│   │   ├── PricingScreen.jsx    # Full cost breakdown
│   │   ├── DashboardScreen.jsx  # Evaluation history + analytics
│   │   ├── AssistantScreen.jsx  # AI assistant for deal queries
│   │   └── ReverseCalculatorScreen.jsx  # Target sell → max buy price
│   ├── context/AppContext.jsx   # Global state provider
│   └── utils/apiValuation.js   # API calls and response normalisation
│
├── run_all_training.py          # Wipes registry and trains all 6 variants sequentially
├── docker-compose.yml
├── vite.config.js
└── index.html
```

---

## Machine Learning Pipeline

### Why an Ensemble?

No single algorithm dominates on all vehicle types and price ranges. By blending XGBoost and LightGBM using SLSQP weight optimisation on the validation R², we consistently outperform any individual model. The ensemble weights are computed per-variant at training time.

### Why Multiple Model Variants?

Different datasets (with/without owner count, with/without pincode) produce models with different strengths. Six variants were trained on different cleaned versions of the Bangalore dataset. The registry records each variant's MAPE and automatically promotes the best one as the default — but dealers can switch to any variant from the UI.

### How the Active Model Is Selected

At training time, `registry_helper.py` writes each variant's MAPE to `registry.json`. After each training run, if the new model achieves a lower MAPE than the current default, the registry updates the `"default"` pointer. The backend loads whichever variant is set as default unless the client specifies a `model_variant` parameter.

### Target Transform

Training uses `log1p(selling_price)` to reduce skew from high-value outliers. Predictions are inverse-transformed using `expm1()`.

### Features (21 total)

**Categorical (10):** `brand`, `model`, `variant`, `city`, `locality`, `rto`, `segment_class`, `fuel_type`, `transmission`, `seller_type`

**Numeric (11):** `vehicle_age`, `odometer_reading`, `km_per_year`, `brand_tier`, `age_km_interaction`, `vehicle_health_score`, `is_high_mileage`, `locality_tier`, `usage_category_num`, `locality_density_norm`, `popularity_score_log`

> Note: `owner_count` was removed — the best-performing variant is trained on owner-agnostic data for broader applicability.

### Training Split

70% train / 30% validation (stratified by price band)

---

## Model Registry

### Registered Variants

| Variant | Dataset | MAPE | R² | MAE | RMSE | Status |
|---------|---------|------|-----|-----|------|--------|
| variant_1 | processed_widown-1.csv | 3.28% | 0.9913 | ₹19,994 | ₹52,673 | Archived |
| **variant_2** | **processed_widoutown-2.csv** | **3.09%** | **0.9918** | **₹19,006** | **₹51,391** | **⭐ Active** |
| variant_3 | processed_pincode_with_owner-3.csv | 3.32% | 0.9911 | ₹20,635 | ₹61,411 | Archived |
| variant_4 | processed_pincode_without_owner-4.csv | 3.11% | 0.9917 | ₹19,616 | ₹60,981 | Archived |
| variant_5 | processed_pincode_with_owner1_filled-5.csv | 3.32% | 0.9911 | ₹20,635 | ₹61,411 | Archived |
| variant_6 | processed_widown1-6.csv | 3.28% | 0.9913 | ₹19,994 | ₹52,673 | Archived |

### Active Model — variant_2 Detail

**Ensemble Weights** (SLSQP optimised on validation R²)

| Model | Weight |
|-------|--------|
| XGBoost | 55.5% |
| LightGBM | 44.5% |
| CatBoost | 0% |

**Price-Band Segment Routing**

| Band | Rows | Routing | Band MAPE | Band R² |
|------|------|---------|-----------|---------|
| ₹0–6L | 20,342 | Global ensemble fallback | 5.96% | 0.964 |
| ₹6–12L | 11,095 | ✅ Dedicated segment model | 4.47% | 0.913 |
| ₹12L+ | 3,251 | ✅ Dedicated segment model | 3.45% | 0.946 |

---

## API Reference

**Base URL:** `http://localhost:9000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, loaded models, active variant |
| `GET` | `/metadata` | Full training metadata and ensemble config |
| `GET` | `/api/brands` | Brand → model catalog for UI dropdowns |
| `GET` | `/api/registry` | List all variants with metrics and status |
| `POST` | `/api/registry/{variant_id}/activate` | Set a variant as the active default |
| `POST` | `/evaluate` | Full dealer evaluation |
| `POST` | `/evaluate-enhanced` | Evaluation with enriched recon/risk breakdown |
| `POST` | `/reverse-calculate` | Target sell price → maximum buy price |
| `POST` | `/bulk-evaluate` | Batch evaluation (array of vehicles) |

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
  "repair_buffer": 25000,
  "model_variant": "variant_2"
}
```

### Sample `/evaluate` Response

```json
{
  "market_value": 735000,
  "price_min": 725000,
  "price_max": 745000,
  "recommended_buy_price": 580000,
  "recommended_sell_price": 771750,
  "expected_profit": 91500,
  "expected_margin_pct": 15.8,
  "risk_score": 22,
  "risk_level": "Low",
  "confidence_score": 78,
  "action": "BUY",
  "recon_cost": 18000,
  "holding_cost": 5500,
  "doc_cost": 5200,
  "risk_buffer": 3200,
  "target_profit": 73500,
  "waterfall": [ ... ],
  "positive_factors": [ ... ],
  "negative_factors": [ ... ],
  "similar_cars": [ ... ]
}
```

### Decision Engine Formula

```
Market Value (ML prediction, condition-adjusted)
  − Reconditioning Cost   [dynamic: segment + age + km + condition + brand tier]
  − Holding Cost          [segment rate × brand popularity days]
  − Documentation Cost    [RC ₹3,500 + NOC ₹500 + insurance ₹1,200 + state transfer if applicable]
  − Risk Buffer           [rupee-based additive + unknown-field penalties]
  − Target Dealer Profit  [dynamic margin, segment-capped]
  ═══════════════════════════════════
  = Recommended Buy Price  (floor: 88% of market value)
```

---

## Training Pipeline

### Train a Single Variant

```bash
# Activate venv
venv\Scripts\activate

# Install ML dependencies
pip install -r ml_training/requirements.txt

# Clean the raw data first
python ml_training/clean_data.py

# Run one training variant
python ml_training/train-2.py
```

### Retrain All 6 Variants (Recommended)

```bash
python run_all_training.py
```

This script:
1. **Clears** all existing `model_registry/` variant folders and `registry.json`
2. **Runs** `train-1.py` through `train-6.py` sequentially (~8–12 min each)
3. **Auto-promotes** the lowest-MAPE variant as the active default
4. **Saves** all 6 variants to `model_registry/variant_1/` … `variant_6/`

Expected total time: **~55–75 minutes**

---

## Deployment

### Option 1 — Local Development (Recommended for Development)

Follow the [Quick Start](#quick-start) steps above.

### Option 2 — Docker (Production-Ready)

```bash
# Build and start all services
docker-compose up --build

# Detached mode (background)
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

Services:
- Backend → **http://localhost:9000**
- Frontend → **http://localhost:5173**

### Option 3 — Production Build (Static Frontend)

```bash
# Build optimised static frontend
npm run build

# The built output is in /dist
# Serve with any static host (Nginx, Vercel, Netlify, etc.)
```

Point your static host to the `/dist` folder and proxy `/api` to the backend.

### Option 4 — Mobile (Flutter)

```bash
# Build React app first
npm run build

# Copy to Flutter assets
npm run build:mobile

# Run Flutter app
cd mobile
flutter pub get
flutter run
```

### Environment Variables

Create `.env` in the project root (see `.env.example`):

```env
VITE_API_URL=http://127.0.0.1:9000
```

---

## Version History

| Version | Dataset Rows | Global MAPE | Highlights |
|---------|-------------|-------------|------------|
| v1.0–v2.1 | 36,956 | 14–12% | Baseline linear + tree models |
| v3.0 | 36,956 | 11.93% | Brand-class segment routing |
| v4.0 | 214,825 | 4.82% | Combined 2026 dataset, luxury R² 0.9987 |
| v5.0–v6.0 | 213,820 | 5.36% | UI overhaul, enterprise dealer portal |
| v8.0 | 34,266 | 7.43% | Phase 7 feature engineering, dynamic decision engine |
| v9.0 | 34,266 | 7.43% | Monetary SHAP, adaptive confidence, negotiation trio |
| **v10.0** | **34,425** | **3.09%** | Owner-agnostic model, price-band routing, 6-variant registry, SLSQP ensemble optimisation |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

