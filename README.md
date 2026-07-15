# PriceRef — Dealership ML Valuation System

> **Scope:** Dealership / manager internal portal only.  
> Seller portal · Buyer portal · Computer vision — all on hold.

PriceRef automates used-car acquisition decisions for dealerships. A dealer enters vehicle details, the system predicts the market value using an ML ensemble, applies a condition calibration, runs a rule-based dealer decision engine, and returns a complete acquisition recommendation — buy price, sell price, profit, risk score, and BUY / NEGOTIATE / REJECT action — in under a second.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How It Works](#2-how-it-works)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [ML Pipeline](#5-ml-pipeline)
6. [Model Performance](#6-model-performance)
7. [Segmented Models](#7-segmented-models)
8. [Backend API](#8-backend-api)
9. [Decision Engine Formula](#9-decision-engine-formula)
10. [Frontend Screens](#10-frontend-screens)
11. [Setup & Run](#11-setup--run)
12. [Demo Login](#12-demo-login)
13. [Known Issues & Next Improvements](#13-known-issues--next-improvements)
14. [Version History](#14-version-history)

---

## 1. Problem Statement

Used-car dealerships must quote a competitive acquisition price the moment a seller walks in. The manual process — market lookup, negotiation experience, rough estimation — is slow, inconsistent, and risky. A competing dealership can win the deal simply by quoting faster.

PriceRef solves this by:
- Predicting market value from vehicle features using a trained ML ensemble
- Routing predictions to brand-class-specific models for accuracy
- Generating a complete dealer decision (price, profit, risk, action) instantly

---

## 2. How It Works

```
Dealer enters vehicle details (brand, model, year, km, condition...)
        ↓
FastAPI backend receives request
        ↓
get_brand_class(brand) → O(1) dict lookup, no price estimation needed
    Budget class  → Maruti, Datsun, Chevrolet, Fiat...  (R² 0.9963)
    Mid class     → Hyundai, Honda, Tata, Ford...        (R² 0.9933)
    Premium class → VW, Toyota, Kia, MG, Skoda...       (R² 0.9982)
    Luxury class  → BMW, Mercedes-Benz, Audi, JLR...    (R² 0.9987)
        ↓
Brand-class ensemble predicts base market value (log1p → expm1)
        ↓
Condition multiplier applied (Excellent/Good/Average/Poor)
        ↓
Dealer decision engine calculates:
    - Recommended buy price
    - Recommended sell price
    - Expected profit
    - Risk score / Confidence score / Deal quality
    - Negotiation trio (opening / target / walk-away)
    - BUY / NEGOTIATE / REJECT action
        ↓
Response returned to React frontend
```

---

## 3. Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 18 + Vite | UI framework and build tool |
| JavaScript / JSX | Component logic |
| CSS3 | Styling (no Tailwind) |
| React Context API | Global app state |
| Recharts | Analytics charts |

### Mobile
| Technology | Purpose |
|---|---|
| Flutter | Android / iOS shell |
| WebView | Wraps the React frontend |
| `npm run build:mobile` | Builds the bundle for Flutter |

### Backend
| Technology | Purpose |
|---|---|
| Python 3.13 | Runtime |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request / response validation |
| Joblib | Segment model loading |

### Machine Learning
| Technology | Purpose |
|---|---|
| CatBoost | Base learner — dominant (handles categoricals natively) |
| LightGBM | Base learner |
| XGBoost | Base learner |
| Scikit-learn | Metrics, train/val/test split, LabelEncoder |
| SciPy | Ensemble weight optimisation (SLSQP) |
| Pandas / NumPy | Data cleaning and feature engineering |

---

## 4. Project Structure

```
Price-Prediction/
├── backend/
│   ├── main.py                  # FastAPI app, segment routing, prediction
│   ├── decision_engine.py       # Rule-based dealer logic
│   ├── ensemble_predictor.py    # Global ensemble loader
│   ├── brand_catalog.py         # Brand/model lookup
│   ├── __init__.py
│   └── requirements.txt
│
├── ml_training/
│   ├── train_ml_model.py        # Full training pipeline (v4.0)
│   ├── clean_data.py            # Multi-schema data cleaner (7 schema variants)
│   ├── requirements.txt
│   └── data/
│       ├── combined_2026.csv    # Raw dataset – gitignored (~35 MB, 217k rows, 7 schemas)
│       └── cleaned.csv          # Output of clean_data.py – gitignored (214,825 rows)
│
├── model_artifacts/             # All large binaries are gitignored; retrain to reproduce
│   ├── ensemble_global.pkl           # Global CatBoost+LightGBM+XGBoost artifact
│   ├── ensemble_budget.pkl           # Budget class segment model
│   ├── ensemble_mid.pkl              # Mid class segment model
│   ├── ensemble_premium.pkl          # Premium class segment model
│   ├── ensemble_luxury.pkl           # Luxury class segment model
│   ├── vehicle_price_catboost.cbm    # Global CatBoost model file
│   ├── vehicle_price_lightgbm.txt    # Global LightGBM model file
│   ├── vehicle_price_xgboost.json    # Global XGBoost model file
│   ├── model_metadata.json           # Training metadata + full metrics (tracked)
│   └── cleaned_training_sample.csv   # 500-row sample for inspection (tracked)
│
├── src/
│   ├── screens/
│   │   ├── AuthScreen.jsx             # Login
│   │   ├── HomeScreen.jsx             # Dashboard overview
│   │   ├── InputScreen.jsx            # Vehicle input form
│   │   ├── ResultScreen.jsx           # ML result + decision
│   │   ├── EnhancedResultScreen.jsx   # Wheelr enrichment result
│   │   ├── EnhancedValuationScreen.jsx
│   │   ├── DashboardScreen.jsx        # Analytics
│   │   ├── ExplainScreen.jsx          # SHAP-style explanation
│   │   ├── PricingScreen.jsx          # Pricing breakdown
│   │   ├── ReverseCalculatorScreen.jsx # Reverse price calculator
│   │   └── AssistantScreen.jsx        # AI assistant
│   ├── components/
│   │   ├── SearchableSelect.jsx
│   │   └── WheelrPanels.jsx
│   ├── context/
│   │   └── AppContext.jsx
│   ├── utils/
│   │   └── apiValuation.js
│   ├── App.jsx
│   └── App.css
│
├── mobile/                       # Flutter shell (WebView wraps the React build)
│   ├── lib/
│   ├── pubspec.yaml
│   └── README.md
├── scripts/
│   └── bundle-web-for-mobile.ps1 # Copies dist/ into Flutter assets
├── public/
├── index.html
├── vite.config.js
├── eslint.config.js
├── package.json
└── README.md
```

---

## 5. ML Pipeline

### 5.1 Datasets & Cleaning
We support two primary datasets located in `ml_training/data/`:
1. **Cell7 Dataset (`cell7_dataset.csv`):** 212,427 raw rows, featuring original ownership details (with some "Unknown" values).
2. **Owner-Assumed Dataset (`owner assumed dataset.csv`):** 212,427 raw rows, where unknown owners have been filled using local imputation/heuristics.

### 5.2 The 34k Row Deduplication Explained
While the raw CSV exports contain **212,427 rows**, the flat preprocessor (`preprocess_flat_csv.py`) outputs **~34,266 cleaned rows** for training. 

This is because the raw files contain **172,439 exact duplicate rows** (where every single column value, including city, price, odometer, year, trim, and locality, is 100% identical). These duplicates are caused by scraping loops appending active listings multiple times. 
* Standard deduplication (`drop_duplicates(keep="first")`) preserves the first unique instance and discards the duplicate copies to prevent the ML model from overfitting on duplicated queries.

| Dataset | Raw Rows | Duplicate Rows Removed | Cleaned/Valid Rows |
|---|---|---|---|
| **Cell7 Dataset** | 212,427 | 177,980 | **34,266** |
| **Owner-Assumed Dataset** | 212,427 | 179,342 | **32,904** |

### 5.3 Features Used (19 ML Features + 8 Enriched Features)
**ML Feature Set:**
* **Categorical (9):** `brand`, `model`, `variant`, `city`, `rto_state`, `color`, `segment_class`, `fuel_type`, `transmission`
* **Numeric (10):** `vehicle_age`, `odometer_reading`, `km_per_year`, `owner_count`, `ownership_trust_score`, `vehicle_health_score`, `inspected`, `high_mileage`, `luxury_brand`, `has_list_price`

**Decision Engine Enriched Features:**
* `Km_Per_Year`, `Depreciation_Bucket`, `Mileage_Tier`, `Brand_Tier`, `Ownership_Category`, `Price_Segment`, `Is_Recent_Model`, `Seller_Type_Clean`

### 5.4 Training Comparison Results (Cell7 vs. Owner-Assumed)
We train the ensemble (`CatBoost`, `LightGBM`, `XGBoost`) on both processed datasets separately. The comparison results are:

| Metric | Cell7 (original owners) | Owner-Assumed (filled) | Winner |
|---|---|---|---|
| **Global R² Score** | **0.9723** | 0.9711 | **Cell7** |
| **Global MAPE** | **7.43%** | 7.79% | **Cell7** |
| **Global MAE** | ₹47,025 | **₹46,720** | **Owner-Assumed** |
| **Global RMSE** | ₹86,842 | **₹78,755** | **Owner-Assumed** |
| **Overfit Gap (Train-Test R²)** | **0.0086** | 0.0094 | **Cell7** |

*Verdict:* **Cell7 is the recommended training dataset** due to superior generalizability, lower MAPE, and a cleaner premium-segment model (R² 0.9791).

### 5.5 Ensemble & Training Rules
- **Split:** 70% Train / 15% Validation / 15% Test.
- **Model Blending:** Weights are optimized using SLSQP (Sequential Least Squares Programming) on the validation set. On the 34k deduplicated row sets, CatBoost gets 100% weight.
- **Target Transformation:** Models train on `log1p(selling_price)` and predict using `expm1()`.

---


## 6. Model Performance

### Global Ensemble (all brands, fallback)
| Split | R² | MAE | MAPE |
|---|---|---|---|
| Train | 0.9900 | — | — |
| Validation | 0.9888 | — | — |
| **Test** | **0.9892** | **₹30,696** | **4.82%** |

**Overfitting gap:** 0.0012 → `healthy_generalization` ✅

### Ensemble Weights (Global)
| Model | Weight |
|---|---|
| **CatBoost** | **100%** |
| LightGBM | 0% |
| XGBoost | 0% |

> CatBoost achieves dominant weight because it handles high-cardinality categorical features (brand, model, variant) natively via ordered target encoding — LightGBM and XGBoost require label encoding which loses ordinal structure.

---

## 7. Segment-Class Models

Three separate ensembles trained per segment class — routing is done by brand name, which is always known at inference time:

| Class | Brands | Routing |
|---|---|---|
| **Economy** | Maruti, Hyundai, Honda, Tata, Ford, Mahindra, Renault, Nissan, Datsun, Fiat, Force... | `brand` maps to `economy` |
| **Premium** | Volkswagen, Skoda, Toyota, MG, Kia, Jeep, Volvo, Lexus... | `brand` maps to `premium` |
| **Luxury** | BMW, Mercedes-Benz, Audi, Jaguar, Land Rover, Porsche, Maserati, Ferrari, Bentley... | `brand` maps to `luxury` |

### Model Performance by Segment (v5.0)

| Class | Rows | R² | MAE | MAPE |
|---|---|---|---|---|
| **Economy** | 207,135 | **0.9872** | ₹32,503 | **5.26%** |
| **Premium** | 3,301 | **0.9056** | ₹80,438 | **15.29%** |
| **Luxury** | 3,384 | **0.9976** | ₹13,709 | **1.08%** |

### Routing Logic
```python
segment_class = BRAND_SEGMENT_MAP.get(brand.lower(), "economy")  # always O(1)
model         = SEGMENT_MODELS[segment_class]                    # direct lookup
```

API response includes:
```json
{
  "segment_class": "economy",
  "segment_model_used": true,
  "routing_note": "economy segment model used"
}
```

---

## 8. Backend API

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server + model status |
| GET | `/metadata` | Full model metadata + metrics |
| GET | `/api/brands` | Brand catalog for frontend dropdowns |
| POST | `/predict` | Market value prediction |
| POST | `/evaluate` | Full dealer evaluation (value + decision) |
| POST | `/evaluate-enhanced` | Wheelr enrichment (recon, risk, negotiation) |
| POST | `/reverse-calculate` | Given sell price → max buy price |
| POST | `/bulk-evaluate` | Array of vehicles, one evaluation each |

### Sample Request — `/evaluate`
```json
{
  "brand": "Honda",
  "model": "City",
  "year": 2021,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "odometer_reading": 28000,
  "fuel_efficiency": 17.5,
  "owner_count": 1,
  "engine_cc": 1497,
  "city": "Bangalore",
  "condition": "Good",
  "seller_asking_price": 750000,
  "target_margin_pct": 15,
  "repair_buffer": 25000
}
```

### Sample Response — `/evaluate`
```json
{
  "base_market_value": 735000,
  "market_value": 735000,
  "condition_multiplier": 1.0,
  "condition_adjustment": 0,
  "condition_score": 75,
  "segment_class": "economy",
  "segment_model_used": true,
  "routing_note": "economy segment model used",
  "recommended_buy_price": 580000,
  "recommended_sell_price": 771750,
  "expected_profit": 91500,
  "risk_score": 22,
  "confidence_score": 78,
  "deal_quality_score": 81,
  "urgency_score": 65,
  "action": "BUY",
  "warnings": []
}
```

---

## 9. Decision Engine Formula

```
Final Market Value   = Base ML Value × Condition Multiplier
Target Profit        = Market Value × target_margin_pct %
Holding Cost         = Market Value × 2.5%
Risk Buffer          = Market Value × (risk_score / 100) × 8%
Recommended Buy      = Market Value − Target Profit − Repair Buffer − Holding Cost − Risk Buffer
Recommended Sell     = Market Value × 1.05
Expected Profit      = Recommended Sell − Recommended Buy − Repair Buffer − Holding Cost

Action thresholds:
  BUY        → confidence ≥ 65 AND risk < 55
  NEGOTIATE  → confidence 50–64 OR risk 55–74
  REJECT     → confidence < 50 OR risk ≥ 75
```

---

## 10. Frontend Screens

| Screen | Purpose |
|---|---|
| `AuthScreen` | Demo login (`dealer@PriceRef.ai / dealer123`) |
| `HomeScreen` | Live dashboard with recent evaluations |
| `InputScreen` | Vehicle details form (brand, model, specs, condition) |
| `ResultScreen` | Market value + BUY/NEGOTIATE/REJECT result |
| `EnhancedResultScreen` | Wheelr enrichment: recon cost, risk deductions, negotiation trio |
| `EnhancedValuationScreen` | Combined valuation with enrichment panel |
| `DashboardScreen` | Analytics: volume, MAPE trends, brand breakdown |
| `ExplainScreen` | SHAP-style feature contribution breakdown |
| `PricingScreen` | Price waterfall breakdown |
| `ReverseCalculatorScreen` | Enter desired sell price → calculate max buy |
| `AssistantScreen` | AI assistant (Q&A about the evaluation) |

---

## 11. Setup & Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip, npm

### 1. Install ML dependencies & clean + train
```bash
pip install -r ml_training/requirements.txt

# Clean the dataset — parses all 7 schema variants, outputs 214,825 rows
python ml_training/clean_data.py

# Train all models (global + 4 brand-class segments, ~20 min on 214k rows)
python ml_training/train_ml_model.py
```

### 2. Run backend
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run frontend
```bash
npm install
npm run dev
```
App: [http://localhost:5173](http://localhost:5173)

### 4. Mobile (Flutter)
```bash
npm run build:mobile
cd mobile
flutter pub get
flutter run
```
See `mobile/README.md` for device-specific instructions.

---

## 12. Demo Login

```
Email:    dealer@PriceRef.ai
Password: dealer123
```

---

## 13. Known Issues & Next Improvements

### 🟡 Medium
- `fuel_efficiency` and `engine_cc` are median-imputed (15.0 km/L, 1200 cc) — adding real values from a vehicle specs DB would improve accuracy further
- `color` has 141k "unknown" values (only the 21-field schema contains it) — incomplete but still useful as a feature
- Luxury class row count (3,179) is thin compared to Mid (122k) — consider data augmentation or weight adjustments

### 🟢 Low
- Add model versioning (timestamp suffix on artifacts) so retrains don't silently overwrite previous metrics
- `training_report.json` and `model_metadata.json` are identical writes — one can be removed
- Backend `ensemble_predictor.py` loads the global models; it should be updated to prefer the `ensemble_global.pkl` artifact

---

## 14. Version History

| Version | Dataset | Rows | Features | Global MAPE | Best Class/Segment R² |
|---|---|---|---|---|---|
| v1.0 | cars.csv | 36,956 | 14 | 14.07% | 0.9136 |
| v2.0 | cars.csv | 36,956 | 14 | 12.31% | 0.9312 |
| v2.1 | cars.csv | 36,956 | 14 | 11.93% | 0.9312 |
| v3.0 | cars.csv | 36,956 | 14 | 11.93% | Mid: 0.9332 |
| v4.0 | combined_2026.csv | 214,825 | 22 | 4.82% | Luxury: 0.9987 |
| v5.0 | cleaned_used_car_dataset.csv | 213,820 | 19 | 5.36% | Luxury: 0.9976 |
| v6.0 | cleaned_used_car_dataset.csv | 213,820 | 19 | 5.36% | Luxury: 0.9976 (UI Overhaul) |
| **v8.0** | **processed_cell7_dataset.csv** | **34,266** | **19 + 8** | **7.43%** | **Premium: 0.9791 (Enriched Preprocessing)** |
| **v9.0** | **processed_cell7_dataset.csv** | **34,266** | **19 + 8** | **7.43%** | **Premium: 0.9791 (Dynamic Engine Spec)** |

**v9.0 highlights:**
- **Dynamic Valuation Upgrades:** Shipped 13 production specs including brand repair multipliers (Toyota/Honda 0.8x vs. Jaguar 2.2x), brand-popularity holding durations, and additive risk penalties for missing fields.
- **Improved Confidence & Safety:** Geometric mean of model confidence × business confidence; adaptive clamp scaling bands based on prediction certainty.
- **Monetary SHAP:** Explains feature impact on resale value in absolute rupee values (e.g. `−₹42,000`).

**v8.0 highlights:**
- **Phase 7 Feature-Enriched Preprocessing:** Engineered 8 new features (`Km_Per_Year`, `Depreciation_Bucket`, `Mileage_Tier`, `Brand_Tier`, `Ownership_Category`, `Price_Segment`, `Is_Recent_Model`, `Seller_Type_Clean`) for advanced analytical rules and metrics.
- **Dynamic Decision Engine:** Replaced static values with segment-aware rupee formulas for holding, recon, documentation, profit margins, and risk buffers.

**v6.0 highlights:**
- **Enterprise SaaS UI/UX Overhaul:** Re-imagined the layout into a professional, high-whitespace enterprise portal (Linear, Stripe, Ramp aesthetics). Responsive sidebar navigation replacing the mobile-first template.
- **Action-centric Color System:** Neutrals for structure, orange for CTAs/actions only, semantic green for profits/success, and red for risk buffers.
- **3-Step Valuation Wizard:** Redesigned InputScreen into a multi-step workflow separating identity, physical state, and commercial parameters.
- **Realistic Pricing Engine:** Updated dealer cost calculations to detail all operational margins (reconditioning, detailing, RC transfer, holding, interest, insurance, sales commissions, buffers). Target net profits adjusted to realistic ₹25,000–₹80,000 range.
- **SHAP-style explainability & chat assistant:** Refined graphics, confidence gauges, and interaction flows.

---

*PriceRef is a dealership-internal prototype. All predictions are ML estimates and should be reviewed by an experienced dealer before finalising any acquisition.*

---

## 15. Docker Setup

The project supports Docker for both **Development** (with hot-reload) and **Production** (Nginx-served static build).

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose) installed and running.

### Quick Start

**1. Copy the environment file:**
```bash
cp .env.example .env
```

**2. Build all containers:**
```bash
docker compose build
```

**3. Start in development mode (hot-reload for both frontend and backend):**
```bash
docker compose up
```

**4. Or run in detached mode (background):**
```bash
docker compose up -d
```

Frontend will be available at: **http://localhost:5173**  
Backend API will be available at: **http://localhost:9000**

---

### All Commands

| Action | Command |
|---|---|
| Build all images | `docker compose build` |
| Start (foreground) | `docker compose up` |
| Start (detached/background) | `docker compose up -d` |
| Stop and remove containers | `docker compose down` |
| Rebuild and start | `docker compose up --build` |
| View logs (all services) | `docker compose logs` |
| View backend logs only | `docker compose logs backend` |
| View frontend logs only | `docker compose logs frontend` |
| Follow logs in real-time | `docker compose logs -f` |
| Open shell in backend container | `docker exec -it price-prediction-backend sh` |
| Open shell in frontend container | `docker exec -it price-prediction-frontend sh` |
| Check health status | `docker compose ps` |

---

### Production Mode

To run the **production build** (Nginx serving optimized static assets):

**1. Build the production frontend image:**
```bash
docker build --target production -t priceref-frontend:prod .
```

**2. Start it manually:**
```bash
docker run -p 5173:5173 priceref-frontend:prod
```

Or update the `docker-compose.yml` to change:
```yaml
target: development
```
to:
```yaml
target: production
```
and re-run `docker compose up --build`.

---

### Architecture Inside Docker

```
Host Machine
 ├── localhost:5173  →  price-prediction-frontend (React / Vite or Nginx)
 └── localhost:9000  →  price-prediction-backend (FastAPI / Uvicorn)
          │
          └── Backend loads ML models from:
              /app/model_artifacts/  (volume-mounted from host)
```

The frontend browser makes direct API calls to `http://localhost:9000` (mapped from the backend container). Both containers share a custom Docker network (`price-prediction-network`).

---

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:9000` | Backend API URL the browser connects to |

To customize, edit your `.env` file:
```env
VITE_API_URL=http://localhost:9000
```
