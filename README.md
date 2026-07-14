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

### Dataset
| Metric | Value |
|---|---|
| Source | `ml_training/data/combined_2026.csv` |
| Raw lines | 217,423 |
| Schema variants | 7 (6-pipe to 20-pipe, all parsed) |
| Clean rows after parsing | **214,825** |
| OEM brands | 28 |
| Price range (clean) | ₹54,000 – ₹75.99 L |

### Why so many schemas?
`combined_2026.csv` is a concatenation of multiple data exports with evolving column schemas. The cleaner (`clean_data.py`) detects each schema by field count and maps all variants to a common feature set — recovering **~207,000 rows** that a naïve single-schema parser would silently discard.

| Schema (fields) | Rows | Extra columns vs baseline |
|---|---|---|
| 7–8 fields | ~3,000 | Full model string in one field |
| 11–12 fields | ~1,900 | No RTO |
| 14 fields (baseline) | 9,877 | — |
| 15 fields | 4,036 | + SEGMENT |
| 16 fields | 20,057 | + SEGMENT, INSPECTED |
| 18 fields | 66,134 | + BRANCH, PINCODE |
| 19 fields | 38,711 | + OWNER_COUNT |
| 21 fields | 73,685 | + COLOR |

### Features Used (22)
| Feature | Type | Source |
|---|---|---|
| `brand` | Categorical | MAKE column |
| `model` | Categorical | MODEL column |
| `variant` | Categorical | TRIM column |
| `city` | Categorical | CITY column |
| `rto_state` | Categorical | RTO prefix (e.g. KA-19 → KA) |
| `color` | Categorical | COLOR column (21-field schema) |
| `segment` | Categorical | SEGMENT column (mass market / luxury / standard) |
| `brand_tier` | Categorical | Derived from brand (budget/mid/premium/luxury) |
| `fuel_type` | Categorical | FUEL column |
| `transmission` | Categorical | TRANS column |
| `vehicle_age` | Numeric | `2026 - YEAR` |
| `odometer_reading` | Numeric | ODOMETER column |
| `km_per_year` | Numeric | `odometer / max(age, 0.5)` |
| `owner_count` | Numeric | Explicit column or parsed from CATEGORY |
| `fuel_efficiency` | Numeric | Median-imputed (15.0 km/L) |
| `engine_cc` | Numeric | Median-imputed (1200 cc) |
| `ownership_trust_score` | Numeric | Composite score (owner, age, km) |
| `vehicle_health_score` | Numeric | Composite score (km, age, owner) |
| `depreciation_ratio` | Numeric | `selling_price / list_price` |
| `listing_month` | Numeric | Month from RECEIVED date (seasonality) |
| `listing_year` | Numeric | Year from RECEIVED date |
| `inspected` | Binary | INSPECTED column (1 = Yes) |

### Training Split
```
Total: 214,825 rows
  ├── Train:      70%  (~150,378 rows)
  ├── Validation: 15%  (~32,224 rows)  ← ensemble weight optimisation
  └── Test:       15%  (~32,224 rows)  ← final unbiased metrics
```

### Ensemble Strategy
- Three base learners trained independently: CatBoost, LightGBM, XGBoost
- Ensemble weights optimised on the validation set using SLSQP (maximise R²)
- Final prediction: `w_cb × pred_cb + w_lgb × pred_lgb + w_xgb × pred_xgb`
- Target transform: `log1p(selling_price)` → `expm1()` at inference
- CatBoost dominates (weight = 1.0) — expected with high-cardinality categoricals (brand/model/variant)

### Condition Calibration (post-ML)
Applied after ensemble prediction to enforce monotonicity:
| Condition | Multiplier |
|---|---|
| Excellent | 1.035 |
| Good | 1.000 |
| Average | 0.940 |
| Poor | 0.860 |

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
| **v5.0** | **cleaned_used_car_dataset.csv** | **213,820** | **19** | **5.36%** | **Luxury: 0.9976** |
| **v6.0** | **cleaned_used_car_dataset.csv** | **213,820** | **19** | **5.36%** | **Luxury: 0.9976 (UI Overhaul)** |

**v6.0 highlights:**
- **Enterprise SaaS UI/UX Overhaul:** Re-imagined the layout into a professional, high-whitespace enterprise portal (Linear, Stripe, Ramp aesthetics). Responsive sidebar navigation replacing the mobile-first template.
- **Action-centric Color System:** Neutrals for structure, orange for CTAs/actions only, semantic green for profits/success, and red for risk buffers.
- **3-Step Valuation Wizard:** Redesigned InputScreen into a multi-step workflow separating identity, physical state, and commercial parameters.
- **Realistic Pricing Engine:** Updated dealer cost calculations to detail all operational margins (reconditioning, detailing, RC transfer, holding, interest, insurance, sales commissions, buffers). Target net profits adjusted to realistic ₹25,000–₹80,000 range.
- **SHAP-style explainability & chat assistant:** Refined graphics, confidence gauges, and interaction flows.

---

*PriceRef is a dealership-internal prototype. All predictions are ML estimates and should be reviewed by an experienced dealer before finalising any acquisition.*

