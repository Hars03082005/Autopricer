# PriceRef — AI-Powered Used Vehicle Valuation Engine

> **Data-driven valuation, acquisition risk assessment, and deal profitability for used vehicles.**

PriceRef is a high-performance machine learning system built for instant vehicle valuation, profit estimation, acquisition risk scoring, and negotiation strategy calculation. Powered by a **dual-mode ML architecture** — a **CatBoost + LightGBM + XGBoost global ensemble (Variant 1)** for broad market coverage, and a **specialized CatBoost + LightGBM S5 model (Variant 4)** for quality shop premium vehicles — PriceRef processes vehicle attributes and returns market valuations in milliseconds across Web and Mobile platforms.

---

## 🔄 Full End-to-End Project Pipeline

PriceRef connects a responsive React frontend, a Flutter mobile shell, a FastAPI REST service, an ML ensemble model, an adaptive financial decision engine, and an optional cloud persistence layer into a single unified workflow:

```mermaid
graph TD
    User([User / Dealer]) -->|Web / App User| UISelector{Platform Interface}
    UISelector -->|Browser| Frontend[1. React Frontend UI]
    UISelector -->|Android / iOS| MobileApp[1b. Flutter Mobile Shell]
    MobileApp -->|WebView Bundle| Frontend
    Frontend -->|POST /evaluate| FastAPI[2. FastAPI Backend Gateway]
    FastAPI -->|Extracts & Sanitizes| FE[3. Feature Engineering & Vectorizer]
    FE -->|Predicts Log Price| ModelRouter{4. Model Router}
    ModelRouter -->|General Vehicles| Ensemble[4a. Variant 1: CatBoost + LightGBM + XGBoost Ensemble]
    ModelRouter -->|S5 Quality Shop: age ≤7 + known model| S5Model[4b. Variant 4: CatBoost + LightGBM S5 Specialist]
    Ensemble -->|Routes Price Tier| SegRouting[5. Segment-Wise Sub-Models]
    S5Model -->|+8% Premium Fallback to Variant 1| SegRouting
    SegRouting -->|Raw Market Value| Decision[6. Adaptive Dealer Financial Decision Engine]
    Decision -->|Market Value, Buy Target, Risk & Decision| APIResp[7. API JSON Response]
    APIResp -->|Displays Dashboard & Analytics| Frontend
    Frontend -.->|Cloud Sync / Offline Fallback| Sync[8. History Persistence: Supabase or LocalStorage]
```

### 1. User Interface & Data Ingestion
- **React + Vite Web App**: Supports Single Vehicle Evaluation, Enhanced Multi-Grade Inspection, VIN Lookup, Bulk Batch Evaluation, AI Dealer Assistant, and Reverse Price Calculation.
- **Flutter Mobile Shell (`mobile/`)**: Native cross-platform mobile application wrapping the Vite build in an embedded WebView with seamless API URL injection.
- **Client Sanitization**: Automatically normalizes user inputs (trim spacing, title-casing brand/model names, fuel type conversion) before constructing API payloads.

### 2. API Gateway & Request Routing (FastAPI REST API)
- High-throughput asynchronous Python web server exposing structured endpoints:
  - `/evaluate`: Standard ML valuation, risk scoring, and profitability breakdown.
  - `/evaluate-enhanced`: Comprehensive evaluation incorporating component grades (Engine, Tyres, Body, Interior, Electricals).
  - `/predict`: Lightweight valuation endpoint returning target market value and soft physical range bounds.
  - `/reverse-calculate`: Computes maximum buy price target given a desired sell price and profit margin.
  - `/api/brands`: Fetches canonical brand catalog and valid model variants.
  - `/api/registry`: Returns active model variant configuration (**Variant 1**).

### 3. Feature Construction & Normalization Pipeline (`backend/main.py`)
- Constructs a 23-column feature DataFrame (`build_features`) in real time:
  - **Categorical Features**: `brand`, `model`, `variant` (trim), `city`, `locality`, `rto`, `segment_class`, `fuel_type`, `transmission`, `seller_type`.
  - **Engineered Numeric Features**: `vehicle_age`, `odometer_reading`, `km_per_year`, `owner_count`, `brand_tier`, `age_km_interaction`, `ownership_trust_score`, `vehicle_health_score`, `locality_tier`, `usage_category_num`, `locality_density_norm`, `popularity_score_log`.

### 4a. Variant 1 — Global Weighted ML Ensemble (Default)
- Predicts log-transformed price $\hat{y}_{\text{log}}$ using optimized model weights:
  $$\hat{y}_{\text{ensemble}} = 0.8918 \times \hat{y}_{\text{LightGBM}} + 0.1082 \times \hat{y}_{\text{CatBoost}} + 0.0000 \times \hat{y}_{\text{XGBoost}}$$
- **LightGBM (89.18% Weight)**: Processes continuous splits, age-km interaction features, and vehicle usage curves.
- **CatBoost (10.82% Weight)**: Handles high-cardinality categorical target encoding for brand, model, and trim combinations.
- **XGBoost (0.00% Weight)**: Included for API compatibility; optimizer assigns zero weight.

### 4b. Variant 4 — S5 Quality Shop Specialist Model
- Activated **only** when `vehicle_age ≤ 7` **AND** the vehicle's brand/model is present in the S5 catalog.
- Small-dataset (173 rows) two-model ensemble trained with heavy regularization to prevent overfitting:
  $$\hat{y}_{\text{S5}} = 0.8061 \times \hat{y}_{\text{CatBoost}} + 0.1939 \times \hat{y}_{\text{LightGBM}}$$
- **CatBoost (80.61% Weight)**: Primary model for high-cardinality brand/model/trim encoding on premium vehicles.
- **LightGBM (19.39% Weight)**: Supports age-mileage curve fitting on young, low-odometer quality stock.
- **Fallback Rule**: When the vehicle is not found in the S5 catalog, falls back to Variant 1 with a `+8% quality premium` applied.
- **Training Script**: `ml_training/train-s5.py` | **Dataset**: `processed_s5.csv` (173 rows, age 0–7 years)

### 5. Price-Band Segment Routing
- Evaluates the initial ensemble quote and routes the vehicle into dedicated price-tier CatBoost sub-models:
  - **Budget Tier (`₹0 – ₹6 Lakhs`)**: `segment_0_6_lakh.cbm`
  - **Mid Tier (`₹6 Lakhs – ₹12 Lakhs`)**: `segment_6_12_lakh.cbm`
  - **Luxury Tier (`₹12 Lakhs+`)**: `segment_12_plus_lakh.cbm`

### 6. Dealer Financial Decision Engine (`backend/decision_engine.py`)
- **Market Value Calculation**: Converts log price back to INR using $\text{Price} = \exp(\hat{y}_{\text{log}}) - 1$, rounded to the nearest ₹500 step.
- **Configurable Adaptive Parameters (`backend/valuation_config.json`)**: Allows zero-code tuning of similarity weights, age/odometer sigmas, confidence limits, and luxury brand thresholds.
- **Dealer Waterfall Model**:
  $$\text{Recommended Buy Price} = \text{Market Value} \times (1 - \text{Margin \%}) - \text{Recon Costs} - \text{Holding/Risk Buffer}$$
- **Locality & RTO Demand Adjustment**: Dynamic geographic price micro-tuning based on intracity demand signals.
- **Risk & Confidence Engine**: Computes Risk Score (0–100 based on mileage, age, owner count, physical inspection) and Confidence Score (0–100 based on comparable market matches and dataset density).
- **Decision Output**: Generates clear dealer actions (`BUY`, `BUY AFTER INSPECTION`, `NEGOTIATE`, `REJECT`).

### 7. Interactive Response Rendering
- Renders key financial metrics, price range visualizers, risk breakdown gauges, negotiation opening/walk-away targets, and counterfactual insights on both web and mobile dashboards.

### 8. Persistence & Dual-Mode History Sync
- **Authenticated Mode**: Automatically syncs completed valuations to Supabase PostgreSQL database using Row-Level Security (`evaluations` table).
- **Offline / Guest Mode**: Fallback persistence using browser `localStorage` if Supabase environment variables are unconfigured.

---

## 📊 Complete Model Results & Benchmarks

PriceRef ships with **all 4 trained model variants** in `model_registry/` — Variants 1–3 are archived benchmarks, Variant 4 is the active S5 specialist.

### 1. Variant 1 — Global Ensemble Metrics (Active Default)

| Metric | Result | Benchmark Quality |
| :--- | :---: | :--- |
| **Active Engine** | `Variant 1 Ensemble` | Active Default |
| **MAPE (Mean Absolute Percentage Error)** | **`6.16%`** | 🌟 Top Precision (< 6.2% error) |
| **R² Score (Variance Explained)** | **`0.9777` (97.77%)** | 🎯 High Overall Accuracy |
| **MAE (Mean Absolute Error)** | **`₹38,273`** | Average deviation per quote |
| **RMSE (Root Mean Squared Error)** | **`₹98,254`** | Outlier-penalized error |
| **Training Dataset** | `processed_overall.csv` | 33,979 listings |

### 2. Variant 1 — Weighted Ensemble Breakdown

| Base Algorithm | Ensemble Weight (%) | Primary Feature Focus |
| :--- | :---: | :--- |
| ⚡ **LightGBM** | **`89.18%`** | Age, Mileage, Age-KM Interactions, Health Scores |
| 🐱 **CatBoost** | **`10.82%`** | Brand, Model, Trim Variant, City, Locality, RTO |
| 🚀 **XGBoost** | **`0.00%`** | Included for API compatibility (optimizer zeroed out) |

### 3. Segment-wise Price-Band Routing Metrics (Variant 1)

| Price Segment Bracket | Dataset Size (Listings) | Segment MAPE | Segment R² | Active Model File |
| :--- | :---: | :---: | :---: | :--- |
| **Budget Tier** (`₹0 – ₹6 Lakhs`) | 11,941 listings | **`8.14%`** | **`0.9154`** | `segment_0_6_lakh.cbm` |
| **Mid Tier** (`₹6L – ₹12 Lakhs`) | 6,525 listings | **`5.64%`** | **`0.8522`** | `segment_6_12_lakh.cbm` |
| **Luxury / High-Value** (`₹12L+`) | 1,993 listings | **`5.09%`** | **`0.8872`** | `segment_12_plus_lakh.cbm` |

### 4. Variant 4 — S5 Quality Shop Specialist Metrics

> **Activation Condition**: `vehicle_age ≤ 7` AND vehicle brand/model exists in S5 catalog.
> Falls back to Variant 1 + 8% premium for vehicles not in S5 catalog.

| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Model Type** | `CatBoost + LightGBM` | No XGBoost (too few rows) |
| **MAPE** | **`16.38%`** | Expected — small dataset (173 rows) |
| **R² Score** | **`0.3429`** | Narrow specialty scope, not general market |
| **MAE** | **`₹2,72,324`** | Premium segment vehicles |
| **RMSE** | **`₹5,04,875`** | High-value vehicle spread |
| **Training Rows** | `138 train / 35 val` | 80/20 split from 173 total rows |
| **Ensemble Weights** | CatBoost 80.61% / LightGBM 19.39% | Optimizer favors CatBoost on small data |
| **Training Dataset** | `processed_s5.csv` | S5 quality shop listings, age 0–7 years |

### 5. Registered Variant Benchmark Comparison

| Rank | Model Variant | Training Dataset | MAPE (%) | R² Score | MAE (₹) | RMSE (₹) | System Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | **`variant_1` (Default)** | `processed_overall.csv` | **`6.16%`** | **`0.9777`** | **₹38,273** | **₹98,254** | **Active Default** |
| 🥈 **2** | **`variant_3`** | `processed_s1_s4_owner_1.csv` | **`6.50%`** | **`0.9755`** | **₹39,829** | **₹79,227** | Archived Variant |
| 🥉 **3** | **`variant_2`** | `processed_s1_s4_owner.csv` | **`6.67%`** | **`0.9741`** | **₹40,661** | **₹83,619** | Archived Variant |
| 🏅 **4** | **`variant_4` (S5 Specialist)** | `processed_s5.csv` | **`16.38%`** | **`0.3429`** | **₹2,72,324** | **₹5,04,875** | **S5 Quality Active** |

---

## 📱 Flutter Mobile Application (`mobile/`)

PriceRef includes a dedicated **Flutter cross-platform shell** located in `mobile/`. It wraps the compiled React web bundle into a native WebView container for deployment on Android and iOS devices.

### Mobile Build & Execution Pipeline

```powershell
# 1. Bundle web UI for mobile
npm run build:mobile

# 2. Run Flutter app on Android emulator
cd mobile
flutter pub get
flutter run

# 3. Run on a physical device connected to your network
flutter run --dart-define=API_URL=http://192.168.1.10:8000

# 4. Generate Production Release Packages
flutter build apk --release
flutter build appbundle --release
flutter build ios --release
```

### Mobile Configuration Flags (`--dart-define`)

| Configuration Flag | Description | Default Value |
| :--- | :--- | :--- |
| `API_URL` | FastAPI backend base URL accessible by emulator/device | `http://10.0.2.2:8000` (Android) / `http://localhost:8000` (iOS) |
| `WEB_URL` | Development live-reload server URL (optional) | Bundled `assets/web/` |

---

## 🛠️ System Architecture & Connection Flow

```mermaid
graph TD
    WebUI[Vite + React Dashboard] -->|HTTP / REST API| FastAPI[FastAPI Backend Server]
    MobileShell[Flutter WebView Shell] -->|Embedded Web Assets| WebUI
    FastAPI -->|Loads Pre-trained Artifacts| Registry[Model Registry: Variant 1]
    Registry -->|Ensemble Ingestion| Predictor[CatBoost + LightGBM + XGBoost Predictor]
    FastAPI -->|Reads Engine Configuration| Config[valuation_config.json / engine_config.json]
    WebUI -.->|Optional Auth & History Sync| Supabase[(Supabase PostgreSQL Database)]
```

### Connection Details
* **Frontend Web**: React (Vite) running on `http://localhost:5173`.
* **Mobile Shell**: Flutter app running on Android / iOS device.
* **Backend API**: FastAPI server running on `http://127.0.0.1:8000`.
* **Swagger API Docs**: `http://localhost:8000/docs`.

---

## 📋 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/evaluate` | `POST` | Core ML valuation request — returns Market Value, Buy/Sell targets, Risk score, and Recommendation. |
| `/evaluate-enhanced` | `POST` | Comprehensive evaluation incorporating physical grade inspection and component condition. |
| `/predict` | `POST` | Fast ML market value prediction with price range bounds. |
| `/reverse-calculate` | `POST` | Calculates maximum buy price target given a desired sell price and profit margin. |
| `/api/brands` | `GET` | Fetches canonical brand catalog and valid models. |
| `/api/registry` | `GET` | Returns active model variant configuration (**Variant 1**). |

---

## ⚙️ Configuration & Utility Scripts

### Configurable Engine Parameters (`backend/valuation_config.json`)
The adaptive decision engine allows zero-code adjustment of similarity weights and thresholds without code modification:
- `similarity_weights`: Feature weights for brand, model, variant, age, odometer, fuel, locality, transmission, owner count.
- `luxury_brands`: Explicit list of luxury brands receiving tailored geographic dampening and similarity thresholds.
- `confidence_weights` & `confidence_labels`: Tuning confidence score ranges and market support thresholds.

### Diagnostic & Operational Helper Scripts (`scripts/`)

| Script File | Command | Description |
| :--- | :--- | :--- |
| `system_health_check.py` | `python scripts/system_health_check.py` | Validates model files, backend imports, decision engine logic, and mock valuation requests. |
| `validate_models.py` | `python scripts/validate_models.py` | Runs automated prediction verification across all variant artifacts. |
| `generate_engine_config.py` | `python scripts/generate_engine_config.py` | Regenerates statistical market percentiles and locality demand tables into `engine_config.json`. |
| `show_buy_price.py` | `python scripts/show_buy_price.py` | CLI tool to calculate dealer buy prices, margins, and risk buffers interactively. |
| `feature_sensitivity_test.py` | `python scripts/feature_sensitivity_test.py` | Tests model sensitivity to individual feature changes (mileage, age, condition). |
| `query_exact_car.py` | `python scripts/query_exact_car.py` | Queries the training dataset for a specific vehicle and returns matching comparable listings. |
| `verify_fixes.py` | `python scripts/verify_fixes.py` | Sanity-checks recent model or engine fixes by running before/after valuation comparisons. |

---

## 🏁 Quick Start (Run Out of the Box)

> 💡 **No model training is required after cloning.** The pre-trained Variant 1 model artifacts are included directly in `model_registry/variant_1`.

### Prerequisites
* **Python 3.10+**
* **Node.js 18+**
* **Flutter SDK 3.44+** *(Optional: required only for mobile app)*
* **Git**

### 1. Clone & Set Up Backend

```bash
# Clone repository
git clone https://github.com/UmaDamotharan/Price-Prediction.git
cd Price-Prediction

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python backend dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables (Optional for Supabase)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Or create `.env` manually:

```env
VITE_API_URL=http://localhost:8000
ACTIVE_VARIANT_ID=variant_1
VITE_SUPABASE_URL=https://placeholder-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2MDA0MDAwMDAsImV4cCI6MTkwMDA0MDAwMH0.placeholder
```

### 3. Run FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8000
```
Backend will start at `http://127.0.0.1:8000` and load pre-trained Variant 1 models automatically.

### 4. Install & Run Frontend Web UI

Open a second terminal:

```bash
# Install Node dependencies
npm install

# Start Vite React development server
npm run dev
```
Frontend will start at `http://localhost:5173`.

### 5. Build & Run Mobile Shell (Optional)

Open a third terminal:

```bash
# Bundle React web assets for mobile WebView
npm run build:mobile

# Launch Flutter mobile application
cd mobile
flutter pub get
flutter run
```

---

## 🍴 How to Fork this Repository

If you want to create your own copy of this repository on GitHub to customize or host under your account:

1. Click the **`Fork`** button at the top right of this repository page ([`github.com/UmaDamotharan/Price-Prediction`](https://github.com/UmaDamotharan/Price-Prediction)).
2. Select your account to create an independent copy under your GitHub profile.
3. You can now clone your forked repository or connect it directly to **Render**, **Railway**, or **Vercel** for 1-click cloud deployment!

---

## 🌐 Deployment & Hosting Guide

Anyone cloning or forking this repository can deploy it online using any of the following methods:

### **Method 1: Render.com (Recommended — 1-Click Auto Blueprint)**

Since `render.yaml` is pre-configured in this repository:

1. Fork this repository to your GitHub account.
2. Sign up at **[render.com](https://render.com)**.
3. Click **New + → Blueprint** and select your repository.
4. Click **Apply**. Render will automatically deploy:
   - **Backend Web Service**: Python FastAPI + Uvicorn server (`price-prediction-backend`).
   - **Frontend Static Site**: React Vite bundle (`price-prediction-frontend`).

---

### **Method 2: Vercel (Frontend) + Railway (Backend)**

1. **Backend (Railway.app):**
   - New Project → Deploy from GitHub → Select Repository.
   - Set Start Command: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
2. **Frontend (Vercel.com):**
   - New Project → Import GitHub Repo.
   - Build Command: `npm run build`, Output Directory: `dist`.
   - Add Environment Variable: `VITE_API_URL=https://your-backend-railway-url.up.railway.app`

---

### **Method 3: Docker Deployment**

Run using Docker locally or on any Cloud VPS (AWS / DigitalOcean / Hetzner):

```bash
# Build & Run using docker-compose
docker-compose up --build
```

---

## ⚡ Supabase Setup (Optional — User Accounts & History Sync)

> **Note:** Core ML valuations work **100% offline without Supabase**. Supabase is only required if you want user authentication (login/signup) and persistent cloud evaluation history.

To enable Supabase integration:

1. Create a free project at [supabase.com](https://supabase.com).
2. Copy your project **URL** and **anon Key** into `.env`:
   ```env
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_ANON_KEY=your-anon-key
   ```
3. Run this SQL in your Supabase **SQL Editor**:

```sql
-- User Profiles
create table if not exists profiles (
  id uuid references auth.users(id) primary key,
  name text not null,
  avatar text not null default 'U',
  role text not null default 'Dealer',
  created_at timestamptz default now()
);

-- Valuation History
create table if not exists evaluations (
  id text primary key,
  user_id uuid references auth.users(id) on delete cascade,
  created_at timestamptz default now(),
  source text, brand text, model text, year int,
  fuel text, transmission text, city text,
  odometer int, fuel_efficiency numeric, owner_count int,
  engine_cc int, condition text, seller_asking_price numeric,
  market_value numeric, buy_price numeric, sell_price numeric,
  expected_profit numeric, margin_pct numeric, risk_score numeric,
  confidence_score numeric, deal_quality_score numeric, action text,
  urgency_score numeric, is_ml_powered boolean,
  positive_factors jsonb, negative_factors jsonb
);

-- Enable RLS
alter table profiles enable row level security;
alter table evaluations enable row level security;

create policy "own profile read"   on profiles   for select using (auth.uid() = id);
create policy "own profile insert" on profiles   for insert with check (auth.uid() = id);
create policy "own evals read"     on evaluations for select using (auth.uid() = user_id);
create policy "own evals insert"   on evaluations for insert with check (auth.uid() = user_id);
```

---

## 🔬 Optional: Retraining Model Variants

> **Note:** This section is completely optional. The app runs immediately without running these scripts.

If you wish to clean a raw dataset and retrain model variants from scratch in the future:

```bash
# ── Variant 1: Full General Market Model (33,979 rows) ──────────────────────
python ml_training/clean-1.py     # Clean raw dataset → processed_overall.csv
python ml_training/train-1.py     # Train CatBoost + LightGBM + XGBoost ensemble

# ── Variant 4: S5 Quality Shop Specialist (173 rows, age 0–7 years) ─────────
python ml_training/clean-s5.py    # Clean S5 shop data → processed_s5.csv
python ml_training/train-s5.py    # Train CatBoost + LightGBM specialist model
```

> **Note:** Variant 2 and Variant 3 training scripts (`clean-2.py`, `train-2.py`, `clean-3.py`, `train-3.py`) are excluded from the public repository. They are archived benchmarks superseded by Variant 1.

*Note: Training outputs will update `model_registry/variant_N` and automatically register in `model_registry/registry.json`. Variant 4 never auto-promotes to default — it is a specialist S5 model only. The active default is always `Variant 1` (`ACTIVE_VARIANT_ID=variant_1`).*

