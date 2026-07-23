# PriceRef — AI-Powered Used Vehicle Valuation Engine

> **Data-driven valuation, acquisition risk assessment, and deal profitability for used vehicles.**

PriceRef is a high-performance machine learning system built for instant vehicle valuation, profit estimation, acquisition risk scoring, and negotiation strategy calculation. Powered by a specialized **CatBoost, LightGBM, and XGBoost ensemble (Variant 2)**, PriceRef processes vehicle attributes and returns market valuations in milliseconds.

---

## 🔄 End-to-End Machine Learning Pipeline

PriceRef implements an end-to-end ML pipeline spanning raw data preprocessing, feature engineering, multi-algorithm ensemble training, segment-wise routing, and financial decision engine processing:

```mermaid
graph TD
    Raw[Raw Vehicle Listings Data] --> Clean[1. Data Cleaning & Normalization]
    Clean --> FE[2. Feature Engineering & Tier Mapping]
    FE --> Train[3. Multi-Algorithm Ensemble Training]
    Train --> Weights[4. SLSQP Constrained Weight Optimization]
    Weights --> SegRouting[5. Price-Band Segment Routing]
    SegRouting --> Backend[6. Inference & Decision Engine]
    Backend --> Output[Market Value, Buy Target, Risk Score & Decision]
```

### 1. Data Preprocessing & Cleaning (`ml_training/clean_data-1.py`)
- **Outlier Filtering**: IQR filtering on raw selling prices (removing erroneous data entries) and extreme odometer readings (> 250,000 km).
- **Text Normalization**: Stripping extra whitespaces, lowercasing, and mapping raw text to canonical OEM brand and model names.
- **Missing Value Handling**: Imputing missing fuel types, transmissions, engine displacement (`cc`), and city names based on model-year medians.

### 2. Feature Engineering & Feature Vector Construction
- **Categorical Encodings**: `brand`, `model`, `variant` (trim), `city`, `locality`, `rto`, `segment_class`, `fuel_type`, `transmission`, `seller_type`.
- **Numeric Features**:
  - `vehicle_age`: `current_year - manufacturing_year`
  - `odometer_reading`: Total kilometers driven
  - `km_per_year`: `odometer_reading / max(1, vehicle_age)`
  - `owner_count`: Number of previous owners (1, 2, 3, 4+)
  - `brand_tier`: Brand classification score (0 = Budget to 4 = Luxury)
  - `age_km_interaction`: `vehicle_age * (odometer_reading / 10,000)`
  - `ownership_trust_score`: Trust weight based on owner count and vehicle age
  - `vehicle_health_score`: Calculated physical condition grade (0–100)
  - `locality_tier` & `locality_density_norm`: Regional price premium factors
  - `popularity_score_log`: Log-transformed demand volume for brand/model

### 3. Ensemble Model Training (`ml_training/train-1.py`)
- **Target Transformation**: `log1p(selling_price)` log-normal target transformation to handle heteroskedasticity and skewed vehicle price distributions.
- **Base Algorithms**:
  - **CatBoost Regressor**: Handles high-cardinality categoricals natively using target statistics.
  - **LightGBM Booster**: Fast gradient boosting handling continuous numeric splits and non-linear interactions.
  - **XGBoost Regressor**: Residual boundary alignment.

### 4. SLSQP Constrained Weight Optimization
- Uses SciPy's `minimize(method='SLSQP')` to solve for non-negative weights $w_1, w_2, w_3$ summing to 1.0 that minimize validation Mean Squared Error (MSE):
  $$\min_{w} \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \sum_{k=1}^{3} w_k \hat{y}_{k,i} \right)^2 \quad \text{s.t.} \quad \sum w_k = 1, \: w_k \ge 0$$
- Resulting Weights: **LightGBM 81.52%**, **CatBoost 18.48%**, **XGBoost < 0.01%**.

### 5. Price-Band Segment Routing
- Evaluates initial ensemble predictions and routes vehicles into 3 specialized price-tier CatBoost sub-models:
  - **Budget Tier (`₹0 – ₹6 Lakhs`)**: `segment_0_6_lakh.cbm`
  - **Mid Tier (`₹6 Lakhs – ₹12 Lakhs`)**: `segment_6_12_lakh.cbm`
  - **Luxury Tier (`₹12 Lakhs+`)**: `segment_12_plus_lakh.cbm`

### 6. Dealer Financial Decision Engine (`backend/decision_engine.py`)
- **Inverse Log Transform**: $\text{Price} = \exp(\hat{y}_{\text{log}}) - 1$, rounded to the nearest ₹500 step.
- **Physical Limits Clamp**: Soft data integrity bounds (₹50k floor, ₹20M cap).
- **Waterfall Profit Calculation**:
  $$\text{Recommended Buy Price} = \text{Market Value} \times (1 - \text{Margin \%}) - \text{Recon Costs} - \text{Holding/Risk Buffer}$$
- **Decision Engine**: Computes Risk Score (0–100), Confidence Score (0–100), and outputs actionable decisions (`BUY`, `BUY AFTER INSPECTION`, `NEGOTIATE`, `REJECT`).

---

## 📊 Model Performance Results (Variant 2)

PriceRef comes pre-packaged with **Variant 2 Model Artifacts** in `model_registry/variant_2`.

### 1. Overall Global Ensemble Metrics

| Metric | Result | Benchmark Quality |
| :--- | :---: | :--- |
| **Active Engine** | `Variant 2 Ensemble` | Active Default |
| **MAPE (Mean Absolute Percentage Error)** | **`6.28%`** | 🌟 Top Precision (< 7% error) |
| **R² Score (Variance Explained)** | **`0.9760` (97.6%)** | 🎯 High Overall Accuracy |
| **MAE (Mean Absolute Error)** | **`₹37,988.79`** | Average deviation per quote |
| **RMSE (Root Mean Squared Error)** | **`₹75,207.06`** | Outlier-penalized error |

### 2. Weighted Ensemble Breakdown

| Base Algorithm | Ensemble Weight (%) | Primary Feature Focus |
| :--- | :---: | :--- |
| ⚡ **LightGBM** | **`81.52%`** | Age, Mileage, Age-KM Interactions, Health Scores |
| 🐱 **CatBoost** | **`18.48%`** | Brand, Model, Trim Variant, City, Locality, RTO |
| 🚀 **XGBoost** | **`< 0.01%`** | Residual boundary adjustment |

### 3. Segment-wise Price-Band Routing Metrics

| Price Segment Bracket | Dataset Size (Listings) | Segment MAPE | Segment R² | Active Model File |
| :--- | :---: | :---: | :---: | :--- |
| **Budget Tier** (`₹0 – ₹6 Lakhs`) | 11,941 listings | **`8.14%`** | **`0.9154`** | `segment_0_6_lakh.cbm` |
| **Mid Tier** (`₹6L – ₹12 Lakhs`) | 6,525 listings | **`5.64%`** | **`0.8522`** | `segment_6_12_lakh.cbm` |
| **Luxury / High-Value** (`₹12L+`) | 1,993 listings | **`5.09%`** | **`0.8872`** | `segment_12_plus_lakh.cbm` |

### 4. 6-Variant Benchmark Comparison

| Rank | Model Variant | Training Dataset | MAPE (%) | R² Score | MAE (₹) | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`variant_2` (Selected)** | `processed_widown-1.csv` | **`6.28%`** | **`0.9760`** | **₹37,988** | **Active Default** |
| 🥈 **2** | **`variant_7`** | `processed_widown1-6.csv` | **`6.28%`** | **`0.9760`** | **₹37,988** | Retrained Checkpoint |
| 🥉 **3** | **`variant_4`** | `processed_pincode_with_owner-3.csv` | **`6.36%`** | **`0.9756`** | **₹38,933** | Comparative Split |
| 4 | **`variant_6`** | `processed_pincode_with_owner1_filled-5.csv` | **`6.36%`** | **`0.9756`** | **₹38,933** | Imputed Split |
| 5 | **`variant_3`** | `processed_widoutown-2.csv` | **`6.37%`** | **`0.9756`** | **₹38,623** | No-Owner Split |
| 6 | **`variant_5`** | `processed_pincode without owner-4.csv` | **`6.47%`** | **`0.9749`** | **₹39,704** | Base Pincode Split |

---

## 🛠️ System Architecture & Connection Flow

```mermaid
graph TD
    UI[Vite + React Dashboard] -->|HTTP / REST API| FastAPI[FastAPI Backend Server]
    FastAPI -->|Loads Pre-trained Artifacts| Registry[Model Registry: Variant 2]
    Registry -->|Ensemble Ingestion| Predictor[CatBoost + LightGBM + XGBoost Predictor]
    UI -.->|Optional Auth & History Sync| Supabase[(Supabase PostgreSQL Database)]
```

### Connection Details
* **Frontend**: React (Vite) running on `http://localhost:5173` (or production port).
* **Backend API**: FastAPI server running on `http://127.0.0.1:8000` (`http://localhost:8000`).
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
| `/api/registry` | `GET` | Returns active model variant configuration (**Variant 2**). |

---

## 🏁 Quick Start (Run Out of the Box)

> 💡 **No model training is required after cloning.** The pre-trained Variant 2 model artifacts are included directly in `model_registry/variant_2`.

### Prerequisites
* **Python 3.10+**
* **Node.js 18+**
* **Git**

### 1. Clone & Set Up Backend

```bash
# Clone repository
git clone https://github.com/Hars03082005/Autopricer.git
cd Autopricer

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
VITE_SUPABASE_URL=https://placeholder-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2MDA0MDAwMDAsImV4cCI6MTkwMDA0MDAwMH0.placeholder
ACTIVE_VARIANT_ID=variant_2
```

### 3. Run FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8000
```
Backend will start at `http://127.0.0.1:8000` and load pre-trained Variant 2 models automatically.

### 4. Install & Run Frontend UI

Open a second terminal:

```bash
# Install Node dependencies
npm install

# Start Vite React development server
npm run dev
```
Frontend will start at `http://localhost:5173`.

---

## 🍴 How to Fork this Repository

If you want to create your own copy of this repository on GitHub to customize or host under your account:

1. Click the **`Fork`** button at the top right of this repository page ([`github.com/Hars03082005/Autopricer`](https://github.com/Hars03082005/Autopricer)).
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

## 🔬 Optional: Retraining Model Variant 2

> **Note:** This section is completely optional. The app runs immediately without running these scripts.

If you wish to clean a raw dataset and retrain Variant 2 from scratch in the future:

```bash
# 1. Clean raw dataset for Variant 2
python ml_training/clean_data-1.py

# 2. Train Variant 2 Ensemble Model
python ml_training/train-1.py
```

*Note: Training outputs will update `model_registry/variant_2`.*
