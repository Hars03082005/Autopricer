# PriceRef — Automotive Dealer Valuation & Acquisition Terminal

> **Professional used-vehicle valuation, acquisition risk assessment, and dealer margin intelligence platform.**

PriceRef is a full-stack valuation terminal designed for used-car dealerships. Powered by a production machine learning ensemble (LightGBM champions with luxury CatBoost specialist routing), the system delivers real-time market value estimates, recommended buy/sell ranges, reconditioning cost waterfalls, expected net margins, deal quality scores, and localized market evidence.

---

## ⚡ Technology Stack

- **Backend**: Python 3.11+, FastAPI, Pydantic v2, Uvicorn
- **Machine Learning**: LightGBM, CatBoost, Scikit-learn (Production bundle in `model_registry/final/ensemble_bundle.pkl`)
- **Frontend**: React 18, Vite, Vanilla CSS design system, Recharts
- **Database & Auth (Optional)**: Supabase / PostgreSQL (Evaluations work fully offline with localStorage fallback)
- **Containerization**: Docker, Docker Compose, Azure Container Apps / Render ready

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+

### 2. Backend Setup
```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# .\venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r backend/requirements.txt

# Start FastAPI backend server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API runs at `http://127.0.0.1:8000`. Swagger API documentation available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
```bash
# Install NPM dependencies
npm install

# Start Vite development server
npm run dev
```
Valuation Terminal runs at `http://localhost:5173`.

---

## 🧪 Testing & Validation

```bash
# Run backend pytest suite (135 tests)
pytest tests/

# Validate frontend production bundle build
npm run build
```

---

## 📁 Repository Structure

```
Price-Prediction/
├── backend/                  # FastAPI backend application & routing
│   ├── main.py               # Main API endpoints (/evaluate, /predict, /health)
│   ├── decision_engine.py    # Acquisition ranges, margin waterfall, adaptive comps
│   ├── champion_predictor.py # Production ensemble loading & inference
│   ├── auth.py               # Token verification & dealer role management
│   └── config.py             # Environment configuration & limits
├── model_registry/           # Production ML model registry
│   ├── final/                # Active frozen production model bundle
│   │   ├── ensemble_bundle.pkl
│   │   └── model_metadata.json
│   └── registry.json         # Active variant metadata
├── src/                      # React frontend application
│   ├── screens/              # Dashboard, New Valuation, Report, Intel, Assistant
│   ├── components/           # Reusable UI components & SearchableDropdown
│   ├── context/              # Global AppContext & AuthContext
│   └── utils/                # CSV exporter, catalog helpers, API clients
├── tests/                    # Comprehensive unit & integration test suite
├── scripts/                  # Operational scripts & health check utilities
├── docker-compose.yml        # Multi-container local deployment configuration
└── Dockerfile                # Production multi-stage Docker build
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
# Backend & API Configuration
APP_ENVIRONMENT=development
PORT=8000
ACTIVE_VARIANT_ID=final
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173

# Optional Supabase Cloud Sync
VITE_API_URL=http://127.0.0.1:8000
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

---

## 📄 License
Internal proprietary dealer terminal. All rights reserved.
