# 🚗 AutoPricer — AI-Powered Used Car Price Predictor

> **Predict pre-owned vehicle prices instantly** using a stacking ensemble of XGBoost, LightGBM, and CatBoost — served via a FastAPI backend and consumed by a sleek Flutter mobile app.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Starting the API](#starting-the-api)
- [API Reference](#api-reference)
- [Flutter App](#flutter-app)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

**AutoPricer** is a full-stack machine learning application that predicts used car prices based on vehicle attributes like brand, age, mileage, fuel type, and transmission. It combines data from two major sources:

- **CarDekho** — Indian used car marketplace
- **Craigslist Vehicles** — US-based listings (converted to INR)

The cleaned and merged dataset powers a **Stacking Ensemble** model that achieves high accuracy, exposed through a production-ready REST API and consumed by a cross-platform Flutter mobile app.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CarDekho    │     │  Craigslist  │     │  Flutter App      │
│  Dataset     │     │  Dataset     │     │  (Mobile Client)  │
└──────┬───────┘     └──────┬───────┘     └────────┬──────────┘
       │                    │                      │
       ▼                    ▼                      │
  ┌─────────────────────────────┐                  │
  │   data_cleaning.py          │                  │
  │   (Clean + Merge + Feature  │                  │
  │    Engineering)             │                  │
  └────────────┬────────────────┘                  │
               ▼                                   │
  ┌─────────────────────────────┐                  │
  │   model.py                  │                  │
  │   (XGBoost + LightGBM +    │                  │
  │    CatBoost Stacking)       │                  │
  └────────────┬────────────────┘                  │
               ▼                                   │
  ┌─────────────────────────────┐                  │
  │   api.py (FastAPI)          │◄─────────────────┘
  │   /predict  /brands         │    HTTP REST
  │   /health   /options        │
  └─────────────────────────────┘
```

---

## Tech Stack

| Layer           | Technology                                          |
| --------------- | --------------------------------------------------- |
| **ML Models**   | XGBoost, LightGBM, CatBoost, Scikit-learn (Stacking)|
| **API**         | FastAPI + Uvicorn                                   |
| **Mobile App**  | Flutter (Dart) with fl_chart, Google Fonts           |
| **Data**        | Pandas, NumPy                                       |
| **Deployment**  | Docker, Procfile (Heroku/Railway), Nixpacks          |

---

## Project Structure

```
autopricer_v2/
├── data/
│   ├── cardekho.csv           # Indian used car dataset
│   ├── vehicles.csv           # US Craigslist dataset (~1.4 GB)
│   └── merged_clean.csv       # Cleaned & merged output
├── models/
│   └── model_artifacts.pkl    # Trained model + encoders + metrics
├── flutter_app/
│   └── autopricer/            # Flutter mobile application
│       ├── lib/
│       │   ├── main.dart
│       │   ├── app_theme.dart
│       │   ├── screens/       # Home, Predict, Dashboard, About
│       │   ├── services/      # API service layer
│       │   └── widgets/       # Reusable UI components
│       └── pubspec.yaml
├── data_cleaning.py           # Data pipeline (clean + merge)
├── model.py                   # Model training & evaluation
├── api.py                     # FastAPI REST server
├── run.py                     # Server entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build
├── Procfile                   # Heroku/Railway deploy
├── nixpacks.toml              # Nixpacks config
├── start.sh                   # Shell startup script
└── .gitignore
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Flutter 3.x** (for mobile app development, optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Hars03082005/Autopricer.git
cd Autopricer

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Step 1 — Data Cleaning (optional if `merged_clean.csv` exists)

```bash
python data_cleaning.py
```

This reads `data/cardekho.csv` and `data/vehicles.csv`, cleans and merges them, then saves the output to `data/merged_clean.csv`.

> ⚠️ `vehicles.csv` is ~1.4 GB. This step requires significant RAM and may take several minutes.

#### Step 2 — Model Training

```bash
python model.py
```

Trains the stacking ensemble on `data/merged_clean.csv` and saves the trained artifacts to `models/model_artifacts.pkl`.

**Output metrics:**
- **R² Score**: ~0.65
- **MAE**: ~Rs.4,38,048
- **RMSE**: ~Rs.7,31,858

#### Step 3 — Start the API

```bash
python api.py
# or
python run.py
```

The API server starts on `http://localhost:8000`.

---

## API Reference

### `GET /`
Health check — returns API status.

### `GET /health`
Returns model health and performance metrics.

```json
{
  "status": "healthy",
  "model_r2": 0.647,
  "model_mae": 438048.0
}
```

### `POST /predict`
Predict the price of a used car.

**Request Body:**
```json
{
  "brand": "Hyundai",
  "year": 2019,
  "km_driven": 35000,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "seller_type": "Individual",
  "owner_num": 1,
  "brand_tier": "Mid"
}
```

**Response:**
```json
{
  "predicted": 550000.0,
  "low": 506000.0,
  "high": 594000.0,
  "metrics": { "r2": 0.647, "mae": 438048.0, "rmse": 731858.0 }
}
```

### `GET /brands`
Returns the list of known car brands.

### `GET /options`
Returns available dropdown values (fuel types, transmissions, seller types, brand tiers).

---

## Flutter App

The mobile app is a cross-platform Flutter application with four main screens:

| Screen        | Description                                    |
| ------------- | ---------------------------------------------- |
| **Home**      | Overview and quick access to prediction        |
| **Predict**   | Enter car details and get instant price estimate|
| **Dashboard** | Market analytics and price distribution charts |
| **About**     | App info and model performance metrics         |

### Running the Flutter App

```bash
cd flutter_app/autopricer

# Install dependencies
flutter pub get

# Run on connected device or emulator
flutter run
```

> **Note:** Update the API base URL in `lib/services/api_service.dart` to point to your running backend.

---

## Model Details

### Approach
- **Target variable**: `price_inr` (log-transformed using `log1p` for better distribution)
- **Ensemble method**: Stacking Regressor with Ridge meta-learner

### Base Models
| Model     | Key Hyperparameters                        |
| --------- | ------------------------------------------ |
| XGBoost   | 1000 estimators, lr=0.05, depth=7          |
| LightGBM  | 1000 estimators, lr=0.05, depth=7          |
| CatBoost  | 1000 iterations, lr=0.05, depth=7          |

### Engineered Features
| Feature           | Description                                   |
| ----------------- | --------------------------------------------- |
| `age`             | Current year − manufacture year               |
| `km_per_year`     | Average kilometers driven per year             |
| `age_km`          | Interaction: age × km_driven                  |
| `is_low_mileage`  | Binary flag: km_per_year < 5000               |
| `brand_tier`      | Categorical: Budget / Mid / Premium / Luxury  |
| `owner_num`       | Ordinal encoding of owner count               |

---

## Deployment

### Docker

```bash
docker build -t autopricer .
docker run -p 8000:8000 autopricer
```

### Railway / Heroku

The project includes `Procfile`, `nixpacks.toml`, and `Dockerfile` for one-click deployment on platforms like **Railway** or **Heroku**.

```bash
# Railway
railway up

# Heroku
git push heroku master
```

---

## License

This project is for educational and personal use.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Hars03082005">Harshavardhana</a>
</p>
