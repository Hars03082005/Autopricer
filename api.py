from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI(title="AutoPricer API", version="2.0")

# Load model once at startup
with open("models/model_artifacts.pkl", "rb") as f:
    ARTIFACTS = pickle.load(f)

MODEL    = ARTIFACTS["model"]
LE_MAP   = ARTIFACTS["le_map"]
FEATURES = ARTIFACTS["features"]
METRICS  = ARTIFACTS["metrics"]


class CarInput(BaseModel):
    brand:        str
    year:         int
    km_driven:    float
    fuel_type:    str   # Petrol / Diesel / Electric / Hybrid
    transmission: str   # Manual / Automatic
    seller_type:  str   # Dealer / Individual / Trustmark Dealer
    owner_num:    int   # 1 / 2 / 3 / 4
    brand_tier:   str   # Budget / Mid / Premium / Luxury


class PredictResponse(BaseModel):
    predicted:    float
    low:          float
    high:         float
    metrics:      dict


def engineer(inp: dict) -> pd.DataFrame:
    df = pd.DataFrame([inp])

    age               = max(2026 - df["year"].iloc[0], 0)
    df["age"]         = age
    df["km_per_year"] = df["km_driven"] / (age if age > 0 else 0.5)
    df["age_km"]      = df["age"] * df["km_driven"]
    df["is_low_mileage"] = (df["km_per_year"] < 5000).astype(int)

    for col, le in LE_MAP.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])

    return df[FEATURES].fillna(0)


@app.get("/")
def root():
    return {"status": "ok", "message": "AutoPricer API is running"}


@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "model_r2": METRICS["r2"],
        "model_mae": METRICS["mae"]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(car: CarInput):
    try:
        X         = engineer(car.dict())
        log_pred  = MODEL.predict(X)[0]
        price     = np.expm1(log_pred)
        return {
            "predicted": round(price, 0),
            "low":       round(price * 0.92, 0),
            "high":      round(price * 1.08, 0),
            "metrics":   METRICS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brands")
def brands():
    le = LE_MAP.get("brand")
    return {"brands": list(le.classes_) if le else []}


@app.get("/options")
def options():
    return {
        "fuel_types":    ["Petrol", "Diesel", "Electric", "Hybrid"],
        "transmissions": ["Manual", "Automatic"],
        "seller_types":  ["Dealer", "Individual", "Trustmark Dealer"],
        "brand_tiers":   ["Budget", "Mid", "Premium", "Luxury"],
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)