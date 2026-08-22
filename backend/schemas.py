"""Pydantic schemas for the Champion API.

Defines strict input validation for VehicleRecord and structured output
types for PredictionResponse and HealthResponse.

Selling price is NEVER accepted as an inference input — attempts to include
it will raise a 422 Validation Error via the `no_selling_price` validator.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Input: Vehicle Record ─────────────────────────────────────────────────────

class VehicleRecord(BaseModel):
    """15 raw vehicle features required for price prediction.

    String fields are automatically lowercased and stripped.
    Missing/None values are accepted for most fields — the predictor
    applies training-median imputation for numerics and 'unknown' for categoricals.

    selling_price MUST NOT be included — any attempt to pass it raises a 422.
    """

    # ── Categorical features ──────────────────────────────────────────────────
    brand:        Optional[str] = Field(None, max_length=50,  description="Vehicle brand e.g. 'hyundai'")
    model:        Optional[str] = Field(None, max_length=50,  description="Vehicle model e.g. 'creta'")
    variant:      Optional[str] = Field(None, max_length=100, description="Vehicle variant e.g. 'sx'")
    locality:     Optional[str] = Field(None, max_length=100, description="Locality name e.g. 'indiranagar'")
    rto:          Optional[str] = Field(None, max_length=30,  description="RTO code e.g. 'ka03'")
    fuel_type:    Optional[str] = Field(None, max_length=30,  description="'petrol' | 'diesel' | 'cng' | 'electric' | 'hybrid'")
    transmission: Optional[str] = Field(None, max_length=30,  description="'manual' | 'automatic'")
    seller_type:  Optional[str] = Field(None, max_length=30,  description="'dealer' | 'individual'")
    color:        Optional[str] = Field(None, max_length=30,  description="Vehicle color e.g. 'white'")

    # ── Numerical features ────────────────────────────────────────────────────
    vehicle_age:      Optional[float] = Field(None, ge=0, le=60,    description="Age of vehicle in years (≥ 0)")
    odometer_reading: Optional[float] = Field(None, ge=0, le=999999, description="Total km driven (≥ 0)")
    km_per_year:      Optional[float] = Field(None, ge=0, le=999999, description="Average km per year (≥ 0)")
    owner_count:      Optional[float] = Field(None, ge=1, le=20,    description="Number of previous owners (≥ 1)")
    certified:        Optional[float] = Field(None, ge=0, le=1,     description="Certified inspection: 0 or 1")
    pincode:          Optional[float] = Field(None, ge=100000, le=999999, description="6-digit Indian PIN code")

    model_config = {"extra": "forbid"}  # Reject unknown fields — including selling_price

    @field_validator("brand", "model", "variant", "locality", "rto",
                     "fuel_type", "transmission", "seller_type", "color",
                     mode="before")
    @classmethod
    def _clean_string(cls, v: object) -> Optional[str]:
        """Auto-lowercase and strip all string fields before validation."""
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s else None

    @model_validator(mode="before")
    @classmethod
    def _reject_selling_price(cls, values: dict) -> dict:
        """Raise immediately if selling_price or any target proxy is present."""
        forbidden = {
            "selling_price", "target", "price", "sale_price", "label",
            "actual_price", "ground_truth",
        }
        found = forbidden & set(str(k).lower() for k in values)
        if found:
            raise ValueError(
                f"Field(s) {sorted(found)} are not allowed in prediction input. "
                "selling_price must never be provided during inference."
            )
        return values

    def to_inference_dict(self) -> dict:
        """Convert to a plain dict for the predictor. None values are kept —
        the predictor handles imputation."""
        return {
            "brand":            self.brand,
            "model":            self.model,
            "variant":          self.variant,
            "locality":         self.locality,
            "rto":              self.rto,
            "fuel_type":        self.fuel_type,
            "transmission":     self.transmission,
            "seller_type":      self.seller_type,
            "color":            self.color,
            "vehicle_age":      self.vehicle_age,
            "odometer_reading": self.odometer_reading,
            "km_per_year":      self.km_per_year,
            "owner_count":      self.owner_count,
            "certified":        self.certified,
            "pincode":          self.pincode,
        }


# ── Output: Prediction Response ───────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Structured response from the final production prediction pipeline."""

    predicted_price:               float            = Field(..., description="Final predicted selling price in INR")
    segment_probability:           float            = Field(0.0, description="Routing indicator / probability (0.0=Champion, 1.0=Specialist)")
    lgbm_prediction:               float            = Field(..., description="Raw 5-seed LightGBM champion prediction in INR")
    catboost_prediction:           float            = Field(..., description="CatBoost specialist prediction in INR")
    final_gate:                    str              = Field(..., description="Routing gate decision description")
    routing:                       str              = Field(..., description="Routing strategy tag e.g. 'Champion' or 'Specialist'")
    champion_prediction:           Optional[float]  = Field(None, description="5-Seed LightGBM Champion prediction")
    luxury_specialist_prediction:  Optional[float]  = Field(None, description="Luxury CatBoost Specialist prediction")
    routing_decision:              Optional[str]    = Field(None, description="'champion' or 'specialist'")

    model_config = {"json_schema_extra": {
        "example": {
            "predicted_price":     1453772.0,
            "segment_probability": 0.0,
            "lgbm_prediction":     1453772.0,
            "catboost_prediction": 1453772.0,
            "final_gate":          "Strategy D: Champion (5-Seed LightGBM)",
            "routing":             "Champion",
            "champion_prediction": 1453772.0,
            "luxury_specialist_prediction": 1420500.0,
            "routing_decision":    "champion",
        }
    }}


# ── Output: Health Response ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response for GET /health."""

    status:            str            = Field(..., description="'ready' | 'artifact_found' | 'missing'")
    model_loaded:      bool           = Field(..., description="True if the model bundle is loaded in memory")
    variant_id:        Optional[str]  = Field(None, description="Active model variant identifier")
    architecture:      Optional[str]  = Field(None, description="Champion architecture description")
    created_at:        Optional[str]  = Field(None, description="Timestamp the model bundle was created")
    artifact_path:     str            = Field(..., description="Absolute path to ensemble_bundle.pkl")
    artifact_exists:   bool           = Field(..., description="True if the artifact file exists on disk")
    artifact_size_mb:  Optional[float]= Field(None, description="Bundle size in megabytes")
    metrics:           Optional[dict] = Field(None, description="Frozen champion benchmark metrics")

    model_config = {"json_schema_extra": {
        "example": {
            "status":            "ready",
            "model_loaded":      True,
            "variant_id":        "final",
            "architecture":      "5-Seed LightGBM + Luxury CatBoost Specialist + Strategy D Routing",
            "created_at":        "2026-08-22 05:25:27",
            "artifact_path":     "/app/model_registry/final/ensemble_bundle.pkl",
            "artifact_exists":   True,
            "artifact_size_mb":  119.9,
            "metrics": {
                "test_mae":     39969.55,
                "test_mape":    6.73,
                "test_rmse":    97821.45,
                "test_r2":      0.9675,
            }
        }
    }}


# ── Output: Error Response ────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error response."""

    error:   str            = Field(..., description="Error type / short description")
    detail:  str            = Field(..., description="Full error message")
    status:  int            = Field(..., description="HTTP status code")

    model_config = {"json_schema_extra": {
        "example": {
            "error":  "ValidationError",
            "detail": "selling_price is not allowed in prediction input.",
            "status": 422,
        }
    }}
