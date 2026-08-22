"""Champion Price Prediction API — app.py

Minimal, self-contained FastAPI application exposing:
  GET  /health  → Model readiness check (with SHA-256 integrity verification)
  POST /predict → Vehicle price prediction

This app is a pure inference wrapper. It has no auth, no DB, and no
state beyond the loaded model.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.champion_predictor import get_health_info, load_champion
from backend.config import get_settings
from backend.schemas import ErrorResponse, HealthResponse, PredictionResponse, VehicleRecord

log = logging.getLogger("champion_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ── Lifespan: eager model load at startup with integrity check ───────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Champion Predictor from frozen bundle with integrity check …")
    try:
        predictor = load_champion()
        log.info(
            "Champion loaded: %s | Architecture: %s | Created: %s",
            predictor.variant_id, predictor.architecture, predictor.created_at,
        )
        metrics = predictor.champion_metrics
        if metrics:
            log.info(
                "Frozen benchmark — MAE: ₹%.2f | MAPE: %.2f%% | RMSE: ₹%.2f | R²: %.4f",
                metrics.get("test_mae", 0),
                metrics.get("test_mape", 0),
                metrics.get("test_rmse", 0),
                metrics.get("test_r2", 0),
            )
    except FileNotFoundError as exc:
        log.error("Champion bundle not found: %s", exc)
    except ValueError as exc:
        log.error("Champion bundle integrity check failed: %s", exc)
    yield
    log.info("Champion API shutting down.")


# ── FastAPI Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Champion Price Prediction API",
    description=(
        "Production inference API for the Fine-Grained Gated Ensemble champion model. "
        "Wraps the frozen ensemble_bundle.pkl artifact. "
        "POST /predict with 15 vehicle features, get back a structured price prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _settings.cors_allow_all else list(_settings.cors_allowed_origins),
    allow_credentials=not _settings.cors_allow_all,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    # Log internal error details server-side
    log.warning("ValueError on %s: %s", request.url.path, exc)
    # Strip any internal filesystem paths or module names from client response
    safe_detail = str(exc)
    if "C:\\" in safe_detail or "/app/" in safe_detail:
        safe_detail = "Input validation failed or artifact integrity error."
    return JSONResponse(
        status_code=400,
        content={"error": "ValueError", "detail": safe_detail, "status": 400},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An internal error occurred. Please try again later.",
            "status": 500,
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    description="Returns the champion model's readiness status, variant ID, artifact info, and frozen benchmark metrics.",
    tags=["System"],
)
def health() -> HealthResponse:
    info = get_health_info()
    return HealthResponse(**info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict vehicle selling price",
    description=(
        "Accepts 15 raw vehicle features and returns the champion model's price prediction. "
        "selling_price must NOT be included in the request body. "
        "Missing categorical fields default to 'unknown'. "
        "Missing numerical fields use training-time medians."
    ),
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction", "model": PredictionResponse},
        400: {"description": "Business rule violation or input error", "model": ErrorResponse},
        422: {"description": "Validation error (e.g. oversized string, selling_price present)", "model": ErrorResponse},
        500: {"description": "Internal inference error", "model": ErrorResponse},
        503: {"description": "Model bundle unavailable or failed integrity check", "model": ErrorResponse},
    },
)
def predict(vehicle: VehicleRecord) -> PredictionResponse:
    """Run the frozen champion pipeline on a single vehicle record."""
    try:
        predictor = load_champion()
    except FileNotFoundError as exc:
        log.error("Model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Model service temporarily unavailable.") from exc
    except ValueError as exc:
        log.error("Model integrity error: %s", exc)
        raise HTTPException(status_code=503, detail="Model service temporarily unavailable.") from exc

    record_dict = vehicle.to_inference_dict()

    try:
        result = predictor.predict_price(record_dict)
    except ValueError as exc:
        log.warning("Validation rejected during prediction: %s", exc)
        safe_msg = str(exc)
        if "C:\\" in safe_msg or "/app/" in safe_msg:
            safe_msg = "Invalid vehicle record provided."
        raise HTTPException(status_code=400, detail=safe_msg) from exc
    except Exception as exc:
        log.error("Inference error for record: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction request could not be processed.") from exc

    log.info(
        "Predicted ₹%.0f | Segment P=%.3f | Gate=%s",
        result["predicted_price"],
        result["segment_probability"],
        result["final_gate"],
    )
    return PredictionResponse(**result)


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8001, reload=True)
