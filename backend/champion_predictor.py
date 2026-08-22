"""Champion Predictor — Production inference wrapper around final/ensemble_bundle.pkl.

This module is the single source of truth for inference using the 5-Seed LightGBM
champion + Luxury CatBoost Specialist + Strategy D brand-aware routing.

Architecture:
  - 5-Seed LightGBM Champion (Seeds: 42, 123, 456, 789, 2024)
  - Luxury CatBoost Specialist (depth=8, lr=0.04, l2=3.0)
  - Strategy D Routing:
      IF (normalized_brand in luxury_brands AND champion_prediction >= Rs16,00,000)
         OR champion_prediction >= Rs22,00,000:
          final_prediction = luxury_specialist_prediction
      ELSE:
          final_prediction = champion_prediction

Production Test Metrics (untouched 3,748 cars):
  MAE       = Rs39,969.55
  MAPE      = 6.73%
  RMSE      = Rs97,821.45
  R2        = 0.9675
  Median AE = Rs20,880.69
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("priceref.champion")

# ── Constants (frozen) ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BUNDLE = _ROOT / "model_registry" / "final" / "ensemble_bundle.pkl"

# Trusted SHA-256 hash for the production final bundle
_FROZEN_FINAL_SHA256 = "5c3a2ccee8efb8d842b0bac7a6380c49d2491a39b403e10dac5a279ed4bf9f3b"

_RAW_CAT_FEATURES = [
    "brand", "model", "variant", "locality", "rto",
    "fuel_type", "transmission", "seller_type", "color",
]
_RAW_NUM_FEATURES = [
    "vehicle_age", "odometer_reading", "km_per_year",
    "owner_count", "certified", "pincode",
]
_ENGINEERED_CATS = ["brand_model", "model_variant"]
_ALL_CAT_FEATURES = _RAW_CAT_FEATURES + _ENGINEERED_CATS
_ALL_FEATURES = _ALL_CAT_FEATURES + _RAW_NUM_FEATURES

# Forbidden inference fields — selling_price must NEVER be in the input
_FORBIDDEN_FIELDS = {"selling_price", "target", "price", "sale_price", "label", "actual_price", "ground_truth"}

# INR bounds for output sanity check
_MIN_PRICE_INR = 10_000.0
_MAX_PRICE_INR = 50_000_000.0

# ── Module-level singleton cache ───────────────────────────────────────────────
_PIPELINE_CACHE: dict[str, "ChampionPredictor"] = {}


def _clean_string_series(s: pd.Series) -> pd.Series:
    """Exact replica of training-time string canonicalization."""
    return (
        s.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unknown")
    )


def _clean_string(val: object) -> str:
    """Clean a single string value."""
    if val is None:
        return "unknown"
    s = str(val).strip().lower()
    return s if s else "unknown"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_expected_sha256(bundle_path: Path) -> str | None:
    """Find the expected SHA-256 for a given bundle from env, metadata, or registry."""
    # 1. Environment variable override
    env_hash = os.environ.get("EXPECTED_BUNDLE_SHA256")
    if env_hash and env_hash.strip():
        return env_hash.strip().lower()

    # 2. Metadata file adjacent to bundle
    meta_path = bundle_path.parent / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            h = meta.get("bundle_sha256") or meta.get("sha256")
            if h:
                return str(h).strip().lower()
        except Exception:
            pass

    # 3. Registry file in parent directory
    reg_path = bundle_path.parent.parent / "registry.json"
    if reg_path.exists():
        try:
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)
            variant_name = bundle_path.parent.name
            h = reg.get("variants", {}).get(variant_name, {}).get("sha256")
            if h:
                return str(h).strip().lower()
        except Exception:
            pass

    # 4. Fallback to frozen constant if final production bundle
    if bundle_path.resolve() == _DEFAULT_BUNDLE.resolve():
        return _FROZEN_FINAL_SHA256

    return None


def verify_bundle_integrity(bundle_path: Path) -> None:
    """Verify SHA-256 hash of the bundle file BEFORE loading with pickle.
    
    Fails closed if hash mismatch or missing expected hash.
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle file not found: {bundle_path.name}")

    actual_hash = compute_sha256(bundle_path)
    expected_hash = _resolve_expected_sha256(bundle_path)

    if not expected_hash:
        log.error("Integrity check failed: no expected SHA-256 configured for %s", bundle_path.name)
        raise ValueError("Model bundle integrity verification failed: missing expected hash.")

    if not secrets.compare_digest(actual_hash.lower(), expected_hash.lower()):
        log.error(
            "Integrity check failed for %s: expected %s, got %s",
            bundle_path.name, expected_hash, actual_hash,
        )
        raise ValueError("Model bundle integrity check failed: artifact is corrupted or modified.")


class ChampionPredictor:
    """Loads and serves the 5-seed LightGBM + Luxury CatBoost Strategy D ensemble."""

    def __init__(self, bundle: dict) -> None:
        self._lgb_models        = bundle["lgb_models"]           # list of 5 Booster
        self._luxury_specialist = bundle["luxury_specialist"]    # CatBoostRegressor
        self._cat_levels        = bundle["cat_levels"]           # {col: [str, ...]}
        self._encoders          = bundle["encoders"]             # {col: LabelEncoder}
        self._medians           = bundle["medians"]              # {col: float}
        self._routing_config    = bundle.get("routing", {})
        self._metadata          = bundle.get("metadata", {})

        # Routing rules
        self._luxury_brands = frozenset(
            self._routing_config.get("luxury_brands", [
                "audi", "bmw", "jaguar", "land rover", "lexus",
                "mercedes-benz", "mini", "porsche", "volvo",
            ])
        )
        self._luxury_brand_threshold = float(self._routing_config.get("luxury_brand_threshold", 1_600_000.0))
        self._global_threshold       = float(self._routing_config.get("global_threshold", 2_200_000.0))

        # Metadata
        self.variant_id        = self._metadata.get("variant_id", "final")
        self.architecture      = self._metadata.get("architecture", "5-Seed LightGBM + Luxury CatBoost Specialist + Strategy D Routing")
        self.champion_metrics  = self._metadata.get("metrics", {})
        self.created_at        = self._metadata.get("training_timestamp", "unknown")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _prepare_inputs(
        self, df_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess a raw input DataFrame into LightGBM and CatBoost frames."""
        df = df_raw.copy()

        # Step 1: Clean raw categoricals
        for col in _RAW_CAT_FEATURES:
            if col in df.columns:
                df[col] = _clean_string_series(df[col])
            else:
                df[col] = "unknown"

        # Step 2: Numerical coercion + median imputation
        for col in _RAW_NUM_FEATURES:
            med = self._medians.get(col, 0.0)
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)
            else:
                df[col] = float(med)

        # Step 3: Engineer interaction features
        df["brand_model"]   = df["brand"] + "__" + df["model"]
        df["model_variant"] = df["model"] + "__" + df["variant"]

        # Step 4: Build LightGBM frame (integer-encoded categoricals)
        X_lgb = pd.DataFrame(index=df.index)
        for col in _ALL_CAT_FEATURES:
            known_set = set(self._cat_levels.get(col, ["unknown"]))
            s = df[col].apply(lambda x: x if x in known_set else "unknown")
            X_lgb[col] = self._encoders[col].transform(s)
        for col in _RAW_NUM_FEATURES:
            X_lgb[col] = df[col].astype(float)

        # Step 5: Build CatBoost frame (native string categoricals)
        X_cb = pd.DataFrame(index=df.index)
        for col in _ALL_CAT_FEATURES:
            X_cb[col] = df[col].astype(str) if col in df.columns else "unknown"
        for col in _RAW_NUM_FEATURES:
            X_cb[col] = df[col].astype(float)

        return X_lgb[_ALL_FEATURES], X_cb[_ALL_FEATURES]

    # ── Prediction ────────────────────────────────────────────────────

    def _predict_batch(self, df_raw: pd.DataFrame) -> dict:
        """Run 5-Seed LightGBM + Luxury CatBoost Strategy D routing."""
        X_lgb, X_cb = self._prepare_inputs(df_raw)
        n = len(df_raw)

        # 5-Seed LightGBM champion prediction in log space
        seed_preds_log = [model.predict(X_lgb) for model in self._lgb_models]
        avg_pred_log   = np.mean(seed_preds_log, axis=0)
        champion_preds = np.expm1(avg_pred_log)

        # Brand extraction
        brands = (
            df_raw["brand"].fillna("unknown").astype(str).str.strip().str.lower().tolist()
            if "brand" in df_raw.columns
            else ["unknown"] * n
        )

        # Strategy D Routing
        final_preds     = champion_preds.copy()
        specialist_preds= np.zeros(n, dtype=float)
        routing_decision= []
        routed_indices  = []

        for i in range(n):
            b  = brands[i]
            cp = float(champion_preds[i])
            if (b in self._luxury_brands and cp >= self._luxury_brand_threshold) or (cp >= self._global_threshold):
                routing_decision.append("specialist")
                routed_indices.append(i)
            else:
                routing_decision.append("champion")

        # Evaluate specialist for routed items
        if routed_indices:
            cb_sub = X_cb.iloc[routed_indices]
            cb_preds_sub = np.expm1(self._luxury_specialist.predict(cb_sub))
            for idx_pos, orig_idx in enumerate(routed_indices):
                sp_val = float(cb_preds_sub[idx_pos])
                specialist_preds[orig_idx] = sp_val
                final_preds[orig_idx]      = sp_val

        # Fill specialist_preds for champion items if requested for introspection
        for i in range(n):
            if routing_decision[i] == "champion":
                specialist_preds[i] = champion_preds[i]

        return {
            "predicted_price":              final_preds,
            "champion_prediction":          champion_preds,
            "luxury_specialist_prediction": specialist_preds,
            "routing_decision":             routing_decision,
            "seed_preds_log":               seed_preds_log,
            "avg_pred_log":                 avg_pred_log,
            "segment_probability":          np.array([1.0 if r == "specialist" else 0.0 for r in routing_decision]),
        }

    def predict_price(self, record: dict) -> dict:
        """Predict the selling price for a single vehicle record."""
        # Safety: reject forbidden target fields
        bad = set(str(k).lower() for k in record) & _FORBIDDEN_FIELDS
        if bad:
            raise ValueError(
                f"Input contains forbidden field(s) {bad}. "
                "selling_price must NOT be provided during inference."
            )

        df_in = pd.DataFrame([record])
        res   = self._predict_batch(df_in)

        predicted = float(res["predicted_price"][0])
        champ_val = float(res["champion_prediction"][0])
        spec_val  = float(res["luxury_specialist_prediction"][0])
        decision  = str(res["routing_decision"][0])
        seg_prob  = float(res["segment_probability"][0])

        # Output sanity checks
        if not (np.isfinite(predicted) and _MIN_PRICE_INR <= predicted <= _MAX_PRICE_INR):
            raise ValueError(
                f"Predicted price Rs{predicted:,.0f} is outside valid INR bounds "
                f"[Rs{_MIN_PRICE_INR:,.0f}, Rs{_MAX_PRICE_INR:,.0f}]."
            )

        gate_label = f"Strategy D: {decision.capitalize()}"
        routing_label = "Specialist" if decision == "specialist" else "Champion"

        return {
            "predicted_price":              round(predicted, 2),
            "segment_probability":          round(seg_prob, 6),
            "lgbm_prediction":              round(champ_val, 2),
            "catboost_prediction":          round(spec_val, 2),
            "champion_prediction":          round(champ_val, 2),
            "luxury_specialist_prediction": round(spec_val, 2),
            "routing_decision":             decision,
            "final_gate":                   gate_label,
            "routing":                      routing_label,
        }

    def predict_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict prices for a batch DataFrame."""
        res = self._predict_batch(df)
        n   = len(df)
        gates   = [f"Strategy D: {d.capitalize()}" for d in res["routing_decision"]]
        routing = ["Specialist" if d == "specialist" else "Champion" for d in res["routing_decision"]]

        return pd.DataFrame({
            "predicted_price":              res["predicted_price"].round(2),
            "segment_probability":          res["segment_probability"].round(6),
            "lgbm_prediction":              res["champion_prediction"].round(2),
            "catboost_prediction":          res["luxury_specialist_prediction"].round(2),
            "champion_prediction":          res["champion_prediction"].round(2),
            "luxury_specialist_prediction": res["luxury_specialist_prediction"].round(2),
            "routing_decision":             res["routing_decision"],
            "final_gate":                   gates,
            "routing":                      routing,
        })

    def predict_log_price(self, df: pd.DataFrame) -> float:
        """Predict log1p price for legacy ensemble compatibility."""
        res = self._predict_batch(df)
        pred_inr = float(res["predicted_price"][0])
        return float(np.log1p(pred_inr))

    def predict_with_variance(self, df: pd.DataFrame) -> dict:
        """Predict log1p price and ensemble variance for legacy consumers."""
        res = self._predict_batch(df)
        pred_inr = float(res["predicted_price"][0])
        log_price = float(np.log1p(pred_inr))
        seed_preds = [float(p[0]) for p in res["seed_preds_log"]]
        variance = float(np.var(seed_preds)) if len(seed_preds) > 1 else 0.0
        return {
            "log_price": log_price,
            "variance":  round(variance, 6),
        }

    @property
    def metadata(self) -> dict:
        """Return bundle metadata dict."""
        return dict(self._metadata)


# ── Module-level factory & singleton access ───────────────────────────────────

def load_champion(
    bundle_path: Path | str | None = None,
    *,
    verify_integrity: bool = True,
) -> ChampionPredictor:
    """Load the production champion predictor from the bundle with SHA-256 verification.
    
    Verifies cryptographic hash BEFORE pickle.load() to prevent deserialization attacks.
    Cached in memory after first successful verification and load.
    """
    path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
    path_str = str(path.resolve())

    if path_str not in _PIPELINE_CACHE:
        if not path.exists():
            log.error("Production bundle not found: %s", path.name)
            raise FileNotFoundError(f"Production bundle not found: {path.name}")

        # Step 1: Cryptographic integrity check BEFORE deserialization
        if verify_integrity:
            verify_bundle_integrity(path)

        # Step 2: Deserialization only after integrity is verified
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        predictor = ChampionPredictor(bundle)
        _PIPELINE_CACHE[path_str] = predictor

    return _PIPELINE_CACHE[path_str]


def clear_champion_cache() -> None:
    """Clear memory cache of champion predictors (useful for testing)."""
    _PIPELINE_CACHE.clear()


def predict_price(vehicle_record: dict, bundle_path: Path | str | None = None) -> dict:
    """Convenience module-level function. Loads champion on first call."""
    predictor = load_champion(bundle_path)
    return predictor.predict_price(vehicle_record)


def get_health_info(bundle_path: Path | str | None = None) -> dict:
    """Return health/readiness information about the production model.
    
    Does NOT leak internal absolute filesystem paths to the client.
    """
    path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
    artifact_exists = path.exists()
    artifact_size_mb = round(path.stat().st_size / 1e6, 1) if artifact_exists else None

    loaded = str(path.resolve()) in _PIPELINE_CACHE
    predictor_info: dict = {}
    health_status = "missing"

    if artifact_exists:
        try:
            pred = load_champion(bundle_path)
            predictor_info = {
                "variant_id":   pred.variant_id,
                "architecture": pred.architecture,
                "created_at":   pred.created_at,
                "metrics":      pred.champion_metrics,
            }
            loaded = True
            health_status = "ready"
        except ValueError as exc:
            log.error("Integrity error during health check: %s", exc)
            health_status = "integrity_error"
            loaded = False
        except Exception as exc:
            log.error("Error loading model during health check: %s", exc)
            health_status = "load_error"
            loaded = False

    return {
        "status":          health_status,
        "model_loaded":    loaded,
        "artifact_path":   path.name,  # filename only, no internal filesystem paths
        "artifact_exists": artifact_exists,
        "artifact_size_mb": artifact_size_mb,
        **predictor_info,
    }
