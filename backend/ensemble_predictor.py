from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class EnsemblePredictor:
    """Load CatBoost / LightGBM / XGBoost and blend predictions."""

    def __init__(self, artifact_dir: Path, metadata: dict) -> None:
        self.artifact_dir = artifact_dir
        self.metadata = metadata
        self.features = metadata.get("features", [])
        self.cat_features = metadata.get("categorical_features", metadata.get("cat_features", []))
        ensemble = metadata.get("ensemble", {})
        self.enabled = bool(ensemble.get("enabled", True))
        self.weights = ensemble.get("weights", metadata.get("ensemble_weights", {"catboost": 1.0}))
        self.category_levels: dict[str, list[str]] = ensemble.get("category_levels", {})

        catboost_path = artifact_dir / "vehicle_price_catboost.cbm"
        self.catboost = CatBoostRegressor()
        self.catboost.load_model(str(catboost_path))

        self.lightgbm = None
        self.xgboost = None
        if self.enabled:
            lgb_path = artifact_dir / "vehicle_price_lightgbm.txt"
            xgb_path = artifact_dir / "vehicle_price_xgboost.json"
            if self.weights.get("lightgbm", 0.0) > 0 and lgb_path.exists():
                if lgb is None:
                    raise ImportError("Ensemble requires lightgbm to be installed.")
                self.lightgbm = lgb.Booster(model_file=str(lgb_path))
            if self.weights.get("xgboost", 0.0) > 0 and xgb_path.exists():
                if xgb is None:
                    raise ImportError("Ensemble requires xgboost to be installed.")
                self.xgboost = xgb.XGBRegressor()
                self.xgboost.load_model(str(xgb_path))

    def _prepare_frame(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        catboost_cols = self.catboost.feature_names_
        frame = features.copy()

        catboost_frame = frame.copy()
        lgb_frame = frame.copy()
        xgb_frame = frame.copy()

        has_lgb_cats = bool(getattr(self.lightgbm, "pandas_categorical", None)) if self.lightgbm is not None else False

        for col in self.cat_features:
            if col not in frame.columns:
                catboost_frame[col] = "unknown"
                lgb_frame[col] = pd.Categorical(["unknown"], categories=self.category_levels.get(col, ["unknown"])) if has_lgb_cats else 0
                xgb_frame[col] = 0
                continue
            values = frame[col].astype(str)
            if self.category_levels.get(col):
                values = values.where(values.isin(self.category_levels[col]), "unknown")
                catboost_frame[col] = values.astype(str)
                mapping = {category: idx for idx, category in enumerate(self.category_levels[col])}
                encoded = values.map(mapping).fillna(0).astype(int)
                if has_lgb_cats:
                    lgb_frame[col] = pd.Categorical(values, categories=self.category_levels[col])
                else:
                    lgb_frame[col] = encoded
                xgb_frame[col] = encoded
            else:
                catboost_frame[col] = values.astype(str)
                if has_lgb_cats:
                    lgb_frame[col] = values.astype("category")
                else:
                    lgb_frame[col] = 0
                xgb_frame[col] = 0

        for col in catboost_cols:
            if col not in catboost_frame.columns:
                catboost_frame[col] = "unknown" if col in self.cat_features else 0.0

        catboost_frame = catboost_frame[catboost_cols]

        if self.lightgbm is not None:
            lgb_cols = self.lightgbm.feature_name()
            for col in lgb_cols:
                if col not in lgb_frame.columns:
                    lgb_frame[col] = 0.0
            lgb_frame = lgb_frame[lgb_cols]

        if self.xgboost is not None:
            xgb_cols = list(self.xgboost.feature_names_in_)
            for col in xgb_cols:
                if col not in xgb_frame.columns:
                    xgb_frame[col] = 0.0
            xgb_frame = xgb_frame[xgb_cols]

        return catboost_frame, lgb_frame, xgb_frame


    def predict_log_price(self, features: pd.DataFrame) -> float:
        catboost_frame, lgb_frame, xgb_frame = self._prepare_frame(features)
        catboost_pred = float(self.catboost.predict(catboost_frame)[0])

        if not self.enabled:
            return catboost_pred

        weights = self.weights
        blended = weights.get("catboost", 0.0) * catboost_pred

        if self.lightgbm is not None and weights.get("lightgbm", 0.0) > 0:
            blended += weights["lightgbm"] * float(self.lightgbm.predict(lgb_frame)[0])

        if self.xgboost is not None and weights.get("xgboost", 0.0) > 0:
            blended += weights["xgboost"] * float(self.xgboost.predict(xgb_frame)[0])

        return float(blended)

    def predict_with_variance(self, features: pd.DataFrame) -> dict:
        """Returns model log-price predictions, blended result, and ensemble variance."""
        catboost_frame, lgb_frame, xgb_frame = self._prepare_frame(features)
        cb_log  = float(self.catboost.predict(catboost_frame)[0])
        lgb_log = None
        xgb_log = None

        if self.lightgbm is not None:
            lgb_log = float(self.lightgbm.predict(lgb_frame)[0])
        if self.xgboost is not None:
            try:
                xgb_log = float(self.xgboost.predict(xgb_frame)[0])
            except Exception:
                xgb_log = None

        if not self.enabled:
            blended = cb_log
        else:
            weights = self.weights
            blended = weights.get("catboost", 0.0) * cb_log
            if lgb_log is not None and weights.get("lightgbm", 0.0) > 0:
                blended += weights["lightgbm"] * lgb_log
            if xgb_log is not None and weights.get("xgboost", 0.0) > 0:
                blended += weights["xgboost"] * xgb_log

        preds = [p for p in [cb_log, lgb_log, xgb_log] if p is not None]
        variance = float(np.std(preds)) if len(preds) > 1 else 0.0

        return {
            "log_price":     float(blended),
            "catboost_log":  cb_log,
            "lightgbm_log":  lgb_log,
            "xgboost_log":   xgb_log,
            "variance":      round(variance, 6),
        }


    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> EnsemblePredictor:
        metadata_path = artifact_dir / "model_metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(artifact_dir, metadata)
