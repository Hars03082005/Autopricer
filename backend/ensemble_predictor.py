from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


class EnsemblePredictor:
    """Load CatBoost / LightGBM / XGBoost and blend predictions."""

    def __init__(self, artifact_dir: Path, metadata: dict) -> None:
        self.artifact_dir = artifact_dir
        self.metadata = metadata
        self.features = metadata["features"]
        self.cat_features = metadata["categorical_features"]
        ensemble = metadata.get("ensemble", {})
        self.enabled = bool(ensemble.get("enabled", False))
        self.weights = ensemble.get("weights", {"catboost": 1.0})
        self.category_levels: Dict[str, list[str]] = ensemble.get("category_levels", {})

        catboost_path = artifact_dir / "vehicle_price_catboost.cbm"
        self.catboost = CatBoostRegressor()
        self.catboost.load_model(str(catboost_path))

        self.lightgbm = None
        self.xgboost = None
        if self.enabled:
            lgb_path = artifact_dir / "vehicle_price_lightgbm.txt"
            xgb_path = artifact_dir / "vehicle_price_xgboost.json"
            if lgb is None or xgb is None:
                raise ImportError("Ensemble requires lightgbm and xgboost to be installed.")
            if not lgb_path.exists() or not xgb_path.exists():
                raise FileNotFoundError("Ensemble model artifacts are missing.")
            self.lightgbm = lgb.Booster(model_file=str(lgb_path))
            self.xgboost = xgb.XGBRegressor()
            self.xgboost.load_model(str(xgb_path))

    def _prepare_frame(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        frame = features[self.features].copy()
        catboost_frame = frame.copy()
        lgb_frame = frame.copy()
        xgb_frame = frame.copy()

        for col in self.cat_features:
            values = frame[col].astype(str)
            if self.category_levels.get(col):
                values = values.where(values.isin(self.category_levels[col]), "unknown")
                catboost_frame[col] = values.astype(str)
                lgb_frame[col] = pd.Categorical(values, categories=self.category_levels[col])
                mapping = {category: idx for idx, category in enumerate(self.category_levels[col])}
                xgb_frame[col] = values.map(mapping).astype(int)
            else:
                catboost_frame[col] = values.astype(str)
                lgb_frame[col] = values.astype("category")

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


    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> "EnsemblePredictor":
        metadata_path = artifact_dir / "model_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(artifact_dir, metadata)
