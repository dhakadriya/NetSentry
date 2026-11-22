"""Abstract base class for all NetSentry intrusion detection machine learning models."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.core.constants import MLAlgorithm
from src.core.exceptions import ModelNotTrainedError, ModelTrainingError


class BaseNetSentryModel(ABC):
    """Abstract base class encapsulating model lifecycle, training timer, and joblib serialization."""

    def __init__(self, model_name: str, algorithm: MLAlgorithm, **hyperparameters):
        self.model_name = model_name
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.model_: Any = None
        self.feature_names_: list[str] = []
        self.training_time_seconds: float = 0.0
        self.prediction_time_ms: float = 0.0
        self.metrics_: dict[str, Any] = {}
        self.model_version: str = "1.0.0"

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the model has undergone successful training."""
        return self.model_ is not None and len(self.feature_names_) > 0

    @property
    def n_features_in_(self) -> int:
        """Returns the number of input features expected by the model."""
        return len(self.feature_names_)

    @abstractmethod
    def _create_estimator(self) -> Any:
        """Instantiates the underlying scikit-learn estimator."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseNetSentryModel":
        """Fits the underlying algorithm on features X and binary target labels y."""
        self.feature_names_ = list(X.columns)
        self.model_ = self._create_estimator()

        start = time.perf_counter()
        try:
            self.model_.fit(X, y)
        except Exception as e:
            raise ModelTrainingError(f"Failed to fit {self.model_name}: {str(e)}") from e
        self.training_time_seconds = round(time.perf_counter() - start, 4)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary intrusion class (0: Normal, 1: Attack)."""
        if not self.is_fitted:
            raise ModelNotTrainedError(f"Model {self.model_name} has not been trained.")

        # Ensure correct column ordering
        X_aligned = self._align_features(X)

        start = time.perf_counter()
        preds = self.model_.predict(X_aligned)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.prediction_time_ms = round(elapsed_ms / max(len(X), 1), 4)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probability scores for intrusion classification."""
        if not self.is_fitted:
            raise ModelNotTrainedError(f"Model {self.model_name} has not been trained.")

        X_aligned = self._align_features(X)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X_aligned)
        elif hasattr(self.model_, "decision_function"):
            df_vals = self.model_.decision_function(X_aligned)
            probs_1 = 1.0 / (1.0 + np.exp(-df_vals))
            return np.column_stack([1.0 - probs_1, probs_1])
        else:
            # Fallback for models without probabilistic estimates
            preds = self.predict(X_aligned)
            return np.column_stack([1.0 - preds, preds.astype(float)])

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures input DataFrame contains exact feature columns used during training."""
        df = X.copy()
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_names_]

    def save(self, output_path: str | Path) -> Path:
        """Serializes trained model artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "BaseNetSentryModel":
        """Loads serialized model instance from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        instance = joblib.load(path)
        return instance
