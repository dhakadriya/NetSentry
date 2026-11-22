"""ModelExperiment ORM model for tracking machine learning training runs and performance."""

import json
from typing import Any

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class ModelExperiment(Base, TimestampMixin):
    """Experiment record tracking ML model configuration, performance metrics, and artifact path."""

    __tablename__ = "model_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    algorithm = Column(String(64), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)

    # Feature Engineering Configuration
    feature_strategy = Column(String(64), default="none", nullable=False)
    selected_features_json = Column(Text, nullable=True)
    hyperparameters_json = Column(Text, nullable=True)

    # Core Evaluation Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True, index=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    confusion_matrix_json = Column(Text, nullable=True)

    # Operational Metrics
    training_time_seconds = Column(Float, nullable=True)
    prediction_time_ms = Column(Float, nullable=True)

    # Persistence
    artifact_path = Column(String(512), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_best_model = Column(Boolean, default=False, nullable=False, index=True)
    notes = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("DatasetRecord", back_populates="experiments")
    prediction_jobs = relationship("PredictionJob", back_populates="model_experiment")

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Deserializes hyperparameters_json into python dictionary."""
        if not self.hyperparameters_json:
            return {}
        try:
            return json.loads(self.hyperparameters_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def selected_features(self) -> list[str]:
        """Deserializes selected_features_json into list of feature names."""
        if not self.selected_features_json:
            return []
        try:
            return json.loads(self.selected_features_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def confusion_matrix(self) -> dict[str, int] | None:
        """Deserializes confusion_matrix_json into dictionary representation."""
        if not self.confusion_matrix_json:
            return None
        try:
            return json.loads(self.confusion_matrix_json)
        except (json.JSONDecodeError, TypeError):
            return None

    def to_dict(self) -> dict[str, Any]:
        """Serializes ModelExperiment to dictionary with formatted metrics."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "dataset_id": self.dataset_id,
            "feature_strategy": self.feature_strategy,
            "selected_features": self.selected_features,
            "hyperparameters": self.hyperparameters,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "false_positive_rate": round(self.false_positive_rate, 4) if self.false_positive_rate is not None else None,
            "false_negative_rate": round(self.false_negative_rate, 4) if self.false_negative_rate is not None else None,
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "training_time_seconds": round(self.training_time_seconds, 3) if self.training_time_seconds is not None else None,
            "prediction_time_ms": round(self.prediction_time_ms, 3) if self.prediction_time_ms is not None else None,
            "artifact_path": self.artifact_path,
            "is_active": self.is_active,
            "is_best_model": self.is_best_model,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        score = f"{self.f1_score:.4f}" if self.f1_score is not None else "None"
        return f"<ModelExperiment(id={self.id}, model='{self.model_name}', algo='{self.algorithm}', f1={score})>"
