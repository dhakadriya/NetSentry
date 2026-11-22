"""Pydantic schemas for machine learning training, evaluation, and model comparison."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.constants import FeatureSelectionStrategy, MLAlgorithm


class TrainModelRequest(BaseModel):
    """Payload to trigger model training job."""

    dataset_id: int = Field(..., description="ID of cataloged dataset to train on", ge=1)
    algorithm: MLAlgorithm = Field(default=MLAlgorithm.RANDOM_FOREST, description="ML algorithm to train")
    feature_strategy: FeatureSelectionStrategy = Field(
        default=FeatureSelectionStrategy.NONE, description="Feature selection strategy"
    )
    hyperparameters: dict[str, Any] | None = Field(default_factory=dict)
    test_size: float = Field(default=0.25, ge=0.1, le=0.5, description="Holdout test proportion")
    random_seed: int = Field(default=42, ge=0, description="Reproducibility seed")
    model_name: str | None = Field(None, min_length=2, max_length=128)


class ModelEvaluationMetrics(BaseModel):
    """Comprehensive performance metrics for an evaluated model."""

    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    false_positive_rate: float = Field(..., ge=0.0, le=1.0)
    false_negative_rate: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float | None = Field(None, ge=0.0, le=1.0)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    training_time_seconds: float = Field(..., ge=0.0)
    prediction_time_ms: float = Field(..., ge=0.0)
    selected_features_count: int = Field(default=0, ge=0)


class ModelExperimentSummary(BaseModel):
    """Detailed summary of a completed model experiment run."""

    id: int = Field(..., ge=1)
    model_name: str
    algorithm: str
    dataset_id: int | None = None
    feature_strategy: str
    metrics: ModelEvaluationMetrics
    artifact_path: str | None = None
    is_best_model: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class CompareModelsRequest(BaseModel):
    """Request payload to train and benchmark all 4 ML models simultaneously."""

    dataset_id: int = Field(..., ge=1)
    algorithms: list[MLAlgorithm] = Field(
        default_factory=lambda: [
            MLAlgorithm.LOGISTIC_REGRESSION,
            MLAlgorithm.DECISION_TREE,
            MLAlgorithm.RANDOM_FOREST,
            MLAlgorithm.KNN,
        ]
    )
    feature_strategy: FeatureSelectionStrategy = Field(default=FeatureSelectionStrategy.NONE)
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)


class ModelComparisonResult(BaseModel):
    """Leaderboard comparison result across all 4 machine learning models."""

    dataset_id: int
    models: list[ModelExperimentSummary]
    best_model_id: int
    best_model_name: str
    best_f1_score: float = Field(..., ge=0.0, le=1.0)
    lowest_fpr: float = Field(..., ge=0.0, le=1.0)
    fastest_inference_model: str
    fastest_inference_ms: float = Field(..., ge=0.0)
    comparison_timestamp: datetime


CompareModelsResponse = ModelComparisonResult


class LeaderboardResponse(BaseModel):
    """Ranked leaderboard of active model experiments."""

    total_models: int = Field(..., ge=0)
    top_models: list[ModelExperimentSummary] = Field(default_factory=list)
