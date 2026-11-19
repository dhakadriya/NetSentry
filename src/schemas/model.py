"""Pydantic schemas for machine learning training, evaluation, and model comparison."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.constants import FeatureSelectionStrategy, MLAlgorithm


class TrainModelRequest(BaseModel):
    """Payload to trigger model training job."""

    dataset_id: int = Field(description="ID of cataloged dataset to train on")
    algorithm: MLAlgorithm = Field(default=MLAlgorithm.RANDOM_FOREST)
    feature_strategy: FeatureSelectionStrategy = Field(default=FeatureSelectionStrategy.NONE)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    random_seed: int = Field(default=42)
    model_name: Optional[str] = None


class ModelEvaluationMetrics(BaseModel):
    """Comprehensive performance metrics for an evaluated model."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: Optional[float] = None
    confusion_matrix: List[List[int]]
    training_time_seconds: float
    prediction_time_ms: float
    selected_features_count: int


class ModelExperimentSummary(BaseModel):
    """Detailed summary of a completed model experiment run."""

    id: int
    model_name: str
    algorithm: str
    dataset_id: Optional[int]
    feature_strategy: str
    metrics: ModelEvaluationMetrics
    artifact_path: Optional[str]
    is_best_model: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CompareModelsRequest(BaseModel):
    """Request payload to train and benchmark all 4 ML models simultaneously."""

    dataset_id: int
    algorithms: List[MLAlgorithm] = Field(
        default_factory=lambda: [
            MLAlgorithm.LOGISTIC_REGRESSION,
            MLAlgorithm.DECISION_TREE,
            MLAlgorithm.RANDOM_FOREST,
            MLAlgorithm.KNN,
        ]
    )
    feature_strategy: FeatureSelectionStrategy = Field(default=FeatureSelectionStrategy.NONE)
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    random_seed: int = Field(default=42)


class CompareModelsResponse(BaseModel):
    """Comparative benchmarking results across tested ML models."""

    dataset_id: int
    total_models_evaluated: int
    best_algorithm: str
    best_model_id: int
    best_f1_score: float
    lowest_fpr: float
    leaderboard: List[Dict[str, Any]]
    feature_strategy: str
    feature_count: int
