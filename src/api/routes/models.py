"""Model training, experiment tracking, and model lifecycle endpoints."""

import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
import pandas as pd
from sqlalchemy.orm import Session

from src.api.deps import get_db_session, get_trainer
from src.database.crud.dataset_repo import get_dataset_by_id
from src.database.crud.experiment_repo import (
    create_experiment,
    get_best_model,
    get_experiment_by_id,
    list_experiments,
)
from src.ml.trainer import NetSentryTrainer
from src.schemas.model import ModelEvaluationMetrics, ModelExperimentSummary, TrainModelRequest

router = APIRouter(prefix="/api/v1/models", tags=["Model Training"])


@router.post("/train", response_model=ModelExperimentSummary, status_code=status.HTTP_201_CREATED)
def train_model(
    request: TrainModelRequest,
    db: Session = Depends(get_db_session),
    trainer: NetSentryTrainer = Depends(get_trainer),
):
    """Trains a machine learning intrusion classifier on a selected dataset."""
    dataset_record = get_dataset_by_id(db, request.dataset_id)
    if not dataset_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} does not exist.",
        )

    file_path = Path(dataset_record.stored_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file missing from disk: {file_path}",
        )

    df = pd.read_csv(file_path, nrows=50000)

    # Train model
    model, preprocessor, selector, metrics, artifact_path = trainer.train(
        df=df,
        algorithm=request.algorithm,
        feature_strategy=request.feature_strategy,
        test_size=request.test_size,
        hyperparameters=request.hyperparameters,
        random_seed=request.random_seed,
        model_name=request.model_name,
    )

    # Check if this model beats existing best model (higher F1 and lower FPR)
    current_best = get_best_model(db)
    is_new_best = False
    if current_best is None:
        is_new_best = True
    elif metrics.f1_score > (current_best.f1_score or 0.0):
        is_new_best = True

    # Persist experiment
    experiment = create_experiment(
        db=db,
        model_name=model.model_name,
        algorithm=request.algorithm.value,
        dataset_id=dataset_record.id,
        feature_strategy=request.feature_strategy.value,
        hyperparameters=request.hyperparameters,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        false_positive_rate=metrics.false_positive_rate,
        false_negative_rate=metrics.false_negative_rate,
        roc_auc=metrics.roc_auc,
        confusion_matrix=metrics.confusion_matrix,
        training_time_seconds=metrics.training_time_seconds,
        prediction_time_ms=metrics.prediction_time_ms,
        artifact_path=str(artifact_path),
        selected_features=model.feature_names_,
        is_best_model=is_new_best,
        notes=f"Trained on {dataset_record.filename} with {request.feature_strategy.value} feature selection.",
    )

    return ModelExperimentSummary(
        id=experiment.id,
        model_name=experiment.model_name,
        algorithm=experiment.algorithm,
        dataset_id=experiment.dataset_id,
        feature_strategy=experiment.feature_strategy,
        metrics=metrics,
        artifact_path=experiment.artifact_path,
        is_best_model=experiment.is_best_model,
        created_at=experiment.created_at,
    )


@router.get("/experiments", response_model=List[ModelExperimentSummary])
def get_experiments(
    limit: int = 50,
    algorithm: Optional[str] = None,
    db: Session = Depends(get_db_session),
):
    """Lists past training experiments and performance benchmarks."""
    experiments = list_experiments(db, limit=limit, algorithm=algorithm)
    results = []
    for exp in experiments:
        cm = json.loads(exp.confusion_matrix_json) if exp.confusion_matrix_json else [[0, 0], [0, 0]]
        features = json.loads(exp.selected_features_json) if exp.selected_features_json else []
        metrics = ModelEvaluationMetrics(
            accuracy=exp.accuracy or 0.0,
            precision=exp.precision or 0.0,
            recall=exp.recall or 0.0,
            f1_score=exp.f1_score or 0.0,
            false_positive_rate=exp.false_positive_rate or 0.0,
            false_negative_rate=exp.false_negative_rate or 0.0,
            roc_auc=exp.roc_auc,
            confusion_matrix=cm,
            training_time_seconds=exp.training_time_seconds or 0.0,
            prediction_time_ms=exp.prediction_time_ms or 0.0,
            selected_features_count=len(features),
        )
        results.append(
            ModelExperimentSummary(
                id=exp.id,
                model_name=exp.model_name,
                algorithm=exp.algorithm,
                dataset_id=exp.dataset_id,
                feature_strategy=exp.feature_strategy,
                metrics=metrics,
                artifact_path=exp.artifact_path,
                is_best_model=exp.is_best_model,
                created_at=exp.created_at,
            )
        )
    return results


@router.get("/experiments/{experiment_id}", response_model=ModelExperimentSummary)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db_session),
):
    """Retrieves detailed metrics for a specific experiment ID."""
    exp = get_experiment_by_id(db, experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with ID {experiment_id} not found.",
        )
    cm = json.loads(exp.confusion_matrix_json) if exp.confusion_matrix_json else [[0, 0], [0, 0]]
    features = json.loads(exp.selected_features_json) if exp.selected_features_json else []
    metrics = ModelEvaluationMetrics(
        accuracy=exp.accuracy or 0.0,
        precision=exp.precision or 0.0,
        recall=exp.recall or 0.0,
        f1_score=exp.f1_score or 0.0,
        false_positive_rate=exp.false_positive_rate or 0.0,
        false_negative_rate=exp.false_negative_rate or 0.0,
        roc_auc=exp.roc_auc,
        confusion_matrix=cm,
        training_time_seconds=exp.training_time_seconds or 0.0,
        prediction_time_ms=exp.prediction_time_ms or 0.0,
        selected_features_count=len(features),
    )
    return ModelExperimentSummary(
        id=exp.id,
        model_name=exp.model_name,
        algorithm=exp.algorithm,
        dataset_id=exp.dataset_id,
        feature_strategy=exp.feature_strategy,
        metrics=metrics,
        artifact_path=exp.artifact_path,
        is_best_model=exp.is_best_model,
        created_at=exp.created_at,
    )
