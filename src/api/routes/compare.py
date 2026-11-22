"""Model comparison, 4-algorithm benchmarking, and leaderboard endpoints."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.deps import get_db_session, get_trainer
from src.database.crud.dataset_repo import get_dataset_by_id
from src.database.crud.experiment_repo import (
    create_experiment,
    get_leaderboard,
    set_best_model,
)
from src.ml.evaluation.comparator import ModelComparator
from src.ml.trainer import NetSentryTrainer
from src.schemas.model import CompareModelsRequest, CompareModelsResponse

router = APIRouter(prefix="/api/v1/models", tags=["Model Comparison"])


@router.post("/compare", response_model=CompareModelsResponse)
def compare_models(
    request: CompareModelsRequest,
    db: Session = Depends(get_db_session),
    trainer: NetSentryTrainer = Depends(get_trainer),
):
    """Benchmarks requested ML algorithms on selected dataset and identifies the optimal intrusion classifier."""
    dataset_record = get_dataset_by_id(db, request.dataset_id)
    if not dataset_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} not found.",
        )

    file_path = Path(dataset_record.stored_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file missing from disk: {file_path}",
        )

    df = pd.read_csv(file_path, nrows=10000)

    evaluated_records: list[dict[str, Any]] = []
    created_experiments = []

    # Benchmark each algorithm
    for algo in request.algorithms:
        algo_name = algo.value
        model, preprocessor, selector, metrics, artifact_path = trainer.train(
            df=df,
            algorithm=algo,
            feature_strategy=request.feature_strategy,
            test_size=request.test_size,
            random_seed=request.random_seed,
            model_name=f"Benchmark-{algo_name.title()}",
        )

        exp = create_experiment(
            db=db,
            model_name=model.model_name,
            algorithm=algo_name,
            dataset_id=dataset_record.id,
            feature_strategy=request.feature_strategy.value,
            hyperparameters={"seed": request.random_seed},
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
            is_best_model=False,
            notes=f"Benchmark run with {request.feature_strategy.value} feature selection.",
        )
        created_experiments.append(exp)

        evaluated_records.append({
            "experiment_id": exp.id,
            "model_name": exp.model_name,
            "algorithm": algo_name,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "false_positive_rate": metrics.false_positive_rate,
            "false_negative_rate": metrics.false_negative_rate,
            "roc_auc": metrics.roc_auc,
            "training_time_seconds": metrics.training_time_seconds,
            "prediction_time_ms": metrics.prediction_time_ms,
            "selected_features_count": len(model.feature_names_),
        })

    # Rank with ModelComparator composite formula
    ranked = ModelComparator.rank_models(evaluated_records)
    best_candidate = ranked[0]

    # Set best candidate as active in DB
    set_best_model(db, best_candidate["experiment_id"])

    return CompareModelsResponse(
        dataset_id=dataset_record.id,
        total_models_evaluated=len(ranked),
        best_algorithm=best_candidate["algorithm"],
        best_model_id=best_candidate["experiment_id"],
        best_f1_score=best_candidate["f1_score"],
        lowest_fpr=min(m["false_positive_rate"] for m in ranked),
        leaderboard=ranked,
        feature_strategy=request.feature_strategy.value,
        feature_count=best_candidate["selected_features_count"],
    )


@router.get("/leaderboard")
def get_model_leaderboard(
    limit: int = 20,
    db: Session = Depends(get_db_session),
):
    """Retrieves current top intrusion models ranked by performance."""
    models = get_leaderboard(db, limit=limit)
    raw = [
        {
            "id": m.id,
            "model_name": m.model_name,
            "algorithm": m.algorithm,
            "f1_score": m.f1_score or 0.0,
            "false_positive_rate": m.false_positive_rate or 0.0,
            "accuracy": m.accuracy or 0.0,
            "prediction_time_ms": m.prediction_time_ms or 0.0,
            "is_best_model": m.is_best_model,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in models
    ]
    ranked = ModelComparator.rank_models(raw)
    return {"leaderboard": ranked}
