"""CRUD operations for machine learning model experiments and artifacts."""

import json
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import ModelExperiment


def create_experiment(
    db: Session,
    model_name: str,
    algorithm: str,
    dataset_id: int | None,
    feature_strategy: str,
    hyperparameters: dict[str, Any] | None,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    false_positive_rate: float,
    false_negative_rate: float,
    roc_auc: float | None,
    confusion_matrix: list[list[int]],
    training_time_seconds: float,
    prediction_time_ms: float,
    artifact_path: str | None = None,
    selected_features: list[str] | None = None,
    is_best_model: bool = False,
    notes: str | None = None,
) -> ModelExperiment:
    """Creates a new model experiment record."""
    if is_best_model:
        # Reset existing best model flags
        db.query(ModelExperiment).update({ModelExperiment.is_best_model: False})

    experiment = ModelExperiment(
        model_name=model_name,
        algorithm=algorithm,
        dataset_id=dataset_id,
        feature_strategy=feature_strategy,
        hyperparameters_json=json.dumps(hyperparameters) if hyperparameters else None,
        selected_features_json=json.dumps(selected_features) if selected_features else None,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        roc_auc=roc_auc,
        confusion_matrix_json=json.dumps(confusion_matrix),
        training_time_seconds=training_time_seconds,
        prediction_time_ms=prediction_time_ms,
        artifact_path=artifact_path,
        is_active=True,
        is_best_model=is_best_model,
        notes=notes,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def get_experiment_by_id(db: Session, experiment_id: int) -> ModelExperiment | None:
    """Retrieves an experiment by its ID."""
    return db.query(ModelExperiment).filter(ModelExperiment.id == experiment_id).first()


def get_best_model(db: Session) -> ModelExperiment | None:
    """Retrieves current best performing model, or highest F1 score model if none flagged."""
    best = db.query(ModelExperiment).filter(ModelExperiment.is_best_model == True).first()
    if not best:
        best = (
            db.query(ModelExperiment)
            .filter(ModelExperiment.is_active == True)
            .order_by(ModelExperiment.f1_score.desc(), ModelExperiment.false_positive_rate.asc())
            .first()
        )
    return best


def set_best_model(db: Session, experiment_id: int) -> ModelExperiment | None:
    """Designates a specific model run as the current production best."""
    db.query(ModelExperiment).update({ModelExperiment.is_best_model: False})
    target = get_experiment_by_id(db, experiment_id)
    if target:
        target.is_best_model = True
        db.commit()
        db.refresh(target)
    return target


def archive_experiment(db: Session, experiment_id: int) -> bool:
    """Soft-deletes or deactivates an experiment from the active leaderboard."""
    target = get_experiment_by_id(db, experiment_id)
    if target:
        target.is_active = False
        target.is_best_model = False
        db.commit()
        db.refresh(target)
        return True
    return False


def count_experiments(
    db: Session, algorithm: str | None = None, is_active: bool | None = None
) -> int:
    """Returns total count of experiments matching optional criteria."""
    query = db.query(func.count(ModelExperiment.id))
    if algorithm:
        query = query.filter(ModelExperiment.algorithm == algorithm)
    if is_active is not None:
        query = query.filter(ModelExperiment.is_active == is_active)
    return query.scalar() or 0


def list_experiments(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    algorithm: str | None = None,
    is_active: bool | None = None,
) -> list[ModelExperiment]:
    """Lists model experiments with pagination and optional filtering."""
    query = db.query(ModelExperiment)
    if algorithm:
        query = query.filter(ModelExperiment.algorithm == algorithm)
    if is_active is not None:
        query = query.filter(ModelExperiment.is_active == is_active)
    return query.order_by(ModelExperiment.id.desc()).offset(offset).limit(limit).all()


def get_leaderboard(db: Session, limit: int = 20) -> list[ModelExperiment]:
    """Returns top performing models ranked by F1-score and low FPR."""
    return (
        db.query(ModelExperiment)
        .filter(ModelExperiment.is_active == True)
        .order_by(ModelExperiment.f1_score.desc(), ModelExperiment.false_positive_rate.asc())
        .limit(limit)
        .all()
    )
