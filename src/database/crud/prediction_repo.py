"""CRUD operations for batch and flow prediction jobs."""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.database.models.prediction_job import PredictionJob


def create_prediction_job(
    db: Session,
    job_type: str,
    dataset_id: Optional[int],
    model_experiment_id: Optional[int],
    total_records: int,
    normal_count: int,
    malicious_count: int,
    detection_rate: float,
    execution_time_seconds: float,
    report_csv_path: Optional[str] = None,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> PredictionJob:
    """Creates a new record for a completed batch flow prediction job."""
    job = PredictionJob(
        job_type=job_type,
        dataset_id=dataset_id,
        model_experiment_id=model_experiment_id,
        total_records=total_records,
        normal_count=normal_count,
        malicious_count=malicious_count,
        detection_rate=detection_rate,
        execution_time_seconds=execution_time_seconds,
        report_csv_path=report_csv_path,
        status=status,
        error_message=error_message,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_prediction_job_by_id(db: Session, job_id: int) -> Optional[PredictionJob]:
    """Retrieves a prediction job by its primary key."""
    return db.query(PredictionJob).filter(PredictionJob.id == job_id).first()


def list_prediction_jobs(
    db: Session, limit: int = 50, offset: int = 0
) -> List[PredictionJob]:
    """Lists recent prediction jobs in descending order."""
    return (
        db.query(PredictionJob)
        .order_by(PredictionJob.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
