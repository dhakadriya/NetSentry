"""CRUD operations for batch and flow prediction jobs."""


from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models.prediction_job import PredictionJob


def create_prediction_job(
    db: Session,
    job_type: str,
    dataset_id: int | None,
    model_experiment_id: int | None,
    total_records: int,
    normal_count: int,
    malicious_count: int,
    detection_rate: float,
    execution_time_seconds: float,
    report_csv_path: str | None = None,
    status: str = "completed",
    error_message: str | None = None,
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


def get_prediction_job_by_id(db: Session, job_id: int) -> PredictionJob | None:
    """Retrieves a prediction job by its primary key."""
    return db.query(PredictionJob).filter(PredictionJob.id == job_id).first()


def update_prediction_job_status(
    db: Session, job_id: int, status: str, error_message: str | None = None
) -> PredictionJob | None:
    """Updates the execution status and error state of a prediction job."""
    job = get_prediction_job_by_id(db, job_id)
    if job:
        job.status = status
        if error_message:
            job.error_message = error_message
        db.commit()
        db.refresh(job)
    return job


def count_prediction_jobs(db: Session, status: str | None = None) -> int:
    """Returns the total number of prediction jobs, optionally filtered by status."""
    query = db.query(func.count(PredictionJob.id))
    if status:
        query = query.filter(PredictionJob.status == status)
    return query.scalar() or 0


def list_prediction_jobs(
    db: Session, limit: int = 50, offset: int = 0, status: str | None = None
) -> list[PredictionJob]:
    """Lists recent prediction jobs in descending order with optional status filter."""
    query = db.query(PredictionJob)
    if status:
        query = query.filter(PredictionJob.status == status)
    return (
        query.order_by(PredictionJob.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
