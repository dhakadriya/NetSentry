"""PredictionJob ORM model for recording batch intrusion detection runs."""

from typing import Any

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class PredictionJob(Base, TimestampMixin):
    """Execution record for network traffic inference jobs and report generation."""

    __tablename__ = "prediction_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(32), default="batch_csv", nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    model_experiment_id = Column(
        Integer, ForeignKey("model_experiments.id", ondelete="SET NULL"), nullable=True
    )

    total_records = Column(Integer, default=0, nullable=False)
    normal_count = Column(Integer, default=0, nullable=False)
    malicious_count = Column(Integer, default=0, nullable=False)
    detection_rate = Column(Float, default=0.0, nullable=False)
    execution_time_seconds = Column(Float, default=0.0, nullable=False)

    report_csv_path = Column(String(512), nullable=True)
    status = Column(String(32), default="completed", nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("DatasetRecord", back_populates="prediction_jobs")
    model_experiment = relationship("ModelExperiment", back_populates="prediction_jobs")
    alerts = relationship("TrafficAlert", back_populates="prediction_job", cascade="all, delete-orphan")

    @property
    def malicious_percentage(self) -> float:
        """Computes malicious percentage of total processed traffic."""
        if self.total_records == 0:
            return 0.0
        return round((self.malicious_count / self.total_records) * 100.0, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serializes PredictionJob to dictionary with formatted metrics."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "dataset_id": self.dataset_id,
            "model_experiment_id": self.model_experiment_id,
            "total_records": self.total_records,
            "normal_count": self.normal_count,
            "malicious_count": self.malicious_count,
            "detection_rate": round(self.detection_rate, 4),
            "malicious_percentage": self.malicious_percentage,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "report_csv_path": self.report_csv_path,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<PredictionJob(id={self.id}, type='{self.job_type}', records={self.total_records}, "
            f"malicious={self.malicious_count}, status='{self.status}')>"
        )
