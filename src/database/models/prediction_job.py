"""PredictionJob ORM model for recording batch intrusion detection runs."""

from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class PredictionJob(Base, TimestampMixin):
    """Execution record for network traffic inference jobs and report generation."""

    __tablename__ = "prediction_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(32), default="batch_csv", nullable=False)
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
    status = Column(String(32), default="completed", nullable=False)
    error_message = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("DatasetRecord", back_populates="prediction_jobs")
    model_experiment = relationship("ModelExperiment", back_populates="prediction_jobs")
    alerts = relationship("TrafficAlert", back_populates="prediction_job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<PredictionJob(id={self.id}, records={self.total_records}, "
            f"malicious={self.malicious_count}, status='{self.status}')>"
        )
