"""ModelExperiment ORM model for tracking machine learning training runs and performance."""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class ModelExperiment(Base, TimestampMixin):
    """Experiment record tracking ML model configuration, performance metrics, and artifact path."""

    __tablename__ = "model_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    algorithm = Column(String(64), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)

    # Feature Engineering Configuration
    feature_strategy = Column(String(64), default="none", nullable=False)
    selected_features_json = Column(Text, nullable=True)
    hyperparameters_json = Column(Text, nullable=True)

    # Core Evaluation Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    confusion_matrix_json = Column(Text, nullable=True)

    # Operational Metrics
    training_time_seconds = Column(Float, nullable=True)
    prediction_time_ms = Column(Float, nullable=True)

    # Persistence
    artifact_path = Column(String(512), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_best_model = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("DatasetRecord", back_populates="experiments")
    prediction_jobs = relationship("PredictionJob", back_populates="model_experiment")

    def __repr__(self) -> str:
        return (
            f"<ModelExperiment(id={self.id}, algo='{self.algorithm}', "
            f"f1={self.f1_score:.4f} if self.f1_score else None)>"
        )
