"""DatasetRecord ORM model representing uploaded and cataloged network flow datasets."""

from sqlalchemy import Column, Integer, String, Boolean, Text
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class DatasetRecord(Base, TimestampMixin):
    """Catalog entry for uploaded and verified network flow datasets."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    dataset_type = Column(String(64), default="unsw_nb15", nullable=False)
    total_rows = Column(Integer, nullable=False, default=0)
    feature_count = Column(Integer, nullable=False, default=0)
    normal_count = Column(Integer, nullable=False, default=0)
    attack_count = Column(Integer, nullable=False, default=0)
    checksum_sha256 = Column(String(64), nullable=False)
    is_validated = Column(Boolean, default=True, nullable=False)
    description = Column(Text, nullable=True)
    columns_json = Column(Text, nullable=True)

    # Relationships
    experiments = relationship("ModelExperiment", back_populates="dataset", cascade="all, delete-orphan")
    prediction_jobs = relationship("PredictionJob", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DatasetRecord(id={self.id}, filename='{self.filename}', rows={self.total_rows})>"
