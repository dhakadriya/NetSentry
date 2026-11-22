"""DatasetRecord ORM model representing uploaded and cataloged network flow datasets."""

import json
from typing import Any

from sqlalchemy import Boolean, Column, Integer, String, Text
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class DatasetRecord(Base, TimestampMixin):
    """Catalog entry for uploaded and verified network flow datasets."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    dataset_type = Column(String(64), default="unsw_nb15", nullable=False, index=True)
    total_rows = Column(Integer, nullable=False, default=0)
    feature_count = Column(Integer, nullable=False, default=0)
    normal_count = Column(Integer, nullable=False, default=0)
    attack_count = Column(Integer, nullable=False, default=0)
    checksum_sha256 = Column(String(64), nullable=False, index=True)
    is_validated = Column(Boolean, default=True, nullable=False)
    description = Column(Text, nullable=True)
    columns_json = Column(Text, nullable=True)

    # Relationships
    experiments = relationship("ModelExperiment", back_populates="dataset", cascade="all, delete-orphan")
    prediction_jobs = relationship("PredictionJob", back_populates="dataset", cascade="all, delete-orphan")

    @property
    def column_names(self) -> list[str]:
        """Deserializes columns_json into list of column name strings."""
        if not self.columns_json:
            return []
        try:
            return json.loads(self.columns_json)
        except (json.JSONDecodeError, TypeError):
            return []

    def to_dict(self) -> dict[str, Any]:
        """Serializes DatasetRecord to dictionary with ISO-formatted timestamps."""
        return {
            "id": self.id,
            "filename": self.filename,
            "stored_path": self.stored_path,
            "dataset_type": self.dataset_type,
            "total_rows": self.total_rows,
            "feature_count": self.feature_count,
            "normal_count": self.normal_count,
            "attack_count": self.attack_count,
            "checksum_sha256": self.checksum_sha256,
            "is_validated": self.is_validated,
            "description": self.description,
            "column_names": self.column_names,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<DatasetRecord(id={self.id}, filename='{self.filename}', rows={self.total_rows})>"
