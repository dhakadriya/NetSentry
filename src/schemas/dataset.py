"""Pydantic schemas for network traffic datasets and exploratory analysis."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    """Concise representation of a cataloged dataset."""

    id: int
    filename: str
    dataset_type: str
    total_rows: int
    feature_count: int
    normal_count: int
    attack_count: int
    checksum_sha256: str
    is_validated: bool
    created_at: datetime
    description: Optional[str] = None

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    """Response returned upon successful CSV upload and registration."""

    dataset_id: int
    filename: str
    dataset_type: str
    total_rows: int
    feature_count: int
    normal_count: int
    attack_count: int
    checksum_sha256: str
    message: str = "Dataset uploaded and cataloged successfully."


class DatasetValidationResult(BaseModel):
    """Validation report inspecting an uploaded CSV dataset."""

    is_valid: bool
    detected_type: str
    total_rows: int
    total_columns: int
    missing_required_columns: List[str] = Field(default_factory=list)
    null_value_counts: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DatasetEDAResponse(BaseModel):
    """Statistical summary and EDA insights for an uploaded dataset."""

    dataset_id: int
    filename: str
    total_rows: int
    feature_count: int
    class_distribution: Dict[str, int]
    attack_category_distribution: Dict[str, int]
    protocol_distribution: Dict[str, int]
    service_distribution: Dict[str, int]
    numerical_summary: Dict[str, Dict[str, float]]
