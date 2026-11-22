"""Pydantic schemas for network traffic datasets and exploratory analysis."""

from datetime import datetime

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    """Concise representation of a cataloged dataset."""

    id: int = Field(..., description="Unique dataset identifier", ge=1)
    filename: str = Field(..., description="Sanitized dataset filename", pattern=r"^[a-zA-Z0-9_\-\.]+\.csv$")
    dataset_type: str = Field(..., description="Dataset benchmark format (e.g. unsw_nb15, cic_ids2017)")
    total_rows: int = Field(..., description="Total network flow records", ge=0)
    feature_count: int = Field(..., description="Number of network flow features", ge=1)
    normal_count: int = Field(..., description="Count of benign traffic flows", ge=0)
    attack_count: int = Field(..., description="Count of malicious traffic flows", ge=0)
    checksum_sha256: str = Field(..., description="Cryptographic SHA-256 hash", min_length=64, max_length=64)
    is_validated: bool = Field(True, description="Whether schema integrity was verified")
    created_at: datetime = Field(..., description="Timestamp of dataset cataloging")
    description: str | None = Field(None, description="Optional dataset description")

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    """Response returned upon successful CSV upload and registration."""

    dataset_id: int = Field(..., ge=1)
    filename: str
    dataset_type: str
    total_rows: int = Field(..., ge=0)
    feature_count: int = Field(..., ge=1)
    normal_count: int = Field(..., ge=0)
    attack_count: int = Field(..., ge=0)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)
    message: str = "Dataset uploaded and cataloged successfully."


class DatasetValidationResult(BaseModel):
    """Validation report inspecting an uploaded CSV dataset."""

    is_valid: bool
    detected_type: str
    total_rows: int = Field(..., ge=0)
    total_columns: int = Field(..., ge=0)
    missing_required_columns: list[str] = Field(default_factory=list)
    null_value_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetEDAResponse(BaseModel):
    """Statistical summary and EDA insights for an uploaded dataset."""

    dataset_id: int = Field(..., ge=1)
    filename: str
    total_rows: int = Field(..., ge=0)
    feature_count: int = Field(..., ge=0)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    attack_category_distribution: dict[str, int] = Field(default_factory=dict)
    protocol_distribution: dict[str, int] = Field(default_factory=dict)
    service_distribution: dict[str, int] = Field(default_factory=dict)
    numerical_summary: dict[str, dict[str, float]] = Field(default_factory=dict)


class DatasetPaginationResponse(BaseModel):
    """Paginated collection of dataset catalog records."""

    total: int = Field(..., ge=0, description="Total matching datasets in database")
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    items: list[DatasetSummary] = Field(default_factory=list)
