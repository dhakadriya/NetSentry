"""Dataset upload, validation, and catalog management endpoints."""

from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
import pandas as pd
from sqlalchemy.orm import Session

from src.api.deps import get_app_settings, get_db_session
from src.core.config import Settings
from src.core.security import compute_sha256, safe_join_path, sanitize_filename, validate_file_size
from src.database.crud.dataset_repo import (
    create_dataset,
    get_dataset_by_checksum,
    get_dataset_by_id,
    list_datasets,
)
from src.ml.validators.dataset_validator import DatasetValidator
from src.schemas.dataset import DatasetSummary, DatasetUploadResponse

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
):
    """Uploads, validates, and catalogs a network flow CSV dataset."""
    safe_name = sanitize_filename(file.filename or "uploaded_traffic.csv")
    content = await file.read()

    # Security: validate file size
    validate_file_size(len(content), max_size_mb=settings.max_upload_size_mb)

    # Compute SHA-256 Checksum
    sha = compute_sha256(content)
    existing = get_dataset_by_checksum(db, sha)
    if existing:
        return DatasetUploadResponse(
            dataset_id=existing.id,
            filename=existing.filename,
            dataset_type=existing.dataset_type,
            total_rows=existing.total_rows,
            feature_count=existing.feature_count,
            normal_count=existing.normal_count,
            attack_count=existing.attack_count,
            checksum_sha256=existing.checksum_sha256,
            message="Identical dataset already uploaded; existing catalog entry returned.",
        )

    # Save file safely to uploads directory
    target_path = safe_join_path(settings.upload_dir, f"{sha[:10]}_{safe_name}")
    target_path.write_bytes(content)

    # Validate dataset structure
    validator = DatasetValidator()
    validation_report, df = validator.validate_file(target_path)

    # Compute counts
    normal_cnt = 0
    attack_cnt = 0
    for lbl_col in ["label", "Label"]:
        if lbl_col in df.columns:
            normal_cnt = int((df[lbl_col] == 0).sum())
            attack_cnt = int((df[lbl_col] == 1).sum())
            break

    # Persist in DB
    record = create_dataset(
        db=db,
        filename=safe_name,
        stored_path=str(target_path),
        dataset_type=validation_report.detected_type,
        total_rows=validation_report.total_rows,
        feature_count=validation_report.total_columns,
        normal_count=normal_cnt,
        attack_count=attack_cnt,
        checksum_sha256=sha,
        columns=list(df.columns),
        description=description,
    )

    return DatasetUploadResponse(
        dataset_id=record.id,
        filename=record.filename,
        dataset_type=record.dataset_type,
        total_rows=record.total_rows,
        feature_count=record.feature_count,
        normal_count=record.normal_count,
        attack_count=record.attack_count,
        checksum_sha256=record.checksum_sha256,
        message="Dataset validated and cataloged successfully.",
    )


@router.get("", response_model=List[DatasetSummary])
def get_datasets(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session),
):
    """Lists all cataloged datasets with row and class statistics."""
    records = list_datasets(db, limit=limit, offset=offset)
    return records


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db_session),
):
    """Retrieves metadata for a specific dataset ID."""
    record = get_dataset_by_id(db, dataset_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )
    return record
