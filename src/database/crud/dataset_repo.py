"""CRUD operations for network traffic dataset catalog."""

import json

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models.dataset import DatasetRecord


def create_dataset(
    db: Session,
    filename: str,
    stored_path: str,
    dataset_type: str,
    total_rows: int,
    feature_count: int,
    normal_count: int,
    attack_count: int,
    checksum_sha256: str,
    columns: list[str] | None = None,
    description: str | None = None,
) -> DatasetRecord:
    """Creates and persists a new dataset catalog entry."""
    record = DatasetRecord(
        filename=filename,
        stored_path=stored_path,
        dataset_type=dataset_type,
        total_rows=total_rows,
        feature_count=feature_count,
        normal_count=normal_count,
        attack_count=attack_count,
        checksum_sha256=checksum_sha256,
        is_validated=True,
        description=description,
        columns_json=json.dumps(columns) if columns else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_dataset_by_id(db: Session, dataset_id: int) -> DatasetRecord | None:
    """Retrieves a dataset record by its primary key."""
    return db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()


def get_dataset_by_checksum(db: Session, checksum: str) -> DatasetRecord | None:
    """Finds an existing dataset with identical SHA-256 hash."""
    return db.query(DatasetRecord).filter(DatasetRecord.checksum_sha256 == checksum).first()


def exists_by_checksum(db: Session, checksum: str) -> bool:
    """Checks whether a dataset with the given SHA-256 checksum exists."""
    return (
        db.query(func.count(DatasetRecord.id))
        .filter(DatasetRecord.checksum_sha256 == checksum)
        .scalar()
        or 0
    ) > 0


def count_datasets(db: Session, dataset_type: str | None = None) -> int:
    """Returns the total number of cataloged datasets matching optional type filter."""
    query = db.query(func.count(DatasetRecord.id))
    if dataset_type:
        query = query.filter(DatasetRecord.dataset_type == dataset_type)
    return query.scalar() or 0


def list_datasets(
    db: Session, limit: int = 50, offset: int = 0, dataset_type: str | None = None
) -> list[DatasetRecord]:
    """Returns cataloged datasets ordered by creation time descending with optional type filtering."""
    query = db.query(DatasetRecord)
    if dataset_type:
        query = query.filter(DatasetRecord.dataset_type == dataset_type)
    return query.order_by(DatasetRecord.id.desc()).offset(offset).limit(limit).all()


def delete_dataset(db: Session, dataset_id: int) -> bool:
    """Deletes a dataset entry and associated cascade records."""
    record = get_dataset_by_id(db, dataset_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False
