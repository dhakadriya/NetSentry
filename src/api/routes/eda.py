"""Exploratory Data Analysis (EDA) API endpoint."""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.api.deps import get_db_session
from src.database.crud.dataset_repo import get_dataset_by_id
from src.schemas.dataset import DatasetEDAResponse

router = APIRouter(prefix="/api/v1/datasets", tags=["Exploratory Data Analysis"])


@router.get("/{dataset_id}/eda", response_model=DatasetEDAResponse)
def get_dataset_eda(
    dataset_id: int,
    db: Session = Depends(get_db_session),
):
    """Computes comprehensive statistical distributions and EDA profile for an uploaded dataset."""
    record = get_dataset_by_id(db, dataset_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )

    file_path = Path(record.stored_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Underlying CSV file does not exist on disk: {file_path}",
        )

    try:
        df = pd.read_csv(file_path, nrows=100000)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    # Class distribution
    class_dist = {}
    for lbl_col in ["label", "Label"]:
        if lbl_col in df.columns:
            counts = df[lbl_col].value_counts().to_dict()
            class_dist = {
                "Normal (0)": int(counts.get(0, 0)),
                "Malicious (1)": int(counts.get(1, 0)),
            }
            break

    # Attack category breakdown
    attack_cat_dist = {}
    for cat_col in ["attack_cat", "Attack_Cat"]:
        if cat_col in df.columns:
            attack_cat_dist = {str(k): int(v) for k, v in df[cat_col].value_counts().items()}
            break

    # Top protocols & services
    proto_dist = {}
    if "proto" in df.columns:
        proto_dist = {str(k): int(v) for k, v in df["proto"].value_counts().head(8).items()}

    service_dist = {}
    if "service" in df.columns:
        service_dist = {str(k): int(v) for k, v in df["service"].value_counts().head(8).items()}

    # Numerical statistics
    key_numerical = ["dur", "sbytes", "dbytes", "sttl", "dttl", "Spkts", "Dpkts", "Sload", "Dload"]
    available_nums = [c for c in key_numerical if c in df.columns]
    num_summary = {}

    if available_nums:
        desc = df[available_nums].describe()
        for col in available_nums:
            num_summary[col] = {
                "mean": round(float(desc.loc["mean", col]), 3),
                "std": round(float(desc.loc["std", col]), 3) if not np.isnan(desc.loc["std", col]) else 0.0,
                "min": round(float(desc.loc["min", col]), 3),
                "50%": round(float(desc.loc["50%", col]), 3),
                "max": round(float(desc.loc["max", col]), 3),
            }

    return DatasetEDAResponse(
        dataset_id=record.id,
        filename=record.filename,
        total_rows=len(df),
        feature_count=len(df.columns),
        class_distribution=class_dist,
        attack_category_distribution=attack_cat_dist,
        protocol_distribution=proto_dist,
        service_distribution=service_dist,
        numerical_summary=num_summary,
    )
