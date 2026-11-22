"""Dataset schema validator inspecting headers, distributions, and integrity."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.constants import (
    UNSW_CATEGORICAL_COLUMNS,
    UNSW_TARGET_LABEL,
    DatasetType,
)
from src.core.exceptions import DatasetValidationError
from src.schemas.dataset import DatasetValidationResult


class DatasetValidator:
    """Validates structural and content compliance of uploaded network traffic datasets."""

    def __init__(self, target_type: DatasetType = DatasetType.UNSW_NB15):
        self.target_type = target_type

    def validate_dataframe(self, df: pd.DataFrame) -> DatasetValidationResult:
        """Validates an in-memory DataFrame against benchmark schema rules."""
        errors: list[str] = []
        warnings: list[str] = []

        total_rows, total_cols = df.shape
        if total_rows == 0:
            errors.append("Dataset contains 0 records.")
            return DatasetValidationResult(
                is_valid=False,
                detected_type="empty",
                total_rows=0,
                total_columns=total_cols,
                errors=errors,
            )

        # Check required columns
        required_cols = UNSW_CATEGORICAL_COLUMNS + ["dur", "sbytes", "dbytes", "sttl", "Spkts", "Dpkts"]
        missing_required = [c for c in required_cols if c not in df.columns]

        if missing_required:
            errors.append(f"Missing mandatory network flow columns: {missing_required}")

        # Check label column
        if UNSW_TARGET_LABEL not in df.columns and "Label" not in df.columns:
            warnings.append("Target label column ('label' or 'Label') not found. Dataset can only be used for inference.")

        # Check missing values
        null_counts = {col: int(count) for col, count in df.isnull().sum().items() if count > 0}
        if null_counts:
            warnings.append(f"Dataset contains null values in {len(null_counts)} columns; automatic imputation will apply.")

        # Check infinite values in numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {col: int(np.isinf(df[col]).sum()) for col in num_cols if np.isinf(df[col]).sum() > 0}
        if inf_counts:
            warnings.append(f"Infinite values detected in {len(inf_counts)} numerical columns.")

        detected_type = "unsw_nb15" if not missing_required else "generic"
        is_valid = len(errors) == 0

        return DatasetValidationResult(
            is_valid=is_valid,
            detected_type=detected_type,
            total_rows=total_rows,
            total_columns=total_cols,
            missing_required_columns=missing_required,
            null_value_counts=null_counts,
            errors=errors,
            warnings=warnings,
        )

    def validate_file(self, file_path: str | Path) -> tuple[DatasetValidationResult, pd.DataFrame]:
        """Reads a CSV file safely and verifies its schema."""
        path = Path(file_path)
        if not path.exists():
            raise DatasetValidationError(f"File does not exist: {file_path}")

        try:
            df = pd.read_csv(path, nrows=50000)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV file: {str(e)}") from e

        validation = self.validate_dataframe(df)
        if not validation.is_valid:
            raise DatasetValidationError(
                f"Dataset validation failed: {'; '.join(validation.errors)}"
            )

        return validation, df
