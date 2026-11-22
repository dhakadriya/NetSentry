"""Data cleaner handling missing values, infinite values, whitespace, and corrupt fields."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for network flow data cleaning."""

    def __init__(
        self,
        drop_duplicates: bool = False,
        fill_numerical: str = "median",
        fill_categorical: str = "unknown",
        clip_non_negative: bool = True,
    ):
        self.drop_duplicates = drop_duplicates
        self.fill_numerical = fill_numerical
        self.fill_categorical = fill_categorical
        self.clip_non_negative = clip_non_negative
        self.numerical_fill_values_: dict = {}
        self.categorical_fill_values_: dict = {}
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataCleaner":
        """Computes imputation values from training data."""
        df = X.copy()
        self.feature_names_in_ = list(df.columns)

        # Clean infinities first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.fill_numerical == "median":
                self.numerical_fill_values_[col] = float(df[col].median(skipna=True))
            else:
                self.numerical_fill_values_[col] = float(df[col].mean(skipna=True))
            # Fallback for empty column
            if np.isnan(self.numerical_fill_values_[col]):
                self.numerical_fill_values_[col] = 0.0

        # Categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            mode_vals = df[col].dropna().mode()
            self.categorical_fill_values_[col] = (
                str(mode_vals.iloc[0]) if len(mode_vals) > 0 else self.fill_categorical
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies data cleaning and sanitization transformations."""
        df = X.copy()

        # Strip whitespaces from column names
        df.columns = [c.strip() for c in df.columns]

        # Strip whitespaces from string columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()

        # Replace infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Impute numerical columns
        for col, fill_val in self.numerical_fill_values_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                # Ensure no negative durations or byte counts
                if self.clip_non_negative and col in [
                    "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Sload", "Dload"
                ]:
                    df[col] = np.maximum(df[col], 0.0)

        # Impute categorical columns
        for col, fill_val in self.categorical_fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        if self.drop_duplicates:
            df.drop_duplicates(inplace=True)

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns feature names after cleaning transformation."""
        return input_features if input_features is not None else self.feature_names_in_
