"""Numerical normalization and feature scaling transformers."""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class NetworkFeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numerical flow features while preserving DataFrame structure and column headers."""

    def __init__(
        self,
        method: str = "standard",
        exclude_columns: list[str] | None = None,
        clip_extreme_values: bool = True,
        clip_range: tuple = (-10.0, 10.0),
    ):
        self.method = method.lower()
        self.exclude_columns = exclude_columns or ["label", "Label", "attack_cat"]
        self.clip_extreme_values = clip_extreme_values
        self.clip_range = clip_range
        self.scaler_: BaseEstimator | None = None
        self.scaled_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NetworkFeatureScaler":
        """Fits numerical scaler on training data."""
        df = X.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        self.scaled_columns_ = [c for c in num_cols if c not in self.exclude_columns]

        if not self.scaled_columns_:
            return self

        if self.method == "robust":
            self.scaler_ = RobustScaler()
        elif self.method == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            self.scaler_ = StandardScaler()

        self.scaler_.fit(df[self.scaled_columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies normalization to numerical features with outlier clipping safeguards."""
        df = X.copy()

        if not self.scaled_columns_ or self.scaler_ is None:
            return df

        # Only scale columns present in both
        active_cols = [c for c in self.scaled_columns_ if c in df.columns]
        if active_cols:
            scaled_arr = self.scaler_.transform(df[active_cols])
            if self.clip_extreme_values and self.method == "standard":
                scaled_arr = np.clip(scaled_arr, self.clip_range[0], self.clip_range[1])
            df[active_cols] = scaled_arr

        return df

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Returns the list of scaled feature columns."""
        return self.scaled_columns_
