"""Correlation filter feature selection removing highly collinear network flow metrics."""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilterSelector(BaseEstimator, TransformerMixin):
    """Eliminates redundant, highly correlated network traffic features to prevent multicollinearity."""

    def __init__(self, threshold: float = 0.85, min_features: int = 8):
        self.threshold = threshold
        self.min_features = min_features
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CorrelationFilterSelector":
        """Calculates pairwise correlations and identifies collinear feature pairs to drop."""
        df = X.copy()
        corr_matrix = df.corr().abs()

        # Identify upper triangle
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = []
        for col in upper_triangle.columns:
            # Check if correlated above threshold with any previously encountered column
            high_corrs = upper_triangle[col][upper_triangle[col] > self.threshold]
            if len(high_corrs) > 0:
                to_drop.append(col)

        # Ensure we do not drop below min_features
        retained = [c for c in df.columns if c not in to_drop]
        if len(retained) < self.min_features:
            # Drop fewer features
            allowed_drops = max(0, len(df.columns) - self.min_features)
            to_drop = to_drop[:allowed_drops]
            retained = [c for c in df.columns if c not in to_drop]

        self.selected_features_ = retained
        self.dropped_features_ = to_drop
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns feature matrix without collinear features."""
        available_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[available_cols].copy()

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.selected_features_
