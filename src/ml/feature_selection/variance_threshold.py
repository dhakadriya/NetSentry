"""Variance threshold feature selection removing zero and near-zero variance features."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """Filters out features exhibiting near-zero variance across network flows."""

    def __init__(self, threshold: float = 0.01, min_features: int = 5):
        self.threshold = threshold
        self.min_features = min_features
        self.selector_: Optional[VarianceThreshold] = None
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VarianceThresholdSelector":
        """Calculates feature variances and determines active mask."""
        df = X.copy()
        variances = df.var()

        # If strict threshold drops too many features, fallback to keeping top min_features
        passing_cols = variances[variances >= self.threshold].index.tolist()
        if len(passing_cols) < self.min_features:
            passing_cols = variances.nlargest(self.min_features).index.tolist()

        self.selected_features_ = passing_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame containing only variance-retained features."""
        available_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[available_cols].copy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.selected_features_
