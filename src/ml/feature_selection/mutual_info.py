"""Mutual Information Gain feature selection capturing non-linear dependencies."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """Selects highest ranking network flow features based on Mutual Information with target labels."""

    def __init__(self, top_k: int = 20, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.mutual_info_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MutualInformationSelector":
        """Computes mutual information classification gain across features."""
        df = X.copy()
        target = y.copy()

        scores = mutual_info_classif(
            df,
            target,
            discrete_features="auto",
            random_state=self.random_state,
        )

        feature_score_map = dict(zip(df.columns, [float(x) for x in scores]))
        sorted_features = sorted(feature_score_map.items(), key=lambda item: item[1], reverse=True)
        self.mutual_info_scores_ = dict(sorted_features)

        k = min(self.top_k, len(sorted_features))
        self.selected_features_ = [feat for feat, _ in sorted_features[:k]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns feature matrix restricted to top mutual information features."""
        available_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[available_cols].copy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.selected_features_
