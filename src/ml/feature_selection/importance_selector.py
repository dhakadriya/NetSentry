"""Random Forest Gini feature importance ranking and selection."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class RandomForestImportanceSelector(BaseEstimator, TransformerMixin):
    """Selects top network traffic features based on tree ensemble Gini importance scores."""

    def __init__(self, top_k: int = 20, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestImportanceSelector":
        """Fits an ensemble classifier to compute feature importance rankings."""
        df = X.copy()
        target = y.copy()

        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(df, target)

        importances = rf.feature_importances_
        feature_score_map = dict(zip(df.columns, [float(x) for x in importances]))

        # Sort descending
        sorted_features = sorted(feature_score_map.items(), key=lambda item: item[1], reverse=True)
        self.feature_importances_ = dict(sorted_features)

        # Select top k
        k = min(self.top_k, len(sorted_features))
        self.selected_features_ = [feat for feat, _ in sorted_features[:k]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns feature matrix restricted to top Gini importance features."""
        available_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[available_cols].copy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.selected_features_
