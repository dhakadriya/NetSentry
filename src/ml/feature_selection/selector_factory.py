"""Unified factory interface for selecting and instantiating feature selection strategies."""

from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.core.constants import FeatureSelectionStrategy
from src.core.exceptions import FeatureSelectionError
from src.ml.feature_selection.variance_threshold import VarianceThresholdSelector
from src.ml.feature_selection.correlation_filter import CorrelationFilterSelector
from src.ml.feature_selection.importance_selector import RandomForestImportanceSelector
from src.ml.feature_selection.mutual_info import MutualInformationSelector


class PassThroughSelector(BaseEstimator, TransformerMixin):
    """Retains all features without modification."""

    def __init__(self):
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PassThroughSelector":
        self.selected_features_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.selected_features_


def get_feature_selector(
    strategy: FeatureSelectionStrategy = FeatureSelectionStrategy.NONE,
    top_k: int = 20,
    **kwargs,
) -> BaseEstimator:
    """Factory creating configured feature selector instance corresponding to strategy."""
    strategy_str = str(strategy).lower()

    if strategy_str in (FeatureSelectionStrategy.NONE.value, "none"):
        return PassThroughSelector()

    elif strategy_str in (FeatureSelectionStrategy.VARIANCE_THRESHOLD.value, "variance_threshold"):
        threshold = kwargs.get("threshold", 0.01)
        return VarianceThresholdSelector(threshold=threshold, min_features=kwargs.get("min_features", 5))

    elif strategy_str in (FeatureSelectionStrategy.CORRELATION_FILTER.value, "correlation_filter"):
        threshold = kwargs.get("threshold", 0.85)
        return CorrelationFilterSelector(threshold=threshold, min_features=kwargs.get("min_features", 8))

    elif strategy_str in (FeatureSelectionStrategy.RF_IMPORTANCE.value, "rf_importance"):
        return RandomForestImportanceSelector(top_k=top_k, random_state=kwargs.get("random_state", 42))

    elif strategy_str in (FeatureSelectionStrategy.MUTUAL_INFO.value, "mutual_info"):
        return MutualInformationSelector(top_k=top_k, random_state=kwargs.get("random_state", 42))

    else:
        raise FeatureSelectionError(f"Unsupported feature selection strategy: '{strategy}'")
