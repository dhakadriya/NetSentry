"""Unified factory interface for selecting and instantiating feature selection strategies."""

from typing import Any, Dict, List, Optional
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

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.selected_features_


def list_available_strategies() -> Dict[str, str]:
    """Returns supported feature selection strategies with descriptions."""
    return {
        FeatureSelectionStrategy.NONE.value: "Preserves all features without reduction.",
        FeatureSelectionStrategy.VARIANCE_THRESHOLD.value: "Removes low-variance or near-constant features.",
        FeatureSelectionStrategy.CORRELATION_FILTER.value: "Eliminates highly collinear and redundant features.",
        FeatureSelectionStrategy.RF_IMPORTANCE.value: "Ranks features by Random Forest Gini impurity importance.",
        FeatureSelectionStrategy.MUTUAL_INFO.value: "Measures mutual information entropy dependency against target.",
    }


def get_feature_selector(
    strategy: FeatureSelectionStrategy = FeatureSelectionStrategy.NONE,
    top_k: int = 20,
    **kwargs: Any,
) -> BaseEstimator:
    """Factory creating configured feature selector instance corresponding to strategy."""
    strategy_val = getattr(strategy, "value", str(strategy)).strip().lower()

    if strategy_val in (FeatureSelectionStrategy.NONE.value, "none"):
        return PassThroughSelector()

    elif strategy_val in (FeatureSelectionStrategy.VARIANCE_THRESHOLD.value, "variance_threshold"):
        threshold = kwargs.get("threshold", 0.01)
        return VarianceThresholdSelector(threshold=threshold, min_features=kwargs.get("min_features", 5))

    elif strategy_val in (FeatureSelectionStrategy.CORRELATION_FILTER.value, "correlation_filter"):
        threshold = kwargs.get("threshold", 0.85)
        return CorrelationFilterSelector(threshold=threshold, min_features=kwargs.get("min_features", 8))

    elif strategy_val in (FeatureSelectionStrategy.RF_IMPORTANCE.value, "rf_importance"):
        k = max(1, top_k)
        return RandomForestImportanceSelector(top_k=k, random_state=kwargs.get("random_state", 42))

    elif strategy_val in (FeatureSelectionStrategy.MUTUAL_INFO.value, "mutual_info"):
        k = max(1, top_k)
        return MutualInformationSelector(top_k=k, random_state=kwargs.get("random_state", 42))

    else:
        raise FeatureSelectionError(f"Unsupported feature selection strategy: '{strategy}'")
