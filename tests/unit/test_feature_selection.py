"""Unit tests for feature selection strategies and research selector factory."""

import numpy as np
import pandas as pd
import pytest

from src.core.constants import FeatureSelectionStrategy
from src.ml.feature_selection.correlation_filter import CorrelationFilterSelector
from src.ml.feature_selection.importance_selector import RandomForestImportanceSelector
from src.ml.feature_selection.mutual_info import MutualInformationSelector
from src.ml.feature_selection.selector_factory import get_feature_selector
from src.ml.feature_selection.variance_threshold import VarianceThresholdSelector


@pytest.fixture
def synthetic_features():
    """Generates synthetic features including zero variance, redundant, and informative features."""
    rng = np.random.default_rng(42)
    n = 200

    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]))

    # Informative
    f_informative1 = y * 3.0 + rng.normal(0, 0.5, size=n)
    f_informative2 = y * (-2.0) + rng.normal(0, 0.5, size=n)

    # Correlated with f_informative1
    f_collinear = f_informative1 * 1.5 + rng.normal(0, 0.01, size=n)

    # Constant / zero variance
    f_constant = np.zeros(n)

    # Low variance
    f_low_var = rng.choice([0.0, 0.0001], size=n, p=[0.99, 0.01])

    # Random noise
    f_noise1 = rng.normal(0, 1, size=n)
    f_noise2 = rng.normal(0, 1, size=n)
    f_noise3 = rng.normal(0, 1, size=n)

    X = pd.DataFrame({
        "info1": f_informative1,
        "info2": f_informative2,
        "collinear": f_collinear,
        "constant": f_constant,
        "low_var": f_low_var,
        "noise1": f_noise1,
        "noise2": f_noise2,
        "noise3": f_noise3,
    })
    return X, y


def test_variance_threshold_selector(synthetic_features):
    """Verify zero and near-zero variance features are eliminated."""
    X, y = synthetic_features
    selector = VarianceThresholdSelector(threshold=0.01, min_features=3)
    X_trans = selector.fit_transform(X)

    assert "constant" not in X_trans.columns
    assert "low_var" not in X_trans.columns
    assert "info1" in X_trans.columns


def test_correlation_filter_selector(synthetic_features):
    """Verify collinear feature is pruned."""
    X, y = synthetic_features
    selector = CorrelationFilterSelector(threshold=0.85, min_features=3)
    X_trans = selector.fit_transform(X)

    assert not ("info1" in X_trans.columns and "collinear" in X_trans.columns)
    assert len(X_trans.columns) < len(X.columns)


def test_rf_importance_selector(synthetic_features):
    """Verify RandomForest selector ranks informative features highest."""
    X, y = synthetic_features
    selector = RandomForestImportanceSelector(top_k=3, random_state=42)
    X_trans = selector.fit_transform(X, y)

    assert len(X_trans.columns) == 3
    # Informative features should be in top 3
    assert "info1" in selector.selected_features_ or "collinear" in selector.selected_features_


def test_mutual_info_selector(synthetic_features):
    """Verify mutual information selector identifies correlated signal."""
    X, y = synthetic_features
    selector = MutualInformationSelector(top_k=3, random_state=42)
    X_trans = selector.fit_transform(X, y)

    assert len(X_trans.columns) == 3
    assert len(selector.mutual_info_scores_) == len(X.columns)


def test_selector_factory_all_strategies(synthetic_features):
    """Verify factory instantiates all supported strategies."""
    X, y = synthetic_features

    for strat in FeatureSelectionStrategy:
        selector = get_feature_selector(strat, top_k=4)
        if strat in (FeatureSelectionStrategy.RF_IMPORTANCE, FeatureSelectionStrategy.MUTUAL_INFO):
            X_out = selector.fit_transform(X, y)
        else:
            X_out = selector.fit_transform(X)
        assert len(X_out) == len(X)
        assert len(X_out.columns) > 0
