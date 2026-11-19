"""Unit tests for the four NetSentry machine learning classification algorithms."""

import numpy as np
import pandas as pd
import pytest

from src.core.constants import MLAlgorithm
from src.ml.models.decision_tree import DecisionTreeModel
from src.ml.models.knn import KNNModel
from src.ml.models.logistic_regression import LogisticRegressionModel
from src.ml.models.random_forest import RandomForestModel


@pytest.fixture
def mock_dataset():
    """Generates a small binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 150
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.6, 0.4]))

    X = pd.DataFrame({
        "sbytes": np.where(y == 1, rng.integers(1000, 5000, size=n), rng.integers(100, 1000, size=n)),
        "dbytes": np.where(y == 1, rng.integers(10, 200, size=n), rng.integers(200, 2000, size=n)),
        "dur": rng.uniform(0.001, 2.0, size=n),
        "sttl": np.where(y == 1, 254, 64),
    })
    return X, y


def test_logistic_regression(mock_dataset):
    """Verify Logistic Regression training, prediction, and coefficients."""
    X, y = mock_dataset
    model = LogisticRegressionModel(C=1.0, max_iter=200, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    coefs = model.get_feature_coefficients()

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert model.training_time_seconds >= 0.0
    assert len(coefs) == len(X.columns)


def test_decision_tree(mock_dataset):
    """Verify Decision Tree training, prediction, and feature importances."""
    X, y = mock_dataset
    model = DecisionTreeModel(max_depth=5, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    importances = model.get_feature_importances()

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert len(importances) == len(X.columns)
    assert sum(importances.values()) > 0.0


def test_random_forest_and_serialization(mock_dataset, tmp_path):
    """Verify Random Forest ensemble training and joblib persistence."""
    X, y = mock_dataset
    model = RandomForestModel(n_estimators=20, max_depth=6, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)

    # Test serialization
    model_path = tmp_path / "test_rf.joblib"
    saved_path = model.save(model_path)
    assert saved_path.exists()

    loaded_model = RandomForestModel.load(saved_path)
    loaded_preds = loaded_model.predict(X)
    assert np.array_equal(preds, loaded_preds)


def test_knn_classifier(mock_dataset):
    """Verify KNN classifier training, prediction, and distance-weighted probabilities."""
    X, y = mock_dataset
    model = KNNModel(n_neighbors=3, weights="distance")
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert (probs[:, 0] + probs[:, 1] <= 1.0001).all()
