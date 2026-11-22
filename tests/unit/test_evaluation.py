"""Unit tests for intrusion evaluation metrics, confusion diagnostics, and model ranking."""

import numpy as np

from src.ml.evaluation.comparator import ModelComparator
from src.ml.evaluation.confusion import format_confusion_matrix
from src.ml.evaluation.metrics import calculate_evaluation_metrics


def test_calculate_evaluation_metrics():
    """Verify precision, recall, f1, fpr, and fnr calculations."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    # TN=3, FP=1, FN=1, TP=3

    metrics = calculate_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        training_time_seconds=0.05,
        prediction_time_ms=0.2,
    )

    assert metrics.accuracy == 0.75
    assert metrics.precision == 0.75  # 3 / (3 + 1)
    assert metrics.recall == 0.75     # 3 / (3 + 1)
    assert metrics.f1_score == 0.75
    assert metrics.false_positive_rate == 0.25  # 1 / (1 + 3)
    assert metrics.false_negative_rate == 0.25  # 1 / (1 + 3)
    assert metrics.confusion_matrix == [[3, 1], [1, 3]]


def test_format_confusion_matrix():
    """Verify row-normalized percentages and diagnostics extraction."""
    cm = [[90, 10], [5, 95]]
    diag = format_confusion_matrix(cm)

    assert diag["counts"]["true_negatives"] == 90
    assert diag["counts"]["false_positives"] == 10
    assert diag["counts"]["false_negatives"] == 5
    assert diag["counts"]["true_positives"] == 95
    assert diag["percentages"][0][0] == 90.0
    assert diag["percentages"][0][1] == 10.0
    assert diag["percentages"][1][1] == 95.0
    assert diag["diagnostics"]["specificity"] == 0.90
    assert diag["diagnostics"]["sensitivity"] == 0.95


def test_model_comparator_ranking():
    """Verify model comparator correctly weights F1 and FPR to rank models."""
    results = [
        {
            "model_name": "High-FPR-Model",
            "algorithm": "model_a",
            "f1_score": 0.92,
            "false_positive_rate": 0.15,
            "accuracy": 0.91,
            "prediction_time_ms": 2.0,
        },
        {
            "model_name": "Low-FPR-Model",
            "algorithm": "model_b",
            "f1_score": 0.94,
            "false_positive_rate": 0.02,
            "accuracy": 0.95,
            "prediction_time_ms": 1.0,
        },
    ]

    ranked = ModelComparator.rank_models(results)
    assert ranked[0]["model_name"] == "Low-FPR-Model"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2

    matrix = ModelComparator.get_comparison_matrix(ranked)
    assert len(matrix["algorithms"]) == 2
    assert "composite_score" in matrix
