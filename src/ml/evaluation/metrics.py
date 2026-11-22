"""Network intrusion detection evaluation metrics calculation."""

from typing import List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.schemas.model import ModelEvaluationMetrics


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates True Negative Rate (Specificity) to measure benign traffic preservation."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Matthews Correlation Coefficient (MCC) for imbalanced cyber threat datasets."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    try:
        return float(matthews_corrcoef(y_t, y_p))
    except Exception:
        return 0.0


def calculate_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    training_time_seconds: float = 0.0,
    prediction_time_ms: float = 0.0,
    selected_features_count: int = 0,
) -> ModelEvaluationMetrics:
    """Calculates comprehensive cybersecurity evaluation metrics including FPR and FNR."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core rates
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Cybersecurity specific rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC AUC
    auc_val = None
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                auc_val = float(roc_auc_score(y_t, y_prob[:, 1]))
            else:
                auc_val = float(roc_auc_score(y_t, y_prob))
        except Exception:
            auc_val = None

    return ModelEvaluationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        roc_auc=round(auc_val, 4) if auc_val is not None else None,
        confusion_matrix=cm.tolist(),
        training_time_seconds=round(training_time_seconds, 4),
        prediction_time_ms=round(prediction_time_ms, 4),
        selected_features_count=selected_features_count,
    )
