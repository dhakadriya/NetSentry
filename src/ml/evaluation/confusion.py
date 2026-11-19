"""Confusion matrix formatter and diagnostic analysis utilities."""

from typing import Any, Dict, List
import numpy as np


def format_confusion_matrix(cm: List[List[int]]) -> Dict[str, Any]:
    """Generates structured confusion matrix diagnostics, rates, and normalized percentages."""
    arr = np.array(cm, dtype=int)
    tn = int(arr[0, 0])
    fp = int(arr[0, 1])
    fn = int(arr[1, 0])
    tp = int(arr[1, 1])

    total_actual_normal = tn + fp
    total_actual_malicious = fn + tp
    total_samples = total_actual_normal + total_actual_malicious

    # Row-normalized percentages (Recall per class)
    norm_normal_tn = round((tn / total_actual_normal) * 100.0, 2) if total_actual_normal > 0 else 0.0
    norm_normal_fp = round((fp / total_actual_normal) * 100.0, 2) if total_actual_normal > 0 else 0.0
    norm_attack_fn = round((fn / total_actual_malicious) * 100.0, 2) if total_actual_malicious > 0 else 0.0
    norm_attack_tp = round((tp / total_actual_malicious) * 100.0, 2) if total_actual_malicious > 0 else 0.0

    # Specificity & Sensitivity
    specificity = round(tn / total_actual_normal, 4) if total_actual_normal > 0 else 0.0
    sensitivity = round(tp / total_actual_malicious, 4) if total_actual_malicious > 0 else 0.0

    return {
        "raw_matrix": cm,
        "counts": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "percentages": [
            [norm_normal_tn, norm_normal_fp],
            [norm_attack_fn, norm_attack_tp],
        ],
        "diagnostics": {
            "specificity": specificity,
            "sensitivity": sensitivity,
            "total_tested": total_samples,
            "normal_samples": total_actual_normal,
            "malicious_samples": total_actual_malicious,
        },
        "labels": ["Normal", "Malicious"],
    }
