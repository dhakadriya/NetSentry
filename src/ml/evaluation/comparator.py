"""Multi-model benchmarking, comparative ranking, and trade-off evaluation engine."""

from typing import Any

import numpy as np


class ModelComparator:
    """Benchmarks and ranks multiple network intrusion models based on detection trade-offs."""

    @staticmethod
    def rank_models(model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ranks models using an enterprise NIDS composite score prioritizing high F1 and minimal FPR."""
        scored = []
        for res in model_results:
            f1 = res.get("f1_score", 0.0)
            fpr = res.get("false_positive_rate", 1.0)
            acc = res.get("accuracy", 0.0)
            pred_time = res.get("prediction_time_ms", 10.0)

            # Latency penalty score (faster is higher score 0 to 1)
            latency_score = 1.0 / (1.0 + np.log1p(max(pred_time, 0.001)))

            # Composite Enterprise Score
            composite = (
                (0.45 * f1)
                + (0.35 * (1.0 - fpr))
                + (0.10 * acc)
                + (0.10 * latency_score)
            )

            res_copy = dict(res)
            res_copy["composite_score"] = round(float(composite), 4)
            scored.append(res_copy)

        # Sort descending by composite score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Assign ranks
        for idx, item in enumerate(scored, 1):
            item["rank"] = idx

        return scored

    @staticmethod
    def get_comparison_matrix(ranked_models: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Generates columnar metric matrix formatted for Plotly multi-bar and radar charts."""
        algorithms = [m.get("algorithm", m.get("model_name", "model")) for m in ranked_models]
        return {
            "algorithms": algorithms,
            "accuracy": [m.get("accuracy", 0.0) for m in ranked_models],
            "f1_score": [m.get("f1_score", 0.0) for m in ranked_models],
            "precision": [m.get("precision", 0.0) for m in ranked_models],
            "recall": [m.get("recall", 0.0) for m in ranked_models],
            "fpr": [m.get("false_positive_rate", 0.0) for m in ranked_models],
            "training_time": [m.get("training_time_seconds", 0.0) for m in ranked_models],
            "prediction_time": [m.get("prediction_time_ms", 0.0) for m in ranked_models],
            "composite_score": [m.get("composite_score", 0.0) for m in ranked_models],
        }
