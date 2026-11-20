"""Model comparison and 4-algorithm benchmark view."""

import streamlit as st
import pandas as pd

from src.database.crud.dataset_repo import list_datasets, get_dataset_by_id
from src.database.crud.experiment_repo import get_leaderboard
from src.frontend.components.charts import render_algorithm_comparison_chart
from src.ml.evaluation.comparator import ModelComparator


def render_comparison_view(db_session) -> None:
    """Renders the 4-algorithm benchmark comparison view."""
    st.markdown("### ⚖️ Multi-Model Comparison & Benchmark Leaderboard")
    st.markdown("Side-by-side empirical performance comparison across Logistic Regression, Decision Tree, Random Forest, and KNN.")

    experiments = get_leaderboard(db_session, limit=20)

    if not experiments:
        st.info("No trained models recorded yet. Train models in the Training Studio to populate leaderboard.")
        return

    # Prepare data for comparator
    raw_results = []
    for exp in experiments:
        raw_results.append({
            "id": exp.id,
            "model_name": exp.model_name,
            "algorithm": exp.algorithm.replace("_", " ").title(),
            "f1_score": exp.f1_score or 0.0,
            "false_positive_rate": exp.false_positive_rate or 0.0,
            "accuracy": exp.accuracy or 0.0,
            "precision": exp.precision or 0.0,
            "recall": exp.recall or 0.0,
            "training_time_seconds": exp.training_time_seconds or 0.0,
            "prediction_time_ms": exp.prediction_time_ms or 0.0,
            "strategy": exp.feature_strategy,
            "is_best": "⭐ Best" if exp.is_best_model else "",
        })

    ranked = ModelComparator.rank_models(raw_results)
    chart_matrix = ModelComparator.get_comparison_matrix(ranked)

    # 1. Bar Chart Comparison
    render_algorithm_comparison_chart(chart_matrix)

    st.markdown("---")
    st.markdown("#### 🏆 Leaderboard (Ranked by Enterprise Security Composite Score)")
    st.caption("Composite score weights F1 (45%), low FPR (35%), Accuracy (10%), and low Latency (10%).")

    table_data = []
    for r in ranked:
        table_data.append({
            "Rank": f"#{r['rank']}",
            "Model Name": r["model_name"],
            "Algorithm": r["algorithm"],
            "Composite Score": f"{r['composite_score']:.3f}",
            "F1-Score": f"{r['f1_score'] * 100:.2f}%",
            "Accuracy": f"{r['accuracy'] * 100:.2f}%",
            "FPR": f"{r['false_positive_rate'] * 100:.2f}%",
            "Inference Latency": f"{r['prediction_time_ms']:.2f}ms",
            "Train Time": f"{r['training_time_seconds']:.2f}s",
            "Feature Strategy": r["strategy"],
            "Status": r["is_best"],
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
