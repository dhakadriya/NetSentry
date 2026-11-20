"""Model training studio view for tuning hyperparameters and evaluating classifiers."""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.core.constants import FeatureSelectionStrategy, MLAlgorithm
from src.database.crud.dataset_repo import list_datasets, get_dataset_by_id
from src.database.crud.experiment_repo import create_experiment, get_best_model
from src.frontend.components.charts import render_confusion_matrix_chart
from src.frontend.components.input import render_hyperparameter_controls
from src.ml.trainer import NetSentryTrainer


def render_training_view(db_session) -> None:
    """Renders the interactive machine learning model training studio."""
    st.markdown("### 🧠 Machine Learning Model Training Studio")
    st.markdown("Train, optimize, and benchmark the four supervised classification models.")

    datasets = list_datasets(db_session)
    if not datasets:
        st.warning("Please upload a dataset in the Datasets tab before training a model.")
        return

    col_config, col_results = st.columns([1, 1])

    with col_config:
        st.markdown("#### 1. Configuration & Algorithm Selection")
        ds_map = {f"#{d.id} - {d.filename} ({d.total_rows:,} records)": d.id for d in datasets}
        selected_ds_str = st.selectbox("Training Dataset", list(ds_map.keys()))
        dataset_id = ds_map[selected_ds_str]

        algo_choice = st.selectbox(
            "Machine Learning Algorithm",
            [
                ("random_forest", "Random Forest (High Accuracy Ensemble)"),
                ("decision_tree", "Decision Tree (Interpretable Rules)"),
                ("logistic_regression", "Logistic Regression (Regularized Linear)"),
                ("knn", "K-Nearest Neighbors (Instance-Based)"),
            ],
            format_func=lambda x: x[1],
        )[0]

        st.markdown("#### 2. Feature Selection Strategy (Research Component)")
        feat_choice = st.selectbox(
            "Feature Engineering & Selection",
            [
                ("none", "None (All Features Retained)"),
                ("rf_importance", "Random Forest Gini Importance (Top 20 Features)"),
                ("mutual_info", "Mutual Information Classification Gain"),
                ("correlation_filter", "Correlation Filter (Collinearity Pruning > 0.85)"),
                ("variance_threshold", "Variance Threshold (Remove Near-Zero Variance)"),
            ],
            format_func=lambda x: x[1],
        )[0]

        test_split = st.slider("Test Set Split Ratio", 0.10, 0.40, 0.25, 0.05)
        hyperparams = render_hyperparameter_controls(algo_choice)

        train_btn = st.button("🚀 Train & Evaluate Classifier", type="primary", use_container_width=True)

    with col_results:
        st.markdown("#### 3. Real-Time Evaluation Diagnostics")
        if train_btn:
            target_ds = get_dataset_by_id(db_session, dataset_id)
            if target_ds and Path(target_ds.stored_path).exists():
                with st.spinner(f"Training {algo_choice.title()} with {feat_choice} feature selection..."):
                    df = pd.read_csv(target_ds.stored_path, nrows=25000)
                    trainer = NetSentryTrainer()

                    algo_enum = MLAlgorithm(algo_choice)
                    feat_enum = FeatureSelectionStrategy(feat_choice)

                    model, preprocessor, selector, metrics, artifact_path = trainer.train(
                        df=df,
                        algorithm=algo_enum,
                        feature_strategy=feat_enum,
                        test_size=test_split,
                        hyperparameters=hyperparams,
                        random_seed=42,
                    )

                    # Persist experiment
                    current_best = get_best_model(db_session)
                    is_new_best = current_best is None or metrics.f1_score > (current_best.f1_score or 0.0)

                    exp = create_experiment(
                        db=db_session,
                        model_name=model.model_name,
                        algorithm=algo_choice,
                        dataset_id=target_ds.id,
                        feature_strategy=feat_choice,
                        hyperparameters=hyperparams,
                        accuracy=metrics.accuracy,
                        precision=metrics.precision,
                        recall=metrics.recall,
                        f1_score=metrics.f1_score,
                        false_positive_rate=metrics.false_positive_rate,
                        false_negative_rate=metrics.false_negative_rate,
                        roc_auc=metrics.roc_auc,
                        confusion_matrix=metrics.confusion_matrix,
                        training_time_seconds=metrics.training_time_seconds,
                        prediction_time_ms=metrics.prediction_time_ms,
                        artifact_path=str(artifact_path),
                        selected_features=model.feature_names_,
                        is_best_model=is_new_best,
                        notes=f"Trained via Studio on {target_ds.filename}.",
                    )

                    st.success(f"🎉 Model Trained Successfully! F1-Score: {metrics.f1_score * 100:.2f}%")
                    if is_new_best:
                        st.balloons()
                        st.info("⭐ New Best Intrusion Detection Model Crowned!")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{metrics.accuracy * 100:.1f}%")
                    m2.metric("Precision", f"{metrics.precision * 100:.1f}%")
                    m3.metric("Recall", f"{metrics.recall * 100:.1f}%")
                    m4.metric("F1-Score", f"{metrics.f1_score * 100:.1f}%")

                    m5, m6, m7 = st.columns(3)
                    m5.metric("False Positive Rate", f"{metrics.false_positive_rate * 100:.2f}%")
                    m6.metric("Train Time", f"{metrics.training_time_seconds:.3f}s")
                    m7.metric("Inference Latency", f"{metrics.prediction_time_ms:.2f}ms")

                    render_confusion_matrix_chart(metrics.confusion_matrix)
            else:
                st.error("Selected dataset file could not be accessed.")
        else:
            st.info("Select parameters and click 'Train & Evaluate Classifier' to view live metrics.")
