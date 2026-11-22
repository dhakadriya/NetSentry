"""End-to-end model training orchestrator managing preprocessing, feature selection, evaluation, and persistence."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import get_settings
from src.core.constants import FeatureSelectionStrategy, MLAlgorithm
from src.core.exceptions import ModelTrainingError
from src.core.logger import get_logger
from src.ml.evaluation.metrics import calculate_evaluation_metrics
from src.ml.feature_selection.selector_factory import get_feature_selector
from src.ml.models.base import BaseNetSentryModel
from src.ml.models.decision_tree import DecisionTreeModel
from src.ml.models.knn import KNNModel
from src.ml.models.logistic_regression import LogisticRegressionModel
from src.ml.models.random_forest import RandomForestModel
from src.ml.preprocessing.pipeline import NetSentryPreprocessor
from src.schemas.model import ModelEvaluationMetrics

logger = get_logger("netsentry.trainer")


class NetSentryTrainer:
    """Orchestrates end-to-end model training, feature selection, and model persistence."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def _instantiate_model(
        self,
        algorithm: MLAlgorithm | str,
        model_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> BaseNetSentryModel:
        """Instantiates algorithm class with hyperparameter bindings."""
        algo_str = str(algorithm).lower()
        params = hyperparameters or {}
        name = model_name or f"{algo_str.replace('_', '-').title()}-{datetime.now(timezone.utc).strftime('%H%M%S')}"

        if algo_str in (MLAlgorithm.LOGISTIC_REGRESSION.value, "logistic_regression"):
            return LogisticRegressionModel(
                model_name=name,
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                solver=params.get("solver", "lbfgs"),
                max_iter=params.get("max_iter", 1000),
                random_state=random_seed,
            )
        elif algo_str in (MLAlgorithm.DECISION_TREE.value, "decision_tree"):
            return DecisionTreeModel(
                model_name=name,
                criterion=params.get("criterion", "gini"),
                max_depth=params.get("max_depth", 12),
                min_samples_split=params.get("min_samples_split", 5),
                random_state=random_seed,
            )
        elif algo_str in (MLAlgorithm.RANDOM_FOREST.value, "random_forest"):
            return RandomForestModel(
                model_name=name,
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 16),
                min_samples_split=params.get("min_samples_split", 4),
                random_state=random_seed,
                n_jobs=-1,
            )
        elif algo_str in (MLAlgorithm.KNN.value, "knn"):
            return KNNModel(
                model_name=name,
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "distance"),
                n_jobs=-1,
            )
        else:
            raise ModelTrainingError(f"Unsupported algorithm type: '{algorithm}'")

    def train(
        self,
        df: pd.DataFrame,
        algorithm: MLAlgorithm | str = MLAlgorithm.RANDOM_FOREST,
        feature_strategy: FeatureSelectionStrategy | str = FeatureSelectionStrategy.NONE,
        test_size: float = 0.25,
        hyperparameters: dict[str, Any] | None = None,
        random_seed: int = 42,
        model_name: str | None = None,
    ) -> tuple[BaseNetSentryModel, NetSentryPreprocessor, Any, ModelEvaluationMetrics, Path]:
        """Executes full training pipeline and persists model package."""
        logger.info(
            "Starting model training [algo=%s, strategy=%s, rows=%d]",
            algorithm,
            feature_strategy,
            len(df),
        )

        # 1. Fit preprocessor and split X, y
        preprocessor = NetSentryPreprocessor(scaling_method="standard")
        preprocessor.fit(df)
        X, y, attack_cats = preprocessor.split_X_y(df)

        if y is None:
            raise ModelTrainingError("Target column 'label' not found in dataset for training.")

        # 2. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y if len(y.unique()) > 1 else None,
        )

        # 3. Feature Selection
        selector = get_feature_selector(feature_strategy, top_k=20, random_state=random_seed)
        if str(feature_strategy).lower() in (
            FeatureSelectionStrategy.RF_IMPORTANCE.value,
            FeatureSelectionStrategy.MUTUAL_INFO.value,
            "rf_importance",
            "mutual_info",
        ):
            X_train_sel = selector.fit_transform(X_train, y_train)
        else:
            X_train_sel = selector.fit_transform(X_train)

        X_test_sel = selector.transform(X_test)
        selected_feature_names = list(X_train_sel.columns)

        # 4. Instantiate & Train Model
        model = self._instantiate_model(
            algorithm=algorithm,
            model_name=model_name,
            hyperparameters=hyperparameters,
            random_seed=random_seed,
        )
        model.fit(X_train_sel, y_train)

        # 5. Evaluate on Test Set
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)

        metrics = calculate_evaluation_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            training_time_seconds=model.training_time_seconds,
            prediction_time_ms=model.prediction_time_ms,
            selected_features_count=len(selected_feature_names),
        )
        model.metrics_ = metrics.model_dump()

        # 6. Persist Model Bundle
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = model.model_name.lower().replace(" ", "_").replace("-", "_")
        artifact_filename = f"{safe_name}_{timestamp}.joblib"
        artifact_path = self.settings.models_dir / artifact_filename

        # Save bundle containing model, preprocessor, and feature selector
        bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "selector": selector,
            "selected_features": selected_feature_names,
            "metrics": metrics.model_dump(),
            "algorithm": str(algorithm),
            "feature_strategy": str(feature_strategy),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        import joblib
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, artifact_path)

        logger.info(
            "Trained %s successfully: F1=%.4f, FPR=%.4f in %.2fs. Saved to %s",
            model.model_name,
            metrics.f1_score,
            metrics.false_positive_rate,
            model.training_time_seconds,
            artifact_path,
        )

        return model, preprocessor, selector, metrics, artifact_path
