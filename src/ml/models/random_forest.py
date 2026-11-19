"""Random Forest ensemble classifier implementation with class weight balancing."""

from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier

from src.core.constants import MLAlgorithm
from src.ml.models.base import BaseNetSentryModel


class RandomForestModel(BaseNetSentryModel):
    """Random Forest ensemble classifier providing high robustness and resilience to noise."""

    def __init__(
        self,
        model_name: str = "RandomForest-NIDS",
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            algorithm=MLAlgorithm.RANDOM_FOREST,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _create_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns ensemble-wide average feature importance scores."""
        if self.model_ is None or not hasattr(self.model_, "feature_importances_"):
            return {}
        importances = self.model_.feature_importances_
        return dict(
            sorted(
                zip(self.feature_names_, [float(x) for x in importances]),
                key=lambda x: x[1],
                reverse=True,
            )
        )
