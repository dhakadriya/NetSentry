"""Decision Tree classifier implementation with tree depth and split optimization."""

from typing import Dict, Optional
from sklearn.tree import DecisionTreeClassifier

from src.core.constants import MLAlgorithm
from src.ml.models.base import BaseNetSentryModel


class DecisionTreeModel(BaseNetSentryModel):
    """Decision Tree classifier offering rule-based interpretability for intrusion patterns."""

    def __init__(
        self,
        model_name: str = "DecisionTree-NIDS",
        criterion: str = "gini",
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            algorithm=MLAlgorithm.DECISION_TREE,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

    def _create_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns Gini importance for each network traffic feature."""
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
