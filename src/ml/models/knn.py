"""K-Nearest Neighbors (KNN) instance-based classifier implementation."""

from typing import Any

from sklearn.neighbors import KNeighborsClassifier

from src.core.constants import MLAlgorithm
from src.ml.models.base import BaseNetSentryModel


class KNNModel(BaseNetSentryModel):
    """K-Nearest Neighbors classifier for instance-based intrusion pattern recognition."""

    def __init__(
        self,
        model_name: str = "KNN-NIDS",
        n_neighbors: int = 5,
        weights: str = "distance",
        algorithm_type: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        neighbors = max(1, int(n_neighbors))
        super().__init__(
            model_name=model_name,
            algorithm=MLAlgorithm.KNN,
            n_neighbors=neighbors,
            weights=weights,
            algorithm_type=algorithm_type,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.n_neighbors = neighbors
        self.weights = weights
        self.algorithm_type = algorithm_type
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.n_jobs = n_jobs

    def _create_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm_type,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
