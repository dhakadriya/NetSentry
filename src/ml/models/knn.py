"""K-Nearest Neighbors (KNN) instance-based classifier implementation."""

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
        metric: str = "minkowski",
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            algorithm=MLAlgorithm.KNN,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm_type=algorithm_type,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm_type = algorithm_type
        self.metric = metric
        self.n_jobs = n_jobs

    def _create_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm_type,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
