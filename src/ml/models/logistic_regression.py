"""Logistic Regression classifier implementation with regularization and coefficient analysis."""

from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression

from src.core.constants import MLAlgorithm
from src.ml.models.base import BaseNetSentryModel


class LogisticRegressionModel(BaseNetSentryModel):
    """Logistic Regression classifier for linear boundary network intrusion detection."""

    def __init__(
        self,
        model_name: str = "LogisticRegression-NIDS",
        C: float = 1.0,
        penalty: Optional[str] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: str = "balanced",
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            algorithm=MLAlgorithm.LOGISTIC_REGRESSION,
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs,
        )
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state

    def _create_estimator(self) -> LogisticRegression:
        params: Dict[str, Any] = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }
        if self.penalty is not None:
            params["penalty"] = self.penalty
        return LogisticRegression(**params)

    def get_feature_coefficients(self) -> Dict[str, float]:
        """Returns feature weights / coefficients learned by the model sorted by magnitude."""
        if self.model_ is None or not hasattr(self.model_, "coef_"):
            return {}
        coefs = self.model_.coef_[0]
        return dict(sorted(zip(self.feature_names_, coefs), key=lambda x: abs(x[1]), reverse=True))
