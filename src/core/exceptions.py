"""Custom domain exceptions for NetSentry."""

from typing import Any


class NetSentryException(Exception):
    """Base exception for all NetSentry domain errors."""
    status_code: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serializes error to structured JSON payload dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class DatasetValidationError(NetSentryException):
    """Raised when uploaded network traffic dataset fails validation."""
    status_code = 400


class ModelTrainingError(NetSentryException):
    """Raised when a machine learning model training job encounters a failure."""
    status_code = 500


class ModelNotFoundError(NetSentryException):
    """Raised when a requested trained model artifact is not located."""
    status_code = 404


class ModelNotTrainedError(NetSentryException):
    """Raised when inference is attempted on an untrained or unfitted model."""
    status_code = 400


class PredictionError(NetSentryException):
    """Raised when inference on single or batch flow records fails."""
    status_code = 500


class FeatureSelectionError(NetSentryException):
    """Raised when a feature selection strategy fails or produces degenerate sets."""
    status_code = 422


class FeatureDimensionMismatchError(NetSentryException):
    """Raised when inference input feature count does not match trained model expectations."""
    status_code = 422


class UnsupportedProtocolError(NetSentryException):
    """Raised when an unrecognized network protocol or service string is provided."""
    status_code = 422


class RateLimitExceededError(NetSentryException):
    """Raised when an IP or client exceeds sliding-window request quotas."""
    status_code = 429


class StorageError(NetSentryException):
    """Raised when file storage, caching, or serialization encounters an I/O failure."""
    status_code = 500


class SecurityViolationError(NetSentryException):
    """Raised when an uploaded file or input payload violates security policies."""
    status_code = 400
