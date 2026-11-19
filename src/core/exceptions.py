"""Custom domain exceptions for NetSentry."""

from typing import Any, Dict, Optional


class NetSentryException(Exception):
    """Base exception for all NetSentry domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetValidationError(NetSentryException):
    """Raised when uploaded network traffic dataset fails validation."""
    pass


class ModelTrainingError(NetSentryException):
    """Raised when a machine learning model training job encounters a failure."""
    pass


class ModelNotFoundError(NetSentryException):
    """Raised when a requested trained model artifact is not located."""
    pass


class PredictionError(NetSentryException):
    """Raised when inference on single or batch flow records fails."""
    pass


class FeatureSelectionError(NetSentryException):
    """Raised when a feature selection strategy fails or produces degenerate sets."""
    pass


class StorageError(NetSentryException):
    """Raised when file storage, caching, or serialization encounters an I/O failure."""
    pass


class SecurityViolationError(NetSentryException):
    """Raised when an uploaded file or input payload violates security policies."""
    pass
