"""Global exception handlers converting domain and runtime exceptions to structured JSON."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    DatasetValidationError,
    ModelNotFoundError,
    ModelTrainingError,
    NetSentryException,
    PredictionError,
    SecurityViolationError,
)
from src.core.logger import get_logger

logger = get_logger("netsentry.api.errors")


def register_error_handlers(app: FastAPI) -> None:
    """Registers exception handlers on the FastAPI application instance."""

    @app.exception_handler(DatasetValidationError)
    async def dataset_validation_handler(request: Request, exc: DatasetValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "DatasetValidationError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(SecurityViolationError)
    async def security_violation_handler(request: Request, exc: SecurityViolationError):
        logger.warning("Security violation: %s from %s", exc.message, request.client.host if request.client else "unknown")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "SecurityViolationError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "ModelNotFoundError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ModelTrainingError)
    async def model_training_handler(request: Request, exc: ModelTrainingError):
        logger.error("Model training failed: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "ModelTrainingError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "PredictionError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(NetSentryException)
    async def generic_netsentry_handler(request: Request, exc: NetSentryException):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Invalid request parameters.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred during processing.",
            },
        )
