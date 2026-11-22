"""FastAPI application factory, lifespan lifecycle manager, and router configuration."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.middleware.cors import setup_cors
from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.logging_middleware import RequestLoggingMiddleware
from src.api.middleware.rate_limiter import InMemoryRateLimiterMiddleware
from src.api.routes.alerts import router as alerts_router
from src.api.routes.compare import router as compare_router
from src.api.routes.datasets import router as datasets_router
from src.api.routes.eda import router as eda_router
from src.api.routes.export import router as export_router

# Route imports
from src.api.routes.health import router as health_router
from src.api.routes.models import router as models_router
from src.api.routes.predict_batch import router as predict_batch_router
from src.api.routes.predict_single import router as predict_single_router
from src.core.config import get_settings
from src.core.logger import get_logger
from src.database.init_db import init_database
from src.database.seed import seed_database

logger = get_logger("netsentry.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    settings = get_settings()
    settings.ensure_directories()
    logger.info("Starting up %s v%s...", settings.app_name, settings.app_version)

    # Initialize database tables and seed sample data if empty
    try:
        init_database()
        seed_database()
    except Exception as e:
        logger.warning("Database startup routine notice: %s", str(e))

    yield

    logger.info("Shutting down %s...", settings.app_name)


def create_app() -> FastAPI:
    """Creates and configures a new FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="NetSentry API",
        description=(
            "Intelligent Machine Learning-Based Network Intrusion Detection System (NIDS) "
            "REST API for dataset ingestion, model training, benchmarking, and real-time flow classification."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS
    setup_cors(app, settings)

    # 2. Rate Limiting
    app.add_middleware(
        InMemoryRateLimiterMiddleware,
        max_requests_per_minute=settings.rate_limit_per_minute,
    )

    # 3. Request Logging & Execution Timing
    app.add_middleware(RequestLoggingMiddleware)

    # 4. Error Handling
    register_error_handlers(app)

    # 5. Route Registration
    app.include_router(health_router)
    app.include_router(datasets_router)
    app.include_router(eda_router)
    app.include_router(models_router)
    app.include_router(compare_router)
    app.include_router(predict_single_router)
    app.include_router(predict_batch_router)
    app.include_router(alerts_router)
    app.include_router(export_router)

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
