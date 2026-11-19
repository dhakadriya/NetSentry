"""FastAPI application factory, lifespan lifecycle manager, and router configuration."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from src.core.config import get_settings
from src.core.logger import get_logger
from src.database.init_db import init_database
from src.database.seed import seed_database
from src.api.middleware.cors import setup_cors
from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.logging_middleware import RequestLoggingMiddleware
from src.api.middleware.rate_limiter import InMemoryRateLimiterMiddleware

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
