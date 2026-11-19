"""FastAPI dependency injection providers."""

from typing import Generator
from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.core.config import Settings, get_settings
from src.database.connection import get_db
from src.ml.trainer import NetSentryTrainer


def get_app_settings() -> Settings:
    """Provides application configuration instance."""
    return get_settings()


def get_db_session(db: Session = Depends(get_db)) -> Generator[Session, None, None]:
    """Provides an active database session for route handlers."""
    yield db


def get_trainer(settings: Settings = Depends(get_app_settings)) -> NetSentryTrainer:
    """Provides an instantiated NetSentry model trainer."""
    return NetSentryTrainer(settings=settings)


def get_client_ip(request: Request) -> str:
    """Extracts client IP address safely, checking proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"
