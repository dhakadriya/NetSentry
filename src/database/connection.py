"""Database connection, engine configuration, and session management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger("netsentry.db")


def get_engine():
    """Initializes SQLAlchemy engine with appropriate settings."""
    settings = get_settings()
    db_url = settings.database_url

    connect_args = {}
    poolclass = None

    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # If in-memory or file path, ensure directory exists
        if "///" in db_url and not db_url.endswith(":memory:"):
            db_path = db_url.split("///")[1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        if db_url == "sqlite:///:memory:":
            poolclass = StaticPool

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        poolclass=poolclass if poolclass else None,
        echo=False,
    )
    return engine


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency generator for database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions in non-request code."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
