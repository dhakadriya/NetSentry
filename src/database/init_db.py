"""Database schema initialization and migration script."""

from src.core.logger import get_logger
from src.database.base import Base
from src.database.connection import engine
# Import all models to ensure registration in Base.metadata
from src.database.models.dataset import DatasetRecord  # noqa: F401
from src.database.models.model_experiment import ModelExperiment  # noqa: F401
from src.database.models.prediction_job import PredictionJob  # noqa: F401
from src.database.models.traffic_alert import TrafficAlert  # noqa: F401
from src.database.models.audit_log import AuditLog  # noqa: F401

logger = get_logger("netsentry.init_db")


def init_database(db_engine=None) -> None:
    """Initializes the database schema by creating all registered tables."""
    target_engine = db_engine or engine
    logger.info("Initializing NetSentry database schema...")
    Base.metadata.create_all(bind=target_engine)
    logger.info("Database schema initialized successfully with tables: %s", list(Base.metadata.tables.keys()))


if __name__ == "__main__":
    init_database()
