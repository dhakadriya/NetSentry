"""Application configuration management for NetSentry."""

import os
from functools import lru_cache
from pathlib import Path

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings, Field  # type: ignore
    except ImportError:
        # Fallback dataclass if pydantic is not installed in current environment
        from dataclasses import dataclass

        @dataclass
        class BaseSettings:  # type: ignore
            pass

        def Field(default=None, **kwargs):  # type: ignore
            return default


class Settings(BaseSettings):
    """Central configuration parameters for NetSentry NIDS."""

    app_name: str = "NetSentry"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # API and Networking
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    allowed_origins: list[str] | str = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")

    # Storage Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", "models/saved"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports/generated"))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", "logs"))

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/netsentry.db")

    # Security & Limits
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120"))
    max_batch_rows: int = int(os.getenv("MAX_BATCH_ROWS", "200000"))

    # Machine Learning Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "random_forest")
    default_test_size: float = float(os.getenv("DEFAULT_TEST_SIZE", "0.25"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    def ensure_directories(self) -> None:
        """Create necessary runtime directories if they do not exist."""
        for directory in [self.upload_dir, self.models_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
