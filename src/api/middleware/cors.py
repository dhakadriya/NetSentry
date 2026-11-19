"""Cross-Origin Resource Sharing (CORS) security configuration."""

from typing import List, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import Settings


def setup_cors(app: FastAPI, settings: Settings) -> None:
    """Configures CORS policies for safe browser communication with Streamlit frontend."""
    origins: List[str] = []
    if isinstance(settings.allowed_origins, str):
        origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    elif isinstance(settings.allowed_origins, list):
        origins = settings.allowed_origins

    if not origins:
        origins = ["http://localhost:8501", "http://127.0.0.1:8501"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms", "X-RateLimit-Remaining"],
    )
