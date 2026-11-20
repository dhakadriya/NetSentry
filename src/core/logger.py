"""Structured JSON and console logging for NetSentry."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_utc_now_iso() -> str:
    """Returns the current UTC timestamp formatted in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


class JSONFormatter(logging.Formatter):
    """Custom logging formatter outputting structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": get_utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str = "netsentry", log_file: str | None = None) -> logging.Logger:
    """Configures and retrieves a singleton logger instance by name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Console handler with clean human-readable output
        console_handler = logging.StreamHandler(sys.stdout)
        console_fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # File handler with JSON structured logs if log_file is specified
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(path), encoding="utf-8")
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)

    return logger


def log_security_audit(
    logger: logging.Logger,
    event_type: str,
    client_ip: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emits a structured security audit log entry."""
    payload = {
        "audit_event": event_type,
        "client_ip": client_ip,
        "severity": severity,
        "timestamp": get_utc_now_iso(),
        "details": details or {},
    }
    logger.warning(f"SECURITY_AUDIT: {json.dumps(payload)}")


def log_ml_event(
    logger: logging.Logger,
    model_name: str,
    action: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Emits an ML training or inference lifecycle log entry."""
    payload = {
        "model_name": model_name,
        "action": action,
        "timestamp": get_utc_now_iso(),
        "metrics": metrics or {},
    }
    logger.info(f"ML_EVENT: {json.dumps(payload)}")
