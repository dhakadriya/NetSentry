"""Unit tests for NetSentry core configuration, constants, logging, and security functions."""

import pytest
import logging
from pathlib import Path
from src.core.config import Settings, get_settings
from src.core.constants import (
    MLAlgorithm,
    ThreatSeverity,
    TrafficClass,
    ATTACK_CATEGORIES,
    MITRE_ATTACK_MAPPING,
    DEFAULT_SERVICE_PORTS,
)
from src.core.exceptions import (
    SecurityViolationError,
    UnsupportedProtocolError,
    FeatureDimensionMismatchError,
    ModelNotTrainedError,
    RateLimitExceededError,
)
from src.core.security import (
    compute_sha256,
    safe_join_path,
    sanitize_filename,
    validate_file_size,
    validate_mime_type,
    is_safe_path,
    strip_ansi_sequences,
)
from src.core.logger import get_logger, log_security_audit, log_ml_event, get_utc_now_iso


def test_settings_initialization():
    """Verify settings loads with valid defaults and directories."""
    settings = get_settings()
    assert settings.app_name == "NetSentry"
    assert settings.default_model == "random_forest"
    assert settings.api_port == 8000
    assert isinstance(settings.upload_dir, Path)


def test_sanitize_filename_valid():
    """Verify clean CSV filename is accepted."""
    clean = sanitize_filename("network_traffic_sample.csv")
    assert clean == "network_traffic_sample.csv"


def test_sanitize_filename_traversal_elimination():
    """Verify directory traversal tokens are stripped or blocked."""
    clean = sanitize_filename("../../etc/passwd.csv")
    assert ".." not in clean
    assert clean.endswith(".csv")


def test_sanitize_filename_invalid_extension():
    """Verify non-CSV files are rejected."""
    with pytest.raises(SecurityViolationError):
        sanitize_filename("malicious_script.exe")

    with pytest.raises(SecurityViolationError):
        sanitize_filename("exploit.sh")


def test_safe_join_path_prevents_escape(tmp_path):
    """Verify safe_join_path blocks attempts to break out of base directory."""
    with pytest.raises(SecurityViolationError):
        safe_join_path(tmp_path, "../outside.csv")

    valid_path = safe_join_path(tmp_path, "subfolder/valid.csv")
    assert valid_path == (tmp_path / "subfolder/valid.csv").resolve()


def test_validate_file_size():
    """Verify file size validation bounds."""
    # Under limit
    validate_file_size(1024 * 1024, max_size_mb=10)

    # Exceed limit
    with pytest.raises(SecurityViolationError):
        validate_file_size(20 * 1024 * 1024, max_size_mb=10)


def test_compute_sha256():
    """Verify consistent sha256 checksum generation."""
    data = b"network intrusion detection test payload"
    checksum = compute_sha256(data)
    assert len(checksum) == 64
    assert checksum == compute_sha256(data)


def test_constants_definitions():
    """Verify core constants and attack categories."""
    assert MLAlgorithm.RANDOM_FOREST.value == "random_forest"
    assert MLAlgorithm.KNN.value == "knn"
    assert ThreatSeverity.CRITICAL.value == "critical"
    assert TrafficClass.NORMAL.value == 0
    assert TrafficClass.MALICIOUS.value == 1
    assert "DoS" in ATTACK_CATEGORIES
    assert "Exploits" in ATTACK_CATEGORIES


def test_mitre_mapping_and_service_ports():
    """Verify MITRE ATT&CK technique IDs and port definitions."""
    assert "DoS" in MITRE_ATTACK_MAPPING
    assert MITRE_ATTACK_MAPPING["DoS"]["technique_id"] == "T1498"
    assert MITRE_ATTACK_MAPPING["Reconnaissance"]["technique_id"] == "T1595"
    assert DEFAULT_SERVICE_PORTS["http"] == 80
    assert DEFAULT_SERVICE_PORTS["ssl"] == 443


def test_mime_type_validation_and_is_safe_path(tmp_path):
    """Verify MIME-type security checking and boolean path safety helper."""
    validate_mime_type("text/csv")
    validate_mime_type("text/csv; charset=utf-8")
    validate_mime_type("application/vnd.ms-excel")

    with pytest.raises(SecurityViolationError):
        validate_mime_type("application/x-sh")

    with pytest.raises(SecurityViolationError):
        validate_mime_type("")

    assert is_safe_path(tmp_path, "safe_dir/test.csv") is True
    assert is_safe_path(tmp_path, "../../etc/shadow") is False

    dirty_text = "\x1b[31mRed Alert\x1b[0m"
    assert strip_ansi_sequences(dirty_text) == "Red Alert"


def test_domain_exceptions_structure():
    """Verify status codes and serialization on domain exceptions."""
    err = UnsupportedProtocolError("Protocol rawip is not supported", details={"proto": "rawip"})
    assert err.status_code == 422
    payload = err.to_dict()
    assert payload["error_type"] == "UnsupportedProtocolError"
    assert payload["details"]["proto"] == "rawip"

    dim_err = FeatureDimensionMismatchError("Feature count mismatch", details={"expected": 42, "got": 10})
    assert dim_err.status_code == 422

    untrained_err = ModelNotTrainedError("Model has not been fitted")
    assert untrained_err.status_code == 400

    rate_err = RateLimitExceededError("Too many requests")
    assert rate_err.status_code == 429


def test_structured_logger_and_events():
    """Verify audit and ML lifecycle log events."""
    logger = get_logger("test_netsentry_core")
    assert isinstance(logger, logging.Logger)
    assert get_utc_now_iso().endswith("+00:00") or "T" in get_utc_now_iso()

    # Verify helper does not throw exception
    log_security_audit(logger, "LOGIN_ATTEMPT", "192.168.1.100", "low")
    log_ml_event(logger, "RandomForest", "TRAINING_COMPLETE", {"f1_score": 0.98})
