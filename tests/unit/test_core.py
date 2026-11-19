"""Unit tests for NetSentry core configuration and security functions."""

import pytest
from pathlib import Path
from src.core.config import Settings, get_settings
from src.core.constants import MLAlgorithm, ThreatSeverity, TrafficClass, ATTACK_CATEGORIES
from src.core.exceptions import SecurityViolationError
from src.core.security import (
    compute_sha256,
    safe_join_path,
    sanitize_filename,
    validate_file_size,
)


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
