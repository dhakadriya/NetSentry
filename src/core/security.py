"""Security utilities for file upload validation, path sanitization, and integrity checks."""

import hashlib
import re
from pathlib import Path

from src.core.exceptions import SecurityViolationError

ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/csv",
    "text/x-csv",
}
MAX_FILENAME_LENGTH = 255


def validate_mime_type(content_type: str | None) -> None:
    """Validates that the uploaded HTTP content type is a recognized CSV format.
    
    Raises:
        SecurityViolationError: If content_type is missing or not in ALLOWED_MIME_TYPES.
    """
    if not content_type:
        raise SecurityViolationError("Content-Type header is required for file uploads.")

    normalized = content_type.split(";")[0].strip().lower()
    if normalized not in ALLOWED_MIME_TYPES:
        raise SecurityViolationError(
            f"Unsupported MIME type '{normalized}'. Expected valid CSV MIME type."
        )


def is_safe_path(base_dir: Path, target_path: str) -> bool:
    """Returns True if target_path resolves strictly within base_dir, False otherwise."""
    try:
        resolved_base = base_dir.resolve()
        resolved_target = (resolved_base / target_path).resolve()
        resolved_target.relative_to(resolved_base)
        return True
    except (ValueError, RuntimeError):
        return False


def strip_ansi_sequences(text: str) -> str:
    """Removes ANSI terminal escape sequences to prevent log injection."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def sanitize_filename(filename: str) -> str:
    """Sanitizes an untrusted filename to prevent path traversal and injection attacks."""
    if not filename or len(filename.strip()) == 0:
        raise SecurityViolationError("Filename cannot be empty.")

    # Extract base name to eliminate directory components
    clean_name = Path(filename).name

    # Remove any null bytes or path traversal markers
    clean_name = clean_name.replace("\x00", "").replace("..", "")

    # Restrict to safe alphanumeric characters, dashes, underscores, and dots
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name)

    if len(clean_name) > MAX_FILENAME_LENGTH:
        suffix = Path(clean_name).suffix
        clean_name = clean_name[: MAX_FILENAME_LENGTH - len(suffix)] + suffix

    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SecurityViolationError(
            f"File extension '{ext}' is not permitted. Allowed: {ALLOWED_EXTENSIONS}"
        )

    return clean_name


def safe_join_path(base_dir: Path, target_path: str) -> Path:
    """Safely joins a base directory with an untrusted user path, ensuring no directory escape."""
    resolved_base = base_dir.resolve()
    resolved_target = (resolved_base / target_path).resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise SecurityViolationError(f"Directory traversal detected: '{target_path}'") from exc

    return resolved_target


def validate_file_size(size_bytes: int, max_size_mb: int = 100) -> None:
    """Validates that a file's size does not exceed the allowed threshold."""
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise SecurityViolationError(
            f"File size ({size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)."
        )


def compute_sha256(content: bytes) -> str:
    """Computes SHA-256 cryptographic checksum of file content."""
    sha = hashlib.sha256()
    sha.update(content)
    return sha.hexdigest()
