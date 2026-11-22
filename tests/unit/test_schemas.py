"""Unit tests for Pydantic request, response, and pagination schemas."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.core.constants import MLAlgorithm, FeatureSelectionStrategy, ThreatSeverity
from src.schemas.dataset import (
    DatasetSummary,
    DatasetUploadResponse,
    DatasetPaginationResponse,
)
from src.schemas.model import (
    TrainModelRequest,
    ModelEvaluationMetrics,
    LeaderboardResponse,
)
from src.schemas.prediction import (
    SinglePacketPredictionRequest,
    SinglePredictionResponse,
    BatchPredictionResponse,
)
from src.schemas.alert import (
    TrafficAlertResponse,
    AlertReviewRequest,
    ThreatStatsSummary,
    AlertPaginationResponse,
)


def test_dataset_summary_schema_validation():
    """Verify validation boundaries and filename constraints on dataset schema."""
    valid_data = {
        "id": 1,
        "filename": "unsw_sample.csv",
        "dataset_type": "unsw_nb15",
        "total_rows": 1000,
        "feature_count": 42,
        "normal_count": 800,
        "attack_count": 200,
        "checksum_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "is_validated": True,
        "created_at": datetime.now(timezone.utc),
    }
    summary = DatasetSummary(**valid_data)
    assert summary.id == 1
    assert summary.total_rows == 1000

    # Negative total_rows should raise ValidationError
    invalid_data = valid_data.copy()
    invalid_data["total_rows"] = -5
    with pytest.raises(ValidationError):
        DatasetSummary(**invalid_data)

    # Invalid filename pattern should raise ValidationError
    invalid_name = valid_data.copy()
    invalid_name["filename"] = "unsw_sample.exe"
    with pytest.raises(ValidationError):
        DatasetSummary(**invalid_name)


def test_train_model_request_bounds():
    """Verify hyperparameter and test split constraints on training schema."""
    req = TrainModelRequest(
        dataset_id=1,
        algorithm=MLAlgorithm.RANDOM_FOREST,
        test_size=0.2,
        random_seed=42,
    )
    assert req.algorithm == MLAlgorithm.RANDOM_FOREST
    assert req.test_size == 0.2

    # test_size out of bounds (< 0.1 or > 0.5)
    with pytest.raises(ValidationError):
        TrainModelRequest(dataset_id=1, test_size=0.05)

    with pytest.raises(ValidationError):
        TrainModelRequest(dataset_id=1, test_size=0.9)


def test_single_packet_ip_regex():
    """Verify IPv4 regex enforcement on single packet predictions."""
    valid_req = SinglePacketPredictionRequest(
        source_ip="192.168.1.100",
        destination_ip="10.0.0.1",
        features={"proto": "tcp", "service": "http", "dur": 0.05},
    )
    assert valid_req.source_ip == "192.168.1.100"

    # Malformed IP address
    with pytest.raises(ValidationError):
        SinglePacketPredictionRequest(
            source_ip="999.999.999.999",
            destination_ip="10.0.0.1",
            features={},
        )


def test_alert_review_request_constraints():
    """Verify max length constraints on analyst triage notes."""
    review = AlertReviewRequest(is_reviewed=True, analyst_notes="Benign ping scan verified.")
    assert review.is_reviewed is True

    # Note exceeding max_length of 1000
    with pytest.raises(ValidationError):
        AlertReviewRequest(is_reviewed=True, analyst_notes="X" * 1001)


def test_pagination_schemas():
    """Verify pagination wrappers for datasets, alerts, and leaderboards."""
    dataset_page = DatasetPaginationResponse(total=0, limit=50, offset=0, items=[])
    assert dataset_page.total == 0

    alert_page = AlertPaginationResponse(total=0, limit=25, offset=0, items=[])
    assert alert_page.limit == 25

    leaderboard = LeaderboardResponse(total_models=0, top_models=[])
    assert leaderboard.total_models == 0
