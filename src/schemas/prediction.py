"""Pydantic schemas for single flow and batch network traffic predictions."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.constants import ThreatSeverity

IPV4_REGEX = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"


class SinglePacketPredictionRequest(BaseModel):
    """Payload to classify a single network flow record."""

    model_id: Optional[int] = Field(default=None, description="Optional model ID; uses best model if omitted", ge=1)
    source_ip: str = Field(default="192.168.1.100", pattern=IPV4_REGEX, description="Source IPv4 address")
    destination_ip: str = Field(default="10.0.0.5", pattern=IPV4_REGEX, description="Destination IPv4 address")
    features: Dict[str, Any] = Field(
        description="Dictionary of UNSW-NB15 flow features (proto, service, state, dur, sbytes, dbytes, etc.)"
    )


class SinglePredictionResponse(BaseModel):
    """Result of single flow classification."""

    is_malicious: bool
    label: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_class: str
    attack_category: Optional[str] = None
    severity: ThreatSeverity
    model_used: str
    latency_ms: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchPredictionRequest(BaseModel):
    """Request to run batch inference on a cataloged or uploaded dataset."""

    dataset_id: int = Field(..., ge=1)
    model_id: Optional[int] = Field(None, ge=1)


class BatchPredictionResponse(BaseModel):
    """Summary of batch classification results."""

    job_id: int = Field(..., ge=1)
    dataset_id: int = Field(..., ge=1)
    model_id: int = Field(..., ge=1)
    model_name: str
    total_records: int = Field(..., ge=0)
    normal_count: int = Field(..., ge=0)
    malicious_count: int = Field(..., ge=0)
    detection_rate: float = Field(..., ge=0.0, le=1.0)
    execution_time_seconds: float = Field(..., ge=0.0)
    report_csv_path: Optional[str] = None
    alerts_created: int = Field(..., ge=0)
    status: str = "completed"


class PredictionJobPaginationResponse(BaseModel):
    """Paginated collection of prediction job records."""

    total: int = Field(..., ge=0)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    items: List[BatchPredictionResponse] = Field(default_factory=list)
