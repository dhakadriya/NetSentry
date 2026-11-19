"""Pydantic schemas for single flow and batch network traffic predictions."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.constants import ThreatSeverity


class SinglePacketPredictionRequest(BaseModel):
    """Payload to classify a single network flow record."""

    model_id: Optional[int] = Field(default=None, description="Optional model ID; uses best model if omitted")
    source_ip: str = Field(default="192.168.1.100")
    destination_ip: str = Field(default="10.0.0.5")
    features: Dict[str, Any] = Field(
        description="Dictionary of UNSW-NB15 flow features (proto, service, state, dur, sbytes, dbytes, etc.)"
    )


class SinglePredictionResponse(BaseModel):
    """Result of single flow classification."""

    is_malicious: bool
    label: int
    confidence: float
    predicted_class: str
    attack_category: Optional[str] = None
    severity: ThreatSeverity
    model_used: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Request to run batch inference on a cataloged or uploaded dataset."""

    dataset_id: int
    model_id: Optional[int] = None


class BatchPredictionResponse(BaseModel):
    """Summary of batch classification results."""

    job_id: int
    dataset_id: int
    model_id: int
    model_name: str
    total_records: int
    normal_count: int
    malicious_count: int
    detection_rate: float
    execution_time_seconds: float
    report_csv_path: Optional[str] = None
    alerts_created: int
    status: str
