"""Pydantic schemas for intrusion alerts, threat intelligence, and filtering."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.constants import ThreatSeverity


class TrafficAlertResponse(BaseModel):
    """Schema representing an individual detected intrusion alert."""

    id: int
    prediction_job_id: Optional[int]
    attack_type: str
    confidence: float
    severity: ThreatSeverity
    source_ip: str
    destination_ip: str
    protocol: str
    service: str
    flow_duration: float
    source_bytes: int
    destination_bytes: int
    is_reviewed: bool
    analyst_notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertReviewRequest(BaseModel):
    """Payload to acknowledge or annotate an intrusion alert."""

    is_reviewed: bool = True
    analyst_notes: Optional[str] = None


class ThreatStatsSummary(BaseModel):
    """Aggregated threat intelligence statistics for dashboard display."""

    total_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int
    attack_type_breakdown: Dict[str, int]
    top_source_ips: List[Dict[str, int]]
    reviewed_count: int
