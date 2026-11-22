"""Pydantic schemas for intrusion alerts, threat intelligence, and filtering."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.constants import ThreatSeverity


class TrafficAlertResponse(BaseModel):
    """Schema representing an individual detected intrusion alert."""

    id: int = Field(..., ge=1)
    prediction_job_id: Optional[int] = Field(None, ge=1)
    attack_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: ThreatSeverity
    source_ip: str
    destination_ip: str
    protocol: str
    service: str
    flow_duration: float = Field(..., ge=0.0)
    source_bytes: int = Field(..., ge=0)
    destination_bytes: int = Field(..., ge=0)
    is_reviewed: bool = False
    analyst_notes: Optional[str] = Field(None, max_length=1000)
    mitre_technique_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertReviewRequest(BaseModel):
    """Payload to acknowledge or annotate an intrusion alert."""

    is_reviewed: bool = Field(True, description="Mark whether alert was inspected by SOC analyst")
    analyst_notes: Optional[str] = Field(None, max_length=1000, description="Analyst triage commentary")


class ThreatStatsSummary(BaseModel):
    """Aggregated threat intelligence statistics for dashboard display."""

    total_alerts: int = Field(..., ge=0)
    critical_alerts: int = Field(..., ge=0)
    high_alerts: int = Field(..., ge=0)
    medium_alerts: int = Field(..., ge=0)
    low_alerts: int = Field(..., ge=0)
    attack_type_breakdown: Dict[str, int] = Field(default_factory=dict)
    top_source_ips: List[Dict[str, Any]] = Field(default_factory=list)
    reviewed_count: int = Field(..., ge=0)


class AlertPaginationResponse(BaseModel):
    """Paginated collection of intrusion alert records."""

    total: int = Field(..., ge=0)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    items: List[TrafficAlertResponse] = Field(default_factory=list)
