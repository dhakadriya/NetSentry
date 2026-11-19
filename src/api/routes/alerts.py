"""Intrusion alert management, analyst reviews, and threat statistics endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.api.deps import get_db_session
from src.database.crud.alert_repo import (
    get_alert_by_id,
    get_threat_stats,
    list_alerts,
    update_alert_review,
)
from src.schemas.alert import AlertReviewRequest, ThreatStatsSummary, TrafficAlertResponse

router = APIRouter(prefix="/api/v1/alerts", tags=["Threat Alerts & Intelligence"])


@router.get("", response_model=List[TrafficAlertResponse])
def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    attack_type: Optional[str] = Query(None, description="Filter by attack type: DoS, Exploits, etc."),
    is_reviewed: Optional[bool] = Query(None, description="Filter by review status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """Retrieves detected network intrusion alerts with optional threat filtering."""
    alerts = list_alerts(
        db=db,
        severity=severity,
        attack_type=attack_type,
        is_reviewed=is_reviewed,
        limit=limit,
        offset=offset,
    )
    return alerts


@router.get("/stats", response_model=ThreatStatsSummary)
def get_alert_statistics(
    db: Session = Depends(get_db_session),
):
    """Computes aggregated threat intelligence metrics across all detected incidents."""
    stats = get_threat_stats(db)
    return ThreatStatsSummary(
        total_alerts=stats["total_alerts"],
        critical_alerts=stats["critical_alerts"],
        high_alerts=stats["high_alerts"],
        medium_alerts=stats["medium_alerts"],
        low_alerts=stats["low_alerts"],
        attack_type_breakdown=stats["attack_type_breakdown"],
        top_source_ips=stats["top_source_ips"],
        reviewed_count=stats["reviewed_count"],
    )


@router.put("/{alert_id}/review", response_model=TrafficAlertResponse)
def review_alert(
    alert_id: int,
    request: AlertReviewRequest,
    db: Session = Depends(get_db_session),
):
    """Allows a security analyst to acknowledge or annotate an intrusion alert."""
    alert = update_alert_review(
        db=db,
        alert_id=alert_id,
        is_reviewed=request.is_reviewed,
        notes=request.analyst_notes,
    )
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with ID {alert_id} not found.",
        )
    return alert
