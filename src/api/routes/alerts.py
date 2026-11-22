"""Intrusion alert management, analyst reviews, and threat statistics endpoints."""


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.deps import get_db_session
from src.database.crud.alert_repo import (
    get_threat_stats,
    list_alerts,
    update_alert_review,
)
from src.schemas.alert import AlertReviewRequest, ThreatStatsSummary, TrafficAlertResponse

router = APIRouter(prefix="/api/v1/alerts", tags=["Threat Alerts & Intelligence"])


@router.get("", response_model=list[TrafficAlertResponse])
def get_alerts(
    severity: str | None = None,
    attack_type: str | None = None,
    is_reviewed: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session),
):
    """Retrieves detected network intrusion alerts with optional threat filtering."""
    # Ensure limit and offset are integers even if FastAPI Query object is passed in tests
    lim = int(getattr(limit, "default", limit)) if hasattr(limit, "default") else int(limit)
    off = int(getattr(offset, "default", offset)) if hasattr(offset, "default") else int(offset)

    alerts = list_alerts(
        db=db,
        severity=severity,
        attack_type=attack_type,
        is_reviewed=is_reviewed,
        limit=lim,
        offset=off,
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
