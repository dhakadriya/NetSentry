"""CRUD operations for intrusion alerts and threat statistics."""

import json
from typing import Any, Dict, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models.traffic_alert import TrafficAlert


def create_alert(
    db: Session,
    attack_type: str,
    confidence: float,
    severity: str,
    source_ip: str = "192.168.1.100",
    destination_ip: str = "10.0.0.1",
    protocol: str = "tcp",
    service: str = "-",
    flow_duration: float = 0.0,
    source_bytes: int = 0,
    destination_bytes: int = 0,
    prediction_job_id: Optional[int] = None,
    packet_details: Optional[Dict[str, Any]] = None,
) -> TrafficAlert:
    """Creates a new intrusion alert."""
    alert = TrafficAlert(
        prediction_job_id=prediction_job_id,
        attack_type=attack_type,
        confidence=confidence,
        severity=severity,
        source_ip=source_ip,
        destination_ip=destination_ip,
        protocol=protocol,
        service=service,
        flow_duration=flow_duration,
        source_bytes=source_bytes,
        destination_bytes=destination_bytes,
        packet_details_json=json.dumps(packet_details) if packet_details else None,
        is_reviewed=False,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def create_alerts_bulk(
    db: Session, alerts_data: List[Dict[str, Any]]
) -> List[TrafficAlert]:
    """Efficiently inserts a batch of threat alerts into the database."""
    alerts = []
    for item in alerts_data:
        packet_json = (
            json.dumps(item.get("packet_details"))
            if item.get("packet_details")
            else None
        )
        alert = TrafficAlert(
            prediction_job_id=item.get("prediction_job_id"),
            attack_type=item.get("attack_type", "Generic"),
            confidence=item.get("confidence", 0.85),
            severity=item.get("severity", "medium"),
            source_ip=item.get("source_ip", "192.168.1.100"),
            destination_ip=item.get("destination_ip", "10.0.0.1"),
            protocol=item.get("protocol", "tcp"),
            service=item.get("service", "-"),
            flow_duration=item.get("flow_duration", 0.0),
            source_bytes=item.get("source_bytes", 0),
            destination_bytes=item.get("destination_bytes", 0),
            packet_details_json=packet_json,
            is_reviewed=item.get("is_reviewed", False),
            analyst_notes=item.get("analyst_notes"),
        )
        alerts.append(alert)

    db.bulk_save_objects(alerts, return_defaults=True)
    db.commit()
    return alerts


def get_alert_by_id(db: Session, alert_id: int) -> Optional[TrafficAlert]:
    """Retrieves an alert by its ID."""
    return db.query(TrafficAlert).filter(TrafficAlert.id == alert_id).first()


def count_alerts(
    db: Session,
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    is_reviewed: Optional[bool] = None,
) -> int:
    """Returns the total number of alerts matching optional filter criteria."""
    query = db.query(func.count(TrafficAlert.id))
    if severity:
        query = query.filter(TrafficAlert.severity == severity)
    if attack_type:
        query = query.filter(TrafficAlert.attack_type == attack_type)
    if is_reviewed is not None:
        query = query.filter(TrafficAlert.is_reviewed == is_reviewed)
    return query.scalar() or 0


def list_alerts(
    db: Session,
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    is_reviewed: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[TrafficAlert]:
    """Retrieves alerts matching optional filtering criteria."""
    query = db.query(TrafficAlert)
    if severity:
        query = query.filter(TrafficAlert.severity == severity)
    if attack_type:
        query = query.filter(TrafficAlert.attack_type == attack_type)
    if is_reviewed is not None:
        query = query.filter(TrafficAlert.is_reviewed == is_reviewed)
    return query.order_by(TrafficAlert.id.desc()).offset(offset).limit(limit).all()


def update_alert_review(
    db: Session, alert_id: int, is_reviewed: bool = True, notes: Optional[str] = None
) -> Optional[TrafficAlert]:
    """Updates the review status and analyst annotations for an alert."""
    alert = get_alert_by_id(db, alert_id)
    if alert:
        alert.is_reviewed = is_reviewed
        if notes:
            alert.analyst_notes = notes
        db.commit()
        db.refresh(alert)
    return alert


def delete_alert(db: Session, alert_id: int) -> bool:
    """Removes an alert record from the database."""
    alert = get_alert_by_id(db, alert_id)
    if alert:
        db.delete(alert)
        db.commit()
        return True
    return False


def get_threat_stats(db: Session) -> Dict[str, Any]:
    """Computes aggregated threat statistics for the dashboard."""
    total = db.query(func.count(TrafficAlert.id)).scalar() or 0
    critical = (
        db.query(func.count(TrafficAlert.id))
        .filter(TrafficAlert.severity == "critical")
        .scalar()
        or 0
    )
    high = (
        db.query(func.count(TrafficAlert.id))
        .filter(TrafficAlert.severity == "high")
        .scalar()
        or 0
    )
    medium = (
        db.query(func.count(TrafficAlert.id))
        .filter(TrafficAlert.severity == "medium")
        .scalar()
        or 0
    )
    low = (
        db.query(func.count(TrafficAlert.id))
        .filter(TrafficAlert.severity == "low")
        .scalar()
        or 0
    )
    reviewed = (
        db.query(func.count(TrafficAlert.id))
        .filter(TrafficAlert.is_reviewed == True)
        .scalar()
        or 0
    )

    attack_counts = (
        db.query(TrafficAlert.attack_type, func.count(TrafficAlert.id))
        .group_by(TrafficAlert.attack_type)
        .all()
    )
    attack_breakdown = {atype: count for atype, count in attack_counts}

    top_ips = (
        db.query(TrafficAlert.source_ip, func.count(TrafficAlert.id))
        .group_by(TrafficAlert.source_ip)
        .order_by(func.count(TrafficAlert.id).desc())
        .limit(5)
        .all()
    )
    top_ip_list = [{"ip": ip, "count": count} for ip, count in top_ips]

    return {
        "total_alerts": total,
        "critical_alerts": critical,
        "high_alerts": high,
        "medium_alerts": medium,
        "low_alerts": low,
        "reviewed_count": reviewed,
        "attack_type_breakdown": attack_breakdown,
        "top_source_ips": top_ip_list,
    }
