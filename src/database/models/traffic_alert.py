"""TrafficAlert ORM model for recording detected intrusion alerts and threat levels."""

import json
from typing import Any, Dict, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship

from src.database.base import Base, TimestampMixin


class TrafficAlert(Base, TimestampMixin):
    """Alert record logged whenever network traffic is flagged as an active intrusion."""

    __tablename__ = "traffic_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_job_id = Column(
        Integer, ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=True
    )

    attack_type = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False, default=0.85)
    severity = Column(String(32), default="medium", nullable=False, index=True)

    source_ip = Column(String(64), default="192.168.1.105", nullable=False, index=True)
    destination_ip = Column(String(64), default="10.0.0.1", nullable=False)
    protocol = Column(String(32), default="tcp", nullable=False)
    service = Column(String(32), default="-", nullable=False)

    flow_duration = Column(Float, default=0.0, nullable=False)
    source_bytes = Column(Integer, default=0, nullable=False)
    destination_bytes = Column(Integer, default=0, nullable=False)

    packet_details_json = Column(Text, nullable=True)
    is_reviewed = Column(Boolean, default=False, nullable=False, index=True)
    analyst_notes = Column(Text, nullable=True)

    # Relationship
    prediction_job = relationship("PredictionJob", back_populates="alerts")

    @property
    def packet_details(self) -> Dict[str, Any]:
        """Deserializes packet_details_json into python dictionary."""
        if not self.packet_details_json:
            return {}
        try:
            return json.loads(self.packet_details_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes TrafficAlert to dictionary representation."""
        return {
            "id": self.id,
            "prediction_job_id": self.prediction_job_id,
            "attack_type": self.attack_type,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "protocol": self.protocol,
            "service": self.service,
            "flow_duration": round(self.flow_duration, 4),
            "source_bytes": self.source_bytes,
            "destination_bytes": self.destination_bytes,
            "packet_details": self.packet_details,
            "is_reviewed": self.is_reviewed,
            "analyst_notes": self.analyst_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<TrafficAlert(id={self.id}, type='{self.attack_type}', "
            f"severity='{self.severity}', confidence={self.confidence:.2f})>"
        )
