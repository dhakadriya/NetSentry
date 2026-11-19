"""TrafficAlert ORM model for recording detected intrusion alerts and threat levels."""

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

    attack_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False, default=0.85)
    severity = Column(String(32), default="medium", nullable=False)

    source_ip = Column(String(64), default="192.168.1.105", nullable=False)
    destination_ip = Column(String(64), default="10.0.0.1", nullable=False)
    protocol = Column(String(32), default="tcp", nullable=False)
    service = Column(String(32), default="-", nullable=False)

    flow_duration = Column(Float, default=0.0, nullable=False)
    source_bytes = Column(Integer, default=0, nullable=False)
    destination_bytes = Column(Integer, default=0, nullable=False)

    packet_details_json = Column(Text, nullable=True)
    is_reviewed = Column(Boolean, default=False, nullable=False)
    analyst_notes = Column(Text, nullable=True)

    # Relationship
    prediction_job = relationship("PredictionJob", back_populates="alerts")

    def __repr__(self) -> str:
        return (
            f"<TrafficAlert(id={self.id}, type='{self.attack_type}', "
            f"severity='{self.severity}', confidence={self.confidence:.2f})>"
        )
