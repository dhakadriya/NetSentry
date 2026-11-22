"""AuditLog ORM model for recording administrative, training, and detection events."""

import json
from typing import Any

from sqlalchemy import Column, Integer, String, Text

from src.database.base import Base, TimestampMixin


class AuditLog(Base, TimestampMixin):
    """Audit log entry capturing system actions, user uploads, model runs, and exports."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=True, index=True)
    entity_id = Column(Integer, nullable=True)
    client_ip = Column(String(64), default="127.0.0.1", nullable=False, index=True)
    status = Column(String(32), default="SUCCESS", nullable=False, index=True)
    details_json = Column(Text, nullable=True)

    @property
    def details(self) -> dict[str, Any]:
        """Deserializes details_json into python dictionary."""
        if not self.details_json:
            return {}
        try:
            return json.loads(self.details_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> dict[str, Any]:
        """Serializes AuditLog to dictionary representation."""
        return {
            "id": self.id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "client_ip": self.client_ip,
            "status": self.status,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', status='{self.status}', ip='{self.client_ip}')>"
