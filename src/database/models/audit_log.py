"""AuditLog ORM model for recording administrative, training, and detection events."""

from sqlalchemy import Column, Integer, String, Text

from src.database.base import Base, TimestampMixin


class AuditLog(Base, TimestampMixin):
    """Audit log entry capturing system actions, user uploads, model runs, and exports."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(64), nullable=False)
    entity_type = Column(String(64), nullable=True)
    entity_id = Column(Integer, nullable=True)
    client_ip = Column(String(64), default="127.0.0.1", nullable=False)
    status = Column(String(32), default="SUCCESS", nullable=False)
    details_json = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', status='{self.status}')>"
