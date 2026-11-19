"""Health check and system diagnostics API endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import get_app_settings, get_db_session
from src.core.config import Settings
from src.database.crud.experiment_repo import get_best_model

router = APIRouter(prefix="/api/v1", tags=["Health & Diagnostics"])


@router.get("/health")
def health_check(
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
):
    """Returns application health status, database connectivity, and active model diagnostics."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    best_model = get_best_model(db)
    best_model_info = None
    if best_model:
        best_model_info = {
            "id": best_model.id,
            "name": best_model.model_name,
            "algorithm": best_model.algorithm,
            "f1_score": best_model.f1_score,
            "fpr": best_model.false_positive_rate,
        }

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "active_best_model": best_model_info,
    }
