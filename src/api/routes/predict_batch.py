"""Batch CSV network traffic flow classification and report generation endpoint."""

from datetime import datetime, timezone
from pathlib import Path
import time
from fastapi import APIRouter, Depends, HTTPException, status
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.api.deps import get_app_settings, get_db_session
from src.core.config import Settings
from src.core.constants import ATTACK_SEVERITY_MAP, ThreatSeverity
from src.database.crud.alert_repo import create_alert
from src.database.crud.dataset_repo import get_dataset_by_id
from src.database.crud.experiment_repo import get_best_model, get_experiment_by_id
from src.database.crud.prediction_repo import create_prediction_job
from src.schemas.prediction import BatchPredictionRequest, BatchPredictionResponse

router = APIRouter(prefix="/api/v1/predict", tags=["Intrusion Prediction"])


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch_traffic(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
):
    """Executes high-throughput batch classification across network flow records in an uploaded CSV."""
    start_time = time.perf_counter()

    # 1. Resolve Dataset
    dataset_record = get_dataset_by_id(db, request.dataset_id)
    if not dataset_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} not found.",
        )

    file_path = Path(dataset_record.stored_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file missing from disk: {file_path}",
        )

    # 2. Resolve Model
    if request.model_id:
        exp = get_experiment_by_id(db, request.model_id)
    else:
        exp = get_best_model(db)

    if not exp or not exp.artifact_path or not Path(exp.artifact_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid trained model artifact found for prediction.",
        )

    # 3. Load Model Artifact
    bundle = joblib.load(exp.artifact_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    selector = bundle["selector"]

    # 4. Load Dataset
    df = pd.read_csv(file_path, nrows=settings.max_batch_rows)

    # 5. Transform and Predict
    X_trans, _, _ = preprocessor.split_X_y(df)
    X_selected = selector.transform(X_trans)

    preds = model.predict(X_selected)
    probs = model.predict_proba(X_selected)

    # Vectorized classification analysis
    confidences = np.where(preds == 1, probs[:, 1], probs[:, 0])

    normal_count = int((preds == 0).sum())
    malicious_count = int((preds == 1).sum())
    total_records = len(preds)
    detection_rate = round(float(malicious_count / max(total_records, 1)), 4)
    elapsed_seconds = round(time.perf_counter() - start_time, 4)

    # 6. Generate Annotated CSV Report
    report_df = df.copy()
    report_df["predicted_label"] = preds
    report_df["predicted_class"] = np.where(preds == 1, "Malicious", "Normal")
    report_df["prediction_confidence"] = np.round(confidences, 4)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_filename = f"detection_report_ds{dataset_record.id}_{timestamp}.csv"
    report_path = settings.reports_dir / report_filename
    report_df.to_csv(report_path, index=False)

    # 7. Persist Prediction Job in DB
    job = create_prediction_job(
        db=db,
        job_type="batch_csv",
        dataset_id=dataset_record.id,
        model_experiment_id=exp.id,
        total_records=total_records,
        normal_count=normal_count,
        malicious_count=malicious_count,
        detection_rate=detection_rate,
        execution_time_seconds=elapsed_seconds,
        report_csv_path=str(report_path),
        status="completed",
    )

    # 8. Create Alerts for Malicious Flows
    alerts_created = 0
    malicious_indices = np.where(preds == 1)[0]
    # Sample up to top 15 malicious records for active alert center inspection
    for idx in malicious_indices[:15]:
        row = df.iloc[idx]
        conf = float(confidences[idx])
        raw_cat = str(row.get("attack_cat", "Generic"))
        attack_type = raw_cat if raw_cat != "Normal" else "Exploits"
        sev = ATTACK_SEVERITY_MAP.get(attack_type, ThreatSeverity.HIGH)

        create_alert(
            db=db,
            prediction_job_id=job.id,
            attack_type=attack_type,
            confidence=conf,
            severity=sev.value,
            source_ip=str(row.get("srcip", f"192.168.1.{100 + (idx % 100)}")),
            destination_ip=str(row.get("dstip", "10.0.0.1")),
            protocol=str(row.get("proto", "tcp")),
            service=str(row.get("service", "-")),
            flow_duration=float(row.get("dur", 0.0)),
            source_bytes=int(row.get("sbytes", 0)),
            destination_bytes=int(row.get("dbytes", 0)),
        )
        alerts_created += 1

    return BatchPredictionResponse(
        job_id=job.id,
        dataset_id=dataset_record.id,
        model_id=exp.id,
        model_name=exp.model_name,
        total_records=total_records,
        normal_count=normal_count,
        malicious_count=malicious_count,
        detection_rate=detection_rate,
        execution_time_seconds=elapsed_seconds,
        report_csv_path=str(report_path),
        alerts_created=alerts_created,
        status="completed",
    )
