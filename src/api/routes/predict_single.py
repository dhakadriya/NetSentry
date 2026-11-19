"""Single packet / flow real-time intrusion classification endpoint."""

import time
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
import joblib
import pandas as pd
from sqlalchemy.orm import Session

from src.api.deps import get_db_session
from src.core.constants import ATTACK_SEVERITY_MAP, ThreatSeverity
from src.core.exceptions import PredictionError
from src.database.crud.alert_repo import create_alert
from src.database.crud.experiment_repo import get_best_model, get_experiment_by_id
from src.schemas.prediction import SinglePacketPredictionRequest, SinglePredictionResponse

router = APIRouter(prefix="/api/v1/predict", tags=["Intrusion Prediction"])


@router.post("/single", response_model=SinglePredictionResponse)
def predict_single_packet(
    request: SinglePacketPredictionRequest,
    db: Session = Depends(get_db_session),
):
    """Classifies a single incoming network flow record in real time."""
    start_time = time.perf_counter()

    # Determine model to use
    if request.model_id:
        exp = get_experiment_by_id(db, request.model_id)
    else:
        exp = get_best_model(db)

    if not exp or not exp.artifact_path or not Path(exp.artifact_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid trained model artifact available for prediction. Please train or seed a model first.",
        )

    # Load bundle
    try:
        bundle = joblib.load(exp.artifact_path)
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        selector = bundle["selector"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model artifact: {str(e)}",
        )

    # Convert single input dict to DataFrame
    input_df = pd.DataFrame([request.features])

    # Transform through preprocessor and selector
    try:
        X_trans, _, _ = preprocessor.split_X_y(input_df)
        X_selected = selector.transform(X_trans)
        preds = model.predict(X_selected)
        probs = model.predict_proba(X_selected)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature extraction failed on input packet: {str(e)}",
        )

    pred_label = int(preds[0])
    is_malicious = pred_label == 1
    confidence = float(probs[0, 1] if is_malicious else probs[0, 0])
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Determine category & severity
    attack_category = "Normal"
    severity = ThreatSeverity.LOW
    if is_malicious:
        # Heuristic inference of category based on features
        sbytes = float(request.features.get("sbytes", 0))
        dur = float(request.features.get("dur", 1.0))
        sload = float(request.features.get("Sload", 0))

        if sbytes > 20000 or sload > 1000000:
            attack_category = "DoS"
        elif request.features.get("proto") == "icmp":
            attack_category = "Reconnaissance"
        elif dur > 1.5:
            attack_category = "Exploits"
        else:
            attack_category = "Generic"

        severity = ATTACK_SEVERITY_MAP.get(attack_category, ThreatSeverity.MEDIUM)

        # Log active threat alert
        create_alert(
            db=db,
            attack_type=attack_category,
            confidence=confidence,
            severity=severity.value,
            source_ip=request.source_ip,
            destination_ip=request.destination_ip,
            protocol=str(request.features.get("proto", "tcp")),
            service=str(request.features.get("service", "-")),
            flow_duration=float(request.features.get("dur", 0.0)),
            source_bytes=int(request.features.get("sbytes", 0)),
            destination_bytes=int(request.features.get("dbytes", 0)),
            packet_details=request.features,
        )

    return SinglePredictionResponse(
        is_malicious=is_malicious,
        label=pred_label,
        confidence=round(confidence, 4),
        predicted_class="Malicious" if is_malicious else "Normal",
        attack_category=attack_category if is_malicious else None,
        severity=severity,
        model_used=exp.model_name,
        latency_ms=round(latency_ms, 2),
    )
