"""Unit tests for database models, migrations, and CRUD operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.crud.dataset_repo import (
    create_dataset,
    get_dataset_by_id,
    list_datasets,
    delete_dataset,
)
from src.database.crud.experiment_repo import (
    create_experiment,
    get_experiment_by_id,
    get_best_model,
    set_best_model,
    get_leaderboard,
)
from src.database.crud.prediction_repo import (
    create_prediction_job,
    get_prediction_job_by_id,
)
from src.database.crud.alert_repo import (
    create_alert,
    get_alert_by_id,
    list_alerts,
    update_alert_review,
    get_threat_stats,
)


@pytest.fixture
def db_session():
    """Provides a fresh isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_dataset_crud(db_session):
    """Verify dataset record creation, retrieval, and deletion."""
    ds = create_dataset(
        db=db_session,
        filename="test_flow.csv",
        stored_path="data/uploads/test_flow.csv",
        dataset_type="unsw_nb15",
        total_rows=500,
        feature_count=42,
        normal_count=350,
        attack_count=150,
        checksum_sha256="abc123hash",
        columns=["proto", "service", "state"],
    )
    assert ds.id is not None
    assert ds.filename == "test_flow.csv"

    retrieved = get_dataset_by_id(db_session, ds.id)
    assert retrieved is not None
    assert retrieved.total_rows == 500

    all_ds = list_datasets(db_session)
    assert len(all_ds) == 1

    deleted = delete_dataset(db_session, ds.id)
    assert deleted is True
    assert get_dataset_by_id(db_session, ds.id) is None


def test_experiment_crud_and_leaderboard(db_session):
    """Verify model experiment recording and leaderboard ranking."""
    exp1 = create_experiment(
        db=db_session,
        model_name="DecisionTree-Test",
        algorithm="decision_tree",
        dataset_id=None,
        feature_strategy="none",
        hyperparameters={"max_depth": 10},
        accuracy=0.91,
        precision=0.89,
        recall=0.88,
        f1_score=0.885,
        false_positive_rate=0.04,
        false_negative_rate=0.12,
        roc_auc=0.92,
        confusion_matrix=[[90, 10], [12, 88]],
        training_time_seconds=0.1,
        prediction_time_ms=0.5,
        is_best_model=False,
    )

    exp2 = create_experiment(
        db=db_session,
        model_name="RandomForest-Test",
        algorithm="random_forest",
        dataset_id=None,
        feature_strategy="rf_importance",
        hyperparameters={"n_estimators": 50},
        accuracy=0.96,
        precision=0.95,
        recall=0.94,
        f1_score=0.945,
        false_positive_rate=0.02,
        false_negative_rate=0.06,
        roc_auc=0.98,
        confusion_matrix=[[98, 2], [6, 94]],
        training_time_seconds=0.5,
        prediction_time_ms=1.1,
        is_best_model=True,
    )

    best = get_best_model(db_session)
    assert best.id == exp2.id
    assert best.f1_score == 0.945

    # Switch best model
    updated_best = set_best_model(db_session, exp1.id)
    assert updated_best.id == exp1.id
    assert updated_best.is_best_model is True

    leaderboard = get_leaderboard(db_session)
    assert len(leaderboard) == 2
    # Highest F1 first
    assert leaderboard[0].id == exp2.id


def test_prediction_job_crud(db_session):
    """Verify batch prediction job persistence."""
    job = create_prediction_job(
        db=db_session,
        job_type="batch_csv",
        dataset_id=None,
        model_experiment_id=None,
        total_records=1000,
        normal_count=650,
        malicious_count=350,
        detection_rate=0.35,
        execution_time_seconds=0.42,
        report_csv_path="reports/test_report.csv",
    )
    assert job.id is not None
    assert job.malicious_count == 350

    retrieved = get_prediction_job_by_id(db_session, job.id)
    assert retrieved is not None
    assert retrieved.detection_rate == 0.35


def test_alert_crud_and_threat_stats(db_session):
    """Verify intrusion alert logging, review status, and stats aggregation."""
    al1 = create_alert(
        db=db_session,
        attack_type="DoS",
        confidence=0.99,
        severity="critical",
        source_ip="192.168.1.150",
    )
    al2 = create_alert(
        db=db_session,
        attack_type="Reconnaissance",
        confidence=0.85,
        severity="medium",
        source_ip="192.168.1.150",
    )

    alerts = list_alerts(db_session)
    assert len(alerts) == 2

    # Filter by severity
    crit_alerts = list_alerts(db_session, severity="critical")
    assert len(crit_alerts) == 1
    assert crit_alerts[0].attack_type == "DoS"

    # Review alert
    reviewed = update_alert_review(db_session, al1.id, is_reviewed=True, notes="False positive test")
    assert reviewed.is_reviewed is True
    assert reviewed.analyst_notes == "False positive test"

    # Aggregated stats
    stats = get_threat_stats(db_session)
    assert stats["total_alerts"] == 2
    assert stats["critical_alerts"] == 1
    assert stats["medium_alerts"] == 1
    assert stats["reviewed_count"] == 1
    assert stats["attack_type_breakdown"]["DoS"] == 1
    assert stats["top_source_ips"][0]["ip"] == "192.168.1.150"
