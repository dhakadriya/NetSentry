"""Unit tests for database models, migrations, and CRUD operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.crud.alert_repo import (
    count_alerts,
    create_alert,
    create_alerts_bulk,
    delete_alert,
    get_threat_stats,
    list_alerts,
    update_alert_review,
)
from src.database.crud.dataset_repo import (
    count_datasets,
    create_dataset,
    delete_dataset,
    exists_by_checksum,
    get_dataset_by_id,
    list_datasets,
)
from src.database.crud.experiment_repo import (
    archive_experiment,
    count_experiments,
    create_experiment,
    get_best_model,
    get_leaderboard,
    set_best_model,
)
from src.database.crud.prediction_repo import (
    count_prediction_jobs,
    create_prediction_job,
    get_prediction_job_by_id,
    update_prediction_job_status,
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


def test_dataset_pagination_and_checksum_exists(db_session):
    """Verify count_datasets and exists_by_checksum functions."""
    create_dataset(
        db=db_session,
        filename="flow_a.csv",
        stored_path="data/uploads/flow_a.csv",
        dataset_type="unsw_nb15",
        total_rows=100,
        feature_count=42,
        normal_count=80,
        attack_count=20,
        checksum_sha256="hash_flow_a",
    )
    create_dataset(
        db=db_session,
        filename="flow_b.csv",
        stored_path="data/uploads/flow_b.csv",
        dataset_type="cic_ids2017",
        total_rows=200,
        feature_count=42,
        normal_count=150,
        attack_count=50,
        checksum_sha256="hash_flow_b",
    )

    assert count_datasets(db_session) == 2
    assert count_datasets(db_session, dataset_type="unsw_nb15") == 1
    assert count_datasets(db_session, dataset_type="cic_ids2017") == 1
    assert exists_by_checksum(db_session, "hash_flow_a") is True
    assert exists_by_checksum(db_session, "nonexistent_hash") is False


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
        false_positive_rate=0.035,
        false_negative_rate=0.12,
        roc_auc=0.94,
        confusion_matrix=[[300, 10], [20, 170]],
        training_time_seconds=0.15,
        prediction_time_ms=0.5,
        artifact_path="models/test_dt.joblib",
    )

    exp2 = create_experiment(
        db=db_session,
        model_name="RandomForest-Test",
        algorithm="random_forest",
        dataset_id=None,
        feature_strategy="variance_threshold",
        hyperparameters={"n_estimators": 50},
        accuracy=0.96,
        precision=0.95,
        recall=0.94,
        f1_score=0.945,
        false_positive_rate=0.015,
        false_negative_rate=0.06,
        roc_auc=0.98,
        confusion_matrix=[[305, 5], [10, 180]],
        training_time_seconds=0.85,
        prediction_time_ms=1.2,
        artifact_path="models/test_rf.joblib",
        is_best_model=True,
    )

    best = get_best_model(db_session)
    assert best is not None
    assert best.id == exp2.id
    assert best.f1_score == 0.945

    # Switch best model manually
    updated = set_best_model(db_session, exp1.id)
    assert updated.is_best_model is True
    assert get_best_model(db_session).id == exp1.id

    # Verify Leaderboard ordering
    leaderboard = get_leaderboard(db_session)
    assert len(leaderboard) == 2
    assert leaderboard[0].f1_score >= leaderboard[1].f1_score


def test_experiment_filtering_and_archive(db_session):
    """Verify count_experiments and archive_experiment functions."""
    exp = create_experiment(
        db=db_session,
        model_name="KNN-Test",
        algorithm="knn",
        dataset_id=None,
        feature_strategy="none",
        hyperparameters={"n_neighbors": 5},
        accuracy=0.90,
        precision=0.88,
        recall=0.87,
        f1_score=0.875,
        false_positive_rate=0.04,
        false_negative_rate=0.13,
        roc_auc=0.92,
        confusion_matrix=[[290, 20], [25, 165]],
        training_time_seconds=0.05,
        prediction_time_ms=2.5,
    )
    assert count_experiments(db_session, algorithm="knn") == 1
    assert count_experiments(db_session, is_active=True) == 1

    archived = archive_experiment(db_session, exp.id)
    assert archived is True
    assert count_experiments(db_session, is_active=True) == 0
    assert count_experiments(db_session, is_active=False) == 1


def test_prediction_job_crud(db_session):
    """Verify batch prediction job tracking."""
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

    assert count_prediction_jobs(db_session, status="completed") == 1

    updated = update_prediction_job_status(db_session, job.id, "failed", "CSV parsing error")
    assert updated.status == "failed"
    assert updated.error_message == "CSV parsing error"


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


def test_alerts_bulk_insert_and_count(db_session):
    """Verify create_alerts_bulk, count_alerts, and delete_alert."""
    batch = [
        {"attack_type": "Fuzzers", "confidence": 0.82, "severity": "medium", "source_ip": "10.0.0.5"},
        {"attack_type": "Exploits", "confidence": 0.95, "severity": "critical", "source_ip": "10.0.0.6"},
        {"attack_type": "Generic", "confidence": 0.78, "severity": "low", "source_ip": "10.0.0.7"},
    ]
    created = create_alerts_bulk(db_session, batch)
    assert len(created) == 3

    assert count_alerts(db_session) == 3
    assert count_alerts(db_session, severity="critical") == 1
    assert count_alerts(db_session, is_reviewed=False) == 3

    del_res = delete_alert(db_session, created[0].id)
    assert del_res is True
    assert count_alerts(db_session) == 2
