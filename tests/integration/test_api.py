"""Integration tests directly exercising FastAPI route handlers and business logic."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.routes.alerts import get_alert_statistics, get_alerts
from src.api.routes.compare import get_model_leaderboard
from src.api.routes.datasets import get_dataset, get_datasets
from src.api.routes.eda import get_dataset_eda
from src.api.routes.health import health_check
from src.api.routes.models import get_experiment, get_experiments
from src.core.config import get_settings
from src.database.base import Base
from src.database.crud.alert_repo import create_alert
from src.database.crud.dataset_repo import create_dataset
from src.database.crud.experiment_repo import create_experiment


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

    # Seed test dataset, experiment, and alerts
    ds = create_dataset(
        db=session,
        filename="unsw_nb15_sample_train.csv",
        stored_path="data/samples/unsw_nb15_sample_train.csv",
        dataset_type="unsw_nb15",
        total_rows=2000,
        feature_count=42,
        normal_count=1300,
        attack_count=700,
        checksum_sha256="test_sha_sample",
        columns=["proto", "service", "state", "dur", "sbytes", "dbytes"],
    )

    exp = create_experiment(
        db=session,
        model_name="RandomForest-IntegrationTest",
        algorithm="random_forest",
        dataset_id=ds.id,
        feature_strategy="none",
        hyperparameters={"seed": 42},
        accuracy=0.962,
        precision=0.954,
        recall=0.948,
        f1_score=0.951,
        false_positive_rate=0.021,
        false_negative_rate=0.052,
        roc_auc=0.987,
        confusion_matrix=[[1270, 30], [36, 664]],
        training_time_seconds=0.45,
        prediction_time_ms=0.8,
        is_best_model=True,
    )

    for i in range(5):
        create_alert(
            db=session,
            attack_type="DoS",
            confidence=0.98,
            severity="high",
            source_ip=f"192.168.1.{100 + i}",
        )

    try:
        yield session
    finally:
        session.close()


def test_health_check_route(db_session):
    """Verify health check handler reports healthy and returns best model info."""
    settings = get_settings()
    res = health_check(db=db_session, settings=settings)

    assert res["status"] == "healthy"
    assert res["database"] == "connected"
    assert res["active_best_model"] is not None
    assert res["active_best_model"]["name"] == "RandomForest-IntegrationTest"


def test_datasets_routes(db_session):
    """Verify dataset listing and single dataset query."""
    datasets = get_datasets(limit=10, offset=0, db=db_session)
    assert len(datasets) == 1
    assert datasets[0].filename == "unsw_nb15_sample_train.csv"

    single = get_dataset(dataset_id=datasets[0].id, db=db_session)
    assert single.id == datasets[0].id


def test_dataset_eda_route(db_session):
    """Verify EDA statistics calculation on seeded sample CSV."""
    eda = get_dataset_eda(dataset_id=1, db=db_session)

    assert eda.total_rows > 0
    assert "Normal (0)" in eda.class_distribution
    assert "Malicious (1)" in eda.class_distribution
    assert "tcp" in eda.protocol_distribution or len(eda.protocol_distribution) > 0


def test_model_experiments_and_leaderboard_routes(db_session):
    """Verify model experiment listing and leaderboard calculation."""
    experiments = get_experiments(limit=10, db=db_session)
    assert len(experiments) == 1
    assert experiments[0].model_name == "RandomForest-IntegrationTest"

    single = get_experiment(experiment_id=experiments[0].id, db=db_session)
    assert single.metrics.f1_score == 0.951

    leaderboard_res = get_model_leaderboard(limit=10, db=db_session)
    assert len(leaderboard_res["leaderboard"]) == 1
    assert leaderboard_res["leaderboard"][0]["rank"] == 1


def test_alerts_routes(db_session):
    """Verify threat alert listing and aggregate intelligence stats."""
    alerts = get_alerts(limit=10, db=db_session)
    assert len(alerts) == 5

    stats = get_alert_statistics(db=db_session)
    assert stats.total_alerts == 5
    assert stats.high_alerts == 5
    assert stats.attack_type_breakdown["DoS"] == 5
