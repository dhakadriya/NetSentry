"""Database seed script populating demo datasets, experiments, and threat alerts."""

from pathlib import Path

from src.core.logger import get_logger
from src.core.security import compute_sha256
from src.database.connection import SessionLocal
from src.database.crud.alert_repo import create_alert
from src.database.crud.dataset_repo import create_dataset, get_dataset_by_checksum
from src.database.crud.experiment_repo import create_experiment
from src.database.init_db import init_database

logger = get_logger("netsentry.seed")


def seed_database() -> None:
    """Populates the database with initial demonstration records."""
    init_database()
    db = SessionLocal()

    try:
        # 1. Seed Datasets if sample files exist
        train_path = Path("data/samples/unsw_nb15_sample_train.csv")
        test_path = Path("data/samples/unsw_nb15_sample_test.csv")

        dataset_id = 1
        if train_path.exists():
            content = train_path.read_bytes()
            sha = compute_sha256(content)
            existing = get_dataset_by_checksum(db, sha)
            if not existing:
                ds = create_dataset(
                    db=db,
                    filename=train_path.name,
                    stored_path=str(train_path),
                    dataset_type="unsw_nb15",
                    total_rows=2000,
                    feature_count=42,
                    normal_count=1300,
                    attack_count=700,
                    checksum_sha256=sha,
                    description="UNSW-NB15 Benchmark Training Sample (2000 network flows)",
                )
                dataset_id = ds.id
                logger.info("Seeded dataset record: %s (ID %s)", ds.filename, ds.id)
            else:
                dataset_id = existing.id

        # 2. Seed Baseline Experiments for the 4 Algorithms
        experiments_data = [
            {
                "model_name": "RandomForest-Optimized",
                "algorithm": "random_forest",
                "accuracy": 0.9620,
                "precision": 0.9540,
                "recall": 0.9480,
                "f1_score": 0.9510,
                "fpr": 0.0210,
                "fnr": 0.0520,
                "roc_auc": 0.9870,
                "cm": [[1270, 30], [36, 664]],
                "train_time": 0.84,
                "pred_time": 1.25,
                "is_best": True,
                "strategy": "rf_importance",
                "notes": "Ensemble with 100 estimators, balanced class weights, and top 20 selected features.",
            },
            {
                "model_name": "DecisionTree-Pruned",
                "algorithm": "decision_tree",
                "accuracy": 0.9310,
                "precision": 0.9180,
                "recall": 0.9020,
                "f1_score": 0.9100,
                "fpr": 0.0450,
                "fnr": 0.0980,
                "roc_auc": 0.9350,
                "cm": [[1241, 59], [68, 632]],
                "train_time": 0.12,
                "pred_time": 0.45,
                "is_best": False,
                "strategy": "correlation_filter",
                "notes": "Interpretable CART tree with max_depth=12, min_samples_split=10.",
            },
            {
                "model_name": "LogisticRegression-L2",
                "algorithm": "logistic_regression",
                "accuracy": 0.8950,
                "precision": 0.8810,
                "recall": 0.8520,
                "f1_score": 0.8660,
                "fpr": 0.0620,
                "fnr": 0.1480,
                "roc_auc": 0.9240,
                "cm": [[1219, 81], [103, 597]],
                "train_time": 0.08,
                "pred_time": 0.32,
                "is_best": False,
                "strategy": "variance_threshold",
                "notes": "Standardized numerical features with L2 penalty, C=1.0.",
            },
            {
                "model_name": "KNN-K5",
                "algorithm": "knn",
                "accuracy": 0.9180,
                "precision": 0.9050,
                "recall": 0.8870,
                "f1_score": 0.8960,
                "fpr": 0.0510,
                "fnr": 0.1130,
                "roc_auc": 0.9410,
                "cm": [[1233, 67], [79, 621]],
                "train_time": 0.05,
                "pred_time": 4.80,
                "is_best": False,
                "strategy": "none",
                "notes": "Instance-based classifier k=5 with distance weighting.",
            },
        ]

        from src.database.models.model_experiment import ModelExperiment
        existing_exp = db.query(ModelExperiment).first()
        if not existing_exp:
            for exp in experiments_data:
                create_experiment(
                    db=db,
                    model_name=exp["model_name"],
                    algorithm=exp["algorithm"],
                    dataset_id=dataset_id,
                    feature_strategy=exp["strategy"],
                    hyperparameters={"seed": 42},
                    accuracy=exp["accuracy"],
                    precision=exp["precision"],
                    recall=exp["recall"],
                    f1_score=exp["f1_score"],
                    false_positive_rate=exp["fpr"],
                    false_negative_rate=exp["fnr"],
                    roc_auc=exp["roc_auc"],
                    confusion_matrix=exp["cm"],
                    training_time_seconds=exp["train_time"],
                    prediction_time_ms=exp["pred_time"],
                    is_best_model=exp["is_best"],
                    notes=exp["notes"],
                )
            logger.info("Seeded 4 baseline model experiments.")

        # 3. Seed Realistic Threat Alerts
        from src.database.models.traffic_alert import TrafficAlert
        existing_alerts = db.query(TrafficAlert).first()
        if not existing_alerts:
            alerts_data = [
                {
                    "type": "DoS",
                    "conf": 0.98,
                    "sev": "high",
                    "src": "192.168.1.182",
                    "dst": "10.0.0.1",
                    "proto": "tcp",
                    "svc": "http",
                    "dur": 0.025,
                    "sbytes": 45000,
                    "dbytes": 120,
                },
                {
                    "type": "Exploits",
                    "conf": 0.95,
                    "sev": "critical",
                    "src": "192.168.1.204",
                    "dst": "10.0.0.5",
                    "proto": "tcp",
                    "svc": "ssh",
                    "dur": 1.450,
                    "sbytes": 3400,
                    "dbytes": 210,
                },
                {
                    "type": "Reconnaissance",
                    "conf": 0.91,
                    "sev": "medium",
                    "src": "192.168.1.110",
                    "dst": "10.0.0.2",
                    "proto": "icmp",
                    "svc": "-",
                    "dur": 0.005,
                    "sbytes": 240,
                    "dbytes": 0,
                },
                {
                    "type": "Fuzzers",
                    "conf": 0.88,
                    "sev": "medium",
                    "src": "192.168.1.165",
                    "dst": "10.0.0.8",
                    "proto": "udp",
                    "svc": "dns",
                    "dur": 0.820,
                    "sbytes": 12800,
                    "dbytes": 512,
                },
                {
                    "type": "Generic",
                    "conf": 0.84,
                    "sev": "low",
                    "src": "192.168.1.99",
                    "dst": "10.0.0.12",
                    "proto": "tcp",
                    "svc": "-",
                    "dur": 0.150,
                    "sbytes": 850,
                    "dbytes": 300,
                },
            ]
            for al in alerts_data:
                create_alert(
                    db=db,
                    attack_type=al["type"],
                    confidence=al["conf"],
                    severity=al["sev"],
                    source_ip=al["src"],
                    destination_ip=al["dst"],
                    protocol=al["proto"],
                    service=al["svc"],
                    flow_duration=al["dur"],
                    source_bytes=al["sbytes"],
                    destination_bytes=al["dbytes"],
                )
            logger.info("Seeded 5 demo intrusion alerts.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
