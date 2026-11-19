"""Synthetic network flow dataset generator adhering to the UNSW-NB15 schema."""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from src.core.constants import (
    COMMON_PROTOCOLS,
    COMMON_SERVICES,
    COMMON_STATES,
    UNSW_CATEGORICAL_COLUMNS,
    UNSW_NUMERICAL_COLUMNS,
)


def generate_unsw_sample(
    num_rows: int = 1500,
    attack_ratio: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic synthetic network flow records compliant with the UNSW-NB15 benchmark."""
    rng = np.random.default_rng(seed)

    num_attacks = int(num_rows * attack_ratio)
    num_normal = num_rows - num_attacks

    labels = np.array([0] * num_normal + [1] * num_attacks)
    rng.shuffle(labels)

    attack_categories = []
    protos = []
    services = []
    states = []

    attack_types = ["DoS", "Exploits", "Reconnaissance", "Fuzzers", "Generic"]

    for lbl in labels:
        if lbl == 0:
            attack_categories.append("Normal")
            protos.append(rng.choice(["tcp", "udp", "icmp"], p=[0.75, 0.20, 0.05]))
            services.append(rng.choice(["http", "dns", "ftp", "smtp", "ssh", "-"], p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.1]))
            states.append(rng.choice(["FIN", "CON", "INT"], p=[0.8, 0.15, 0.05]))
        else:
            cat = rng.choice(attack_types, p=[0.35, 0.25, 0.20, 0.10, 0.10])
            attack_categories.append(cat)
            if cat == "DoS":
                protos.append(rng.choice(["tcp", "udp", "icmp"], p=[0.5, 0.3, 0.2]))
                services.append(rng.choice(["http", "dns", "-"], p=[0.5, 0.2, 0.3]))
                states.append(rng.choice(["REQ", "INT", "RST"], p=[0.5, 0.3, 0.2]))
            elif cat == "Reconnaissance":
                protos.append(rng.choice(["tcp", "icmp"], p=[0.8, 0.2]))
                services.append("-")
                states.append("INT")
            else:
                protos.append(rng.choice(["tcp", "udp"], p=[0.85, 0.15]))
                services.append(rng.choice(["http", "ssh", "ftp", "-"], p=[0.4, 0.2, 0.2, 0.2]))
                states.append(rng.choice(["FIN", "REQ", "RST"], p=[0.4, 0.4, 0.2]))

    # Generate numerical features with distinct statistical distributions for normal vs attack
    data_dict = {
        "proto": protos,
        "service": services,
        "state": states,
    }

    is_attack = labels == 1

    # Duration
    dur_normal = rng.exponential(scale=1.2, size=num_rows)
    dur_attack = rng.exponential(scale=0.3, size=num_rows)
    data_dict["dur"] = np.where(is_attack, dur_attack, dur_normal).round(4)

    # Source & Dest bytes
    sbytes_normal = rng.integers(100, 3000, size=num_rows)
    sbytes_attack = rng.integers(500, 50000, size=num_rows)
    data_dict["sbytes"] = np.where(is_attack, sbytes_attack, sbytes_normal)

    dbytes_normal = rng.integers(100, 8000, size=num_rows)
    dbytes_attack = rng.integers(0, 1500, size=num_rows)
    data_dict["dbytes"] = np.where(is_attack, dbytes_attack, dbytes_normal)

    # TTL
    sttl_normal = rng.choice([31, 62, 64, 254], size=num_rows, p=[0.1, 0.3, 0.5, 0.1])
    sttl_attack = rng.choice([62, 64, 254], size=num_rows, p=[0.2, 0.2, 0.6])
    data_dict["sttl"] = np.where(is_attack, sttl_attack, sttl_normal)

    dttl_normal = rng.choice([29, 31, 60, 252], size=num_rows)
    dttl_attack = rng.choice([0, 29, 252], size=num_rows, p=[0.5, 0.2, 0.3])
    data_dict["dttl"] = np.where(is_attack, dttl_attack, dttl_normal)

    # Packet counts
    spkts_normal = rng.integers(2, 40, size=num_rows)
    spkts_attack = rng.integers(1, 200, size=num_rows)
    data_dict["Spkts"] = np.where(is_attack, spkts_attack, spkts_normal)

    dpkts_normal = rng.integers(2, 50, size=num_rows)
    dpkts_attack = rng.integers(0, 10, size=num_rows)
    data_dict["Dpkts"] = np.where(is_attack, dpkts_attack, dpkts_normal)

    # Loss
    data_dict["sloss"] = (data_dict["Spkts"] * rng.uniform(0, 0.05, size=num_rows)).astype(int)
    data_dict["dloss"] = (data_dict["Dpkts"] * rng.uniform(0, 0.05, size=num_rows)).astype(int)

    # Load
    dur_safe = np.maximum(data_dict["dur"], 0.001)
    data_dict["Sload"] = (data_dict["sbytes"] * 8 / dur_safe).round(2)
    data_dict["Dload"] = (data_dict["dbytes"] * 8 / dur_safe).round(2)

    # Window sizes
    data_dict["swin"] = np.where(is_attack, rng.choice([0, 255], size=num_rows), 255)
    data_dict["dwin"] = np.where(is_attack, rng.choice([0, 255], size=num_rows), 255)

    # TCP sequence base
    data_dict["stcpb"] = rng.integers(1000000, 4000000000, size=num_rows, dtype=np.int64)
    data_dict["dtcpb"] = rng.integers(1000000, 4000000000, size=num_rows, dtype=np.int64)

    # Mean packet size
    data_dict["smeansz"] = (data_dict["sbytes"] / np.maximum(data_dict["Spkts"], 1)).astype(int)
    data_dict["dmeansz"] = (data_dict["dbytes"] / np.maximum(data_dict["Dpkts"], 1)).astype(int)

    # Jitter, transaction depth, response body length
    data_dict["trans_depth"] = np.where(is_attack, 0, rng.choice([0, 1], size=num_rows, p=[0.7, 0.3]))
    data_dict["res_bdy_len"] = np.where(is_attack, 0, rng.integers(0, 2048, size=num_rows))
    data_dict["sjit"] = rng.uniform(0, 150, size=num_rows).round(4)
    data_dict["djit"] = rng.uniform(0, 150, size=num_rows).round(4)
    data_dict["sintpkt"] = rng.uniform(0.001, 50.0, size=num_rows).round(4)
    data_dict["dintpkt"] = rng.uniform(0.001, 50.0, size=num_rows).round(4)
    data_dict["tcprtt"] = rng.uniform(0.0001, 0.5, size=num_rows).round(5)
    data_dict["synack"] = (data_dict["tcprtt"] * 0.4).round(5)
    data_dict["ackdat"] = (data_dict["tcprtt"] * 0.6).round(5)

    # Connection and state counters
    data_dict["is_sm_ips_ports"] = np.where(is_attack, rng.choice([0, 1], size=num_rows, p=[0.9, 0.1]), 0)
    data_dict["ct_state_ttl"] = rng.integers(0, 6, size=num_rows)
    data_dict["ct_flw_http_mthd"] = np.where(is_attack, 0, rng.choice([0, 1, 2], size=num_rows, p=[0.8, 0.15, 0.05]))
    data_dict["is_ftp_login"] = np.where(is_attack, 0, rng.choice([0, 1], size=num_rows, p=[0.95, 0.05]))
    data_dict["ct_ftp_cmd"] = data_dict["is_ftp_login"]
    data_dict["ct_srv_src"] = np.where(is_attack, rng.integers(5, 45, size=num_rows), rng.integers(1, 10, size=num_rows))
    data_dict["ct_srv_dst"] = np.where(is_attack, rng.integers(5, 45, size=num_rows), rng.integers(1, 10, size=num_rows))
    data_dict["ct_dst_ltm"] = rng.integers(1, 25, size=num_rows)
    data_dict["ct_src_ltm"] = rng.integers(1, 25, size=num_rows)
    data_dict["ct_src_dport_ltm"] = np.where(is_attack, rng.integers(5, 30, size=num_rows), rng.integers(1, 5, size=num_rows))
    data_dict["ct_dst_sport_ltm"] = rng.integers(1, 15, size=num_rows)
    data_dict["ct_dst_src_ltm"] = np.where(is_attack, rng.integers(5, 40, size=num_rows), rng.integers(1, 10, size=num_rows))

    # Targets
    data_dict["attack_cat"] = attack_categories
    data_dict["label"] = labels

    df = pd.DataFrame(data_dict)
    return df


def create_sample_datasets(output_dir: str = "data/samples") -> Path:
    """Generates and writes train, test, and benchmark CSV datasets."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training set: 2,000 flows
    train_df = generate_unsw_sample(num_rows=2000, attack_ratio=0.35, seed=42)
    train_path = out_dir / "unsw_nb15_sample_train.csv"
    train_df.to_csv(train_path, index=False)

    # 2. Test / inference evaluation set: 500 flows
    test_df = generate_unsw_sample(num_rows=500, attack_ratio=0.40, seed=100)
    test_path = out_dir / "unsw_nb15_sample_test.csv"
    test_df.to_csv(test_path, index=False)

    # 3. Unlabeled live traffic simulation for batch prediction testing
    unlabeled_df = test_df.drop(columns=["label", "attack_cat"])
    unlabeled_path = out_dir / "unlabeled_live_traffic_feed.csv"
    unlabeled_df.to_csv(unlabeled_path, index=False)

    return out_dir
