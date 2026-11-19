"""Unit tests for network traffic data cleaning, categorical encoding, and scaling."""

import numpy as np
import pandas as pd
import pytest

from src.ml.preprocessing.cleaner import DataCleaner
from src.ml.preprocessing.encoder import NetworkCategoricalEncoder
from src.ml.preprocessing.scaler import NetworkFeatureScaler
from src.ml.preprocessing.pipeline import NetSentryPreprocessor


@pytest.fixture
def sample_raw_dataframe():
    """Generates a small test DataFrame with edge cases: NaNs, Infs, whitespaces, unseen categories."""
    return pd.DataFrame({
        "proto": ["tcp ", "udp", "icmp", "tcp", "unknown_proto"],
        "service": ["http", "dns", "-", "http", "ftp"],
        "state": ["FIN", "CON", "INT", "FIN", "REQ"],
        "dur": [0.12, np.inf, 1.5, np.nan, 0.05],
        "sbytes": [1000, 2500, 150, 4000, -10],  # negative to test clipping
        "dbytes": [200, 450, 0, 1200, 30],
        "sttl": [64, 64, 254, 62, 31],
        "Spkts": [10, 20, 2, 35, 1],
        "Dpkts": [8, 15, 0, 28, 1],
        "label": [0, 1, 0, 1, 0],
        "attack_cat": ["Normal", "DoS", "Normal", "Exploits", "Normal"],
    })


def test_data_cleaner(sample_raw_dataframe):
    """Verify cleaner eliminates infs, handles negative values, and imputes nulls."""
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(sample_raw_dataframe)

    assert not df_clean.isnull().any().any()
    assert not np.isinf(df_clean["dur"]).any()
    assert (df_clean["sbytes"] >= 0).all()
    assert df_clean["proto"].iloc[0] == "tcp"


def test_categorical_encoder(sample_raw_dataframe):
    """Verify categorical encoder creates dummy columns and handles unseen categories."""
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(sample_raw_dataframe)

    encoder = NetworkCategoricalEncoder()
    df_encoded = encoder.fit_transform(df_clean)

    assert "proto" not in df_encoded.columns
    assert any(c.startswith("proto_") for c in df_encoded.columns)
    assert any(c.startswith("service_") for c in df_encoded.columns)

    # Test inference on previously unseen category
    unseen_df = pd.DataFrame({
        "proto": ["super_new_protocol"],
        "service": ["rare_svc"],
        "state": ["FIN"],
        "dur": [0.5],
        "sbytes": [100],
        "dbytes": [100],
        "sttl": [64],
        "Spkts": [2],
        "Dpkts": [2],
    })
    encoded_unseen = encoder.transform(unseen_df)
    assert len(encoded_unseen) == 1


def test_feature_scaler(sample_raw_dataframe):
    """Verify feature scaler normalizes numerical features without altering label."""
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(sample_raw_dataframe)

    scaler = NetworkFeatureScaler(method="standard")
    df_scaled = scaler.fit_transform(df_clean)

    assert np.isclose(df_scaled["sttl"].mean(), 0.0, atol=1e-1)
    # Ensure label column was untouched
    assert (df_scaled["label"] == sample_raw_dataframe["label"]).all()


def test_full_pipeline_split(sample_raw_dataframe):
    """Verify complete preprocessor pipeline transforms and splits X, y, attack_cat."""
    preprocessor = NetSentryPreprocessor(scaling_method="standard")
    preprocessor.fit(sample_raw_dataframe)

    X, y, attack_cats = preprocessor.split_X_y(sample_raw_dataframe)

    assert len(X) == 5
    assert len(y) == 5
    assert len(attack_cats) == 5
    assert y.iloc[0] == 0
    assert y.iloc[1] == 1
    assert "label" not in X.columns
    assert "attack_cat" not in X.columns
