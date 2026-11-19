"""End-to-end data preprocessing pipeline orchestrating cleaning, encoding, and scaling."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.core.constants import UNSW_TARGET_LABEL, UNSW_ATTACK_CAT
from src.ml.preprocessing.cleaner import DataCleaner
from src.ml.preprocessing.encoder import NetworkCategoricalEncoder
from src.ml.preprocessing.scaler import NetworkFeatureScaler


class NetSentryPreprocessor(BaseEstimator, TransformerMixin):
    """Unified preprocessing pipeline transforming raw network flows into ML-ready feature matrices."""

    def __init__(
        self,
        scaling_method: str = "standard",
        drop_duplicates: bool = False,
        categorical_columns: Optional[List[str]] = None,
    ):
        self.scaling_method = scaling_method
        self.drop_duplicates = drop_duplicates
        self.categorical_columns = categorical_columns or ["proto", "service", "state"]

        self.cleaner = DataCleaner(drop_duplicates=drop_duplicates)
        self.encoder = NetworkCategoricalEncoder(categorical_columns=self.categorical_columns)
        self.scaler = NetworkFeatureScaler(method=scaling_method)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NetSentryPreprocessor":
        """Fits cleaner, categorical encoder, and scaler sequentially on training flow records."""
        # 1. Clean
        df_clean = self.cleaner.fit_transform(X)

        # 2. Encode
        df_encoded = self.encoder.fit_transform(df_clean)

        # 3. Scale
        df_scaled = self.scaler.fit_transform(df_encoded)

        # Store resulting feature column names
        exclude = [UNSW_TARGET_LABEL, UNSW_ATTACK_CAT, "Label", "attack_cat", "id", "srcip", "dstip"]
        self.feature_names_ = [c for c in df_scaled.columns if c not in exclude]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies sequential transformation pipeline to input records."""
        df_clean = self.cleaner.transform(X)
        df_encoded = self.encoder.transform(df_clean)
        df_scaled = self.scaler.transform(df_encoded)
        return df_scaled

    def split_X_y(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """Splits processed DataFrame into feature matrix X, binary target y, and multiclass attack_cat."""
        df_trans = self.transform(df)

        y = None
        for lbl_col in [UNSW_TARGET_LABEL, "Label"]:
            if lbl_col in df_trans.columns:
                y = df_trans[lbl_col].astype(int)
                break

        attack_cats = None
        for cat_col in [UNSW_ATTACK_CAT, "attack_cat"]:
            if cat_col in df_trans.columns:
                attack_cats = df_trans[cat_col].astype(str)
                break

        # Keep only trained feature columns if available, or numerical/encoded columns
        if self.feature_names_:
            available_cols = [c for c in self.feature_names_ if c in df_trans.columns]
            X = df_trans[available_cols].copy()
            # If any features missing, fill with 0
            for missing_col in self.feature_names_:
                if missing_col not in X.columns:
                    X[missing_col] = 0.0
            X = X[self.feature_names_]
        else:
            exclude = [UNSW_TARGET_LABEL, UNSW_ATTACK_CAT, "Label", "attack_cat", "id", "srcip", "dstip"]
            X = df_trans.drop(columns=[c for c in exclude if c in df_trans.columns])

        return X, y, attack_cats
