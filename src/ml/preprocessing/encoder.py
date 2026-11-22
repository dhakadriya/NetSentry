"""Categorical encoding for network protocol, service, and connection state features."""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class NetworkCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical flow features with robust unseen category handling."""

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        max_categories: int = 20,
    ):
        self.categorical_columns = categorical_columns or ["proto", "service", "state"]
        self.max_categories = max_categories
        self.encoder_: Optional[OneHotEncoder] = None
        self.active_cat_cols_: List[str] = []
        self.top_categories_: Dict[str, Set[str]] = {}
        self.encoded_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NetworkCategoricalEncoder":
        """Fits OneHotEncoder on active categorical columns, pruning rare tail categories."""
        df = X.copy()
        self.active_cat_cols_ = [c for c in self.categorical_columns if c in df.columns]

        if not self.active_cat_cols_:
            return self

        # Extract top categories to prevent dimensional explosion on rare protocols/services
        processed_data = {}
        for col in self.active_cat_cols_:
            top_vals = set(df[col].astype(str).value_counts().nlargest(self.max_categories).index)
            self.top_categories_[col] = top_vals
            processed_data[col] = df[col].astype(str).where(df[col].isin(top_vals), other="other")

        cat_df = pd.DataFrame(processed_data, index=df.index)

        self.encoder_ = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.encoder_.fit(cat_df)
        self.encoded_feature_names_ = list(self.encoder_.get_feature_names_out(self.active_cat_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame by encoding categorical columns with unseen level fallbacks."""
        df = X.copy()

        if not self.active_cat_cols_ or self.encoder_ is None:
            return df

        processed_data = {}
        for col in self.active_cat_cols_:
            if col in df.columns:
                top_vals = self.top_categories_.get(col, set())
                processed_data[col] = df[col].astype(str).where(df[col].isin(top_vals), other="other")
            else:
                processed_data[col] = ["other"] * len(df)

        cat_df = pd.DataFrame(processed_data, index=df.index)
        encoded_arr = self.encoder_.transform(cat_df)
        encoded_df = pd.DataFrame(
            encoded_arr,
            columns=self.encoded_feature_names_,
            index=df.index,
        )

        # Drop original raw categorical columns and join encoded ones
        non_cat_df = df.drop(columns=[c for c in self.active_cat_cols_ if c in df.columns])
        result = pd.concat([non_cat_df, encoded_df], axis=1)
        return result

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns the list of encoded one-hot column names."""
        return self.encoded_feature_names_
