"""Feature construction for mixed categorical / numeric flow records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SkewReport:
    """EDA-driven skew diagnostics used to justify log1p transforms."""

    column: str
    skew_before: float
    skew_after: float


def numeric_skewness(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number])
    return num.skew(axis=0, skipna=True).sort_values(ascending=False)


def apply_log1p_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[SkewReport]]:
    """Apply log1p to non-negative heavy-tailed count/byte columns (EDA-motivated)."""
    out = df.copy()
    reports: list[SkewReport] = []
    for col in columns:
        if col not in out.columns:
            continue
        before = float(out[col].skew(skipna=True))
        shifted = out[col].astype(float).clip(lower=0.0)
        out[col] = np.log1p(shifted)
        after = float(out[col].skew(skipna=True))
        reports.append(SkewReport(column=col, skew_before=before, skew_after=after))
    return out, reports


def build_preprocess_pipeline(x_train: pd.DataFrame) -> ColumnTransformer:
    """
    Encode categorical protocol fields and scale numeric statistics.

    One-hot expansion can increase dimensionality; for tree/kernel methods this is acceptable
    here because SA subset cardinality is moderate and sklearn handles sparsity internally.
    """
    num_cols = list(x_train.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    if not cat_cols:
        return ColumnTransformer(
            transformers=[("num", num_pipe, num_cols)],
            remainder="drop",
        )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
