"""Load public network-intrusion benchmarks for semi-supervised anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99


def _binary_labels_from_kdd(series: pd.Series) -> np.ndarray:
    """Return y in {0,1} with 1 = attack / anomaly, 0 = normal."""

    def _as_label_str(v: object) -> str:
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="replace").strip()
        return str(v).strip()

    s = series.map(_as_label_str).str.lower()
    y = (~s.eq("normal.")).astype(np.int32).to_numpy()
    return y


def load_kddcup99_sa(*, percent10: bool = True, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    KDDCup99 'SA' subset: normal traffic plus a fixed-size abnormal sample (see sklearn docs).

    Semi-supervised framing: models are fit only on *normal* connections; evaluation uses
    the full held-out mix (normal + attack).
    """
    bunch = fetch_kddcup99(
        subset="SA",
        percent10=percent10,
        shuffle=True,
        random_state=random_state,
        as_frame=True,
    )
    frame = bunch.frame
    if frame is None:
        raise RuntimeError("Expected pandas frame when as_frame=True.")
    label_col = "labels" if "labels" in frame.columns else frame.columns[-1]
    y = _binary_labels_from_kdd(frame[label_col])
    x = frame.drop(columns=[label_col])
    return x, y


def train_val_test_indices(
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    val_fraction_of_trainval: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split: (train, val, test). Val is a fraction of the pre-test pool."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr_va, te = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    tr, va = train_test_split(
        tr_va,
        test_size=val_fraction_of_trainval,
        stratify=y[tr_va],
        random_state=random_state,
    )
    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int), np.asarray(te, dtype=int)
