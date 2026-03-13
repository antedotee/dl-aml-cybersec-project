"""Combine reconstruction error with One-Class SVM margin on latent vectors."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def minmax_on_reference(values_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values_ref))
    hi = float(np.max(values_ref))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def fit_ocsvm_latent(
    z_train_normal: np.ndarray,
    *,
    nu: float = 0.06,
    gamma: str | float = "scale",
) -> tuple[OneClassSVM, StandardScaler]:
    """RBF OCSVM on benign latent embeddings (z-scored for scale invariance)."""
    sc = StandardScaler()
    zn = sc.fit_transform(z_train_normal)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    ocsvm.fit(zn)
    return ocsvm, sc


def ocsvm_latent_scores(ocsvm: OneClassSVM, sc: StandardScaler, z: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (outliers have more negative decision_function in sklearn)."""
    zs = sc.transform(z)
    return -ocsvm.decision_function(zs).ravel()


def hybrid_score(
    recon: np.ndarray,
    ocsvm_s: np.ndarray,
    *,
    recon_ref: np.ndarray,
    ocsvm_ref: np.ndarray,
    w_recon: float = 0.5,
) -> np.ndarray:
    """Convex combination of min–max normalized scores (fit min/max on benign validation)."""
    r = minmax_on_reference(recon_ref, recon)
    o = minmax_on_reference(ocsvm_ref, ocsvm_s)
    return w_recon * r + (1.0 - w_recon) * o
