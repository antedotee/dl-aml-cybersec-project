"""
Classical statistical anomaly baselines.

Mahalanobis distance assumes approximately elliptical normal traffic in the *scaled* feature
space; Ledoit--Wolf shrinkage mitigates ill-conditioned empirical covariance when p is large.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf


def max_zscore_anomaly_score(x: np.ndarray, *, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-row max absolute z-score vs normal reference (simple multivariate guardrail)."""
    z = (x - mu) / np.maximum(sigma, 1e-12)
    return np.max(np.abs(z), axis=1)


class MahalanobisAnomalyDetector:
    """
    Anomaly score = squared Mahalanobis distance to the normal centroid.

    Under multivariate Gaussian assumptions, squared distances are asymptotically chi-square
    with d degrees of freedom; we primarily use empirical quantiles on held-out *normal* data.
    """

    def __init__(self) -> None:
        self._lw = LedoitWolf(store_precision=True, assume_centered=False)

    def fit(self, x_normal: np.ndarray) -> MahalanobisAnomalyDetector:
        self._lw.fit(x_normal)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        loc = self._lw.location_.astype(float)
        prec = self._lw.precision_.astype(float)
        xc = x.astype(float) - loc
        # (n, d) @ (d, d) -> (n, d); row-wise dot with xc
        md2 = np.sum(xc @ prec * xc, axis=1)
        return md2

    def chi2_pvalue(self, x: np.ndarray) -> np.ndarray:
        """Convert Mahalanobis^2 to tail p-values under chi-square(d)."""
        d = x.shape[1]
        s = self.score_samples(x)
        return 1.0 - stats.chi2.cdf(s, df=d)
