"""
Advanced (non-deep) probabilistic / kernel detectors fit on benign traffic only.

GaussianMixture:
  - EM alternates E-step (responsibilities) and M-step (weighted MLE) with monotonic
    incomplete-data log-likelihood until convergence tolerances are met.
  - We rank anomalies by *negative* log-likelihood; lower density under the benign mixture
    implies higher anomaly score.

OneClassSVM:
  - RBF kernel maps flows to an RKHS where a tight enclosing hyperplane separates the origin
    (anomalies) from the bulk of training points; `nu` upper-bounds the fraction of training
    outliers and SV ratio.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM


class GaussianMixtureAnomalyDetector:
    def __init__(self, *, n_components: int = 6, random_state: int = 42) -> None:
        self._model = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            init_params="k-means++",
            n_init=5,
            max_iter=300,
            reg_covar=1e-6,
            random_state=random_state,
        )

    def fit(self, x_normal: np.ndarray) -> GaussianMixtureAnomalyDetector:
        self._model.fit(x_normal)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        # Higher score => more anomalous
        return -self._model.score_samples(x)


class OneClassSvmAnomalyDetector:
    def __init__(self, *, nu: float = 0.08, gamma: str | float = "scale") -> None:
        self._model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    def fit(self, x_normal: np.ndarray) -> OneClassSvmAnomalyDetector:
        self._model.fit(x_normal)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        # sklearn: decision_function > 0 for inliers; flip so higher => more anomalous
        return -self._model.decision_function(x)
