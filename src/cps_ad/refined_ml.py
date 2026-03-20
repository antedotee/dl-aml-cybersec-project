"""Refined Model A: calibrated ensemble of one-class detectors.

Three benign-only detectors with complementary inductive biases:

1. **One-Class SVM (RBF)** with ``gamma`` chosen by the median heuristic on benign
   pairwise distances (Schölkopf 2001; Caputo et al. 2002).
2. **Isolation Forest** — partitions the feature space and scores by average path
   length (Liu et al. 2008). Robust to high cardinality and irrelevant features.
3. **Robust Mahalanobis (MCD)** — Minimum Covariance Determinant gives a robust
   estimate even when training is mildly contaminated (Rousseeuw 1984).

Score fusion uses **isotonic regression** of each detector's raw score against the
empirical CDF on a benign validation set. Because isotonic calibration is rank
preserving for a fixed detector, fusion never *re-orders* a single detector's
scores — this directly avoids the AUC-degradation pathology we hit in Phase 1's
naive min-max hybrid (see ``hybrid_scores.py``, which we leave untouched).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

LOG = logging.getLogger(__name__)


def median_heuristic_gamma(
    x: np.ndarray,
    *,
    sample_size: int = 2000,
    seed: int = 42,
) -> float:
    """gamma = 1 / (2 * sigma^2) where sigma is the median pairwise distance.

    Returns a usable scalar for sklearn's RBF kernel parameter ``gamma``.
    Falls back to 'scale' (1/n_features) when the sample is degenerate.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        s = x[idx]
    else:
        s = x
    diffs = s[:, None, :] - s[None, :, :]
    d2 = np.sum(diffs ** 2, axis=-1)
    iu = np.triu_indices(d2.shape[0], k=1)
    if iu[0].size == 0:
        return float(1.0 / max(x.shape[1], 1))
    median_sq = float(np.median(d2[iu]))
    if median_sq <= 0.0:
        return float(1.0 / max(x.shape[1], 1))
    return float(1.0 / (2.0 * median_sq))


@dataclass
class _CalibratedDetector:
    name: str
    fitter: object
    calibrator: Optional[IsotonicRegression]


class RefinedAnomalyEnsemble:
    """Three benign-only detectors with isotonic calibration & mean fusion.

    Usage
    -----
    >>> ens = RefinedAnomalyEnsemble().fit(x_train_benign, x_val_benign)
    >>> scores = ens.score(x_test)             # higher = more anomalous (calibrated mean)
    >>> per = ens.score_components(x_test)     # dict of per-detector calibrated scores
    """

    def __init__(
        self,
        *,
        nu: float = 0.05,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        mcd_support_fraction: float | None = None,
        scaler: bool = True,
        weights: dict[str, float] | None = None,
        random_state: int = 42,
    ) -> None:
        self.nu = nu
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.mcd_support_fraction = mcd_support_fraction
        self.use_scaler = scaler
        self.weights = weights or {"ocsvm": 1.0, "iforest": 1.0, "mcd": 1.0}
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._detectors: dict[str, _CalibratedDetector] = {}
        self._gamma: float | None = None
        self._fitted: bool = False

    def _maybe_scale_fit(self, x: np.ndarray) -> np.ndarray:
        if not self.use_scaler:
            return x
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(x)

    def _maybe_scale(self, x: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return x
        return self._scaler.transform(x)

    def _fit_one(self, name: str, fitter: object, x_tr: np.ndarray, val_scores: np.ndarray) -> None:
        # Build an isotonic calibrator: map raw score -> rank in [0, 1] on benign val.
        ranks = np.argsort(np.argsort(val_scores)).astype(np.float64)
        ranks = ranks / max(len(ranks) - 1, 1)
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(val_scores, ranks)
        self._detectors[name] = _CalibratedDetector(name=name, fitter=fitter, calibrator=cal)

    def _raw_scores(self, name: str, x: np.ndarray) -> np.ndarray:
        det = self._detectors[name]
        if name == "ocsvm":
            return -det.fitter.decision_function(x).ravel()
        if name == "iforest":
            return -det.fitter.score_samples(x).ravel()
        if name == "mcd":
            return det.fitter.mahalanobis(x).ravel()
        raise KeyError(name)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray) -> "RefinedAnomalyEnsemble":
        x_tr = self._maybe_scale_fit(x_train)
        x_va = self._maybe_scale(x_val)

        # OCSVM (RBF, median-heuristic gamma).
        self._gamma = median_heuristic_gamma(x_tr, seed=self.random_state)
        ocsvm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self._gamma)
        LOG.info("Fitting OCSVM (gamma=%.4g, nu=%.3g) on %d benign rows",
                 self._gamma, self.nu, x_tr.shape[0])
        ocsvm.fit(x_tr)
        self._detectors["ocsvm"] = _CalibratedDetector("ocsvm", ocsvm, None)

        # Isolation Forest.
        iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        LOG.info("Fitting IsolationForest (n_estimators=%d)", self.n_estimators)
        iforest.fit(x_tr)
        self._detectors["iforest"] = _CalibratedDetector("iforest", iforest, None)

        # Robust Mahalanobis (MCD).
        kwargs = {"random_state": self.random_state}
        if self.mcd_support_fraction is not None:
            kwargs["support_fraction"] = self.mcd_support_fraction
        mcd = MinCovDet(**kwargs)
        LOG.info("Fitting MinCovDet")
        mcd.fit(x_tr)
        self._detectors["mcd"] = _CalibratedDetector("mcd", mcd, None)

        # Calibrate each detector on the benign val set.
        for name in ["ocsvm", "iforest", "mcd"]:
            val_scores = self._raw_scores(name, x_va)
            self._fit_one(name, self._detectors[name].fitter, x_tr, val_scores)
        self._fitted = True
        return self

    def _calibrated(self, name: str, x: np.ndarray) -> np.ndarray:
        cal = self._detectors[name].calibrator
        raw = self._raw_scores(name, x)
        if cal is None:
            return raw
        base = cal.predict(raw)
        # Isotonic clips out-of-range values; that destroys rank information
        # for true outliers (everything maps to 1.0). We add a tiny rank-based
        # tiebreak so the overall mapping is *strictly* rank-preserving while
        # the dominant signal remains the calibrated probability.
        if raw.size > 1:
            r_lo, r_hi = float(raw.min()), float(raw.max())
            span = max(r_hi - r_lo, 1e-12)
            tiebreak = (raw - r_lo) / span
            base = base + 1e-6 * tiebreak
        return base

    def score_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("RefinedAnomalyEnsemble.fit() must be called first.")
        x_t = self._maybe_scale(x)
        return {name: self._calibrated(name, x_t) for name in self._detectors}

    def score(self, x: np.ndarray) -> np.ndarray:
        comps = self.score_components(x)
        total_w = 0.0
        accum: np.ndarray | None = None
        for name, val in comps.items():
            w = float(self.weights.get(name, 1.0))
            if w <= 0.0:
                continue
            if accum is None:
                accum = w * val
            else:
                accum = accum + w * val
            total_w += w
        if accum is None:
            return np.zeros(x.shape[0], dtype=np.float64)
        return accum / max(total_w, 1e-12)

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise RuntimeError("Ensemble not fitted yet.")
        return self._gamma
