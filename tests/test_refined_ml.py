"""Tests for the refined ML ensemble (OCSVM + IsoForest + MCD with calibration)."""

from __future__ import annotations

import numpy as np
import pytest

from cps_ad.refined_ml import RefinedAnomalyEnsemble, median_heuristic_gamma


def _benign(n: int, d: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


def test_median_heuristic_gamma_positive() -> None:
    x = _benign(64, 4)
    g = median_heuristic_gamma(x, sample_size=64)
    assert g > 0


def test_ensemble_fit_score_shapes() -> None:
    x_tr = _benign(200, 6, seed=1)
    x_va = _benign(50, 6, seed=2)
    ens = RefinedAnomalyEnsemble(nu=0.05, n_estimators=50).fit(x_tr, x_va)
    s = ens.score(x_va)
    assert s.shape == (50,)
    parts = ens.score_components(x_va)
    assert set(parts) == {"ocsvm", "iforest", "mcd"}
    for v in parts.values():
        assert v.shape == (50,)
        assert np.all(np.isfinite(v))


def test_ensemble_separates_outliers() -> None:
    x_tr = _benign(400, 5, seed=3)
    x_va = _benign(80, 5, seed=4)
    rng = np.random.default_rng(99)
    x_out = rng.standard_normal((80, 5)) * 4.0 + 6.0  # heavily shifted

    ens = RefinedAnomalyEnsemble(nu=0.05, n_estimators=80).fit(x_tr, x_va)
    s_in = ens.score(x_va)
    s_out = ens.score(x_out)
    assert s_out.mean() > s_in.mean()


def test_calibration_is_rank_preserving_per_detector() -> None:
    """Isotonic calibration must not invert order on a single detector's output."""
    x_tr = _benign(200, 4, seed=5)
    x_va = _benign(60, 4, seed=6)
    test = np.vstack([_benign(40, 4, seed=7), _benign(40, 4, seed=8) + 5.0])

    ens = RefinedAnomalyEnsemble(nu=0.05, n_estimators=60).fit(x_tr, x_va)
    parts = ens.score_components(test)

    # Compare ranks of calibrated scores vs raw scores per detector.
    for name in parts:
        det = ens._detectors[name]
        raw = ens._raw_scores(name, ens._maybe_scale(test))
        cal = parts[name]
        order_raw = np.argsort(np.argsort(raw))
        order_cal = np.argsort(np.argsort(cal))
        # Pearson correlation between rank arrays should be ~1 (allow ties).
        corr = np.corrcoef(order_raw, order_cal)[0, 1]
        assert corr > 0.99, f"{name} calibration scrambled order (corr={corr:.4f})"
