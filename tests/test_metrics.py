import numpy as np

from cps_ad.metrics import best_f1_threshold, evaluate_scores


def test_evaluate_scores_at_threshold() -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    s = np.array([0.0, 0.1, 0.9, 1.0], dtype=float)
    m = evaluate_scores(y, s, threshold=0.5)
    assert m.f1 >= 0.99


def test_best_f1_threshold_reasonable() -> None:
    y = np.array([0, 0, 0, 1, 1], dtype=int)
    s = np.array([0.05, 0.1, 0.15, 0.85, 0.95], dtype=float)
    thr, f1 = best_f1_threshold(y, s)
    assert 0.15 < thr <= 0.85
    assert f1 > 0.5
