import numpy as np

from cps_ad.baselines import MahalanobisAnomalyDetector, max_zscore_anomaly_score


def test_max_zscore_increases_for_shift() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(500, 4))
    mu = x0.mean(axis=0)
    sigma = x0.std(axis=0)
    x1 = x0.copy()
    x1[:, 0] += 12.0
    s0 = max_zscore_anomaly_score(x0, mu=mu, sigma=sigma)
    s1 = max_zscore_anomaly_score(x1, mu=mu, sigma=sigma)
    assert float(s1.mean()) > float(s0.mean())


def test_mahalanobis_ranks_outliers() -> None:
    rng = np.random.default_rng(1)
    xn = rng.normal(scale=0.5, size=(800, 3))
    det = MahalanobisAnomalyDetector().fit(xn)
    x_out = np.vstack([xn, np.array([[6.0, 6.0, 6.0]])])
    scores = det.score_samples(x_out)
    assert float(scores[-1]) > float(np.quantile(scores[:-1], 0.99))
