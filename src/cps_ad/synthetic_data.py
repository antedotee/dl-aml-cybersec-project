"""Deterministic synthetic flow-like tabular data for offline runs when remote fetch fails."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_synthetic_intrusion_sa_like(*, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Produce a mixed categorical/numeric table with separable benign vs attack clusters.

    This is **not** a security benchmark; it exists to validate the training/eval plumbing
    when `fetch_kddcup99` cannot download (air-gapped machines, SSL misconfiguration, etc.).
    """
    rng = np.random.default_rng(random_state)
    n_normal = 12_000
    n_attack = 3_500
    d_num = 28
    mu_n = rng.normal(scale=1.0, size=d_num)
    mu_a = mu_n + rng.normal(scale=2.2, size=d_num)

    xn = rng.normal(loc=mu_n, scale=0.35, size=(n_normal, d_num))
    xa = rng.normal(loc=mu_a, scale=0.55, size=(n_attack, d_num))
    x = np.vstack([xn, xa])
    y = np.concatenate([np.zeros(n_normal, dtype=np.int32), np.ones(n_attack, dtype=np.int32)])

    perm = rng.permutation(len(y))
    x = x[perm]
    y = y[perm]

    cols = {
        "duration": np.abs(rng.normal(scale=40.0, size=len(y))),
        "src_bytes": np.abs(rng.normal(scale=500.0, size=len(y))),
        "dst_bytes": np.abs(rng.normal(scale=800.0, size=len(y))),
    }
    for j in range(d_num):
        cols[f"stat_{j}"] = x[:, j]

    df = pd.DataFrame(cols)
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], size=len(y))
    df["service"] = rng.choice(["http", "smtp", "ftp", "private", "domain", "other"], size=len(y))
    df["flag"] = rng.choice(["SF", "S0", "REJ", "RSTO"], size=len(y))
    return df, y
