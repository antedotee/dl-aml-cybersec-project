"""End-to-end Phase-1 experiment: EDA-motivated features + baselines + advanced ML."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cps_ad.advanced_ml import GaussianMixtureAnomalyDetector, OneClassSvmAnomalyDetector
from cps_ad.baselines import MahalanobisAnomalyDetector, max_zscore_anomaly_score
from cps_ad.data import load_kddcup99_sa, train_val_test_indices
from cps_ad.synthetic_data import load_synthetic_intrusion_sa_like
from cps_ad.metrics import DetectionMetrics, best_f1_threshold, evaluate_scores
from cps_ad.preprocess import apply_log1p_columns, build_preprocess_pipeline, numeric_skewness


@dataclass(frozen=True)
class ModelResult:
    name: str
    val_threshold: float
    val_f1_at_threshold: float
    test: DetectionMetrics


def _dense(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def select_gmm_components_by_bic(
    x_normal: np.ndarray,
    *,
    components_range: range,
    random_state: int,
) -> tuple[int, dict[int, float]]:
    from sklearn.mixture import GaussianMixture

    bics: dict[int, float] = {}
    for k in components_range:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            init_params="k-means++",
            n_init=5,
            max_iter=300,
            reg_covar=1e-6,
            random_state=random_state,
        )
        gm.fit(x_normal)
        bics[k] = float(gm.bic(x_normal))
    best_k = int(min(bics, key=bics.get))
    return best_k, bics


def run_phase1(
    *,
    percent10: bool = True,
    random_state: int = 42,
    top_skew_log1p: int = 4,
    gmm_components_search: range | None = None,
    synthetic: bool = False,
) -> tuple[list[ModelResult], dict[str, Any]]:
    if synthetic:
        x_raw, y = load_synthetic_intrusion_sa_like(random_state=random_state)
        dataset_name = "synthetic_intrusion_tabular"
    else:
        x_raw, y = load_kddcup99_sa(percent10=percent10, random_state=random_state)
        dataset_name = "kddcup99_sa"
    skew = numeric_skewness(x_raw)
    log_cols = list(skew.head(top_skew_log1p).index)
    x_fe, skew_reports = apply_log1p_columns(x_raw, log_cols)

    tr_idx, va_idx, te_idx = train_val_test_indices(y, random_state=random_state)
    x_tr, y_tr = x_fe.iloc[tr_idx], y[tr_idx]
    x_va, y_va = x_fe.iloc[va_idx], y[va_idx]
    x_te, y_te = x_fe.iloc[te_idx], y[te_idx]

    normal_mask = (y_tr == 0)
    x_fit = x_tr.iloc[normal_mask]

    pre = build_preprocess_pipeline(x_fit)
    pre.fit(x_fit)

    x_tr_t = _dense(pre.transform(x_tr))
    x_va_t = _dense(pre.transform(x_va))
    x_te_t = _dense(pre.transform(x_te))

    x_fit_t = x_tr_t[normal_mask]

    mu = x_fit_t.mean(axis=0)
    sigma = x_fit_t.std(axis=0)

    # --- Models (fit on benign only) ---
    maha = MahalanobisAnomalyDetector().fit(x_fit_t)

    if gmm_components_search is None:
        gmm_components_search = range(2, 9)
    best_k, bic_map = select_gmm_components_by_bic(
        x_fit_t,
        components_range=gmm_components_search,
        random_state=random_state,
    )
    gmm = GaussianMixtureAnomalyDetector(n_components=best_k, random_state=random_state).fit(x_fit_t)

    ocsvm = OneClassSvmAnomalyDetector(nu=0.06, gamma="scale").fit(x_fit_t)

    def pack(name: str, scores_va: np.ndarray, scores_te: np.ndarray) -> ModelResult:
        thr, f1v = best_f1_threshold(y_va, scores_va)
        test_metrics = evaluate_scores(y_te, scores_te, threshold=thr)
        return ModelResult(name=name, val_threshold=thr, val_f1_at_threshold=f1v, test=test_metrics)

    s_va_z = max_zscore_anomaly_score(x_va_t, mu=mu, sigma=sigma)
    s_te_z = max_zscore_anomaly_score(x_te_t, mu=mu, sigma=sigma)

    results = [
        pack("baseline_max_zscore", s_va_z, s_te_z),
        pack("advanced_mahalanobis_shrunk", maha.score_samples(x_va_t), maha.score_samples(x_te_t)),
        pack("advanced_gmm_nll", gmm.score_samples(x_va_t), gmm.score_samples(x_te_t)),
        pack("advanced_one_class_svm", ocsvm.score_samples(x_va_t), ocsvm.score_samples(x_te_t)),
    ]

    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "synthetic": bool(synthetic),
        "percent10": percent10,
        "random_state": random_state,
        "log1p_columns": log_cols,
        "skew_reports": [asdict(s) for s in skew_reports],
        "gmm_bic_by_k": bic_map,
        "gmm_selected_k": best_k,
        "class_balance": {
            "train": {"normal": int((y_tr == 0).sum()), "attack": int((y_tr == 1).sum())},
            "val": {"normal": int((y_va == 0).sum()), "attack": int((y_va == 1).sum())},
            "test": {"normal": int((y_te == 0).sum()), "attack": int((y_te == 1).sum())},
        },
    }
    return results, meta


def results_to_markdown_table(results: list[ModelResult]) -> str:
    lines = [
        "| Model | Val F1 (thr tuned) | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r.test
        lines.append(
            f"| {r.name} | {r.val_f1_at_threshold:.4f} | {m.roc_auc:.4f} | {m.pr_auc:.4f} | "
            f"{m.f1:.4f} | {m.precision:.4f} | {m.recall:.4f} | {m.threshold:.6g} |"
        )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
