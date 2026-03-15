# Repository architecture

This document describes the **on-disk layout** of the adaptive cyber-physical security (anomaly detection) project: what each path is for, how code is layered, and what is **generated locally** versus **committed to Git**.

---

## Directory tree (what you commit)

Everything below is under the project root `dl-aml-cybersec-project/`.

```
.
├── architecture.md              # This file — layout and module reference
├── README.md                    # Quick start, setup, how to run the notebook
├── requirements.txt             # Python dependencies (includes Jupyter stack)
├── pytest.ini                   # Pytest: pythonpath=src, “slow” marker
├── .gitignore                   # Ignores venv, caches, downloads, generated reports
│
├── notebooks/
│   └── phase1_eval_nsl_mae_ocsvm.ipynb   # Main deliverable: EDA + MAE + OCSVM + ablation A/B/C
│
├── src/
│   └── cps_ad/                  # Importable package (`PYTHONPATH=src` or `pip install -e .` pattern)
│       ├── __init__.py          # Package version string
│       ├── nsl_kdd.py           # NSL-KDD download + CSV parse + binary labels
│       ├── preprocess.py        # Skew audit, log1p, sklearn ColumnTransformer builder
│       ├── metrics.py           # ROC/PR-AUC, F1, threshold tuning helpers
│       ├── baselines.py         # Max-|z| scores, Ledoit–Wolf Mahalanobis detector
│       ├── advanced_ml.py       # GMM NLL anomaly score, One-Class SVM wrapper
│       ├── torch_mae.py         # PyTorch tabular masked autoencoder + training loop
│       ├── hybrid_scores.py     # OCSVM on latents + min–max fusion for hybrid score
│       ├── data.py              # sklearn KDDCup99 “SA” loader + stratified index splits
│       ├── synthetic_data.py    # Offline tabular synthetic intrusion-like data
│       └── phase1.py            # End-to-end KDDCup99/synthetic baseline pipeline (used by tests)
│
├── tests/                       # Pytest suite (no network by default)
│   ├── test_nsl_kdd.py          # NSL column schema + label parsing
│   ├── test_metrics.py          # Threshold / metric helpers
│   ├── test_baselines.py        # Mahalanobis + z-score behavior
│   ├── test_phase1_synthetic.py # Full phase1 run on synthetic data
│   └── test_phase1_slow.py      # Optional: real KDDCup99 download (marked slow)
│
├── data/
│   └── nsl/
│       └── .gitkeep             # Keeps directory in Git; actual *.txt datasets are ignored
│
└── reports/
    ├── figures/
    │   └── .gitkeep             # Keeps directory; PNGs written by notebook are ignored
    └── (no committed JSON)      # Ablation JSON is written here when you run the notebook
```

---

## Runtime / local-only paths (not in Git)

| Path | Origin |
|------|--------|
| `.venv/` | Local virtual environment — never commit |
| `data/nsl/KDDTrain+_20Percent.txt`, `KDDTest+.txt`, … | Downloaded by `nsl_kdd.load_nsl_kdd_frames()` |
| `~/scikit_learn_data/` | sklearn cache if you run `test_phase1_slow` or `fetch_kddcup99` |
| `reports/figures/phase1_nsl_*.png` | Written when you execute the notebook (correlation + KDE plots) |
| `reports/phase1_ablation_nsl.json` | Written by the notebook’s metrics cell |
| `.pytest_cache/`, `__pycache__/` | Tooling caches |

---

## Data and control flow (high level)

1. **Notebook** loads NSL-KDD via `cps_ad.nsl_kdd` → pandas frames → `frames_to_xy`.
2. **EDA** uses raw frames; **feature engineering** applies `log1p` (skew-driven) then `build_preprocess_pipeline`, **fit only on benign** training rows.
3. **Model A (tabular):** scaled vectors → Mahalanobis / GMM / One-Class SVM (`baselines`, `advanced_ml`).
4. **Model B (DL):** same scaled matrix → `train_mae_benign` → reconstruction MSE per row (`torch_mae`).
5. **Model C (hybrid):** frozen encoder → latent `z` → OCSVM on `z` + normalized fusion with recon (`hybrid_scores`).
6. **Metrics** from `metrics.py`: validation threshold for F1, test ROC-AUC / PR-AUC / classification metrics.

The **mermaid diagram** inside the notebook matches this flow for coursework submissions.

---

## Module reference (`src/cps_ad/`)

| File | Responsibility |
|------|----------------|
| `__init__.py` | Package metadata (`__version__`). |
| `nsl_kdd.py` | Column names for NSL-KDD; HTTPS download (certifi); correct `label` / `difficulty` order; `frames_to_xy` drops label/difficulty and returns binary attack labels. |
| `preprocess.py` | `numeric_skewness`, `apply_log1p_columns`, `SkewReport`, `build_preprocess_pipeline` (numeric vs categorical branches). |
| `metrics.py` | `evaluate_scores`, `best_f1_threshold`, ROC helpers — higher score = more anomalous. |
| `baselines.py` | `max_zscore_anomaly_score`, `MahalanobisAnomalyDetector` (Ledoit–Wolf). |
| `advanced_ml.py` | `GaussianMixtureAnomalyDetector` (diag cov, EM), `OneClassSvmAnomalyDetector` (RBF, flipped decision function as score). |
| `torch_mae.py` | `TabularMAE`, masked MSE training, `train_mae_benign`, `reconstruction_mse_per_sample`, `latent_matrix`. |
| `hybrid_scores.py` | `fit_ocsvm_latent`, `ocsvm_latent_scores`, `hybrid_score` (min–max on benign-val references). |
| `data.py` | KDDCup99 SA loader; `train_val_test_indices` (stratified train/val/test). |
| `synthetic_data.py` | Deterministic synthetic tabular data for offline tests / demos. |
| `phase1.py` | Older **KDDCup99 / synthetic** pipeline: GMM BIC selection, combined result table — exercised by `tests/test_phase1_*.py`. |

---

## Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_nsl_kdd.py` | Ensures NSL 43-column layout (`label` before `difficulty`) and `frames_to_xy` correctness. |
| `test_metrics.py` | Sanity checks for F1 thresholding. |
| `test_baselines.py` | Mahalanobis and z-score ordering properties on synthetic arrays. |
| `test_phase1_synthetic.py` | Imports `run_phase1(synthetic=True)` — full sklearn path without NSL download. |
| `test_phase1_slow.py` | Real KDDCup99 fetch — run with `pytest -m slow` only. |

---

## Configuration files

| File | Role |
|------|------|
| `requirements.txt` | Pinned stack: numpy, scipy, pandas, sklearn, torch, matplotlib, seaborn, pytest, certifi, jupyter, nbconvert, ipykernel. |
| `pytest.ini` | Adds `src` to `pythonpath`; registers `slow` marker. |
| `.gitignore` | Prevents committing environments, NSL raw files, figure PNGs, ablation JSON, sklearn data cache. |

---

## Single entrypoint for coursework

**Authoritative runnable narrative:** `notebooks/phase1_eval_nsl_mae_ocsvm.ipynb` (execute locally, then commit the `.ipynb` with outputs if your course requires it).

All other Python under `src/cps_ad/` exists as **imported library code** so the notebook stays readable and tests stay small.
