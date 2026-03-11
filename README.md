# Adaptive Cyber-Physical Security — Phase 1 (Anomaly Detection)

Industrial and enterprise networks must flag **zero-day** behavior without attack signatures. This repository implements a **semi-supervised** intrusion-detection baseline: models see **benign traffic only** at training time, then score held-out traffic containing attacks.

Phase 1 focuses on **dataset understanding**, **transparent feature construction**, **classical statistical baselines**, and **advanced probabilistic / kernel detectors** (Gaussian Mixture + One-Class SVM). Deep learning and the hybrid system are reserved for later phases, but the evaluation hooks and ablation scaffolding match the course specification.

## Literature anchors (what to cite in your report)

- Chandola, Banerjee & Kumar, *Anomaly Detection: A Survey* (ACM Computing Surveys, 2009) — taxonomy of anomaly detection and evaluation pitfalls.
- Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution* (Neural Computation, 2001) — One-Class SVM / SVDD perspective.
- Tax & Duin, *Support Vector Data Description* (Machine Learning, 2004) — geometric interpretation of one-class kernels.
- Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006) — EM for mixture models, Gaussian mixtures, model selection (BIC).
- Lakhina, Crovella & Diot, *Mining Anomalies Using Traffic Feature Distributions* (SIGCOMM MineNet, 2005) — multivariate flow-feature anomaly motivation.
- Mahoney & Chan, *Learning Nonstationary Models of Normal Network Traffic* (RAID, 2002) — cautionary notes on KDDCup99 evaluation non-stationarity (acknowledge limitations).

## Theory checklist (viva / report)

- **Semi-supervised IDS framing**: training distribution is *predominantly normal*; attacks are **out-of-distribution** under the learned benign model.
- **Gaussian Mixture (EM)**: E-step computes responsibilities \(q_{ik}\propto \pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\); M-step updates \(\pi_k,\mu_k,\Sigma_k\) by weighted MLE. The incomplete-data log-likelihood is non-decreasing; convergence is to a **local** optimum → multiple `n_init` restarts.
- **Model selection**: **BIC** penalizes complexity; we use it to pick `n_components` on *benign-only* training data (your report should justify why this is reasonable for density estimation).
- **One-Class SVM (RBF)**: separates the training bulk from the origin in the RKHS induced by the kernel; **`nu`** upper-bounds the training false-positive fraction and influences support-vector sparsity.
- **Mahalanobis distance + Ledoit–Wolf**: if benign traffic is approximately elliptical after scaling, squared distances follow a \(\chi^2_d\) law under Gaussian assumptions; shrinkage stabilizes \(\Sigma^{-1}\) when \(d\) grows (many one-hot dimensions).

## Data

We load the sklearn-hosted **KDDCup99** extract (`fetch_kddcup99`, `subset="SA"`). This keeps the project **reproducible** without manual mirrors. Your report should still discuss dataset age and **concept drift** as limitations (link to Mahoney & Chan).

## Methods implemented (Phase 1)

1. **EDA-driven transforms**: identify heavy-tailed numeric flow statistics via skewness; apply `log1p` where it materially reduces skew (see `run_eda.py` output).
2. **Preprocessing**: median imputation + `StandardScaler` for numeric fields; `OneHotEncoder(handle_unknown="ignore")` for symbolic fields (`protocol_type`, `service`, `flag`, …).
3. **Baselines**
   - **Max |z|** anomaly score vs benign column means (simple multivariate guardrail).
   - **Shrunk Mahalanobis** anomaly score (`LedoitWolf`).
4. **Advanced ML**
   - **GMM** with **diagonal** covariances per component (scalable under high-dimensional one-hot expansions; still a full EM mixture).
   - **One-Class SVM** (RBF, tuned `nu` in code; extend with grid search in Phase 2).

**Thresholding protocol (transparent, defensible):** each model produces a scalar **anomaly score** (higher ⇒ more suspicious). We tune a single threshold on the **validation** split to maximize **F1**, then freeze it for **test** reporting. ROC-AUC / PR-AUC are reported **without** thresholding.

## Architecture (Phase 1 data flow)

```mermaid
flowchart LR
  D[KDDCup99 SA] --> E[EDA skew audit]
  E --> F[log1p on top-skew numerics]
  F --> P[Impute + scale + one-hot]
  P -->|fit on benign train only| B[Baselines + GMM + OCSVM]
  B --> S[Anomaly scores]
  S --> T[Threshold tune on val]
  T --> M[Test metrics + JSON artifact]
```

For the full course deliverable, extend this diagram in Phase 3 with a **neural encoder** feeding the same scoring head (hybrid).

## Ablation scaffolding (per course spec)

Phase 1 emits a **precursor** table (statistical / probabilistic detectors). Populate the final rows after Phase 2/3:

| Model | What to report |
| --- | --- |
| **A — Advanced ML only** | Best of {Mahalanobis, GMM, OCSVM} after honest tuning |
| **B — Deep learning only** | Sequence / representation model (e.g., Transformer or LSTM autoencoder) |
| **C — Hybrid** | Shared representation + explicit density / margin boundary |

Run `scripts/run_phase1.py` and paste the Markdown table it prints into your LaTeX report.

## Setup

```bash
cd /path/to/dl-aml-cybersec-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### If sklearn cannot download KDDCup99 (SSL errors on macOS)

Some Python installs lack the system certificate bundle. Fix the interpreter certificates (example on macOS: run the “Install Certificates” command shipped with python.org builds), or rely on the **offline** path:

```bash
python scripts/run_phase1.py --synthetic
```

Use `--synthetic` only for plumbing validation; your graded report should still analyze **real** intrusion traffic and discuss dataset limitations.

## Commands

```bash
# Fast unit tests (no dataset download)
pytest -m "not slow"

# Optional integration test (downloads KDDCup99 10% sample on first run)
pytest -m slow

# EDA figures -> reports/figures/
python scripts/run_eda.py
python scripts/run_eda.py --synthetic

# Train + evaluate + write reports/phase1_metrics.json
python scripts/run_phase1.py
python scripts/run_phase1.py --synthetic
```

## Repository layout

```
src/cps_ad/        # library code (data, preprocessing, models, metrics, phase1 runner)
scripts/           # CLI entrypoints
tests/             # pytest suite
reports/           # generated artifacts (json + figures; png/json gitignored by default)
```

## Authorship / reproducibility notes

- Pin `numpy<2` in `requirements.txt` to avoid broken binary wheels on some environments.
- First sklearn fetch may download data into `~/scikit_learn_data` (or `$SCIKIT_LEARN_DATA`).

## License

Educational use — add your institutional license statement if required.
