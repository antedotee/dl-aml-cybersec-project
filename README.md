# Adaptive Cyber-Physical Security — Project 2 (Phase 1 + hybrid stack)

Industrial and enterprise networks must flag **zero-day** behavior without attack signatures. This repository implements a **semi-supervised** pipeline aligned with the **“hypersphere of normalcy”** story:

1. **NSL-KDD** tabular connection records (same statistical view as many IDS backends).  
2. **Masked autoencoder (PyTorch)** — BERT-style *feature* masking on benign traffic; reconstruction error flags distributional drift.  
3. **One-Class SVM (scikit-learn)** — RBF margin on **raw features** (Model A) and on **frozen latent embeddings** (hybrid leg).  
4. **Hybrid (Model C)** — combines normalized reconstruction MSE with OCSVM-on-*z* (weight tuned on validation).

**Scapy:** offline work uses NSL rows; **Scapy** is the documented bridge from **PCAP / live packets** → flow features (see the main notebook).  

**Primary Phase-1 artifact for GitHub upload:** executed notebook `notebooks/phase1_eval_nsl_mae_ocsvm.ipynb` (regenerate via `scripts/build_phase1_rubric_notebook.py` + `jupyter nbconvert --execute`).

## Literature anchors (what to cite in your report)

- Chandola, Banerjee & Kumar, *Anomaly Detection: A Survey* (ACM Computing Surveys, 2009) — taxonomy of anomaly detection and evaluation pitfalls.
- Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution* (Neural Computation, 2001) — One-Class SVM / SVDD perspective.
- Tax & Duin, *Support Vector Data Description* (Machine Learning, 2004) — geometric interpretation of one-class kernels.
- Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006) — EM for mixture models, Gaussian mixtures, model selection (BIC).
- Vincent et al., *Stacked Denoising Autoencoders* (ICML 2008) — reconstruction-based representation learning.
- Devlin et al., *BERT* (NAACL 2019) — masked-token pre-training (we apply masking to **tabular** flow features).
- Tavallaee et al., *Toward a Reliable Evaluation of Network IDS* (2012) — NSL-KDD motivation.
- Lakhina, Crovella & Diot, *Mining Anomalies Using Traffic Feature Distributions* (SIGCOMM MineNet, 2005) — multivariate flow-feature anomaly motivation.
- Mahoney & Chan, *Learning Nonstationary Models of Normal Network Traffic* (RAID, 2002) — cautionary notes on KDDCup99 evaluation non-stationarity (acknowledge limitations).

## Theory checklist (viva / report)

- **Semi-supervised IDS framing**: training distribution is *predominantly normal*; attacks are **out-of-distribution** under the learned benign model.
- **Gaussian Mixture (EM)**: E-step computes responsibilities \(q_{ik}\propto \pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\); M-step updates \(\pi_k,\mu_k,\Sigma_k\) by weighted MLE. The incomplete-data log-likelihood is non-decreasing; convergence is to a **local** optimum → multiple `n_init` restarts.
- **Model selection**: **BIC** penalizes complexity; we use it to pick `n_components` on *benign-only* training data (your report should justify why this is reasonable for density estimation).
- **One-Class SVM (RBF)**: separates the training bulk from the origin in the RKHS induced by the kernel; **`nu`** upper-bounds the training false-positive fraction and influences support-vector sparsity.
- **Mahalanobis distance + Ledoit–Wolf**: if benign traffic is approximately elliptical after scaling, squared distances follow a \(\chi^2_d\) law under Gaussian assumptions; shrinkage stabilizes \(\Sigma^{-1}\) when \(d\) grows (many one-hot dimensions).

## Data

- **NSL-KDD** (`KDDTrain+` / `KDDTest+`) is downloaded automatically to `data/nsl/` (gitignored) from a [mirrored copy](https://github.com/jmnwong/NSL-KDD-Dataset) using **certifi** for TLS. Use `train_variant="twenty"` in code for faster iteration (`KDDTrain+_20Percent.txt`).  
- **KDDCup99 SA** via sklearn remains available in `scripts/run_phase1.py` / `cps_ad/phase1.py` as an alternate benchmark.  
- Discuss **concept drift** and protocol non-stationarity in your report (Mahoney & Chan; NSL train/test protocol limits).

## Methods implemented

1. **EDA-driven transforms**: skew audit → `log1p` on top heavy-tailed numerics.  
2. **Preprocessing**: median imputation + `StandardScaler`; `OneHotEncoder(handle_unknown="ignore")` on symbolic fields — **fit on benign train only**.  
3. **Baselines / advanced ML (tabular):** max-|z|, Ledoit–Wolf Mahalanobis, **GMM** (EM + **BIC** for `K`), **One-Class SVM** on raw scaled vectors.  
4. **Deep learning:** **Masked autoencoder** (dropout, BatchNorm, AdamW, early stopping, Xavier init); anomaly score = **reconstruction MSE** (RMSE for reporting).  
5. **Hybrid:** RBF **OCSVM on frozen encoder latents** + normalized recon error; **validation grid** over convex weight `w_recon`.

**Thresholding:** per model, tune one threshold on **train-split validation** for **F1**, report **NSL KDDTest+** metrics at that threshold; ROC-AUC / PR-AUC are threshold-free.

## Architecture (high level)

See the **mermaid** diagram inside `notebooks/phase1_eval_nsl_mae_ocsvm.ipynb` (course deliverable). Data flow: **NSL row** → skew / `log1p` → **ColumnTransformer** → **MAE (benign)** → **latent z** → **OCSVM(z)**; parallel **OCSVM(x)** on raw features; **hybrid** fuses recon + margin.

## Ablation (required A / B / C)

The executed notebook prints a table for:

| Model | Role in this repo |
| --- | --- |
| **A** | One-Class SVM on **scaled raw x** (plus auxiliary Mahalanobis / GMM / max-z rows) |
| **B** | **MAE reconstruction MSE** only |
| **C** | **Hybrid**: tuned blend of recon + **OCSVM on frozen z** |

Also run `python scripts/run_phase1.py` for the KDDCup99 SA quick table if needed.

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

## Jupyter (upload-ready executed notebook)

```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-notebooks.txt
# Regenerate source cells (optional) then execute and save all outputs into the .ipynb:
python scripts/build_phase1_rubric_notebook.py
jupyter nbconvert --execute --to notebook --inplace notebooks/phase1_eval_nsl_mae_ocsvm.ipynb --ExecutePreprocessor.timeout=900
```

Commit **`notebooks/phase1_eval_nsl_mae_ocsvm.ipynb`** to GitHub so markers see plots/tables without re-running.  
Legacy exploratory notebook: `notebooks/phase1_adaptive_cps.ipynb` (KDDCup99 / synthetic).

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
