# Adaptive Cyber-Physical Security — Project 2 (Phase 1 + hybrid stack)

Industrial and enterprise networks must flag **zero-day** behavior without attack signatures. This repository implements a **semi-supervised** pipeline aligned with the **“hypersphere of normalcy”** story:

1. **NSL-KDD** tabular connection records (same statistical view as many IDS backends).  
2. **Masked autoencoder (PyTorch)** — BERT-style *feature* masking on benign traffic; reconstruction error flags distributional drift.  
3. **One-Class SVM (scikit-learn)** — RBF margin on **raw features** (Model A) and on **frozen latent embeddings** (hybrid leg).  
4. **Hybrid (Model C)** — combines normalized reconstruction MSE with OCSVM-on-*z* (weight tuned on validation).

**Scapy:** offline work uses NSL rows; **Scapy** is the documented bridge from **PCAP / live packets** → flow features (see the notebook).

**Primary deliverable:** the Jupyter notebook `notebooks/phase1_eval_nsl_mae_ocsvm.ipynb` — run it locally with **Run All** or `jupyter nbconvert --execute` so outputs are saved for GitHub.

**Repository layout (every file explained):** see [architecture.md](architecture.md).  
**PDF report (LaTeX):** see [report/README.md](report/README.md) — authors Kartik Yadav \& Agrapujya Lashkari (Rishihood University).

## Literature anchors (what to cite in your report)

- Chandola, Banerjee & Kumar, *Anomaly Detection: A Survey* (ACM Computing Surveys, 2009) — taxonomy of anomaly detection and evaluation pitfalls.
- Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution* (Neural Computation, 2001) — One-Class SVM / SVDD perspective.
- Tax & Duin, *Support Vector Data Description* (Machine Learning, 2004) — geometric interpretation of one-class kernels.
- Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006) — EM for mixture models, Gaussian mixtures, model selection (BIC).
- Vincent et al., *Stacked Denoising Autoencoders* (ICML 2008) — reconstruction-based representation learning.
- Devlin et al., *BERT* (NAACL 2019) — masked-token pre-training (we apply masking to **tabular** flow features).
- Tavallaee et al., *Toward a Reliable Evaluation of Network IDS* (2012) — NSL-KDD motivation.
- Lakhina, Crovella & Diot, *Mining Anomalies Using Traffic Feature Distributions* (SIGCOMM MineNet, 2005) — multivariate flow-feature anomaly motivation.
- Mahoney & Chan, *Learning Nonstationary Models of Normal Network Traffic* (RAID, 2002) — cautionary notes on KDD-era benchmarks (acknowledge limitations).

## Theory checklist (viva / report)

- **Semi-supervised IDS framing**: training distribution is *predominantly normal*; attacks are **out-of-distribution** under the learned benign model.
- **Gaussian Mixture (EM)**: E-step computes responsibilities; M-step weighted MLE; local optima → multiple `n_init`.
- **Model selection**: **BIC** penalizes complexity for choosing `n_components` on benign-only data.
- **One-Class SVM (RBF)**: margin in RKHS; **`nu`** bounds training outliers / SV fraction.
- **Mahalanobis + Ledoit–Wolf**: shrinkage when dimensionality is high after one-hot encoding.

## Data

- **NSL-KDD** is downloaded automatically to `data/nsl/` (see `.gitignore`; not committed) from a [mirrored copy](https://github.com/jmnwong/NSL-KDD-Dataset) using **certifi** for TLS. The notebook uses `TRAIN_VARIANT = "twenty"` (`KDDTrain+_20Percent.txt`) by default for faster runs; set `"full"` for the complete training set.
- Optional **KDDCup99** path for tests lives in `src/cps_ad/phase1.py` (`run_phase1`, synthetic mode) — used by `pytest`, not required for the main notebook.

## Methods (implemented in `src/cps_ad/` + notebook)

1. EDA-driven `log1p` on skewed numerics.  
2. ColumnTransformer (median impute, StandardScaler, one-hot) **fit on benign train only**.  
3. **Model A:** OCSVM on raw *x*; auxiliary Mahalanobis / GMM / max-|z|.  
4. **Model B:** MAE reconstruction MSE.  
5. **Model C:** Hybrid recon + OCSVM on frozen latent *z*.

## Ablation (A / B / C)

Printed inside the notebook as DataFrames; also writes `reports/phase1_ablation_nsl.json` when you run it (json path is gitignored).

## Setup

```bash
cd /path/to/dl-aml-cybersec-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook locally (save all cell outputs)

From the **repository root** (so `data/nsl/` and `src/` paths resolve):

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace notebooks/phase1_eval_nsl_mae_ocsvm.ipynb --ExecutePreprocessor.timeout=900
```

Or open the notebook in Jupyter / VS Code / Cursor, select the `.venv` kernel, and **Run All** → **Save**.

## Tests

```bash
pytest -m "not slow"          # default
pytest -m slow               # optional: downloads KDDCup99 via sklearn
```

## Repository layout

```
src/cps_ad/        # importable library (NSL loader, MAE, OCSVM, metrics, …)
notebooks/         # phase1_eval_nsl_mae_ocsvm.ipynb (main artifact)
tests/             # pytest
data/nsl/          # downloaded NSL files (gitignored)
reports/figures/   # plots from the notebook (png gitignored)
```

## Notes

- Pin `numpy<2` avoids some broken wheels on certain setups.
- `torch` is CPU-first; GPU is used automatically if available in the MAE training helpers.

## License

Educational use — add your institutional license statement if required.
