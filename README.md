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

## Phase 2 — refined deep model on CIC-IDS2017

Phase 2 replaces the MLP-MAE with an **FT-Transformer** pretrained by **Masked
Feature Modeling** and finetuned with a **Deep SVDD** objective on **CIC-IDS2017**
(Sharafaldin et al. 2018). Phase 1 (NSL-KDD pipeline + Model A baselines + Model
C hybrid) is **left intact**; Model C is *deliberately untouched* per the Phase-2
brief so the comparison stays honest.

### Files added for Phase 2

| Path | Purpose |
|------|---------|
| `src/cps_ad/cic_ids.py` | CIC-IDS2017 downloader, schema normalizer, `make_zero_day_split` (leave-one-attack-family-out). |
| `src/cps_ad/flow_extractor.py` | PCAP / live-NIC -> CICFlowMeter-v3 78-feature flow extractor (pure Python; uses `scapy`). |
| `src/cps_ad/torch_ft_svdd.py` | `FTTransformerSVDD` model + MFM and Deep SVDD trainers + attention attribution. |
| `src/cps_ad/refined_ml.py` | Refined Model A: OCSVM (median-gamma) + IsolationForest + MCD-Mahalanobis with isotonic calibration. |
| `scripts/train_phase2.py` | End-to-end training CLI (Colab-friendly, checkpoints every epoch). |
| `scripts/derive_thresholds.py` | Compute SUSPECT / ATTACK quantiles from benign val. |
| `scripts/make_demo_pcaps.py` | Synthesize benign + bruteforce demo PCAPs (no live capture needed). |
| `scripts/demo_score_packets.py` | PCAP-replay or live-NIC scorer with attention-based per-flow explanation. |
| `notebooks/phase2_colab_train.ipynb` | Colab launcher: clones repo, mounts Drive, trains, plots curves. |
| `notebooks/phase2_eval.ipynb` | Validation suite: ROC/PR/F1, per-family confusion, ablation, robustness, attention heatmap. |
| `notebooks/phase2_local_demo.ipynb` | Mac-CPU live demo using PCAP replay. |

### Training on Google Colab (recommended)

1. Push this repo to GitHub.
2. Open `notebooks/phase2_colab_train.ipynb` in Colab. Set `REPO_URL` in §0.
3. Choose **Runtime -> Change runtime type -> GPU (T4)**. Run all cells.
4. The trainer mounts Drive, downloads CIC-IDS2017 once (~500 MB), runs 200
   epochs MFM + 100 epochs SVDD, and persists checkpoints to
   `/content/drive/MyDrive/cps_ad_phase2/checkpoints/`.
5. If Colab disconnects, re-run the train cell — `--resume` will pick up from
   `ft_svdd_mfm_last.pt`.

### Local live demo (Mac CPU)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_pcaps.py                             # one-time
python scripts/derive_thresholds.py --ckpt models/ft_svdd_final.pt
python scripts/demo_score_packets.py --pcap data/demo_pcaps/ssh_bruteforce.pcap \
                                     --ckpt models/ft_svdd_final.pt
```

For a real capture:

```bash
sudo -E python scripts/demo_score_packets.py --live en0 --duration 30 \
       --ckpt models/ft_svdd_final.pt --quiet-benign
```

### Why this design (rubric crosswalk)

* **Architecture**: feature-token attention (FT-Transformer), pre-norm residuals,
  modified loss (MFM + Deep SVDD + InfoNCE auxiliary). The SVDD center sits on a
  bias-free projection per Ruff et al. 2018 §3 to prevent hypersphere collapse.
* **Theory**: SVDD loss is derivable from one-class SVM in RKHS; attention is a
  scaled dot-product softmax (Vaswani 2017); MFM is a denoising objective
  (Vincent 2008) on a per-feature `[MASK]` token (Devlin 2019).
* **Dataset & regularization**: CIC-IDS2017 with `leave_one_family_out`
  (Heartbleed by default) for genuine zero-day evaluation, DropFeature
  augmentation for the contrastive view, scaler fit on benign train only.
* **Validation**: ROC / PR / F1 on random + zero-day splits, per-family
  recall, ablation of MFM-only / SVDD-only / fused, robustness to Gaussian
  feature noise, CLS->feature attention heatmap for interpretability.

### Literature anchors added in Phase 2

* Sharafaldin, Lashkari & Ghorbani, *Toward Generating a New Intrusion Detection
  Dataset* (ICISSP 2018) — CIC-IDS2017.
* Ruff et al., *Deep One-Class Classification* (ICML 2018) — Deep SVDD.
* Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (NeurIPS
  2021) — FT-Transformer.
* Vaswani et al., *Attention Is All You Need* (NeurIPS 2017).
* van den Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive
  Coding* (2018) — InfoNCE auxiliary.
* Engelen, Rimmer & Joosen, *Troubleshooting an Intrusion Detection Dataset:
  the CICIDS2017 Case Study* (IEEE SPW 2021) — label-noise hygiene.

## License

Educational use — add your institutional license statement if required.
