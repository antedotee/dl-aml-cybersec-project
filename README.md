# CICIDS2017 anomaly IDS: classical ML, autoencoder, and hybrid fusion

This repository holds a **benign-only anomaly detection** pipeline on **CICIDS2017 flow CSVs**: classical one-class models, a reconstruction autoencoder, and a **hybrid** score fusion. Training is split across **three self-contained Jupyter notebooks** (no dependency on a shared `src` package inside those notebooks). The written report is `report/main.pdf`.

## Local setup

1. **Python**  
   Use Python 3.11+ (3.12 is fine). A virtual environment is recommended.

2. **Create and activate a venv** (example):

   ```bash
   cd /path/to/dl-aml-cybersec-project
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Optional: **ThunderSVM** (`thundersvm`) on a GPU machine speeds up One-Class SVM in notebook 01; sklearn’s RBF OC-SVM on CPU may subsample large benign sets.

4. **Data layout**  
   Download the **eight ISCX flow CSVs** for CICIDS2017 and place them in a folder named **`cic_ids_data`** at the project root (same layout as in Colab’s `MyDrive/cic_ids_data`):

   ```text
   cic_ids_data/
     Monday-WorkingHours.pcap_ISCX.csv
     Tuesday-WorkingHours.pcap_ISCX.csv
     Wednesday-workingHours.pcap_ISCX.csv
     Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
     Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
     Friday-WorkingHours-Morning.pcap_ISCX.csv
     Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
     Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

   Large CSVs are commonly gitignored; you must provide them locally.

5. **Run notebooks**  
   Open `notebooks/` in Jupyter, VS Code, or upload to Colab. Each notebook sets `ROOT_ARTIFACTS` to `./artifacts` when not on Colab.

## Project layout (current)

```text
notebooks/
  01_classical_ids_colab.ipynb      # Classical ML baselines
  02_deep_learning_autoencoder_colab.ipynb   # Deep autoencoder IDS
  03_hybrid_fusion_colab_updated.ipynb     # OC-SVM + AE + fusion (end-to-end)



report/
  main.pdf                  # submission PDF

requirements.txt
```

## What the three notebooks do

### 01 — `01_classical_ids_colab.ipynb` (classical IDS)

**Approach:** Unsupervised detectors trained **only on benign** flows after correlation pruning, median imputation, and standard scaling fit on benign data. The notebook evaluates **One-Class SVM** (RBF kernel) over a **ν sweep** and optional **Isolation Forest** for comparison. Per-class metrics and tables (e.g. accuracy vs. ν) summarize how well each attack family is separated from benign in feature space.

**Outputs (typical):** `artifacts/classical/` including figures and `oc_svm_model.pkl` when saved.

### 02 — `02_deep_learning_autoencoder_colab.ipynb` (deep learning)

**Approach:** A **fully connected autoencoder** (`torch.nn`) learns to reconstruct benign flows; anomaly scores come from **reconstruction error** (e.g. MSE). Thresholds are swept to expose the precision–recall trade-off. The notebook exports per-class summaries and plots (e.g. per-class macro-F1 vs. threshold), analogous to reporting in autoencoder-based IDS papers.

**Outputs (typical):** `artifacts/dl/` including `ae_weights.pt`, `dl_manifest.pkl`, and figure/table CSVs under `artifacts/dl/figures/`.

### 03 — `03_hybrid_fusion_colab_updated.ipynb` (hybrid)

**Approach:** Single end-to-end pipeline: trains **OC-SVM** and **autoencoder** on the same preprocessing, then **fuses** normalized scores with calibrated weights (e.g. validation-tuned mix of SVM and AE terms) and an operating threshold. Includes an explicit **ablation** comparing **ML-only**, **DL-only**, and **hybrid fusion** on aligned folds.

**Outputs:** Writes `artifacts/hybrid/` (`ablation_summary.csv`, `hybrid_results.json`, `fusion_config.json`, figures), mirrors under `artifacts/classical/` and `artifacts/dl/`, and prefers **`artifacts/phase3/`** as the complete bundle for a local realtime web UI when that app is available (command is documented in the notebook manifest).

## Live capture / realtime notes

Live sniffing (Scapy) usually needs **elevated privileges** on the host (`sudo` on macOS/Linux), the correct interface name (`en0`, `eth0`, etc.), or **PCAP file** input. Notebook **03** also documents how to run a realtime web UI (`uvicorn` and `artifacts/phase3/`) when that application code is present in your environment.

