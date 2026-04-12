# CIC-IDS Training and Local Realtime Inference

This repository is simplified to one clear pipeline:

1. Train ML baseline on CIC-IDS2017 in Colab
2. Train DL model on CIC-IDS2017 in Colab
3. Run local Scapy inference with exported `.pth` model files

`report/` and `reports/` are intentionally kept.

## Repository Structure

```text
notebooks/
  train_phase1_ml_colab.ipynb
  train_phase2_dl_colab.ipynb
  local_realtime_inference_scapy.ipynb

models/
  phase1_ml/
  phase2_dl/

data/
  cic_ids/     # keep empty in git, upload CSVs to Google Drive

report/
reports/
```

## Notebook Roles

- `train_phase1_ml_colab.ipynb`
  - Isolation Forest + One-Class SVM
  - Exports `phase1_ml_model.pth`

- `train_phase2_dl_colab.ipynb`
  - Autoencoder training
  - Exports `ft_ae.pth`

- `local_realtime_inference_scapy.ipynb`
  - Local replay/live inference
  - Loads both model artifacts and flags anomalies

## Google Drive Upload (for both training notebooks)

Create:
- `MyDrive/cic_ids_data/`

Upload these 8 CSV files:
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

## Artifacts To Keep In Repo After Training

- From phase1:
  - `models/phase1_ml/phase1_ml_model.pth`
  - `models/phase1_ml/phase1_ml_results.json`

- From phase2:
  - `models/phase2_dl/ft_ae.pth`
  - `models/phase2_dl/history.json`

## Local Inference Quick Start

1. Put model files in:
   - `models/phase1_ml/phase1_ml_model.pth`
   - `models/phase2_dl/ft_ae.pth`
2. Open `notebooks/local_realtime_inference_scapy.ipynb`
3. Choose PCAP replay or live sniffing
4. Run all cells

## Notes

- Legacy NSL/hybrid code, tests, and old scripts were removed.
- Do not commit raw CIC-IDS CSV files.
