# CIC-IDS Training and Local Realtime Inference

This repo is my end-to-end anomaly detection workflow on CIC-IDS2017. I kept it split into three training notebooks so I can show progression clearly from classical ML to baseline DL and then to an improvised DL run with stronger validation.

The final written output for submission is `report/main.pdf`.

## Project layout

```text
notebooks/
  train_phase1_ml_colab.ipynb
  train_phase2_dl_colab.ipynb
  train_phase2_dl_colab_improvised.ipynb
  local_realtime_inference_scapy.ipynb

figures/
  phase1/
  phase2/
  phase2_improvised/

models/
  phase1_ml/
  phase2_dl/

report/
  main.pdf
```

## Notebook overview

- `train_phase1_ml_colab.ipynb`  
  Classical unsupervised baselines (Isolation Forest and One-Class SVM), saved as `phase1_ml_model.pth` with result JSON.

- `train_phase2_dl_colab.ipynb`  
  Baseline deep autoencoder training, saved as `ft_ae.pth` and `history.json`.

- `train_phase2_dl_colab_improvised.ipynb`  
  Improvised Phase 2 run with ablation, diagnostics, robustness stress tests, and saved outputs:
  - `ft_ae_level7.pth`
  - `level7_results.json`
  - `history_level7.json`

- `local_realtime_inference_scapy.ipynb`  
  Realtime/local packet-flow inference using exported model artifacts.

## Latest result snapshot

### Phase 1
- Isolation Forest (known): AUC `0.6984`, F1 `0.0284`
- Isolation Forest (zero-day): AUC `0.9996`, F1 `0.0366`
- One-Class SVM (known): AUC `0.7588`, F1 `0.5835`
- One-Class SVM (zero-day): AUC `0.9995`, F1 `0.0330`

### Phase 2 baseline
- Best epoch: `49`
- Autoencoder (known): AUC `0.8302`, F1 `0.6035`
- Autoencoder (zero-day): AUC `0.9994`, F1 `0.0166`

### Phase 2 improvised
- Selected deployment candidate: `BaselineAE`
- BaselineAE (known): AUC `0.8815`, F1 `0.8985`
- ResidualDenoisingAE (known): AUC `0.8045`, F1 `0.8976`
- Trivial comparator (known): AUC `0.3268`, F1 `0.0046`
- Robustness (AUC):
  - known + noise: `0.8707`
  - known + feature dropout: `0.7325`
  - zero-day + noise: `0.9998`
  - zero-day + feature dropout: `0.9874`

## Data and artifacts

Training data source folder used in Colab:
- `MyDrive/cic_ids_data/`

CSV set used:
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
