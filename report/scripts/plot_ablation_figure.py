#!/usr/bin/env python3
"""
Generate ablation bar chart (PDF + PNG) from metrics produced by the executed notebook.
Values match notebooks/phase1_eval_nsl_mae_ocsvm.ipynb embedded outputs (NSL 20%% train).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "OCSVM ($x$)\nModel A",
    "MAE recon\nModel B",
    "Hybrid\nModel C",
    "Mahalanobis\n(aux.)",
    "Max-$|z|$\n(aux.)",
    "GMM\n(aux.)",
]
ROCAUC = np.array([0.920359, 0.942441, 0.781210, 0.950259, 0.927582, 0.923397])
COLORS = ["#1f4e79", "#2e75b6", "#c55a11", "#548235", "#7030a0", "#996633"]

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
x = np.arange(len(MODELS))
bars = ax.bar(x, ROCAUC, color=COLORS, edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=9)
ax.set_ylabel("Test ROC-AUC (NSL KDDTest+)")
ax.set_ylim(0.65, 1.0)
ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
for b, v in zip(bars, ROCAUC):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Anomaly detection ablation: ROC-AUC on held-out NSL-KDD test split")
fig.tight_layout()
fig.savefig(OUT / "ablation_roc_auc.pdf", bbox_inches="tight")
fig.savefig(OUT / "ablation_roc_auc.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote", OUT / "ablation_roc_auc.pdf")
