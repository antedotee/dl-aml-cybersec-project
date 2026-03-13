"""Tabular masked autoencoder (BERT-style feature masking) for benign pre-training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class TabularMAE(nn.Module):
    """
    MLP encoder–decoder with optional dropout and BatchNorm.

    Training: random subset of features is masked to zero; MSE is taken only on masked
    coordinates (denoising / MAE objective). At scoring time, use full vectors to measure
    reconstruction error and latent shape for One-Class SVM.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 192,
        latent_dim: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x_in)
        return self.decode(z), z

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-feature pass (no masking) for anomaly scoring."""
        recon, z = self.forward(x)
        return recon, z


def mae_masked_batch(x: torch.Tensor, mask_prob: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-out a random fraction of features; MSE later applies only where mask==1."""
    mask = (torch.rand_like(x) < mask_prob).float()
    if mask.sum() < 1:
        mask.reshape(-1)[0] = 1.0
    x_in = x * (1.0 - mask)
    return x_in, mask


@dataclass
class MAETrainResult:
    model: TabularMAE
    history: list[float]
    best_epoch: int


def train_mae_benign(
    x_train_normal: np.ndarray,
    x_val_normal: np.ndarray,
    *,
    mask_prob: float = 0.25,
    epochs: int = 45,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    device: str | None = None,
    seed: int = 42,
) -> MAETrainResult:
    """AdamW + early stopping on validation masked reconstruction loss (benign only)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dim = x_train_normal.shape[1]
    model = TabularMAE(input_dim=dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_tr = torch.tensor(x_train_normal, dtype=torch.float32, device=dev)
    x_va = torch.tensor(x_val_normal, dtype=torch.float32, device=dev)
    n = x_tr.shape[0]

    history: list[float] = []
    best = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    stall = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_tr[idx]
            x_in, m = mae_masked_batch(xb, mask_prob=mask_prob)
            recon, _ = model(x_in)
            loss = (((recon - xb) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            x_in, m = mae_masked_batch(x_va, mask_prob=mask_prob)
            recon, _ = model(x_in)
            vloss = float((((recon - x_va) ** 2) * m).sum() / m.sum().clamp(min=1.0).cpu())
        history.append(vloss)

        if vloss < best - 1e-7:
            best = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev)
    return MAETrainResult(model=model, history=history, best_epoch=best_epoch)


@torch.no_grad()
def reconstruction_mse_per_sample(model: TabularMAE, x: np.ndarray, device: str | None = None) -> np.ndarray:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(dev)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    recon, _ = model.reconstruct(xt)
    mse = ((recon - xt) ** 2).mean(dim=1).cpu().numpy()
    return mse


@torch.no_grad()
def latent_matrix(model: TabularMAE, x: np.ndarray, device: str | None = None) -> np.ndarray:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(dev)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    z = model.encode(xt).cpu().numpy()
    return z
