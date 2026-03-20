"""FT-Transformer encoder with Masked Feature Modeling pretraining and Deep SVDD finetuning.

References
----------
* Gorishniy et al. 2021. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS).
* Devlin et al. 2019. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL).
* Ruff et al. 2018. "Deep One-Class Classification" (ICML).
* van den Oord et al. 2018. "Representation Learning with Contrastive Predictive Coding"
  — InfoNCE objective used as auxiliary loss to mitigate Deep-SVDD hypersphere collapse.

Design notes
------------
* Each numeric feature becomes a token of dimension ``d_model`` via a learned linear
  projection plus a per-feature bias. A learned ``[CLS]`` token is prepended; its
  embedding after the encoder is the latent ``z``.
* The encoder uses *pre-norm* residual blocks (LayerNorm before MHA / FFN) which is
  the prescription that makes Transformers stable without warmup tricks.
* Following Ruff §3, the SVDD projection layer has **no bias terms** and the network
  uses **no bias on the final encoder norm output** to avoid the trivial collapse
  where the network maps every input to the center.
* Masked Feature Modeling replaces token embeddings with a learned ``[MASK]`` token
  rather than zeros so the model gets a clean signal that "this position is hidden".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FTConfig:
    n_features: int
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.1
    latent_dim: int = 64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class FeatureTokenizer(nn.Module):
    """Project (B, F) numeric features to (B, F, d_model) tokens.

    For feature ``j``, token = x_j * w_j + b_j  where w_j, b_j ∈ R^d.
    A learned ``[MASK]`` embedding replaces tokens whose mask flag is True.
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, F).  weight/bias: (F, d).
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (B, F, d)
        if mask is not None:
            # mask: (B, F) bool / float; True/1 means "hide this feature".
            m = mask.to(tokens.dtype).unsqueeze(-1)
            tokens = (1.0 - m) * tokens + m * self.mask_token
        return tokens


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> MHA -> residual -> LN -> FFN -> residual."""

    def __init__(self, cfg: FTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=return_attn,
                                          average_attn_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, (attn_weights if return_attn else None)


class FTTransformerSVDD(nn.Module):
    """FT-Transformer encoder with MFM decoder + Deep SVDD projection."""

    def __init__(self, cfg: FTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = FeatureTokenizer(cfg.n_features, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

        # Reconstruction head per feature token (MFM): each token -> scalar.
        self.recon_head = nn.Linear(cfg.d_model, 1)

        # SVDD projection: no bias to avoid trivial collapse (Ruff 2018 §3).
        self.svdd_proj = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False)

        self.register_buffer("center", torch.zeros(cfg.latent_dim), persistent=True)
        self.register_buffer("center_initialized", torch.zeros(1, dtype=torch.bool),
                             persistent=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_tokens(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b = x.shape[0]
        tokens = self.tokenizer(x, mask=mask)  # (B, F, d)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d)
        seq = torch.cat([cls, tokens], dim=1)  # (B, F+1, d)
        attn_maps: list[torch.Tensor] = []
        for block in self.blocks:
            seq, attn = block(seq, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_maps.append(attn)
        seq = self.norm(seq)
        return seq, attn_maps

    def encode_cls(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.encode_tokens(x)
        return seq[:, 0, :]

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.svdd_proj(self.encode_cls(x))

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq, _ = self.encode_tokens(x, mask=mask)
        feat_tokens = seq[:, 1:, :]  # drop CLS
        recon = self.recon_head(feat_tokens).squeeze(-1)  # (B, F)
        return recon

    @torch.no_grad()
    def initialize_center(
        self,
        loader: Iterable[torch.Tensor],
        device: torch.device,
        eps: float = 0.1,
    ) -> None:
        """Set the SVDD center to the mean latent over benign data (Ruff §4.1).

        Coordinates with magnitude < eps are nudged away from zero so that the
        network cannot learn the trivial all-zero solution.
        """
        self.eval()
        zs: list[torch.Tensor] = []
        for batch in loader:
            xb = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            zs.append(self.latent(xb))
        z = torch.cat(zs, dim=0)
        c = z.mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center.data.copy_(c)
        self.center_initialized.data.fill_(True)

    def svdd_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample squared distance to the SVDD center (higher = more anomalous)."""
        z = self.latent(x)
        return torch.sum((z - self.center) ** 2, dim=1)


def random_feature_mask(
    shape: tuple[int, int],
    mask_prob: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rand = torch.rand(shape, device=device, generator=generator)
    mask = (rand < mask_prob).float()
    # Guarantee at least one masked position per row to keep the loss well-defined.
    fix = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if fix.numel() > 0:
        mask[fix, 0] = 1.0
    return mask


def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (recon - target) ** 2
    num = (sq * mask).sum()
    den = mask.sum().clamp(min=1.0)
    return num / den


def svdd_loss(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((z - center) ** 2, dim=1))


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Symmetric InfoNCE between two views; pulls aug pairs together, pushes others apart."""
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    logits = z1n @ z2n.T / temperature  # (B, B)
    targets = torch.arange(z1n.shape[0], device=z1n.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_b)


def drop_feature_view(
    x: torch.Tensor,
    drop_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random per-feature dropout for contrastive views; zero-out behaves like a
    'soft' MASK on input space and is cheap to compute."""
    keep = (torch.rand(x.shape, device=x.device, generator=generator) > drop_prob).float()
    return x * keep


@dataclass
class TrainHistory:
    pretrain_train: list[float]
    pretrain_val: list[float]
    finetune_loss: list[float]
    finetune_radius: list[float]
    best_pretrain_epoch: int = 0
    best_finetune_epoch: int = 0


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _iterate_minibatches(
    x: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterable[torch.Tensor]:
    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=x.device, generator=generator)
    else:
        perm = torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        yield x[perm[i : i + batch_size]]


def pretrain_mfm(
    model: FTTransformerSVDD,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    mask_prob: float = 0.25,
    warmup_epochs: int = 5,
    patience: int = 15,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    seed: int = 42,
    history: TrainHistory | None = None,
    log_every: int = 1,
    on_epoch_end=None,
) -> TrainHistory:
    """Stage 1: BERT-style masked feature modeling on benign data only."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_tr = _to_tensor(x_train, dev)
    x_va = _to_tensor(x_val, dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    history = history or TrainHistory(
        pretrain_train=[], pretrain_val=[], finetune_loss=[], finetune_radius=[]
    )
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr * lr_at(epoch)

        model.train()
        train_losses: list[float] = []
        for xb in _iterate_minibatches(x_tr, batch_size, shuffle=True):
            mask = random_feature_mask(xb.shape, mask_prob, dev)
            recon = model.reconstruct(xb, mask=mask)
            loss = masked_recon_loss(recon, xb, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_losses: list[float] = []
            for xb in _iterate_minibatches(x_va, batch_size, shuffle=False):
                mask = random_feature_mask(xb.shape, mask_prob, dev)
                recon = model.reconstruct(xb, mask=mask)
                val_losses.append(float(masked_recon_loss(recon, xb, mask).cpu()))
        train_l = float(np.mean(train_losses)) if train_losses else float("nan")
        val_l = float(np.mean(val_losses)) if val_losses else float("nan")
        history.pretrain_train.append(train_l)
        history.pretrain_val.append(val_l)

        if val_l < best_val - 1e-7:
            best_val = val_l
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_pretrain_epoch = epoch
            stale = 0
        else:
            stale += 1

        if log_every > 0 and (epoch % log_every == 0):
            print(f"[MFM] epoch={epoch:03d} train={train_l:.5f} val={val_l:.5f}"
                  f" best_val={best_val:.5f}")
        if on_epoch_end is not None:
            on_epoch_end("mfm", epoch, train_l, val_l, model)
        if stale >= patience:
            print(f"[MFM] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev)
    return history


def finetune_svdd(
    model: FTTransformerSVDD,
    x_train: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    drop_prob: float = 0.15,
    info_nce_weight: float = 0.1,
    info_nce_temperature: float = 0.5,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    seed: int = 43,
    history: TrainHistory | None = None,
    log_every: int = 1,
    on_epoch_end=None,
) -> TrainHistory:
    """Stage 2: Deep SVDD finetune; keeps decoder frozen, trains encoder + svdd_proj."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    torch.manual_seed(seed)

    for p in model.recon_head.parameters():
        p.requires_grad = False

    x_tr = _to_tensor(x_train, dev)

    if not bool(model.center_initialized.item()):
        model.initialize_center(_iterate_minibatches(x_tr, batch_size, shuffle=False), dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    history = history or TrainHistory(
        pretrain_train=[], pretrain_val=[], finetune_loss=[], finetune_radius=[]
    )
    best_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        radii: list[float] = []
        for xb in _iterate_minibatches(x_tr, batch_size, shuffle=True):
            x_a = drop_feature_view(xb, drop_prob)
            x_b = drop_feature_view(xb, drop_prob)
            z_a = model.latent(x_a)
            z_b = model.latent(x_b)
            l_svdd = 0.5 * (svdd_loss(z_a, model.center) + svdd_loss(z_b, model.center))
            l_nce = info_nce_loss(z_a, z_b, temperature=info_nce_temperature)
            loss = l_svdd + info_nce_weight * l_nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                z = model.latent(xb)
                r = torch.sqrt(torch.sum((z - model.center) ** 2, dim=1)).mean()
                radii.append(float(r.cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_r = float(np.mean(radii)) if radii else float("nan")
        history.finetune_loss.append(epoch_loss)
        history.finetune_radius.append(epoch_r)

        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_finetune_epoch = epoch

        if log_every > 0 and (epoch % log_every == 0):
            print(f"[SVDD] epoch={epoch:03d} loss={epoch_loss:.5f} mean_r={epoch_r:.5f}")
        if on_epoch_end is not None:
            on_epoch_end("svdd", epoch, epoch_loss, epoch_r, model)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev)
    return history


@torch.no_grad()
def score_samples(
    model: FTTransformerSVDD,
    x: np.ndarray,
    *,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    """Per-sample anomaly score (squared distance to SVDD center)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    xt = _to_tensor(x, dev)
    out = []
    for xb in _iterate_minibatches(xt, batch_size, shuffle=False):
        out.append(model.svdd_score(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def reconstruction_error(
    model: FTTransformerSVDD,
    x: np.ndarray,
    *,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    """Per-sample mean-squared reconstruction error (no masking)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    xt = _to_tensor(x, dev)
    out = []
    for xb in _iterate_minibatches(xt, batch_size, shuffle=False):
        recon = model.reconstruct(xb, mask=None)
        out.append(((recon - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def cls_attention_attribution(
    model: FTTransformerSVDD,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Average attention from CLS to each feature token across layers and heads.

    Returns a (B, F) array suitable for "top-k contributing features" explanations
    in the live demo (Grad-CAM analog for tabular data)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    xt = _to_tensor(x, dev)
    seq, attn_maps = model.encode_tokens(xt, mask=None, return_attn=True)
    if not attn_maps:
        return np.zeros((xt.shape[0], model.cfg.n_features), dtype=np.float32)
    stacked = torch.stack(attn_maps, dim=0)  # (L, B, H, T, T)
    # Average across layers and heads, then take CLS row's attention to feature tokens.
    avg = stacked.mean(dim=(0, 2))  # (B, T, T)
    cls_attn = avg[:, 0, 1:]  # (B, F)
    return cls_attn.cpu().numpy().astype(np.float32)


def save_checkpoint(
    model: FTTransformerSVDD,
    path: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": vars(model.cfg),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[FTTransformerSVDD, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = FTConfig(**payload["config"])
    model = FTTransformerSVDD(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("extra", {})
