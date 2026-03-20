"""Smoke tests for the FT-Transformer + Deep SVDD model.

Kept fast: a tiny model, tiny synthetic data, a handful of epochs. Goal is to
exercise every code path (init, MFM step, SVDD step, scoring, attention) so a
regression breaks CI immediately.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cps_ad.torch_ft_svdd import (
    FTConfig,
    FTTransformerSVDD,
    cls_attention_attribution,
    drop_feature_view,
    finetune_svdd,
    info_nce_loss,
    masked_recon_loss,
    pretrain_mfm,
    random_feature_mask,
    reconstruction_error,
    save_checkpoint,
    load_checkpoint,
    score_samples,
    svdd_loss,
)


@pytest.fixture()
def tiny_model() -> FTTransformerSVDD:
    cfg = FTConfig(n_features=12, d_model=16, n_heads=4, n_layers=2, d_ff=32, latent_dim=8)
    return FTTransformerSVDD(cfg)


def _benign(n: int = 64, f: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, f)).astype(np.float32)


def test_forward_shapes(tiny_model: FTTransformerSVDD) -> None:
    x = torch.tensor(_benign(8, 12), dtype=torch.float32)
    seq, _ = tiny_model.encode_tokens(x)
    assert seq.shape == (8, 13, 16)  # CLS + 12 features
    z = tiny_model.latent(x)
    assert z.shape == (8, 8)
    recon = tiny_model.reconstruct(x)
    assert recon.shape == (8, 12)
    score = tiny_model.svdd_score(x)
    assert score.shape == (8,)


def test_random_feature_mask_at_least_one_per_row() -> None:
    m = random_feature_mask((4, 6), mask_prob=0.0, device=torch.device("cpu"))
    # Even with prob=0, the helper guarantees one masked position per row.
    assert (m.sum(dim=1) >= 1).all()


def test_masked_recon_loss_zero_when_perfect_recon() -> None:
    x = torch.zeros(3, 5)
    mask = torch.ones(3, 5)
    assert float(masked_recon_loss(x, x, mask)) == 0.0


def test_svdd_loss_zero_at_center() -> None:
    z = torch.zeros(4, 6)
    c = torch.zeros(6)
    assert float(svdd_loss(z, c)) == 0.0


def test_info_nce_handles_small_batch() -> None:
    z = torch.randn(4, 6)
    loss = info_nce_loss(z, z + 1e-3 * torch.randn_like(z))
    assert torch.isfinite(loss)


def test_drop_feature_view_zeros_with_high_prob() -> None:
    x = torch.ones(4, 8)
    out = drop_feature_view(x, drop_prob=1.0)
    assert torch.all(out == 0)


def test_pretrain_then_finetune_runs_and_scores(tiny_model: FTTransformerSVDD) -> None:
    x_tr = _benign(128, 12, seed=1)
    x_va = _benign(32, 12, seed=2)
    history = pretrain_mfm(
        tiny_model, x_tr, x_va,
        epochs=3, batch_size=32, lr=1e-3, warmup_epochs=1, patience=10,
        device=torch.device("cpu"), log_every=0,
    )
    assert len(history.pretrain_train) == 3
    assert all(np.isfinite(history.pretrain_train))

    history = finetune_svdd(
        tiny_model, x_tr,
        epochs=2, batch_size=32, lr=5e-4, info_nce_weight=0.1,
        device=torch.device("cpu"), history=history, log_every=0,
    )
    assert len(history.finetune_loss) == 2
    assert tiny_model.center_initialized.item()

    s = score_samples(tiny_model, x_va, device=torch.device("cpu"))
    assert s.shape == (32,)
    assert np.all(np.isfinite(s))

    err = reconstruction_error(tiny_model, x_va, device=torch.device("cpu"))
    assert err.shape == (32,) and np.all(np.isfinite(err))


def test_attention_attribution_shape(tiny_model: FTTransformerSVDD) -> None:
    x = _benign(4, 12, seed=3)
    attr = cls_attention_attribution(tiny_model, x, device=torch.device("cpu"))
    assert attr.shape == (4, 12)
    assert np.all(np.isfinite(attr))


def test_save_and_load_roundtrip(tiny_model: FTTransformerSVDD, tmp_path) -> None:
    p = tmp_path / "ft.pt"
    save_checkpoint(tiny_model, str(p), extra={"feature_names": [f"f{i}" for i in range(12)]})
    model2, extra = load_checkpoint(str(p), map_location="cpu")
    assert extra["feature_names"][0] == "f0"
    x = _benign(2, 12, seed=4)
    s1 = score_samples(tiny_model, x, device=torch.device("cpu"))
    s2 = score_samples(model2, x, device=torch.device("cpu"))
    np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_anomaly_score_separates_shifted_distribution() -> None:
    """Score on heavily shifted data should be larger on average than on clean data."""
    cfg = FTConfig(n_features=8, d_model=16, n_heads=4, n_layers=2, d_ff=32, latent_dim=8)
    model = FTTransformerSVDD(cfg)
    x_tr = _benign(256, 8, seed=10)
    x_va = _benign(64, 8, seed=11)
    pretrain_mfm(model, x_tr, x_va, epochs=5, batch_size=64, lr=1e-3,
                 warmup_epochs=1, patience=10, device=torch.device("cpu"), log_every=0)
    finetune_svdd(model, x_tr, epochs=5, batch_size=64, lr=5e-4,
                  info_nce_weight=0.1, device=torch.device("cpu"), log_every=0)
    rng = np.random.default_rng(99)
    x_outlier = rng.standard_normal((64, 8)).astype(np.float32) * 5.0 + 10.0
    s_in = score_samples(model, x_va, device=torch.device("cpu"))
    s_out = score_samples(model, x_outlier, device=torch.device("cpu"))
    assert s_out.mean() > s_in.mean()
