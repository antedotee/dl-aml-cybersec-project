"""End-to-end Phase 2 training: CIC-IDS2017 -> FT-Transformer + Deep SVDD.

Designed to be run from Colab (T4 GPU) or any machine with PyTorch. Persists
incremental checkpoints to a directory you can mount on Google Drive so a 12 h
session timeout never costs you the run.

Example
-------
$ python scripts/train_phase2.py \
    --out-dir /content/drive/MyDrive/cps_ad_ckpts \
    --held-out-family Heartbleed \
    --epochs-mfm 200 --epochs-svdd 100 --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Allow running as a stand-alone script (no PYTHONPATH=src required).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cps_ad.cic_ids import (
    class_balance_report,
    load_cic_ids2017,
    make_zero_day_split,
)
from cps_ad.torch_ft_svdd import (
    FTConfig,
    FTTransformerSVDD,
    finetune_svdd,
    load_checkpoint,
    pretrain_mfm,
    save_checkpoint,
)


LOG = logging.getLogger("phase2.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 trainer.")
    p.add_argument("--data-dir", type=Path, default=None, help="Where CIC-IDS2017 CSVs live.")
    p.add_argument(
        "--cic-files",
        type=str,
        nargs="*",
        default=None,
        help="Optional CIC day CSV file names to load (subset for RAM-safe runs).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Checkpoint / metrics output.")
    p.add_argument("--held-out-family", type=str, default="Heartbleed",
                   choices=["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack",
                            "Infiltration", "Bot", "Heartbleed"])
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-benign-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs-mfm", type=int, default=200)
    p.add_argument("--epochs-svdd", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr-mfm", type=float, default=3e-4)
    p.add_argument("--lr-svdd", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-prob", type=float, default=0.25)
    p.add_argument("--drop-prob", type=float, default=0.15)
    p.add_argument("--info-nce-weight", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=5)

    p.add_argument("--device", type=str, default="auto",
                   help="'cuda', 'mps', 'cpu' or 'auto'.")
    p.add_argument("--max-train-rows", type=int, default=0,
                   help="Optional cap on benign-train rows (debug / quick smoke). 0 = all.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint to resume MFM from.")
    p.add_argument("--skip-mfm", action="store_true",
                   help="Skip MFM pretrain (assumes --resume points to a finished MFM ckpt).")
    p.add_argument("--skip-svdd", action="store_true")
    return p.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def fit_scaler_and_save(x_train: np.ndarray, out_path: Path) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(x_train)
    np.savez(out_path, mean=sc.mean_, scale=sc.scale_, var=sc.var_)
    LOG.info("Wrote scaler stats to %s", out_path)
    return sc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    LOG.info("Device: %s", device)

    LOG.info("Loading CIC-IDS2017 ...")
    df = load_cic_ids2017(data_dir=args.data_dir, download=True, files=args.cic_files)
    LOG.info("Loaded %d rows, %d feature columns.", len(df), df.shape[1] - 2)
    LOG.info("Class balance:\n%s", class_balance_report(df).to_string(index=False))

    split = make_zero_day_split(
        df,
        held_out_family=args.held_out_family,
        val_frac=args.val_frac,
        test_benign_frac=args.test_benign_frac,
        seed=args.seed,
    )
    del df
    gc.collect()
    LOG.info("Split sizes: train=%d  val=%d  test_known=%d  test_zero_day=%d (held=%s)",
             len(split.train), len(split.val), len(split.test_known),
             len(split.test_zero_day), split.held_out_family)

    x_tr = split.train.x.to_numpy(dtype=np.float32)
    x_va = split.val.x.to_numpy(dtype=np.float32)

    if args.max_train_rows and args.max_train_rows < x_tr.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(x_tr.shape[0], size=args.max_train_rows, replace=False)
        x_tr = x_tr[idx]
        LOG.info("Subsampled train -> %d rows.", x_tr.shape[0])

    sc = fit_scaler_and_save(x_tr, args.out_dir / "scaler.npz")
    x_tr_s = sc.transform(x_tr).astype(np.float32, copy=False)
    x_va_s = sc.transform(x_va).astype(np.float32, copy=False)

    cfg = FTConfig(
        n_features=x_tr_s.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
    )
    LOG.info("Model config: %s", cfg)

    if args.resume is not None and args.resume.exists():
        LOG.info("Resuming model from %s", args.resume)
        model, _ = load_checkpoint(str(args.resume), map_location=str(device))
    else:
        model = FTTransformerSVDD(cfg)
    model.to(device)

    feature_names = list(split.feature_names)
    (args.out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (args.out_dir / "split_meta.json").write_text(json.dumps({
        "held_out_family": split.held_out_family,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test_known": len(split.test_known),
        "n_test_zero_day": len(split.test_zero_day),
        "feature_names": feature_names,
        "config": vars(cfg),
    }, indent=2))

    def checkpoint_cb(stage: str, epoch: int, *_unused, model: FTTransformerSVDD) -> None:
        # Save every epoch so a Colab disconnect costs at most one epoch.
        path = args.out_dir / f"ft_svdd_{stage}_last.pt"
        save_checkpoint(model, str(path), extra={"stage": stage, "epoch": epoch,
                                                  "feature_names": feature_names})

    if not args.skip_mfm:
        LOG.info("=== Stage 1: MFM pretrain ===")
        t0 = time.time()
        history = pretrain_mfm(
            model, x_tr_s, x_va_s,
            epochs=args.epochs_mfm,
            batch_size=args.batch_size,
            lr=args.lr_mfm,
            weight_decay=args.weight_decay,
            mask_prob=args.mask_prob,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            device=device,
            seed=args.seed,
            on_epoch_end=lambda s, e, *_, model=model: checkpoint_cb(s, e, model=model),
        )
        LOG.info("MFM finished in %.1f s; best epoch=%d val=%.5f",
                 time.time() - t0, history.best_pretrain_epoch,
                 history.pretrain_val[history.best_pretrain_epoch] if history.pretrain_val else float("nan"))
        save_checkpoint(model, str(args.out_dir / "ft_svdd_mfm_best.pt"),
                        extra={"feature_names": feature_names})
    else:
        history = None

    if not args.skip_svdd:
        LOG.info("=== Stage 2: Deep SVDD finetune ===")
        t0 = time.time()
        history = finetune_svdd(
            model, x_tr_s,
            epochs=args.epochs_svdd,
            batch_size=args.batch_size,
            lr=args.lr_svdd,
            drop_prob=args.drop_prob,
            info_nce_weight=args.info_nce_weight,
            device=device,
            seed=args.seed + 1,
            history=history,
            on_epoch_end=lambda s, e, *_, model=model: checkpoint_cb(s, e, model=model),
        )
        LOG.info("SVDD finetune finished in %.1f s; best epoch=%d loss=%.5f",
                 time.time() - t0, history.best_finetune_epoch,
                 history.finetune_loss[history.best_finetune_epoch] if history.finetune_loss else float("nan"))
        save_checkpoint(model, str(args.out_dir / "ft_svdd_final.pt"),
                        extra={"feature_names": feature_names,
                               "scaler_path": str(args.out_dir / "scaler.npz")})

    if history is not None:
        (args.out_dir / "history.json").write_text(json.dumps(asdict(history), indent=2))

    LOG.info("Done. Artifacts under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
