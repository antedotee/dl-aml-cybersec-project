"""Derive demo verdict thresholds from a benign validation distribution.

Loads the trained checkpoint, scores the benign validation split, and writes
``thresholds.json`` next to the checkpoint with two quantiles:

* ``suspect`` — 95th percentile of benign val scores
* ``attack``  — 99.5th percentile of benign val scores

These feed into ``scripts/demo_score_packets.py`` for verdict colors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/cic_ids"))
    p.add_argument("--scaler", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--suspect-q", type=float, default=0.95)
    p.add_argument("--attack-q", type=float, default=0.995)
    args = p.parse_args()

    import sys

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from cps_ad.cic_ids import load_cic_ids2017, make_zero_day_split
    from cps_ad.torch_ft_svdd import load_checkpoint, score_samples

    model, extra = load_checkpoint(str(args.ckpt), map_location="cpu")
    meta_path = args.ckpt.parent / "split_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"split_meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())

    scaler_path = args.scaler or (args.ckpt.parent / "scaler.npz")
    sc = np.load(scaler_path)
    mean, scale = sc["mean"], sc["scale"]

    df = load_cic_ids2017(data_dir=args.data_dir, download=True)
    split = make_zero_day_split(df, held_out_family=meta["held_out_family"])
    x_va = split.val.x.to_numpy(dtype=np.float32)
    x_va = ((x_va - mean) / np.where(scale > 0, scale, 1.0)).astype(np.float32)

    s = score_samples(model, x_va, device="cpu")
    thr = {
        "suspect": float(np.quantile(s, args.suspect_q)),
        "attack": float(np.quantile(s, args.attack_q)),
        "median_benign": float(np.median(s)),
        "p05": float(np.quantile(s, 0.05)),
    }
    out_path = args.out or (args.ckpt.parent / "thresholds.json")
    out_path.write_text(json.dumps(thr, indent=2))
    print(f"wrote {out_path}: {thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
