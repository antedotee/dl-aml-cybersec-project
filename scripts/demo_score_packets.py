"""Live-ish demo: read packets, extract CICFlowMeter features, score with the
trained FT-Transformer + Deep SVDD model.

Two modes
---------
$ python scripts/demo_score_packets.py --pcap data/demo_pcaps/benign_http.pcap \
        --ckpt models/ft_svdd_final.pt
$ sudo python scripts/demo_score_packets.py --live en0 --duration 30 \
        --ckpt models/ft_svdd_final.pt

For each flow we print:
* anomaly score (squared distance to SVDD center)
* verdict (BENIGN / SUSPECT / ATTACK) using thresholds derived from a benign
  validation distribution stored alongside the checkpoint
* the top-3 features by CLS attention (per-flow attribution)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score real packets against the Phase-2 model.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", type=Path, help="PCAP file to replay (no sudo needed).")
    src.add_argument("--live", type=str, help="Network interface to sniff (requires sudo).")

    p.add_argument("--ckpt", type=Path, required=True, help="Path to ft_svdd_final.pt")
    p.add_argument("--scaler", type=Path, default=None,
                   help="Path to scaler.npz (defaults to <ckpt>/../scaler.npz).")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="Path to thresholds.json (suspect/attack quantiles).")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Live capture duration in seconds (live mode only).")
    p.add_argument("--bpf", type=str, default=None,
                   help="Optional BPF filter for --live (e.g. 'tcp port 22').")
    p.add_argument("--top-k", type=int, default=3, help="Top features per flagged flow.")
    p.add_argument("--max-flows", type=int, default=200, help="Hard cap on flows to print.")
    p.add_argument("--quiet-benign", action="store_true",
                   help="Only print SUSPECT / ATTACK verdicts.")
    return p.parse_args(argv)


def _autodetect_scaler(ckpt: Path) -> Path:
    cand = ckpt.parent / "scaler.npz"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"Could not auto-locate scaler.npz next to {ckpt}; pass --scaler."
    )


def _load_thresholds(path: Path | None, ckpt: Path) -> dict[str, float]:
    cand = path or (ckpt.parent / "thresholds.json")
    if cand.exists():
        return json.loads(cand.read_text())
    # Defaults are conservative; used until the user runs evaluate to derive proper ones.
    return {"suspect": 1.0, "attack": 4.0}


def _verdict(score: float, thr: dict[str, float]) -> tuple[str, str]:
    if score >= thr["attack"]:
        return "ATTACK", "\033[31m"   # red
    if score >= thr["suspect"]:
        return "SUSPECT", "\033[33m"  # yellow
    return "BENIGN", "\033[32m"       # green


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from cps_ad.flow_extractor import (
        align_to_training_columns,
        extract_from_live,
        extract_from_pcap,
        to_numpy_features,
    )
    from cps_ad.torch_ft_svdd import (
        cls_attention_attribution,
        load_checkpoint,
        score_samples,
    )

    ckpt_path: Path = args.ckpt
    if not ckpt_path.exists():
        print(f"!! checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"-- loading model from {ckpt_path}")
    model, extra = load_checkpoint(str(ckpt_path), map_location="cpu")
    feature_names = extra.get("feature_names")
    if not feature_names:
        meta = (ckpt_path.parent / "split_meta.json")
        if meta.exists():
            feature_names = json.loads(meta.read_text()).get("feature_names")
    if not feature_names:
        print("!! could not recover feature_names; refusing to score.", file=sys.stderr)
        return 3

    scaler_path = args.scaler or _autodetect_scaler(ckpt_path)
    sc = np.load(scaler_path)
    mean, scale = sc["mean"], sc["scale"]
    thr = _load_thresholds(args.thresholds, ckpt_path)
    print(f"-- thresholds: suspect={thr['suspect']:.3f}  attack={thr['attack']:.3f}")

    if args.pcap is not None:
        print(f"-- reading PCAP {args.pcap}")
        df_flows = extract_from_pcap(args.pcap)
    else:
        print(f"-- sniffing {args.live} for {args.duration}s (BPF={args.bpf})")
        df_flows = extract_from_live(args.live, duration=args.duration, bpf_filter=args.bpf)

    if df_flows.empty:
        print("-- no flows extracted; nothing to score.")
        return 0

    df_flows = align_to_training_columns(df_flows, feature_names)
    x_raw = to_numpy_features(df_flows)
    x = ((x_raw - mean) / np.where(scale > 0, scale, 1.0)).astype(np.float32)

    scores = score_samples(model, x, device="cpu")
    attn = cls_attention_attribution(model, x, device="cpu")

    n_print = min(len(scores), args.max_flows)
    order = np.argsort(-scores)[:n_print]
    print(f"-- {len(scores)} flows extracted; printing top {n_print} by score")
    for i in order:
        verdict, color = _verdict(float(scores[i]), thr)
        if args.quiet_benign and verdict == "BENIGN":
            continue
        top_idx = np.argsort(-attn[i])[: args.top_k]
        feats = ", ".join(f"{feature_names[j]} ({attn[i][j]:.2f})" for j in top_idx)
        reset = "\033[0m"
        print(f"{color}[{verdict:7s}] score={scores[i]:7.3f}  top: {feats}{reset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
