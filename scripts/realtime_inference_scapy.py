#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scapy.all import IP, TCP, UDP, rdpcap, sniff


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims=(128, 64), bottleneck_dim: int = 32):
        super().__init__()
        enc, prev = [], n_features
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        enc += [nn.Linear(prev, bottleneck_dim), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], bottleneck_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec += [nn.Linear(prev, n_features)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Realtime/PCAP zero-day anomaly inference with model selection."
    )
    parser.add_argument(
        "--model",
        default="model2",
        choices=["model1", "model2", "phase1", "phase2", "both"],
        help="model1/phase1=ML baseline, model2/phase2=DL autoencoder, both=OR fusion.",
    )
    parser.add_argument(
        "--pcap",
        type=Path,
        default=None,
        help="Optional pcap path. If omitted, live sniff mode is used.",
    )
    parser.add_argument(
        "--iface",
        default="en0",
        help="Network interface for live sniff mode.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Sniff timeout (seconds) for live mode.",
    )
    parser.add_argument(
        "--phase1-model",
        type=Path,
        default=repo / "models" / "phase1_ml" / "phase1_ml_model.pth",
        help="Path to phase1 ML artifact.",
    )
    parser.add_argument(
        "--phase2-model",
        type=Path,
        default=repo / "models" / "phase2_dl" / "ft_ae.pth",
        help="Path to phase2 DL artifact (or level7 checkpoint).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save scored flow table as CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="Max rows to print in terminal output.",
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    if name in {"model1", "phase1"}:
        return "phase1"
    if name in {"model2", "phase2"}:
        return "phase2"
    return "both"


def capture_packets(pcap: Path | None, iface: str, timeout: int):
    if pcap is not None:
        if not pcap.exists():
            raise FileNotFoundError(f"Missing pcap: {pcap}")
        return rdpcap(str(pcap))
    return sniff(iface=iface, timeout=timeout)


def flow_key(pkt):
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = int(ip.proto)
    sport = dport = 0
    if TCP in pkt:
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif UDP in pkt:
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    return (ip.src, ip.dst, sport, dport, proto)


def packets_to_df(pkts) -> pd.DataFrame:
    flows = {}
    for pkt in pkts:
        key = flow_key(pkt)
        if key is None:
            continue
        ts = float(pkt.time)
        ip = pkt[IP]
        plen = len(pkt)

        rec = flows.get(key)
        if rec is None:
            rec = {
                "start": ts,
                "end": ts,
                "pkts": 0,
                "bytes": 0,
                "lens": [],
                "fwd_pkts": 0,
                "bwd_pkts": 0,
                "fwd_bytes": 0,
                "bwd_bytes": 0,
                "fin": 0,
                "syn": 0,
                "rst": 0,
                "psh": 0,
                "ack": 0,
                "urg": 0,
                "src0": ip.src,
            }
            flows[key] = rec

        rec["end"] = ts
        rec["pkts"] += 1
        rec["bytes"] += plen
        rec["lens"].append(plen)

        if ip.src == rec["src0"]:
            rec["fwd_pkts"] += 1
            rec["fwd_bytes"] += plen
        else:
            rec["bwd_pkts"] += 1
            rec["bwd_bytes"] += plen

        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            rec["fin"] += 1 if (flags & 0x01) else 0
            rec["syn"] += 1 if (flags & 0x02) else 0
            rec["rst"] += 1 if (flags & 0x04) else 0
            rec["psh"] += 1 if (flags & 0x08) else 0
            rec["ack"] += 1 if (flags & 0x10) else 0
            rec["urg"] += 1 if (flags & 0x20) else 0

    rows = []
    for (src, dst, sport, dport, proto), rec in flows.items():
        dur = max(rec["end"] - rec["start"], 1e-6)
        lens = np.asarray(rec["lens"], dtype=np.float32)
        mean_len = float(lens.mean()) if len(lens) else 0.0
        rows.append(
            {
                "src": src,
                "dst": dst,
                "sport": sport,
                "dport": dport,
                "proto": proto,
                "Flow Duration": dur,
                "Total Fwd Packets": float(rec["fwd_pkts"]),
                "Total Backward Packets": float(rec["bwd_pkts"]),
                "Total Length of Fwd Packets": float(rec["fwd_bytes"]),
                "Total Length of Bwd Packets": float(rec["bwd_bytes"]),
                "Flow Bytes/s": float(rec["bytes"] / dur),
                "Flow Packets/s": float(rec["pkts"] / dur),
                "Packet Length Max": float(lens.max()) if len(lens) else 0.0,
                "Packet Length Min": float(lens.min()) if len(lens) else 0.0,
                "Packet Length Mean": mean_len,
                "Packet Length Std": float(lens.std()) if len(lens) else 0.0,
                "FIN Flag Count": float(rec["fin"]),
                "SYN Flag Count": float(rec["syn"]),
                "RST Flag Count": float(rec["rst"]),
                "PSH Flag Count": float(rec["psh"]),
                "ACK Flag Count": float(rec["ack"]),
                "URG Flag Count": float(rec["urg"]),
                "Average Packet Size": mean_len,
            }
        )
    return pd.DataFrame(rows)


def align(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if len(df) == 0:
        return np.zeros((0, len(cols)), dtype=np.float32)
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            out[c] = pd.Series(np.zeros(len(df), dtype=np.float32))
    return pd.DataFrame(out)[cols].to_numpy(dtype=np.float32)


def run_phase1(flow_df: pd.DataFrame, artifact_path: Path):
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing phase1 artifact: {artifact_path}")
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)

    cols = list(art["features"])
    x = align(flow_df, cols)
    xs = art["scaler"].transform(x)
    iso_score = -art["isolation_forest"].score_samples(xs)
    svm_score = -art["one_class_svm"].decision_function(xs)
    iso_pred = iso_score >= float(art["iso_threshold"])
    svm_pred = svm_score >= float(art["svm_threshold"])
    attack = (iso_pred | svm_pred).astype(int)
    return attack, iso_score, svm_score


def run_phase2(flow_df: pd.DataFrame, model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing phase2 artifact: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")

    cfg = ckpt["config"]
    model = Autoencoder(
        int(cfg["n_features"]),
        tuple(cfg["hidden_dims"]),
        int(cfg["bottleneck_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feat = list(ckpt["feature_names"])
    mean = np.asarray(ckpt["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler"]["scale"], dtype=np.float32)
    thr = float(ckpt["threshold"])

    x = align(flow_df, feat)
    xs = (x - mean) / np.where(scale == 0, 1.0, scale)
    with torch.no_grad():
        xt = torch.from_numpy(xs.astype(np.float32))
        rec = model(xt)
        err = ((rec - xt) ** 2).mean(dim=1).cpu().numpy()
    attack = (err >= thr).astype(int)
    return attack, err


def main():
    args = parse_args()
    mode = normalize_model_name(args.model)
    pkts = capture_packets(args.pcap, args.iface, args.timeout)
    flow_df = packets_to_df(pkts)

    print(f"packets: {len(pkts)}  flows: {len(flow_df)}")
    if len(flow_df) == 0:
        print("No IP/TCP/UDP flows extracted.")
        return

    out = flow_df[["src", "dst", "sport", "dport", "proto"]].copy()

    if mode in {"phase1", "both"}:
        p1_attack, iso_score, svm_score = run_phase1(flow_df, args.phase1_model)
        out["model1_attack"] = p1_attack
        out["model1_iso_score"] = iso_score
        out["model1_svm_score"] = svm_score

    if mode in {"phase2", "both"}:
        p2_attack, dl_err = run_phase2(flow_df, args.phase2_model)
        out["model2_attack"] = p2_attack
        out["model2_error"] = dl_err

    if mode == "phase1":
        out["final_attack"] = out["model1_attack"].astype(int)
    elif mode == "phase2":
        out["final_attack"] = out["model2_attack"].astype(int)
    else:
        out["final_attack"] = (
            (out["model1_attack"].fillna(0).astype(int) == 1)
            | (out["model2_attack"].fillna(0).astype(int) == 1)
        ).astype(int)

    flagged = int(out["final_attack"].sum())
    print(f"flows flagged: {flagged} / {len(out)}")
    print(out.head(max(1, args.max_rows)).to_string(index=False))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"saved csv: {args.out_csv}")


if __name__ == "__main__":
    main()
