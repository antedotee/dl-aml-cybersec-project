"""Generate two small synthetic PCAPs for the live demo (no network required).

* benign_http.pcap     — short HTTP-like fetches (slow IAT, mixed packet sizes).
* ssh_bruteforce.pcap  — many tiny SYN/ACK exchanges to port 22 in rapid bursts.

These are *crafted* packets, not real captures, but they hit the same flow
features the model sees in CIC-IDS2017 (high SYN flag count, low Down/Up ratio,
tight Flow IAT distribution for the bruteforce case). Use these as a fallback
when you don't have a real capture handy.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _check_scapy() -> None:
    try:
        import scapy  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "scapy is required to generate demo PCAPs. Install with 'pip install scapy'."
        ) from e


def _now_ts() -> float:
    import time
    return time.time()


def make_benign_http(out_path: Path, n_flows: int = 20, seed: int = 1) -> None:
    from scapy.all import IP, TCP, Raw, wrpcap
    rng = random.Random(seed)
    pkts = []
    t0 = _now_ts()
    src_base = "10.0.0."
    for f in range(n_flows):
        src = src_base + str(rng.randint(2, 254))
        dst = "192.0.2." + str(rng.randint(2, 254))
        sport = rng.randint(20000, 40000)
        dport = 80
        seq = rng.randint(1000, 100000)
        # 3-way handshake.
        pkts.append(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="S", seq=seq))
        pkts[-1].time = t0 + f * 0.5
        pkts.append(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="SA",
                                              seq=seq + 100, ack=seq + 1))
        pkts[-1].time = t0 + f * 0.5 + 0.05
        pkts.append(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A",
                                              seq=seq + 1, ack=seq + 101))
        pkts[-1].time = t0 + f * 0.5 + 0.10
        # GET request.
        get_payload = ("GET /page%d HTTP/1.1\r\nHost: example.com\r\n\r\n" % f).encode()
        pkts.append(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="PA",
                                              seq=seq + 1, ack=seq + 101) / Raw(get_payload))
        pkts[-1].time = t0 + f * 0.5 + 0.15
        # Response chunks.
        for k in range(rng.randint(2, 6)):
            sz = rng.randint(200, 1400)
            pkts.append(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="PA",
                                                  seq=seq + 101 + k * sz,
                                                  ack=seq + 1 + len(get_payload))
                         / Raw(b"x" * sz))
            pkts[-1].time = t0 + f * 0.5 + 0.20 + k * 0.03
        # FIN.
        pkts.append(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="FA"))
        pkts[-1].time = t0 + f * 0.5 + 0.5
    wrpcap(str(out_path), pkts)
    print(f"wrote {len(pkts)} packets across {n_flows} flows to {out_path}")


def make_ssh_bruteforce(out_path: Path, n_attempts: int = 200, seed: int = 2) -> None:
    from scapy.all import IP, TCP, wrpcap
    rng = random.Random(seed)
    pkts = []
    t0 = _now_ts()
    attacker = "203.0.113." + str(rng.randint(2, 254))
    victim = "10.0.0.42"
    for i in range(n_attempts):
        sport = rng.randint(40000, 65000)
        seq = rng.randint(1000, 100000)
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=22, flags="S", seq=seq))
        pkts[-1].time = t0 + i * 0.01
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="SA",
                                                      seq=seq + 100, ack=seq + 1))
        pkts[-1].time = t0 + i * 0.01 + 0.001
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=22, flags="A",
                                                      seq=seq + 1, ack=seq + 101))
        pkts[-1].time = t0 + i * 0.01 + 0.002
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="R",
                                                      seq=seq + 101, ack=seq + 1))
        pkts[-1].time = t0 + i * 0.01 + 0.003
    wrpcap(str(out_path), pkts)
    print(f"wrote {len(pkts)} packets across {n_attempts} bruteforce attempts to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/demo_pcaps"))
    args = p.parse_args()
    _check_scapy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_benign_http(args.out_dir / "benign_http.pcap")
    make_ssh_bruteforce(args.out_dir / "ssh_bruteforce.pcap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
