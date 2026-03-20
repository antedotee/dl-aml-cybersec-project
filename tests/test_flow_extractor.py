"""Flow extractor tests.

We deliberately avoid requiring scapy in CI: all tests use the dict-based
``add_packet`` interface that bypasses scapy parsing. A separate scapy-marked
test exercises the PCAP code path when scapy is present.
"""

from __future__ import annotations

import pytest

from cps_ad.flow_extractor import (
    CIC_FEATURE_COLUMNS,
    FlowAggregator,
    align_to_training_columns,
    to_numpy_features,
)


def _pkt(t, src, sport, dst, dport, length, flags="", header=20, payload=0, win=0):
    return {
        "time": t, "src": src, "dst": dst, "sport": sport, "dport": dport,
        "len": length, "ihl": 20, "header_len": header, "flags": flags,
        "win": win, "payload_len": payload, "proto": "TCP",
    }


def test_flow_aggregator_collects_bidirectional() -> None:
    agg = FlowAggregator()
    agg.add_packet(_pkt(1.0, "a", 100, "b", 80, 200, flags="S", win=8192))
    agg.add_packet(_pkt(1.1, "b", 80, "a", 100, 60, flags="SA", win=8192))
    agg.add_packet(_pkt(1.2, "a", 100, "b", 80, 1500, flags="PA"))
    agg.add_packet(_pkt(1.3, "b", 80, "a", 100, 1500, flags="PA"))
    rows = list(agg.drain())
    assert len(rows) == 1
    r = rows[0]
    assert r["Total Fwd Packets"] == 2
    assert r["Total Backward Packets"] == 2
    assert r["SYN Flag Count"] == 2  # one from each side
    assert r["Flow Duration"] > 0
    assert set(CIC_FEATURE_COLUMNS).issubset(set(r.keys()))


def test_align_to_training_columns_pads_and_orders() -> None:
    import pandas as pd
    df = pd.DataFrame([{"Total Fwd Packets": 1.0, "Flow Duration": 100.0}])
    aligned = align_to_training_columns(df, list(CIC_FEATURE_COLUMNS))
    assert list(aligned.columns) == list(CIC_FEATURE_COLUMNS)
    assert aligned["Total Fwd Packets"].iloc[0] == 1.0
    assert aligned["Flow Bytes/s"].iloc[0] == 0.0


def test_to_numpy_features_replaces_non_finite() -> None:
    import numpy as np
    import pandas as pd
    df = pd.DataFrame([{c: 0.0 for c in CIC_FEATURE_COLUMNS}])
    df.loc[0, "Flow Bytes/s"] = float("inf")
    arr = to_numpy_features(df)
    assert arr.dtype == np.float32
    assert np.all(np.isfinite(arr))


def test_aggregator_handles_unidirectional_only() -> None:
    agg = FlowAggregator()
    for k in range(5):
        agg.add_packet(_pkt(1.0 + 0.05 * k, "a", 1234, "b", 22, 60, flags="S"))
    rows = list(agg.drain())
    assert len(rows) == 1
    r = rows[0]
    assert r["Total Fwd Packets"] == 5
    assert r["Total Backward Packets"] == 0
    assert r["SYN Flag Count"] == 5
    assert r["Down/Up Ratio"] == 0.0


@pytest.mark.skipif(
    pytest.importorskip("scapy", reason="scapy not installed") is None,
    reason="scapy missing"
)
def test_extract_from_pcap_roundtrip(tmp_path) -> None:
    from scapy.all import IP, TCP, wrpcap

    from cps_ad.flow_extractor import extract_from_pcap

    pkts = [
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="S"),
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=12345, flags="SA"),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="A"),
    ]
    for i, p in enumerate(pkts):
        p.time = 1700000000.0 + i * 0.1
    path = tmp_path / "tiny.pcap"
    wrpcap(str(path), pkts)
    df = extract_from_pcap(path)
    assert len(df) == 1
    assert df["Total Fwd Packets"].iloc[0] >= 1
