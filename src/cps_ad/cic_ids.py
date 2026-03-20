"""CIC-IDS2017 loader: download, schema normalization, binary labels, zero-day splits.

The dataset (Sharafaldin et al., 2018) consists of eight day-files of CICFlowMeter-v3
features extracted from real captures. We:

* Provide an auto-downloader that tries a list of mirrors and falls back gracefully
  with explicit instructions for manual placement.
* Normalize column names (the published CSVs ship with leading whitespace and a few
  inconsistent forms across days) and drop non-feature meta columns.
* Apply the *Engelen 2021* hygiene rules: drop infinity / NaN, clip negatives, and
  optionally relabel a small set of mislabelled flows.
* Expose binary BENIGN-vs-attack labels plus an attack-family taxonomy that supports
  **leave-one-family-out** zero-day evaluation.
"""

from __future__ import annotations

import io
import logging
import re
import ssl
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


CIC_DAY_FILES: tuple[str, ...] = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)


# Attack family taxonomy used for leave-one-family-out zero-day eval.
# Keys are the human-readable family names we surface in the API; values are the
# raw label strings the published CSVs use (after .strip()).
ATTACK_FAMILIES: dict[str, tuple[str, ...]] = {
    "DoS": ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"),
    "DDoS": ("DDoS",),
    "PortScan": ("PortScan",),
    "BruteForce": ("FTP-Patator", "SSH-Patator"),
    "WebAttack": (
        "Web Attack \xe2\x80\x93 Brute Force",
        "Web Attack \xe2\x80\x93 XSS",
        "Web Attack \xe2\x80\x93 Sql Injection",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "Web Attack \u2013 Brute Force",
        "Web Attack \u2013 XSS",
        "Web Attack \u2013 Sql Injection",
    ),
    "Infiltration": ("Infiltration",),
    "Bot": ("Bot",),
    "Heartbleed": ("Heartbleed",),
}


# Columns the published CSVs include but that are NOT model features:
#   * Flow Bytes/s and Flow Packets/s contain `inf` for instantaneous flows;
#     we keep them but cleanse below.
#   * Identifier columns leak the source / destination of each flow and would
#     make the problem trivial; we drop them everywhere.
_DROP_LEAKY_COLS: tuple[str, ...] = (
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Fwd Header Length.1",  # Duplicated header column in the published files.
)


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _normalize_label_text(s: str) -> str:
    """Collapse whitespace/dash variants used inconsistently across days."""
    if s is None:
        return ""
    out = str(s).strip()
    out = out.replace("\u2013", "-")
    out = re.sub(r"\s+", " ", out)
    return out


def _family_lookup_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for fam, raw_labels in ATTACK_FAMILIES.items():
        for raw in raw_labels:
            table[_normalize_label_text(raw)] = fam
    return table


_FAMILY_TABLE = _family_lookup_table()


def label_to_family(label: str) -> str:
    """Map a raw label string to its attack family or 'BENIGN'."""
    norm = _normalize_label_text(label)
    if norm.lower() == "benign":
        return "BENIGN"
    fam = _FAMILY_TABLE.get(norm)
    if fam is not None:
        return fam
    LOG.warning("Unknown CIC-IDS2017 label %r treated as 'Other'", norm)
    return "Other"


@dataclass(frozen=True)
class CicSplit:
    """Materialized features and labels for one phase (train/val/test)."""

    x: pd.DataFrame
    y: np.ndarray  # 0 = BENIGN, 1 = attack
    family: np.ndarray  # str per row; 'BENIGN' for negatives.

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.x.shape[0])


@dataclass(frozen=True)
class CicZeroDaySplit:
    """Train/Val/Test split with one attack family held out for zero-day eval.

    `train` and `val` contain only BENIGN flows.
    `test_known` contains BENIGN + every non-held-out attack (sanity baseline).
    `test_zero_day` contains BENIGN + only the held-out family (the real metric).
    """

    train: CicSplit
    val: CicSplit
    test_known: CicSplit
    test_zero_day: CicSplit
    held_out_family: str
    feature_names: tuple[str, ...]


# Mirrors that historically host the CIC-IDS2017 CSVs as a single zip.
# The first one that responds wins. Users can override via env / function arg.
DEFAULT_MIRRORS: tuple[str, ...] = (
    # GitHub mirror (no auth required, ~500 MB).
    "https://github.com/cybersecurity-utrgv/CIC-IDS2017/releases/download/v1.0/MachineLearningCSV.zip",
    # University of Florence mirror (occasionally up).
    "https://intrusion-detection.distributedsystems.it/data/CIC-IDS-2017/MachineLearningCSV.zip",
)


def _http_download(url: str, dest: Path, *, timeout: int = 600) -> None:
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "cps-ad/2.0"})
    LOG.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _try_download_zip(mirrors: Sequence[str], dest_zip: Path) -> bool:
    last_err: Exception | None = None
    for url in mirrors:
        try:
            _http_download(url, dest_zip)
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            LOG.warning("Mirror failed (%s): %s", url, e)
            last_err = e
    LOG.error("All CIC-IDS2017 mirrors failed (last error: %s)", last_err)
    return False


def _extract_csvs(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            base = Path(member).name
            if not base.lower().endswith(".csv"):
                continue
            target = out_dir / base
            if target.exists():
                continue
            with z.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())


def ensure_cic_ids2017(
    *,
    data_dir: Path | None = None,
    download: bool = True,
    mirrors: Sequence[str] = DEFAULT_MIRRORS,
) -> Path:
    """Make sure all eight day-CSVs live under ``data_dir``; return that directory."""
    root = Path(__file__).resolve().parents[2]
    cic_dir = data_dir or (root / "data" / "cic_ids")
    cic_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CIC_DAY_FILES if not (cic_dir / name).exists()]
    if not missing:
        return cic_dir

    if not download:
        raise FileNotFoundError(
            f"CIC-IDS2017 CSVs missing under {cic_dir} and download=False. "
            f"Missing: {missing}"
        )

    zip_path = cic_dir / "_MachineLearningCSV.zip"
    if not zip_path.exists():
        ok = _try_download_zip(mirrors, zip_path)
        if not ok:
            raise RuntimeError(
                "Could not download CIC-IDS2017 from any configured mirror. "
                "Place 'MachineLearningCSV.zip' (the eight day CSVs) under "
                f"{cic_dir} or pass mirrors=... to ensure_cic_ids2017()."
            )
    _extract_csvs(zip_path, cic_dir)

    still_missing = [n for n in CIC_DAY_FILES if not (cic_dir / n).exists()]
    if still_missing:
        raise FileNotFoundError(
            f"After extraction, files are still missing: {still_missing}"
        )
    return cic_dir


def _read_one_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df = _strip_columns(df)
    return df


def _coerce_numeric(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Force every feature column to numeric, replace +-inf with NaN, fill 0."""
    out = df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    arr = out[list(feature_cols)].to_numpy(dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out.loc[:, list(feature_cols)] = arr
    return out


def load_cic_ids2017(
    *,
    data_dir: Path | None = None,
    download: bool = True,
    files: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate the 8 day-CSVs into a single normalized DataFrame.

    Output columns:
      * the CICFlowMeter v3 feature columns (numeric, NaN/inf cleansed)
      * ``Label``  — raw string label (whitespace-trimmed)
      * ``Family`` — coarse attack family or 'BENIGN'
    """
    cic_dir = ensure_cic_ids2017(data_dir=data_dir, download=download)
    selected = list(files) if files is not None else list(CIC_DAY_FILES)

    frames: list[pd.DataFrame] = []
    for name in selected:
        path = cic_dir / name
        if not path.exists():
            LOG.warning("Missing day file %s, skipping", name)
            continue
        frames.append(_read_one_day(path))
    if not frames:
        raise FileNotFoundError("No CIC-IDS2017 day-files were readable.")

    # Some files have an extra/duplicate column; align on the intersection.
    common_cols = set.intersection(*(set(f.columns) for f in frames))
    frames = [f[sorted(common_cols)] for f in frames]
    df = pd.concat(frames, axis=0, ignore_index=True)

    label_col = "Label" if "Label" in df.columns else None
    if label_col is None:
        raise KeyError("CIC-IDS2017 frames are missing the 'Label' column.")

    drop_cols = [c for c in _DROP_LEAKY_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    df["Label"] = df["Label"].map(_normalize_label_text)
    df["Family"] = df["Label"].map(label_to_family)

    feature_cols = [c for c in df.columns if c not in {"Label", "Family"}]
    df = _coerce_numeric(df, feature_cols)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in df.columns if c not in {"Label", "Family"})


def make_zero_day_split(
    df: pd.DataFrame,
    *,
    held_out_family: str = "Heartbleed",
    val_frac: float = 0.10,
    test_benign_frac: float = 0.20,
    seed: int = 42,
) -> CicZeroDaySplit:
    """Carve a train/val/test_known/test_zero_day split.

    * BENIGN rows are split into (train, val, test_benign) using ``val_frac`` and
      ``test_benign_frac``.
    * `held_out_family` rows go ENTIRELY into ``test_zero_day`` (joined with the
      benign test pool).
    * All other attack rows go into ``test_known`` (joined with the benign test
      pool).
    """
    if held_out_family not in ATTACK_FAMILIES and held_out_family != "BENIGN":
        raise ValueError(
            f"Unknown family {held_out_family!r}; choose from {sorted(ATTACK_FAMILIES)}"
        )

    feats = feature_columns(df)
    rng = np.random.default_rng(seed)

    benign_mask = df["Family"].to_numpy() == "BENIGN"
    benign_idx = np.flatnonzero(benign_mask)
    rng.shuffle(benign_idx)
    n_benign = benign_idx.size
    n_test_benign = int(round(n_benign * test_benign_frac))
    n_val = int(round(n_benign * val_frac))
    test_benign_idx = benign_idx[:n_test_benign]
    val_idx = benign_idx[n_test_benign : n_test_benign + n_val]
    train_idx = benign_idx[n_test_benign + n_val :]

    family = df["Family"].to_numpy()
    zero_attack_idx = np.flatnonzero(family == held_out_family)
    known_attack_idx = np.flatnonzero((family != "BENIGN") & (family != held_out_family))

    def make_split(idx: np.ndarray) -> CicSplit:
        sub = df.iloc[idx]
        x = sub[list(feats)].reset_index(drop=True)
        y = (sub["Family"].to_numpy() != "BENIGN").astype(np.int32)
        fam = sub["Family"].to_numpy().astype(str)
        return CicSplit(x=x, y=y, family=fam)

    test_known_idx = np.concatenate([test_benign_idx, known_attack_idx])
    test_zero_idx = np.concatenate([test_benign_idx, zero_attack_idx])
    rng.shuffle(test_known_idx)
    rng.shuffle(test_zero_idx)

    return CicZeroDaySplit(
        train=make_split(train_idx),
        val=make_split(val_idx),
        test_known=make_split(test_known_idx),
        test_zero_day=make_split(test_zero_idx),
        held_out_family=held_out_family,
        feature_names=feats,
    )


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-family counts; useful for logging during training."""
    out = df["Family"].value_counts().rename_axis("family").reset_index(name="count")
    out["fraction"] = out["count"] / out["count"].sum()
    return out
