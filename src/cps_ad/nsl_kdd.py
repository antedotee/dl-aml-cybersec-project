"""NSL-KDD tabular loader (train/test) with benign-vs-attack binary labels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

# NSL-KDD Train+ / Test+ schema (41 features + difficulty + label for Train+)
NSL_FEATURE_COLS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


def _binary_attack_label(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    y = (~s.eq("normal")).astype(np.int32).to_numpy()
    return y


def _read_nsl_table(path: Path) -> pd.DataFrame:
    """NSL files are comma-separated without a header row."""
    df = pd.read_csv(path, header=None, low_memory=False)
    n = df.shape[1]
    # NSL raw files use: 41 features, then ``label``, then ``difficulty`` (Train+/Test+).
    if n == len(NSL_FEATURE_COLS) + 2:
        cols = NSL_FEATURE_COLS + ["label", "difficulty"]
    elif n == len(NSL_FEATURE_COLS) + 1:
        cols = NSL_FEATURE_COLS + ["label"]
    else:
        raise ValueError(f"Unexpected column count {n} in {path}")
    df.columns = cols[: df.shape[1]]
    return df


def load_nsl_kdd_frames(
    *,
    data_dir: Path | None = None,
    download: bool = True,
    train_variant: str = "twenty",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load KDDTrain+ and KDDTest+ as DataFrames.

    Looks under ``data_dir`` (default: ``<project>/data/nsl``). If missing and
    ``download=True``, attempts HTTPS download using certifi for the CA bundle.
    """
    root = Path(__file__).resolve().parents[2]
    nsl_dir = data_dir or (root / "data" / "nsl")
    nsl_dir.mkdir(parents=True, exist_ok=True)

    if train_variant == "full":
        train_name = "KDDTrain+.txt"
    elif train_variant == "twenty":
        train_name = "KDDTrain+_20Percent.txt"
    else:
        raise ValueError("train_variant must be 'full' or 'twenty'")

    train_p = nsl_dir / train_name
    test_p = nsl_dir / "KDDTest+.txt"

    url_base = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
    train_urls = (f"{url_base}/{train_name}",)
    test_urls = (f"{url_base}/KDDTest+.txt",)

    def fetch_first(urls_try: tuple[str, ...], dest: Path) -> None:
        import ssl
        import urllib.error
        import urllib.request

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        last_err: Exception | None = None
        for url in urls_try:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cps-ad/1.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                    dest.write_bytes(resp.read())
                return
            except (urllib.error.URLError, OSError) as e:
                last_err = e
        raise RuntimeError(f"Could not download {dest.name}: {last_err}") from last_err

    if download:
        if not train_p.exists():
            fetch_first(train_urls, train_p)
        if not test_p.exists():
            fetch_first(test_urls, test_p)

    train_df = _read_nsl_table(train_p)
    test_df = _read_nsl_table(test_p)
    return train_df, test_df


def frames_to_xy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Drop difficulty/label; return (X_train, y_train, X_test, y_test) with y=1 attack."""
    def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if "difficulty" in df.columns:
            lab = df["label"]
            x = df.drop(columns=["label", "difficulty"])
        else:
            lab = df["label"]
            x = df.drop(columns=["label"])
        y = _binary_attack_label(lab)
        return x, y

    x_tr, y_tr = xy(train_df)
    x_te, y_te = xy(test_df)
    return x_tr, y_tr, x_te, y_te
