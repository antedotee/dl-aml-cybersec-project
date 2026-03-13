"""NSL-KDD column layout: label precedes difficulty in raw Train+/Test+ files."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from cps_ad.nsl_kdd import NSL_FEATURE_COLS, _binary_attack_label, _read_nsl_table, frames_to_xy


def test_read_nsl_table_label_before_difficulty(tmp_path: Path) -> None:
    # One numeric feature column + rest dummy: build minimal 43-col row
    n_feat = len(NSL_FEATURE_COLS)
    vals = ["0"] * n_feat
    vals[-1] = "5"
    line = ",".join(vals) + ",normal,20\n"
    p = tmp_path / "row.txt"
    p.write_text(line, encoding="utf-8")
    df = _read_nsl_table(p)
    assert "label" in df.columns and "difficulty" in df.columns
    assert str(df.loc[0, "label"]).strip().lower() == "normal"
    assert int(df.loc[0, "difficulty"]) == 20


def test_binary_attack_label() -> None:
    s = pd.Series(["normal", "neptune", "Normal "])
    y = _binary_attack_label(s)
    assert list(y) == [0, 1, 0]


def test_frames_to_xy_drops_meta() -> None:
    n = len(NSL_FEATURE_COLS)
    row = {c: 0 for c in NSL_FEATURE_COLS}
    row["label"] = "normal"
    row["difficulty"] = 1
    tr = pd.DataFrame([row])
    te = pd.DataFrame([{**row, "label": "neptune"}])
    xtr, ytr, xte, yte = frames_to_xy(tr, te)
    assert ytr[0] == 0 and yte[0] == 1
    assert "label" not in xtr.columns and "difficulty" not in xtr.columns
