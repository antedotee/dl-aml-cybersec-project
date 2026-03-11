#!/usr/bin/env python3
"""Generate interpretable EDA summaries and figures (Phase 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cps_ad.data import load_kddcup99_sa
from cps_ad.synthetic_data import load_synthetic_intrusion_sa_like
from cps_ad.preprocess import numeric_skewness


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--full-data",
        action="store_true",
        help="Use the full KDDCup99 extract (slow/large). Default uses the 10% sample.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic tabular data (offline).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    args = p.parse_args()

    if args.synthetic:
        x, y = load_synthetic_intrusion_sa_like(random_state=args.random_state)
    else:
        x, y = load_kddcup99_sa(percent10=not args.full_data, random_state=args.random_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.where(y == 0, "normal", "attack")
    bal = pd.Series(labels).value_counts(normalize=True)
    print("Class balance (fraction):")
    print(bal.to_string())
    print()

    skew = numeric_skewness(x)
    print("Top absolute skew (numeric features):")
    print(skew.head(12).to_string())
    print()

    num = x.select_dtypes(include=[np.number])
    if len(num.columns):
        corr = num.corr(numeric_only=True)
        plt.figure(figsize=(11, 9))
        sns.heatmap(corr, cmap="vlag", center=0.0)
        plt.title("Numeric feature correlation (KDDCup99 SA)")
        plt.tight_layout()
        plt.savefig(args.out_dir / "corr_numeric.png", dpi=160)
        plt.close()

    sample_cols = [c for c in ["duration", "src_bytes", "dst_bytes", "count"] if c in x.columns]
    for c in sample_cols:
        plt.figure(figsize=(7, 4))
        sns.kdeplot(data=x.assign(_label=labels), x=c, hue="_label", common_norm=False)
        plt.title(f"Distribution by label: {c}")
        plt.tight_layout()
        safe = c.replace("/", "_")
        plt.savefig(args.out_dir / f"kde_{safe}.png", dpi=160)
        plt.close()

    print(f"Wrote figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
