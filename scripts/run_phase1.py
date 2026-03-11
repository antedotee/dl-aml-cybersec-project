#!/usr/bin/env python3
"""Train/evaluate Phase-1 detectors and emit a metrics table + JSON artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cps_ad.phase1 import results_to_markdown_table, run_phase1, write_json


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
        help="Use built-in synthetic tabular data (offline / CI). For coursework, prefer real KDDCup99.",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=ROOT / "reports" / "phase1_metrics.json")
    args = p.parse_args()

    results, meta = run_phase1(
        percent10=not args.full_data,
        random_state=args.random_state,
        synthetic=args.synthetic,
    )
    payload = {
        "meta": meta,
        "rows": [
            {
                "name": r.name,
                "val_threshold": r.val_threshold,
                "val_f1_at_threshold": r.val_f1_at_threshold,
                "test": r.test.__dict__,
            }
            for r in results
        ],
    }
    write_json(args.out_json, payload)

    print(results_to_markdown_table(results))
    print()
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
