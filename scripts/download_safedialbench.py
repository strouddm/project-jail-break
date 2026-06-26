"""Download HongyeCao/SafeDialBench to data/raw/safedialbench/ as parquet.

License: Apache-2.0 (ungated, no HuggingFace login required).
Source: https://huggingface.co/datasets/HongyeCao/SafeDialBench
Paper: SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models
       in Multi-Turn Dialogues with Diverse Jailbreak Attacks

Usage:
    python scripts/download_safedialbench.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset

REPO = "HongyeCao/SafeDialBench"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "safedialbench"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")  # not required, but used if present

    print(f"=== {REPO} ===")
    ds = load_dataset(REPO, token=token)
    for split, dset in ds.items():
        out = OUT_DIR / f"{split}.parquet"
        dset.to_parquet(out)
        print(f"  wrote {out}  ({len(dset):,} rows, {len(dset.column_names)} cols)")
        print(f"  columns: {dset.column_names}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
