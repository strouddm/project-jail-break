"""Download allenai/wildguardmix to data/raw/wildguardmix/ as parquet.

Two configs: `wildguardtrain` (~87K) and `wildguardtest` (~1.7K).
Schema (per dataset card):
  - prompt: str
  - adversarial: bool
  - response: str | None  (None for prompt-only items)
  - prompt_harm_label: "harmful" | "unharmful" | None
  - response_harm_label: "harmful" | "unharmful" | None
  - response_refusal_label: "refusal" | "compliance" | None
  - subcategory: str

License: ODC-BY (gated; must accept AI2 Responsible Use Guidelines).

Prereqs:
  1. Visit https://huggingface.co/datasets/allenai/wildguardmix while logged in
     and accept the terms (one-time form).
  2. `hf auth login` or `export HF_TOKEN=hf_...`.

Usage:
  python scripts/download_wildguardmix.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset

REPO = "allenai/wildguardmix"
CONFIGS = ["wildguardtrain", "wildguardtest"]
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "wildguardmix"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")  # falls back to cached login if None

    for cfg in CONFIGS:
        print(f"\n=== {REPO} :: {cfg} ===", flush=True)
        ds = load_dataset(REPO, cfg, token=token)
        for split, dset in ds.items():
            out = OUT_DIR / f"{cfg}-{split}.parquet"
            dset.to_parquet(out)
            print(f"  wrote {out}  ({len(dset):,} rows, {len(dset.column_names)} cols)")
            print(f"  columns: {dset.column_names}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf this is a 401/403, accept the dataset terms at "
            "https://huggingface.co/datasets/allenai/wildguardmix and re-auth.",
            file=sys.stderr,
        )
        sys.exit(1)
