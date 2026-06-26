"""Download ONE LMSYS-Chat-1M parquet shard and filter for benign multi-turn convos.

Each shard is ~250 MB / ~167K rows. After English + >=MIN_TURNS user-turns +
unflagged moderation filters, expect ~5-15K rows kept — more than enough for
our 3K target.

This avoids the slow HF streaming row-group decoder by downloading one full
shard with a progress bar, then iterating locally with pyarrow.

Filters:
  - language == "English"
  - >= MIN_TURNS user/assistant message pairs
  - openai_moderation: no message flagged in any category
  - drop redacted=True (PII redaction artifacts)

Prereqs:
  1. Accept license at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
  2. `hf auth login` or `export HF_TOKEN=hf_...`

Usage:
  python scripts/download_lmsys_benign.py [--target 3000] [--min-turns 3] [--shard 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "lmsys/lmsys-chat-1m"
SHARDS = [
    "data/train-00000-of-00006-4feeb3f83346a0e9.parquet",
    "data/train-00001-of-00006-4030672591c2f478.parquet",
    "data/train-00002-of-00006-1779b7cec9462180.parquet",
    "data/train-00003-of-00006-2fa862bfed56af1f.parquet",
    "data/train-00004-of-00006-18f4bdd50c103e71.parquet",
    "data/train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
]
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "benign_multiturn"


def is_clean_unflagged(moderation) -> bool:
    if moderation is None or len(moderation) == 0:
        return True
    for m in moderation:
        cats = m.get("categories") if isinstance(m, dict) else None
        if not cats:
            continue
        for v in cats.values():
            if bool(v):
                return False
    return True


def num_user_turns(conversation) -> int:
    if conversation is None:
        return 0
    return sum(1 for m in conversation if isinstance(m, dict) and m.get("role") == "user")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=3000)
    p.add_argument("--min-turns", type=int, default=3)
    p.add_argument("--shard", type=int, default=0, help="which of the 6 shards to use")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    shard_path = SHARDS[args.shard]

    print(f"downloading shard {args.shard}: {shard_path}", flush=True)
    local_path = hf_hub_download(
        repo_id=REPO,
        filename=shard_path,
        repo_type="dataset",
        token=token,
    )
    print(f"  cached at {local_path}", flush=True)

    pf = pq.ParquetFile(local_path)
    total_rows = pf.metadata.num_rows
    print(f"  shard rows: {total_rows:,}  row_groups: {pf.num_row_groups}", flush=True)

    kept: list[dict] = []
    scanned = 0
    en_count = 0
    long_count = 0
    clean_count = 0

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        df_chunk = tbl.to_pandas()

        for row in df_chunk.itertuples(index=False):
            scanned += 1
            if getattr(row, "language", None) != "English":
                continue
            en_count += 1

            conv = getattr(row, "conversation", None)
            if num_user_turns(conv) < args.min_turns:
                continue
            long_count += 1

            mod = getattr(row, "openai_moderation", None)
            if not is_clean_unflagged(mod):
                continue
            if getattr(row, "redacted", False):
                continue
            clean_count += 1

            kept.append({
                "conversation_id": getattr(row, "conversation_id", None),
                "model": getattr(row, "model", None),
                "language": getattr(row, "language", None),
                "turn": getattr(row, "turn", None),
                "n_user_turns": num_user_turns(conv),
                "conversation": list(conv) if conv is not None else [],
            })

            if len(kept) >= args.target:
                break

        print(
            f"  rg {rg_idx + 1}/{pf.num_row_groups}  scanned={scanned:>7,}  "
            f"en={en_count:>6,}  long={long_count:>6,}  clean={clean_count:>6,}  "
            f"kept={len(kept):>5,}",
            flush=True,
        )
        if len(kept) >= args.target:
            break

    print(
        f"\nDONE  scanned={scanned:,}  en={en_count:,}  long={long_count:,}  "
        f"clean={clean_count:,}  kept={len(kept):,}"
    )

    if not kept:
        print("nothing kept; aborting", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(kept)
    out_parquet = OUT_DIR / "lmsys_benign_multiturn.parquet"
    df.to_parquet(out_parquet)
    print(f"wrote {out_parquet}  ({len(df):,} rows)")

    sample_path = OUT_DIR / "lmsys_benign_multiturn.sample.json"
    sample_path.write_text(
        json.dumps(df.head(3).to_dict(orient="records"), indent=2, default=str)
    )
    print(f"wrote {sample_path}  (3 rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf this is a 401/403, accept the dataset license at "
            "https://huggingface.co/datasets/lmsys/lmsys-chat-1m and re-auth.",
            file=sys.stderr,
        )
        sys.exit(1)
