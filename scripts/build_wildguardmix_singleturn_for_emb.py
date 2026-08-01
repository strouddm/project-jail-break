"""Build a single-turn dataset from WildGuardMix as a second column for comparison
against the raw text.

Usage:
  python scripts/build_wildguardmix_singleturn.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1234
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wildguardmix"
OUT_DIR = REPO_ROOT / "data" / "processed"


def build_singleturn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["prompt_harm_label"]).drop_duplicates(subset="prompt").copy()
    df["harm"] = df["prompt_harm_label"] == "harmful"
    return df


def main() -> None:
    train_raw = pd.read_parquet(RAW_DIR / "train" / "wildguard_train.parquet")
    test_raw = pd.read_parquet(RAW_DIR / "test" / "wildguard_test.parquet")

    train_df = build_singleturn(train_raw)
    test_df = build_singleturn(test_raw)

    train_part, val_part = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df["harm"],
    )

    parts = {"train": train_part, "val": val_part, "test": test_df}
    for split, part in parts.items():
        out = pd.DataFrame({
            "conversation_id": part.index.astype(str),
            "conversation": part["prompt"],
        })
        out.to_csv(OUT_DIR / f"wildguardmix_X_{split}.csv", index=False)
        part[["harm"]].to_csv(OUT_DIR / f"wildguardmix_Y_{split}.csv", index=False)
        print(f"{split:>5}: {len(out):>6} rows -> wildguardmix_X_{split}.csv / wildguardmix_Y_{split}.csv")


if __name__ == "__main__":
    main()
