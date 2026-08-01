"""Build a multi-turn dataset from the merged SafeDial+LMSYS data, for embedding
models. Text is flattened to a plain string but NOT lowercased/stopword-stripped
(that's still what train_test_split.py's multiturn_X_*.csv contains).

Requires data/processed/safedial_enriched_with_benign.csv to already exist
(run build_multiturn_dataset.py first).

Usage:
  python scripts/build_multiturn_for_emb.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1234
np.random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed"


def conversation_to_text(conversation):
    """Flatten a list of {'role', 'content'} turns into a single string, excluding role labels."""
    if isinstance(conversation, str):
        conversation = json.loads(conversation)
    return " ".join(
        turn["content"]
        for turn in conversation
        if isinstance(turn, dict)
    )


def main() -> None:
    X_multi = pd.read_csv(OUT_DIR / "safedial_enriched_with_benign.csv")
    X_multi["conversation"] = X_multi["conversation"].apply(json.loads)
    X_multi["conversation"] = X_multi["conversation"].apply(conversation_to_text)
    X_multi = X_multi[X_multi["conversation"].str.strip() != ""]

    idx = X_multi.index.to_list()
    np.random.shuffle(idx)
    data_shuffled = X_multi.loc[idx].reset_index(drop=True)

    X = data_shuffled[["conversation_id", "conversation"]]
    y = data_shuffled[["harm"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y["harm"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train["harm"],
    )

    parts = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for split, (Xp, yp) in parts.items():
        Xp.to_csv(OUT_DIR / f"multiturn_X_{split}_raw.csv", index=False)
        yp.to_csv(OUT_DIR / f"multiturn_Y_{split}_raw.csv", index=False)
        print(f"{split:>5}: {len(Xp):>6} rows -> multiturn_X_{split}_raw.csv / multiturn_Y_{split}_raw.csv")


if __name__ == "__main__":
    main()
