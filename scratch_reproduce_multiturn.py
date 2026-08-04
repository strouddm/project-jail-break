"""Faithful reproduction of Rachel's multiturn dataset pipeline.

Stage 1 (from 02_rachel_eda_lmsys.ipynb cells 0-14): enrich SafeDialBench
        (harmful) with a turn-stratified LMSYS sample (benign) -> writes
        data/processed/safedial_enriched_with_benign.csv   [DETERMINISTIC]

Stage 2 (from 02_rachel_preprocess_data.ipynb multiturn cells): preprocess +
        split -> multiturn_X_{train,test,val}.csv           [NON-DETERMINISTIC:
        her np.random.shuffle is unseeded]

Run from anywhere; paths are absolute to the repo.
"""
import glob
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = "/Users/flo/project1/207 - Applied ML/project-jail-break"
LMSYS_GLOB = f"{REPO}/data/raw/lmsys-chat/data/*.parquet"
# NOTE: Rachel's code globbed data/by_task/english/*.jsonl (from a GitHub-style
# layout). The HF repo is flat, so we read the equivalent English file. All
# safedial rows are used as the harmful half (no random sampling), so layout
# only affects row order, not content or benign sample counts.
SAFEDIAL_JSONL = f"{REPO}/data/raw/safedialbench/datasets_en.jsonl"
OUT_ENRICHED = f"{REPO}/data/processed/safedial_enriched_with_benign.csv"
OUT_TRAIN = f"{REPO}/data/processed/multiturn_X_train.csv"
OUT_TEST = f"{REPO}/data/processed/multiturn_X_test.csv"
OUT_VAL = f"{REPO}/data/processed/multiturn_X_val.csv"

SEED = 1234


# ---------------------------------------------------------------------------
# STAGE 1 — enrichment (her eda_lmsys cells 0-14, verbatim logic)
# ---------------------------------------------------------------------------
def normalize_conversation(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            x = eval(x)
    normalized = []
    for turn in x:
        if isinstance(turn, np.ndarray):
            turn = turn.tolist()
        if "role" in turn and "content" in turn:
            normalized.append({"role": turn["role"], "content": turn["content"]})
        elif "user" in turn and "bot" in turn:
            normalized.append({"role": "user", "content": turn["user"]})
            normalized.append({"role": "assistant", "content": turn["bot"]})
        else:
            normalized.append(dict(turn))
    return normalized


def stage1_enrich():
    # --- lmsys (cell 1) ---
    files = sorted(glob.glob(LMSYS_GLOB))
    print(f"[stage1] lmsys shards found: {len(files)}")
    assert files, f"no lmsys parquet at {LMSYS_GLOB}"
    dfs = [pd.read_parquet(f, engine="pyarrow") for f in files]
    data = pd.concat(dfs).reset_index().drop(columns=["index"])
    print(f"[stage1] lmsys total rows: {len(data):,}")

    # --- pre-process (cell 2): English only ---
    data_proc = data[(data["language"] == "English")].copy()
    print(f"[stage1] lmsys English rows: {len(data_proc):,}")

    # --- safedial (cell 3, adapted to flat HF layout) ---
    import os
    assert os.path.exists(SAFEDIAL_JSONL), f"no safedial jsonl at {SAFEDIAL_JSONL}"
    rows = []
    with open(SAFEDIAL_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    safedial = pd.DataFrame(rows)
    safedial["turn"] = safedial["history"].apply(lambda x: len(x))
    print(f"[stage1] safedial rows: {len(safedial):,}")

    # --- normalize (cell 11) ---
    safedial["history"] = safedial["history"].apply(normalize_conversation)
    data_proc["conversation"] = data_proc["conversation"].apply(normalize_conversation)

    # --- stratified benign sample (cell 12) ---
    turn_distr = safedial["turn"].value_counts().to_dict()
    print("[stage1] per-turn feasibility (need vs available in lmsys English pool):")
    samples = []
    shortfalls = {}
    for turn, count in turn_distr.items():
        pool = data_proc[data_proc["turn"] == turn]
        avail = len(pool)
        flag = "OK" if avail >= count else f"SHORT by {count - avail}"
        if avail < count:
            shortfalls[turn] = count - avail
        print(f"    turn={turn:>3}: need {count:>5}  avail {avail:>6}  {flag}")
        # replicate her call exactly (will raise if short — that is informative)
        samples.append(pool.sample(count, random_state=SEED))
    stratified_sample = pd.concat(samples)
    stratified_sample["harm"] = False
    safedial["harm"] = True

    # --- merge (cell 14) ---
    merged = pd.concat(
        [
            safedial[["id", "turn", "history", "harm"]].rename(
                columns={"id": "conversation_id", "history": "conversation"}
            ),
            stratified_sample[["conversation_id", "turn", "conversation", "harm"]],
        ]
    ).reset_index().drop(columns=["index"])
    merged["conversation"] = merged["conversation"].apply(json.dumps)
    merged.to_csv(OUT_ENRICHED, index=False)
    print(f"[stage1] wrote {OUT_ENRICHED}")
    print(f"[stage1] merged rows: {len(merged):,}  "
          f"(harmful={int((merged.harm==True).sum())}, benign={int((merged.harm==False).sum())})")
    return merged


# ---------------------------------------------------------------------------
# STAGE 2 — preprocess + split (her preprocess_data multiturn cells, verbatim)
# ---------------------------------------------------------------------------
def conversation_to_text(conversation):
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except Exception:
            conversation = eval(conversation)
    return " ".join(f"{t['role']}: {t['content']}" for t in conversation)


def preprocessor(conversation):
    text = conversation_to_text(conversation) if isinstance(conversation, list) else conversation
    text = text.replace("\n", " ")
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\\)|\\(|D|P)", text)
    text = re.sub("[\\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    text = text.replace("user", "USER:")
    text = text.replace("assistant", "ASSISTANT:")
    return text


def stage2_split():
    X_multi = pd.read_csv(OUT_ENRICHED)
    X_multi["conversation"] = X_multi["conversation"].apply(json.loads)
    X_multi["conversation"] = X_multi["conversation"].apply(
        lambda x: preprocessor(conversation_to_text(x))
    )

    # her code: unseeded np.random.shuffle (NON-DETERMINISTIC)
    idx = X_multi.index.to_list()
    np.random.shuffle(idx)
    data_shuffled = X_multi.loc[idx].reset_index(drop=True)
    X = data_shuffled.drop(columns=["harm"])
    y = data_shuffled[["harm"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print(f"[stage2] split sizes  train={len(X_train)}  test={len(X_test)}  val={len(X_val)}")
    X_train.to_csv(OUT_TRAIN, index=False)
    X_test.to_csv(OUT_TEST, index=False)
    X_val.to_csv(OUT_VAL, index=False)
    print(f"[stage2] wrote multiturn_X_train/test/val.csv")


if __name__ == "__main__":
    stage1_enrich()
    stage2_split()
    print("REPRODUCE_COMPLETE")
