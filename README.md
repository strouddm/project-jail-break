# Project: Detection of Harmful and Jailbreak-Style Prompts

## Goal

Build a classifier to flag harmful / jailbreak-style prompts before they reach an LLM, comparing classical ML (Logistic Regression, XGBoost) against a transformer-based classifier using contextual embeddings.

---

## Directory Structure

```
proj-jail-break/
├── README.md            # project overview + data-download instructions
├── requirements.txt     # pandas, scikit-learn, etc.
├── .gitignore           # ignores venv, caches, data contents, model artifacts (*.pkl/*.pt/*.bin) — keeps dirs via .gitkeep
├── data/
│   ├── raw/             # contents gitignored, dir tracked — three empty dataset dirs staged:
│   │   ├── wildguardmix/    # ~92K single-turn prompts (primary, benign+harmful)
│   │   ├── safedialbench/   # ~4K multi-turn attacks
│   │   └── lmsys-chat/      # ~1M real-world LLM conversations
│   └── processed/       # contents gitignored, dir tracked — unified/cleaned output
├── models/              # contents gitignored, dir tracked — trained model artifacts
├── notebooks/           # EDA + experiments (empty)
├── src/                 # flat Python modules: data prep, features, models, eval (empty)
├── scripts/             # runnable CLIs: download, train, evaluate (empty)
└── reports/
    └── figures/         # plots & result artifacts (empty)
```

Empty tracked directories get a `.gitkeep`: `notebooks/`, `src/`, `scripts/`, `reports/figures/`, `data/raw/`, `data/processed/`, and `models/`. For the gitignored dirs (`data/raw/`, `data/processed/`, `models/`), `.gitignore` ignores their *contents* (e.g. `data/raw/*`) while un-ignoring `.gitkeep`, so the directory structure travels with a clone but data/model files stay out of git.

---

## Data Sources

This project aggregates and standardizes data from three primary datasets to create a unified, multi-category classification resource:

**WildGuardMix** ~92K single-turn prompts - Primary source; provides benign and harmful examples.
**LMSYS-Chat-1M** ~1M real-world LLM conversations - Large-scale in-the-wild prompts (benign + unsafe).
**SafeDialBench** ~4K multi-turn attacks - Multi-turn conversational threats.

---

## Setup & Data Download

1. Clone the repository

```bash
git clone https://github.com/username/proj-jail-break.git
cd proj-jail-break
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the datasets into `data/raw/`

    All three datasets download in a single step. Two of them are **gated** on Hugging Face, so you must accept their terms (while logged in) before your token can pull them. Access is granted automatically — there's no approval wait.

    1. Install prerequisites: `pip install datasets huggingface_hub`

    2. Create a Hugging Face account (or log in): https://huggingface.co/join

    3. Accept the terms on each gated dataset page (one click each):

        - WildGuardMix — https://huggingface.co/datasets/allenai/wildguardmix
        - LMSYS-Chat-1M — https://huggingface.co/datasets/lmsys/lmsys-chat-1m
        - (SafeDialBench is open — no terms required: https://huggingface.co/datasets/HongyeCao/SafeDialBench)

    4. Create an access token: Settings -> Access Tokens -> New Token

    5. Authenticate locally:

        ```bash
        huggingface-cli login   # paste your token when prompted
        ```

    6. Download all three datasets. The simplest way is to run the script:

        ```bash
        python scripts/download_datasets.py
        ```

        It runs the equivalent of:

        ```python
        from huggingface_hub import snapshot_download

        # WildGuardMix
        snapshot_download(
        repo_id="allenai/wildguardmix",
        repo_type="dataset",
        local_dir="data/raw/wildguardmix",)

        # SafeDialBench
        snapshot_download(
        repo_id="HongyeCao/SafeDialBench",
        repo_type="dataset",
        local_dir="data/raw/safedialbench",)

        # LMSYS-Chat-1M
        snapshot_download(
        repo_id="lmsys/lmsys-chat-1m",
        repo_type="dataset",
        local_dir="data/raw/lmsys-chat",)
        ```

## Notes

After running the steps above you should have the raw data in the following structure. 
The data lives in these directories (gitignored)

```
data/raw/
├── wildguardmix/
├── safedialbench/
└── lmsys-chat/
```

