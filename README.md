# Project: Detection of Harmful and Jailbreak-Style Prompts

## Goal

Build a classifier to flag harmful / jailbreak-style prompts before they reach an LLM, comparing classical ML (Logistic Regression, FFNN) against a transformer-based classifier.

---

## Directory Structure

proj-jail-break/
├── README.md                        # project overview + setup instructions
├── CONTRIBUTING.md                  # contribution guidelines
├── requirements.txt                 # python dependencies
├── .gitignore                       # excludes venv, data contents, model artifacts
├── data/
│   ├── processed/                   # cleaned/unified datasets (gitignored contents)
│   └── raw/                         # original downloaded datasets (gitignored contents)
│       ├── wildguardmix/            # ~92K single-turn prompts (benign + harmful)
│       └── lmsys-chat/              # ~1M real-world LLM conversations
├── notebooks/                       # EDA + model experiments (per-contributor)
├── reports/                         # presentation deck + figures
└── scripts/                         # data download + preprocessing pipelines

Empty tracked directories get a `.gitkeep`. For the gitignored dirs (`data/raw/`, `data/processed/`), `.gitignore` ignores their *contents* (e.g. `data/raw/*`) while un-ignoring `.gitkeep`, so the directory structure travels with a clone but data/processed files stay out of git.

---

## Data Sources

This project aggregates and standardizes data from two primary datasets to create a unified, multi-category classification resource:

**WildGuardMix** ~92K single-turn prompts - Primary source; provides benign and harmful examples.
**LMSYS-Chat-1M** ~1M real-world LLM conversations - Large-scale in-the-wild prompts (benign + unsafe).

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

    Both datasets download in a single step. They are **gated** on Hugging Face, so you must accept their terms (while logged in) before your token can pull them. Access is granted automatically — there's no approval wait.

    1. Install prerequisites: `pip install datasets huggingface_hub`

    2. Create a Hugging Face account (or log in): https://huggingface.co/join

    3. Accept the terms on each gated dataset page (one click each):

        - WildGuardMix — https://huggingface.co/datasets/allenai/wildguardmix
        - LMSYS-Chat-1M — https://huggingface.co/datasets/lmsys/lmsys-chat-1m

    4. Create an access token: Settings -> Access Tokens -> New Token

    5. Authenticate locally:

        ```bash
        huggingface-cli login   # paste your token when prompted
        ```

    6. Download both datasets. The simplest way is to run the script:

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
└── lmsys-chat/
```

