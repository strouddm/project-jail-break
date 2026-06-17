# Project: Detection of Harmful and Jailbreak-Style Prompts

## Goal

Build a classifier to flag harmful / jailbreak-style prompts before they reach an LLM, comparing classical ML (Logistic Regression, XGBoost) against a transformer-based classifier using contextual embeddings.

---

## Directory Structure

```
proj-jail-break/
├── README.md            # project overview + data-download instructions
├── requirements.txt     # pandas, scikit-learn, etc.
├── .gitignore           # ignores venv, caches, data/raw, data/processed, model artifacts (*.pkl/*.pt/*.bin)
├── data/
│   ├── raw/             # gitignored — three empty dataset dirs staged:
│   │   ├── wildguardmix/    # ~92K single-turn prompts (primary, benign+harmful)
│   │   ├── safedialbench/   # ~4K multi-turn attacks
│   │   └── multibreak/      # ~10K adversarial jailbreaks
│   └── processed/       # gitignored — unified/cleaned output
├── notebooks/           # EDA + experiments (empty)
├── src/                 # flat Python modules: data prep, features, models, eval (empty)
├── scripts/             # runnable CLIs: download, train, evaluate (empty)
└── reports/
    └── figures/         # plots & result artifacts (empty)
```

Empty tracked directories get a `.gitkeep`: `notebooks/`, `src/`, `scripts/`, `reports/figures/`, `data/processed/`.

---

## Data Sources

This project aggregates and standardizes data from three primary datasets to create a unified, multi-category classification resource:

**WildGuardMix** ~92K single-turn prompts - Primary source; provides benign and harmful examples.
**MultiBreak** ~10K adversarial prompts - Adversarial, jailbreak-style tactics. 
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

3. Add the WildGuardMix dataset to `data/raw/wildguardmix`

    1. Install prerequisites: `pip install datasets huggingface_hub`

    2. Create a Hugging Face account (or log in): https://huggingface.co/join

    3. Open the dataset page and accept the terms: https://huggingface.co/datasets/allenai/wildguardmix

    4. Create an access token: Settings -> Access Tokens -> New Token

    5. Authenticate locally:

        ```bash
        huggingface-cli login   # paste your token when prompted
        ```

    6. Download the files into the local directory

        ```python 
        from huggingface_hub import snapshot_download

        # Add WildGuardMix
        snapshot_download(
        repo_id="allenai/wildguardmix",
        repo_type="dataset",
        local_dir="data/raw/wildguardmix",)

        # SafeDialBench
        snapshot_download(
        repo_id="HongyeCao/SafeDialBench",
        repo_type="dataset",
        local_dir="data/raw/safedialbench",)

        # MultiBreak
        snapshot_download(
        repo_id="microsoft/MultiBreak",
        repo_type="dataset",
        local_dir="data/raw/multibreak",)

        ```
        * Of note, there is a script available at /scripts/download_datasets.py built to do this

## Notes

After running the steps above you should have the raw data in the following structure. 
The data lives in these directories (gitignored)

```
data/raw/
├── wildguardmix/
├── safedialbench/
└── multibreak/
```

