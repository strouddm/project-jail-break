from huggingface_hub import snapshot_download
from pyprojroot import here

# Absolute paths anchored to repo root
REPO_ROOT = here()
RAW_DATA_DIR = REPO_ROOT / 'data' / 'raw'


snapshot_download(repo_id="allenai/wildguardmix",
                  repo_type="dataset",
                  local_dir=RAW_DATA_DIR / "wildguardmix",)

# LMSYS-Chat-1M
snapshot_download(repo_id="lmsys/lmsys-chat-1m",
                  repo_type="dataset",
                  local_dir=RAW_DATA_DIR / "lmsys-chat",)