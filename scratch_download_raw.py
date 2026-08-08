import sys
from huggingface_hub import snapshot_download

print("=== LMSYS-Chat-1M (data/*.parquet only) ===", flush=True)
p = snapshot_download(
    repo_id="lmsys/lmsys-chat-1m", repo_type="dataset",
    local_dir="data/raw/lmsys-chat",
    allow_patterns=["data/*.parquet"],
)
print("lmsys ->", p, flush=True)

print("=== SafeDialBench (full repo, small) ===", flush=True)
p2 = snapshot_download(
    repo_id="HongyeCao/SafeDialBench", repo_type="dataset",
    local_dir="data/raw/safedialbench",
)
print("safedial ->", p2, flush=True)
print("DOWNLOAD_COMPLETE", flush=True)
