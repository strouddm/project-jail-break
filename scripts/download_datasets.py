from huggingface_hub import snapshot_download


snapshot_download(repo_id="allenai/wildguardmix",
                  repo_type="dataset",
                  local_dir="data/raw/wildguardmix",)

# SafeDialBench
snapshot_download(repo_id="HongyeCao/SafeDialBench",
                  repo_type="dataset",
                  local_dir="data/raw/safedialbench",)

# LMSYS-Chat-1M
snapshot_download(repo_id="lmsys/lmsys-chat-1m",
                  repo_type="dataset",
                  local_dir="data/raw/lmsys-chat",)