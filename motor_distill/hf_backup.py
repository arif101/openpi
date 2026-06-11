"""Back up irreplaceable motor-distill assets (rollout datasets + checkpoints + scripts
+ specs) to a PRIVATE HF dataset repo before the GPU box is stopped."""
import os, glob
from huggingface_hub import HfApi, create_repo, get_token

TOKEN = get_token()  # read from ~/.cache/huggingface/token (no secret on cmdline)
assert TOKEN, "no stored HF token found"
REPO = "arif101/openpi-motordistill-backup-20260611"
api = HfApi(token=TOKEN)

create_repo(REPO, repo_type="dataset", private=True, exist_ok=True, token=TOKEN)
print(f"repo ready: {REPO}", flush=True)

# 1) the big irreplaceable asset: data/ (rollout npz dirs + *.pt checkpoints)
print("uploading data/ (rollouts + checkpoints) ...", flush=True)
api.upload_folder(folder_path="/root/openpi/data", path_in_repo="data",
                  repo_id=REPO, repo_type="dataset",
                  ignore_patterns=["dino_cache/*", "_stale_*/*"])
print("data/ done", flush=True)

# 2) box-only scripts + specs (cheap, for full reproducibility)
print("uploading scripts + specs ...", flush=True)
api.upload_folder(folder_path="/root/openpi/motor_distill", path_in_repo="motor_distill",
                  repo_id=REPO, repo_type="dataset",
                  allow_patterns=["*.py", "*.md", "*.sh"])
for f in glob.glob("/root/openpi/*.md") + glob.glob("/root/openpi/run_*.sh"):
    api.upload_file(path_or_fileobj=f, path_in_repo="root_" + os.path.basename(f),
                    repo_id=REPO, repo_type="dataset")
print("ALL BACKUP DONE ->", REPO, flush=True)
