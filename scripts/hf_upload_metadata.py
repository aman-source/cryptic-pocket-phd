#!/usr/bin/env python3
"""Upload metadata/ directory to HuggingFace dataset."""
import os
import sys

from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "aman-gpt/cryptic-pocket-task-a1"
BASE = os.environ.get("METADATA_DIR", "/workspace/cryptic-pocket-phd/results/task_a1/metadata")

api = HfApi(token=TOKEN)

jsons = sorted(f for f in os.listdir(BASE) if f.endswith(".json"))
print(f"Uploading {len(jsons)} metadata files to {REPO_ID}/metadata/", flush=True)

api.upload_folder(
    folder_path=BASE,
    path_in_repo="metadata",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message=f"Add metadata for {len(jsons)} proteins (backbone extraction)",
)
print("DONE", flush=True)
