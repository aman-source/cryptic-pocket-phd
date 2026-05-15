#!/usr/bin/env python3
"""Upload task_a1 results to HuggingFace dataset, one protein at a time."""
import os, sys
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "aman-gpt/cryptic-pocket-task-a1"
BASE = "/workspace/cryptic-pocket-phd/results/task_a1"
SKIP = {'.cache', '.progress', 'logs'}

api = HfApi(token=TOKEN)

proteins = sorted(
    p for p in os.listdir(BASE)
    if not p.startswith('.') and p not in SKIP and os.path.isdir(os.path.join(BASE, p))
)
print(f"Uploading {len(proteins)} proteins to {REPO_ID}", flush=True)

for i, p in enumerate(proteins):
    folder = os.path.join(BASE, p)
    try:
        api.upload_folder(
            folder_path=folder,
            path_in_repo=p,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"{p} ({i+1}/{len(proteins)})",
        )
        print(f"[{i+1}/{len(proteins)}] {p} OK", flush=True)
    except Exception as e:
        print(f"[{i+1}/{len(proteins)}] {p} FAILED: {e}", flush=True)

print("ALL DONE", flush=True)
