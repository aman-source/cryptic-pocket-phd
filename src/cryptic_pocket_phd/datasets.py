"""Dataset for noise-aware PocketMiner training.

Loads Boltz intermediate npz files + per-protein metadata.
Supports two modes:
  1. local_data_dir: read directly from local disk (fast, preferred)
  2. HF streaming: download per-file from HuggingFace (slow, fallback)

Use snapshot_download_dataset() to pre-download everything to local disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .pocketminer_torch import AA_LOOKUP

TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]


def snapshot_download_dataset(
    hf_repo: str = "aman-gpt/cryptic-pocket-task-a1",
    local_dir: str = "/workspace/data/cryptic-pocket-task-a1",
    hf_token: Optional[str] = None,
) -> str:
    """Download entire HF dataset to local disk. Returns local_dir path."""
    from huggingface_hub import snapshot_download

    local_dir = str(local_dir)
    if os.path.exists(os.path.join(local_dir, "metadata")):
        n_meta = len([f for f in os.listdir(os.path.join(local_dir, "metadata")) if f.endswith(".json")])
        if n_meta >= 120:
            print(f"Dataset already downloaded: {local_dir} ({n_meta} proteins)")
            return local_dir

    print(f"Downloading {hf_repo} to {local_dir}...")
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=local_dir,
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    print(f"Download complete: {local_dir}")
    return local_dir


class NoisyPocketDataset(Dataset):
    """Dataset of noisy Boltz intermediates with pocket labels.

    If local_data_dir is set, reads from local disk (fast).
    Otherwise falls back to HF streaming (slow).
    """

    def __init__(
        self,
        protein_list: list[str],
        local_data_dir: Optional[str] = None,
        hf_repo: str = "aman-gpt/cryptic-pocket-task-a1",
        cache_dir: str = "/tmp/hf_cache",
        hf_token: Optional[str] = None,
        n_frames: int = 100,
        timesteps: Optional[list[float]] = None,
        max_proteins: Optional[int] = None,
    ):
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.timesteps = timesteps or TIMESTEPS
        self.n_frames = n_frames

        if max_proteins is not None:
            protein_list = protein_list[:max_proteins]

        # Load metadata
        self.metadata: dict[str, dict] = {}
        self._load_metadata(protein_list)

        # Precompute backbone indices as numpy arrays for speed
        self._bb_indices: dict[str, np.ndarray] = {}
        for pid, meta in self.metadata.items():
            flat = []
            for indices in meta["backbone_indices"]:
                flat.extend(indices)
            self._bb_indices[pid] = np.array(flat, dtype=np.int64)

        # Precompute sequence indices
        self._seq_indices: dict[str, np.ndarray] = {}
        for pid, meta in self.metadata.items():
            self._seq_indices[pid] = np.array(
                [AA_LOOKUP.get(aa, 0) for aa in meta["sequence"]], dtype=np.int64
            )

        # Build index
        self.index: list[tuple[str, int, float]] = []
        for pid in protein_list:
            if pid not in self.metadata:
                continue
            for frame_idx in range(n_frames):
                for t in self.timesteps:
                    self.index.append((pid, frame_idx, t))

    def _load_metadata(self, protein_list: list[str]):
        """Load metadata from local dir or HF."""
        if self.local_data_dir:
            meta_dir = self.local_data_dir / "metadata"
            for pid in protein_list:
                path = meta_dir / f"{pid}.json"
                if path.exists():
                    with open(path) as f:
                        self.metadata[pid] = json.load(f)
                else:
                    print(f"WARNING: metadata not found: {path}")
        else:
            from huggingface_hub import hf_hub_download
            for pid in protein_list:
                try:
                    path = hf_hub_download(
                        repo_id=self.hf_repo,
                        filename=f"metadata/{pid}.json",
                        repo_type="dataset",
                        cache_dir=self.cache_dir,
                        token=self.hf_token,
                    )
                    with open(path) as f:
                        self.metadata[pid] = json.load(f)
                except Exception as e:
                    print(f"WARNING: metadata for {pid} not found: {e}")

    def _get_npz_path(self, protein_id: str, frame_idx: int, t: float) -> str:
        """Get path to npz file (local or HF download)."""
        if self.local_data_dir:
            return str(self.local_data_dir / protein_id / f"{frame_idx:04d}_t{t:.1f}.npz")
        else:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=self.hf_repo,
                filename=f"{protein_id}/{frame_idx:04d}_t{t:.1f}.npz",
                repo_type="dataset",
                cache_dir=self.cache_dir,
                token=self.hf_token,
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        protein_id, frame_idx, t = self.index[idx]
        meta = self.metadata[protein_id]

        npz_path = self._get_npz_path(protein_id, frame_idx, t)
        data = np.load(npz_path)
        noisy_coords = data["noisy_coords"]
        pocket_labels = data["pocket_labels"]

        # Extract backbone using precomputed indices
        backbone = noisy_coords[self._bb_indices[protein_id]].reshape(
            meta["n_residues"], 4, 3
        )

        return {
            "coords": torch.from_numpy(backbone).float(),
            "seq": torch.from_numpy(self._seq_indices[protein_id].copy()).long(),
            "t": torch.tensor(t, dtype=torch.float32),
            "labels": torch.from_numpy(pocket_labels).float(),
            "n_res": meta["n_residues"],
            "protein_id": protein_id,
        }


def collate_variable_length(batch: list[dict]) -> dict:
    """Collate samples with variable N_residues into padded batch."""
    B = len(batch)
    N_max = max(b["n_res"] for b in batch)

    coords = torch.zeros(B, N_max, 4, 3)
    seq = torch.zeros(B, N_max, dtype=torch.long)
    t = torch.stack([b["t"] for b in batch])
    labels = torch.zeros(B, N_max)
    mask = torch.zeros(B, N_max)

    for i, b in enumerate(batch):
        n = b["n_res"]
        coords[i, :n] = b["coords"]
        seq[i, :n] = b["seq"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = 1.0

    return {
        "coords": coords,
        "seq": seq,
        "t": t,
        "labels": labels,
        "mask": mask,
    }
