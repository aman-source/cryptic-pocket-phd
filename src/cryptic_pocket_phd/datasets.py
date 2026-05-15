"""Dataset for noise-aware PocketMiner training.

Loads Boltz intermediate npz files + per-protein metadata from HuggingFace.
Extracts backbone (N/CA/C/O) using CCD atom offsets stored in metadata.

Each sample: (backbone_coords [N_res, 4, 3], t, pocket_labels [N_res], sequence)
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


# Timesteps captured during Task A1
TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]


class NoisyPocketDataset(Dataset):
    """Dataset of noisy Boltz intermediates with pocket labels.

    Index: (protein_id, frame_idx, timestep) tuples.
    Each __getitem__ returns backbone coords extracted via metadata.
    """

    def __init__(
        self,
        protein_list: list[str],
        hf_repo: str = "aman-gpt/cryptic-pocket-task-a1",
        cache_dir: str = "/tmp/hf_cache",
        hf_token: Optional[str] = None,
        n_frames: int = 100,
        timesteps: Optional[list[float]] = None,
        max_proteins: Optional[int] = None,
    ):
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.timesteps = timesteps or TIMESTEPS
        self.n_frames = n_frames

        if max_proteins is not None:
            protein_list = protein_list[:max_proteins]

        # Load metadata for each protein
        self.metadata: dict[str, dict] = {}
        self._load_metadata(protein_list)

        # Build index: list of (protein_id, frame_idx, t)
        self.index: list[tuple[str, int, float]] = []
        for pid in protein_list:
            if pid not in self.metadata:
                continue
            for frame_idx in range(n_frames):
                for t in self.timesteps:
                    self.index.append((pid, frame_idx, t))

    def _load_metadata(self, protein_list: list[str]):
        """Download and cache metadata JSONs from HF."""
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

    def _download_npz(self, protein_id: str, frame_idx: int, t: float) -> str:
        """Download single npz from HF. Returns local path."""
        from huggingface_hub import hf_hub_download

        filename = f"{protein_id}/{frame_idx:04d}_t{t:.1f}.npz"
        return hf_hub_download(
            repo_id=self.hf_repo,
            filename=filename,
            repo_type="dataset",
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        protein_id, frame_idx, t = self.index[idx]
        meta = self.metadata[protein_id]

        # Load npz
        npz_path = self._download_npz(protein_id, frame_idx, t)
        data = np.load(npz_path)
        noisy_coords = data["noisy_coords"]  # (N_atoms_padded, 3)
        pocket_labels = data["pocket_labels"]  # (N_res,)

        # Extract backbone using metadata indices
        bb_flat = []
        for indices in meta["backbone_indices"]:
            bb_flat.extend(indices)
        backbone = noisy_coords[bb_flat].reshape(meta["n_residues"], 4, 3)

        # Sequence → AA indices
        seq = meta["sequence"]
        s_indices = np.array(
            [AA_LOOKUP.get(aa, 0) for aa in seq], dtype=np.int64
        )

        return {
            "coords": torch.from_numpy(backbone).float(),       # (N_res, 4, 3)
            "seq": torch.from_numpy(s_indices).long(),           # (N_res,)
            "t": torch.tensor(t, dtype=torch.float32),           # scalar
            "labels": torch.from_numpy(pocket_labels).float(),   # (N_res,)
            "n_res": meta["n_residues"],
            "protein_id": protein_id,
        }


def collate_variable_length(batch: list[dict]) -> dict:
    """Collate samples with variable N_residues into padded batch.

    Returns:
        coords: (B, N_max, 4, 3) padded backbone
        seq: (B, N_max) padded AA indices
        t: (B,) timesteps
        labels: (B, N_max) padded labels
        mask: (B, N_max) 1 for real residues, 0 for padding
    """
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
