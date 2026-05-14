#!/usr/bin/env python3
"""
Task A0: Compute LIGSITE per-residue pocket labels for preprocessed frames.

For each frame npz:
  - Load all-atom coords + CA coords
  - Run LIGSITE grid scan (min_rank=7, grid_spacing=1.0)
  - Assign pocket grid points to nearest residue
  - Threshold at pos_thresh=20 for binary labels
  - Save: data/md_labels/{protein_id}/{frame_idx}_labels.npz

Uses same LIGSITE settings as PocketMiner published training pipeline.

Usage:
  python scripts/compute_ligsite_labels.py \
    --frames_dir data/md_frames/atlas \
    --out_dir data/md_labels/atlas \
    --pos_thresh 20 \
    --min_rank 7
"""
import click
import json
import os
import time
from pathlib import Path

import numpy as np

# Import our LIGSITE implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cryptic_pocket_phd.ligsite import ligsite_labels, compute_ligsite_grid, assign_pocket_to_residues


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--pos_thresh", type=int, default=20, help="Grid point count for positive label")
@click.option("--min_rank", type=int, default=7, help="Min PSP directions (max 7)")
@click.option("--grid_spacing", type=float, default=1.0, help="Grid spacing in Angstroms")
def main(frames_dir: str, out_dir: str, pos_thresh: int, min_rank: int, grid_spacing: float):
    frames_path = Path(frames_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all protein directories
    protein_dirs = sorted([
        d for d in frames_path.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    print(f"Found {len(protein_dirs)} proteins in {frames_path}")
    print(f"LIGSITE settings: min_rank={min_rank}, pos_thresh={pos_thresh}, grid={grid_spacing} A")

    total_frames = 0
    total_positive = 0
    total_residues = 0

    for i, pdir in enumerate(protein_dirs):
        protein_id = pdir.name
        label_dir = out_path / protein_id
        label_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        with open(pdir / "metadata.json") as f:
            meta = json.load(f)

        # Find all frame files
        frame_files = sorted(pdir.glob("*.npz"))
        if not frame_files:
            print(f"[{i+1}/{len(protein_dirs)}] {protein_id}: no frames, skipping")
            continue

        n_done = 0
        n_new = 0
        t0 = time.time()

        for frame_file in frame_files:
            frame_idx = frame_file.stem  # e.g., "0042"
            label_file = label_dir / f"{frame_idx}_labels.npz"

            if label_file.exists():
                n_done += 1
                continue

            # Load frame
            data = np.load(frame_file, allow_pickle=True)
            coords = data["coords"]      # (N_atoms, 3) in Angstroms
            ca_coords = data["ca_coords"]  # (N_res, 3) in Angstroms
            n_residues = len(ca_coords)

            # Compute LIGSITE labels
            labels = ligsite_labels(
                coords, ca_coords, n_residues,
                pos_thresh=pos_thresh,
                min_rank=min_rank,
                grid_spacing=grid_spacing,
            )

            # Also save raw scores (before thresholding) for analysis
            pocket_points, pocket_ranks = compute_ligsite_grid(
                coords, grid_spacing=grid_spacing, min_rank=min_rank
            )
            scores = assign_pocket_to_residues(
                pocket_points, pocket_ranks, ca_coords, n_residues
            )

            np.savez_compressed(
                label_file,
                labels=labels,
                scores=scores,
                n_pocket_points=len(pocket_points),
            )

            total_frames += 1
            total_positive += labels.sum()
            total_residues += n_residues
            n_new += 1

        elapsed = time.time() - t0
        total_done = n_done + n_new
        print(
            f"[{i+1}/{len(protein_dirs)}] {protein_id}: "
            f"{total_done} frames ({n_new} new, {n_done} cached), "
            f"{elapsed:.1f}s"
        )

    if total_residues > 0:
        pos_rate = total_positive / total_residues
        print(f"\nOverall: {total_frames} frames, {total_positive}/{total_residues} "
              f"positive residues ({pos_rate:.3f} rate)")
    else:
        print("\nNo frames processed.")


if __name__ == "__main__":
    main()
