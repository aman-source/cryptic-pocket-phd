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
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

# Import our LIGSITE implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cryptic_pocket_phd.ligsite import ligsite_labels, compute_ligsite_grid, assign_pocket_to_residues


def _process_protein(args):
    """Worker function: compute LIGSITE labels for all frames of one protein."""
    pdir, out_path, pos_thresh, min_rank, grid_spacing = args
    protein_id = pdir.name
    label_dir = out_path / protein_id
    label_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(pdir.glob("*.npz"))
    n_new = 0
    n_done = 0
    t0 = time.time()

    for frame_file in frame_files:
        frame_idx = frame_file.stem
        label_file = label_dir / f"{frame_idx}_labels.npz"
        if label_file.exists():
            n_done += 1
            continue

        data = np.load(frame_file, allow_pickle=True)
        coords = data["coords"]
        ca_coords = data["ca_coords"]
        n_residues = len(ca_coords)

        labels = ligsite_labels(
            coords, ca_coords, n_residues,
            pos_thresh=pos_thresh, min_rank=min_rank, grid_spacing=grid_spacing,
        )
        pocket_points, pocket_ranks = compute_ligsite_grid(
            coords, grid_spacing=grid_spacing, min_rank=min_rank
        )
        scores = assign_pocket_to_residues(pocket_points, pocket_ranks, ca_coords, n_residues)

        np.savez_compressed(
            label_file,
            labels=labels,
            scores=scores,
            n_pocket_points=len(pocket_points),
        )
        n_new += 1

    elapsed = time.time() - t0
    pos_total = sum(
        int(np.load(label_dir / f"{f.stem}_labels.npz")["labels"].sum())
        for f in frame_files
        if (label_dir / f"{f.stem}_labels.npz").exists()
    )
    res_total = sum(
        int(len(np.load(f, allow_pickle=True)["ca_coords"]))
        for f in frame_files
    )
    return protein_id, n_new, n_done, pos_total, res_total, elapsed


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--pos_thresh", type=int, default=20, help="Grid point count for positive label")
@click.option("--min_rank", type=int, default=7, help="Min PSP directions (max 7)")
@click.option("--grid_spacing", type=float, default=1.0, help="Grid spacing in Angstroms")
@click.option("--workers", type=int, default=1,
              help="Number of parallel worker processes (per-protein parallelism). "
                   "Use os.cpu_count()//2 as a safe default.")
def main(frames_dir: str, out_dir: str, pos_thresh: int, min_rank: int,
         grid_spacing: float, workers: int):
    frames_path = Path(frames_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    protein_dirs = sorted([
        d for d in frames_path.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    print(f"Found {len(protein_dirs)} proteins in {frames_path}")
    print(f"LIGSITE settings: min_rank={min_rank}, pos_thresh={pos_thresh}, grid={grid_spacing} A")
    print(f"Workers: {workers}")

    args_list = [
        (pdir, out_path, pos_thresh, min_rank, grid_spacing)
        for pdir in protein_dirs
        if sorted(pdir.glob("*.npz"))  # skip empty dirs
    ]

    total_frames = 0
    total_positive = 0
    total_residues = 0

    if workers == 1:
        results = [_process_protein(a) for a in args_list]
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_process_protein, args_list)

    for i, (protein_id, n_new, n_done, pos_total, res_total, elapsed) in enumerate(results):
        total_frames += n_new
        total_positive += pos_total
        total_residues += res_total
        print(
            f"[{i+1}/{len(results)}] {protein_id}: "
            f"{n_new + n_done} frames ({n_new} new, {n_done} cached), "
            f"{elapsed:.1f}s"
        )

    if total_residues > 0:
        pos_rate = total_positive / total_residues
        print(f"\nOverall: {total_frames} new frames, {total_positive}/{total_residues} "
              f"positive residues ({pos_rate:.3f} rate)")
    else:
        print("\nNo frames processed.")


if __name__ == "__main__":
    main()
