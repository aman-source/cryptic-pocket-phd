#!/usr/bin/env python3
"""Sanity check: load npz + metadata, extract backbone, verify CA-CA distances.

Run after metadata generation + HF upload to confirm backbone extraction works.

Usage:
    python scripts/sanity_check_backbone.py \
        --metadata_dir results/task_a1/metadata \
        --npz_dir results/task_a1 \
        --n_proteins 5
"""
import json
import os
import sys
from pathlib import Path

import click
import numpy as np


def extract_backbone(noisy_coords: np.ndarray, backbone_indices: list) -> np.ndarray:
    """Extract backbone (N, CA, C, O) from all-atom coords using metadata.

    Args:
        noisy_coords: (N_atoms_padded, 3) all-atom Boltz coords
        backbone_indices: list of [N_idx, CA_idx, C_idx, O_idx] per residue

    Returns:
        (N_residues, 4, 3) backbone coords
    """
    bb_flat = []
    for indices in backbone_indices:
        bb_flat.extend(indices)
    return noisy_coords[bb_flat].reshape(len(backbone_indices), 4, 3)


@click.command()
@click.option("--metadata_dir", type=click.Path(exists=True), required=True)
@click.option("--npz_dir", type=click.Path(exists=True), required=True)
@click.option("--n_proteins", type=int, default=5)
def main(metadata_dir, npz_dir, n_proteins):
    metadata_dir = Path(metadata_dir)
    npz_dir = Path(npz_dir)

    # Find proteins with both metadata and npz data
    meta_files = sorted(f for f in os.listdir(metadata_dir) if f.endswith(".json"))
    click.echo(f"Found {len(meta_files)} metadata files")

    checked = 0
    passed = 0

    for meta_file in meta_files:
        domain_id = meta_file.replace(".json", "")
        protein_npz_dir = npz_dir / domain_id

        if not protein_npz_dir.exists():
            continue

        # Find first npz
        npz_files = sorted(f for f in os.listdir(protein_npz_dir) if f.endswith(".npz"))
        if not npz_files:
            continue

        # Load metadata
        with open(metadata_dir / meta_file) as f:
            meta = json.load(f)

        # Load npz
        npz_path = protein_npz_dir / npz_files[0]
        data = np.load(npz_path)
        noisy_coords = data["noisy_coords"]
        t = float(data["t"])
        pocket_labels = data["pocket_labels"]

        # Verify shapes
        assert noisy_coords.shape[0] == meta["n_atoms_padded"], \
            f"{domain_id}: noisy_coords {noisy_coords.shape[0]} != padded {meta['n_atoms_padded']}"
        assert pocket_labels.shape[0] == meta["n_residues"], \
            f"{domain_id}: labels {pocket_labels.shape[0]} != residues {meta['n_residues']}"

        # Extract backbone
        bb = extract_backbone(noisy_coords, meta["backbone_indices"])
        assert bb.shape == (meta["n_residues"], 4, 3), \
            f"{domain_id}: backbone shape {bb.shape} != ({meta['n_residues']}, 4, 3)"

        # Check CA-CA distances
        ca = bb[:, 1, :]  # CA is index 1
        ca_dists = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        mean_d = ca_dists.mean()
        min_d = ca_dists.min()
        max_d = ca_dists.max()
        reasonable = ((ca_dists > 2.0) & (ca_dists < 6.0)).sum()
        frac = reasonable / len(ca_dists) * 100

        ok = frac > 90  # >90% reasonable CA-CA distances
        status = "PASS" if ok else "FAIL"
        click.echo(
            f"[{status}] {domain_id} (t={t:.1f}): "
            f"{meta['n_residues']} res, "
            f"CA-CA mean={mean_d:.2f} min={min_d:.2f} max={max_d:.2f}, "
            f"reasonable={frac:.0f}%"
        )

        if ok:
            passed += 1
        checked += 1

        if checked >= n_proteins:
            break

    click.echo(f"\n{passed}/{checked} passed sanity check")
    if passed < checked:
        sys.exit(1)


if __name__ == "__main__":
    main()
