#!/usr/bin/env python3
"""
Test Task A0 pipeline with synthetic data locally.

Creates fake protein trajectories, runs preprocessing and LIGSITE labeling.
Validates code works end-to-end without downloading real ATLAS data.

Usage:
  python scripts/test_pipeline_synthetic.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_synthetic_protein(
    n_residues: int = 100,
    n_atoms_per_residue: int = 8,
    n_frames: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Generate a fake protein with a cavity.

    Returns:
        all_coords: (n_frames, n_atoms, 3)
        ca_coords: (n_frames, n_residues, 3)
        sequence: amino acid sequence
    """
    rng = np.random.default_rng(seed)

    # Generate a helical backbone
    t = np.linspace(0, 4 * np.pi, n_residues)
    ca_x = 10 * np.cos(t)
    ca_y = 10 * np.sin(t)
    ca_z = np.linspace(0, 50, n_residues)
    ca_template = np.column_stack([ca_x, ca_y, ca_z])  # (n_res, 3)

    # Generate all-atom coords around each CA
    n_atoms = n_residues * n_atoms_per_residue
    all_atom_template = np.zeros((n_atoms, 3))
    for i in range(n_residues):
        start = i * n_atoms_per_residue
        end = start + n_atoms_per_residue
        all_atom_template[start:end] = ca_template[i] + rng.normal(0, 0.5, (n_atoms_per_residue, 3))

    # Create a cavity by pushing atoms away from center at residues 40-60
    cavity_residues = range(40, 60)
    for i in cavity_residues:
        start = i * n_atoms_per_residue
        end = start + n_atoms_per_residue
        # Push atoms outward from z-axis
        direction = all_atom_template[start:end, :2].copy()
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms < 0.1] = 1.0
        direction = direction / norms
        all_atom_template[start:end, :2] += direction * 5.0  # 5 Å outward

    # Generate frames with small perturbations
    all_coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    ca_coords_all = np.zeros((n_frames, n_residues, 3), dtype=np.float32)

    for f in range(n_frames):
        noise = rng.normal(0, 0.3, (n_atoms, 3))
        all_coords[f] = all_atom_template + noise
        # CA coords = first atom of each residue
        ca_coords_all[f] = all_coords[f, ::n_atoms_per_residue]

    # Random sequence
    aas = "ACDEFGHIKLMNPQRSTVWY"
    sequence = "".join(rng.choice(list(aas)) for _ in range(n_residues))

    return all_coords, ca_coords_all, sequence


def test_full_pipeline():
    """Run full pipeline on synthetic data."""
    from cryptic_pocket_phd.ligsite import ligsite_labels, compute_ligsite_grid

    out_base = Path("data/synthetic_test")
    frames_dir = out_base / "frames"
    labels_dir = out_base / "labels"

    n_proteins = 5
    n_frames = 10
    n_residues = 80

    print("=== Phase 2 Task A0 Pipeline Test (Synthetic) ===\n")

    # Step 1: Generate synthetic proteins
    print("Step 1: Generating synthetic proteins...")
    for i in range(n_proteins):
        protein_id = f"synth_{i:03d}"
        protein_dir = frames_dir / protein_id
        protein_dir.mkdir(parents=True, exist_ok=True)

        all_coords, ca_coords, sequence = create_synthetic_protein(
            n_residues=n_residues + i * 10,
            n_frames=n_frames,
            seed=42 + i,
        )

        # Save frames (simulating preprocess_md_to_frames.py output)
        for f in range(n_frames):
            np.savez_compressed(
                protein_dir / f"{f:04d}.npz",
                coords=all_coords[f],
                ca_coords=ca_coords[f],
                sequence=sequence,
                frame_index=f,
                protein_id=protein_id,
            )

        # Save metadata
        meta = {
            "protein_id": protein_id,
            "source": "synthetic",
            "total_frames": n_frames,
            "n_extracted": n_frames,
            "n_residues": len(sequence),
            "n_atoms": all_coords.shape[1],
            "sequence": sequence,
        }
        with open(protein_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        print(f"  {protein_id}: {len(sequence)} residues, {n_frames} frames")

    # Step 2: Run LIGSITE labeling
    print("\nStep 2: Computing LIGSITE labels...")
    total_pos = 0
    total_res = 0

    for protein_dir in sorted(frames_dir.iterdir()):
        if not protein_dir.is_dir():
            continue

        protein_id = protein_dir.name
        label_dir = labels_dir / protein_id
        label_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(protein_dir.glob("*.npz"))
        t0 = time.time()

        for frame_file in frame_files:
            data = np.load(frame_file, allow_pickle=True)
            coords = data["coords"]
            ca_coords = data["ca_coords"]
            n_res = len(ca_coords)

            labels = ligsite_labels(
                coords, ca_coords, n_res,
                pos_thresh=20, min_rank=7, grid_spacing=1.0,
            )

            label_file = label_dir / f"{frame_file.stem}_labels.npz"
            np.savez_compressed(label_file, labels=labels)

            total_pos += labels.sum()
            total_res += n_res

        elapsed = time.time() - t0
        n_pos = sum(
            np.load(f)["labels"].sum()
            for f in label_dir.glob("*_labels.npz")
        )
        print(f"  {protein_id}: {len(frame_files)} frames, "
              f"{n_pos} positive residues, {elapsed:.1f}s")

    pos_rate = total_pos / total_res if total_res > 0 else 0
    print(f"\n  Overall positive rate: {pos_rate:.3f} ({total_pos}/{total_res})")

    # Step 3: Verify sequence split logic (no MMseqs2 needed)
    print("\nStep 3: Verifying split logic...")
    sequences = {}
    for protein_dir in sorted(frames_dir.iterdir()):
        if not (protein_dir / "metadata.json").exists():
            continue
        with open(protein_dir / "metadata.json") as f:
            meta = json.load(f)
        sequences[protein_dir.name] = meta["sequence"]

    print(f"  {len(sequences)} sequences collected")
    print(f"  (MMseqs2 split skipped — requires mmseqs2 binary)")

    # Step 4: Verify outputs
    print("\nStep 4: Verifying outputs...")
    checks = [
        ("Frames exist", len(list(frames_dir.glob("*/*.npz"))) > 0),
        ("Labels exist", len(list(labels_dir.glob("*/*_labels.npz"))) > 0),
        ("Metadata exists", len(list(frames_dir.glob("*/metadata.json"))) == n_proteins),
        ("Label shape matches", True),  # checked below
        ("Positive rate > 0", pos_rate > 0),
    ]

    # Check label shape
    sample_frame = np.load(list(frames_dir.glob("*/*.npz"))[0], allow_pickle=True)
    sample_label = np.load(list(labels_dir.glob("*/*_labels.npz"))[0])
    n_ca = len(sample_frame["ca_coords"])
    n_lab = len(sample_label["labels"])
    checks[3] = ("Label shape matches CA count", n_ca == n_lab)

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = test_full_pipeline()
    sys.exit(0 if ok else 1)
