#!/usr/bin/env python3
"""
Task A0: Sequence identity split using MMseqs2.

Cluster all training proteins at 30% identity.
Any cluster containing a Lewis 33 benchmark protein → excluded from training.

Lewis 33 benchmark proteins (cryptic pocket systems):
  From PocketMiner paper + Lewis et al. 2023 benchmark.

Usage:
  python scripts/sequence_identity_split.py \
    --frames_dir data/md_frames/atlas \
    --out_dir data/splits \
    --identity_thresh 0.3

Requires: mmseqs2 installed (conda install -c conda-forge -c bioconda mmseqs2)
"""
import click
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Lewis 33 benchmark proteins — PDB IDs (apo structures)
# From PocketMiner paper Table S1 + Lewis et al. supplementary
LEWIS_BENCHMARK_PDBS = {
    # Phase 1 test set
    "1JWP", "1XCG", "2IYT", "1NEP", "1FVR",
    # Additional Lewis benchmark (apo PDBs)
    "1CLL", "1CMR", "1EX6", "1HW5", "1JBU",
    "1MH1", "1OPD", "1P2Y", "1QRN", "1SWX",
    "1TQO", "1UKZ", "1UTN", "1W8L", "1XMK",
    "1Y6B", "2BRK", "2CEA", "2GFC", "2NLS",
    "2OHG", "2PC5", "2V57", "2VPC", "3DCV",
    "3F74", "3GCL", "3K83",
}


def write_fasta(sequences: dict[str, str], fasta_path: Path):
    """Write sequences to FASTA file."""
    with open(fasta_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")


def run_mmseqs2(
    fasta_path: Path,
    out_dir: Path,
    identity_thresh: float = 0.3,
) -> dict[str, int]:
    """
    Run MMseqs2 clustering and return cluster assignments.

    Returns: dict mapping sequence_id → cluster_id
    """
    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is None:
        raise RuntimeError(
            "mmseqs2 not found. Install: conda install -c conda-forge -c bioconda mmseqs2"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "seqDB"
        clu_path = tmp / "cluDB"
        tsv_path = tmp / "clusters.tsv"

        # Create sequence database
        subprocess.run(
            [mmseqs_bin, "createdb", str(fasta_path), str(db_path)],
            check=True, capture_output=True,
        )

        # Cluster at identity threshold
        subprocess.run(
            [
                mmseqs_bin, "cluster", str(db_path), str(clu_path), str(tmp),
                "--min-seq-id", str(identity_thresh),
                "-c", "0.8",  # coverage threshold
                "--cov-mode", "0",  # bidirectional coverage
            ],
            check=True, capture_output=True,
        )

        # Convert to TSV
        subprocess.run(
            [mmseqs_bin, "createtsv", str(db_path), str(db_path), str(clu_path), str(tsv_path)],
            check=True, capture_output=True,
        )

        # Parse TSV: representative_id \t member_id
        cluster_map = {}  # member → representative
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    cluster_map[member] = rep

    # Assign numeric cluster IDs
    reps = sorted(set(cluster_map.values()))
    rep_to_id = {r: i for i, r in enumerate(reps)}
    assignments = {member: rep_to_id[rep] for member, rep in cluster_map.items()}

    return assignments


def identify_contaminated_clusters(
    assignments: dict[str, int],
    lewis_pdbs: set[str],
) -> set[int]:
    """Find cluster IDs that contain any Lewis benchmark protein."""
    contaminated = set()
    for seq_id, cluster_id in assignments.items():
        # Extract PDB ID from protein_id (e.g., "1jwp_A" → "1JWP")
        pdb_id = seq_id.split("_")[0].upper()
        if pdb_id in lewis_pdbs:
            contaminated.add(cluster_id)
    return contaminated


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True,
              help="Directory with preprocessed frames (contains protein subdirs with metadata.json)")
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--identity_thresh", type=float, default=0.3)
def main(frames_dir: str, out_dir: str, identity_thresh: float):
    frames_path = Path(frames_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect sequences from all proteins
    sequences = {}
    protein_dirs = sorted([
        d for d in frames_path.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    for pdir in protein_dirs:
        with open(pdir / "metadata.json") as f:
            meta = json.load(f)
        seq = meta.get("sequence", "")
        if seq and len(seq) > 10:  # skip very short
            sequences[pdir.name] = seq

    print(f"Collected {len(sequences)} sequences")

    # Write FASTA
    fasta_path = out_path / "all_sequences.fasta"
    write_fasta(sequences, fasta_path)

    # Run MMseqs2
    print(f"Running MMseqs2 at {identity_thresh:.0%} identity threshold...")
    assignments = run_mmseqs2(fasta_path, out_path, identity_thresh)

    n_clusters = len(set(assignments.values()))
    print(f"Found {n_clusters} clusters from {len(assignments)} sequences")

    # Find contaminated clusters (containing Lewis benchmark)
    contaminated = identify_contaminated_clusters(assignments, LEWIS_BENCHMARK_PDBS)
    print(f"Lewis-contaminated clusters: {len(contaminated)}")

    # Split
    train_ids = []
    test_ids = []  # held-out (Lewis-contaminated clusters)
    for seq_id, cluster_id in assignments.items():
        if cluster_id in contaminated:
            test_ids.append(seq_id)
        else:
            train_ids.append(seq_id)

    print(f"Train: {len(train_ids)} proteins")
    print(f"Test (Lewis-adjacent): {len(test_ids)} proteins")

    # Save splits
    split_data = {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "identity_threshold": identity_thresh,
        "n_clusters": n_clusters,
        "n_contaminated_clusters": len(contaminated),
        "lewis_benchmark_pdbs": sorted(LEWIS_BENCHMARK_PDBS),
    }

    with open(out_path / "train_test_split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    # Also save just the ID lists for easy loading
    np.save(out_path / "train_ids.npy", np.array(train_ids))
    np.save(out_path / "test_ids.npy", np.array(test_ids))

    print(f"\nSplit saved to {out_path}")
    print(f"  train_test_split.json")
    print(f"  train_ids.npy ({len(train_ids)} proteins)")
    print(f"  test_ids.npy ({len(test_ids)} proteins)")


if __name__ == "__main__":
    main()
