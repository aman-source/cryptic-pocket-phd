"""Residue index mapping: Boltz sequential ↔ PDB resSeq.

The Problem
-----------
Boltz-1 outputs atom coordinates indexed sequentially 0..N-1 (residue
positions in the input sequence).  Lewis / PocketMiner annotations use
original PDB residue sequence numbers (resSeq), which may have:
  - gaps (missing density)
  - non-sequential numbering (e.g. starts at 23, jumps from 63→65)
  - insertion codes (e.g. 100A, 100B)

Without an explicit per-protein mapping, cryptic-pocket residue filters
are silently wrong.

Mapping convention
------------------
  boltz_0idx : int  — 0-based position in the Boltz sequence (= mdtraj
                      residue index after selecting only protein residues)
  pdb_resSeq : int  — original PDB residue sequence number (no iCode;
                      insertion codes are not tracked here as the Lewis
                      benchmark does not use them)

Storage
-------
Saved to: data/lewis_subset/<uniprot_id>/residue_mapping.json
Format:
    {
      "0": 23,
      "1": 24,
      ...
    }
Keys are stringified ints (JSON requirement); values are PDB resSeq ints.

Usage
-----
>>> mapping = build_residue_mapping("data/1JWP_A.pdb")
>>> mapping[0]   # boltz index 0 → PDB resSeq
23
>>> pdb_resid = boltz_idx_to_pdb_resid("P62593", 5, "data/lewis_subset")
28
"""

from __future__ import annotations

import json
from pathlib import Path


def build_residue_mapping(pdb_path: str) -> dict[int, int]:
    """Build 0-based sequential index → PDB resSeq mapping from a PDB file.

    Uses mdtraj to read backbone CA atoms, which preserves PDB chain/residue
    order and exposes the original resSeq from the PDB ATOM records.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file (apo or holo structure).

    Returns
    -------
    dict[int, int]
        {boltz_0idx: pdb_resSeq, ...}  for every protein residue in order.
    """
    import mdtraj as md

    traj = md.load(str(pdb_path))
    ca_iis = traj.top.select("protein and name CA")
    ca_traj = traj.atom_slice(ca_iis)

    mapping: dict[int, int] = {}
    for i, res in enumerate(ca_traj.top.residues):
        mapping[i] = int(res.resSeq)
    return mapping


def save_residue_mapping(
    uniprot_id: str,
    pdb_path: str,
    data_dir: str,
) -> dict[int, int]:
    """Build and persist residue mapping for one protein.

    Writes: {data_dir}/{uniprot_id}/residue_mapping.json

    Parameters
    ----------
    uniprot_id : str
    pdb_path : str   — path to the apo PDB file for this protein
    data_dir : str   — parent directory for per-protein subdirectories

    Returns
    -------
    dict[int, int]  — the mapping (also written to disk)
    """
    mapping = build_residue_mapping(pdb_path)

    out_dir = Path(data_dir) / uniprot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "residue_mapping.json"

    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

    return mapping


def load_residue_mapping(uniprot_id: str, data_dir: str) -> dict[int, int]:
    """Load a previously saved residue mapping from JSON.

    Parameters
    ----------
    uniprot_id : str
    data_dir : str

    Returns
    -------
    dict[int, int]  — {boltz_0idx: pdb_resSeq}
    """
    mapping_path = Path(data_dir) / uniprot_id / "residue_mapping.json"
    with open(mapping_path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def boltz_idx_to_pdb_resid(
    uniprot_id: str,
    boltz_residue_index: int,
    data_dir: str,
) -> int:
    """Convert a Boltz 0-based residue index to PDB resSeq.

    Loads the saved mapping JSON for this protein.

    Parameters
    ----------
    uniprot_id : str
    boltz_residue_index : int  — 0-based (0..N-1)
    data_dir : str

    Returns
    -------
    int  — PDB resSeq
    """
    mapping = load_residue_mapping(uniprot_id, data_dir)
    if boltz_residue_index not in mapping:
        raise KeyError(
            f"boltz_residue_index={boltz_residue_index} not in mapping for "
            f"{uniprot_id}. Mapping covers 0..{max(mapping)}"
        )
    return mapping[boltz_residue_index]


def pocket_residue_indices(
    pocket_residue_ranges: list[list[int]],
    n_residues: int,
) -> list[int]:
    """Return 0-based Boltz indices for residues in the cryptic pocket region.

    Parameters
    ----------
    pocket_residue_ranges : list of [start, end] pairs (inclusive).
        Uses 1-based SEQUENTIAL POSITION in the protein chain (same convention
        as the Lewis / bioemu-benchmarks annotations and configs/phase0_proteins.yaml).
        Sequential position 1 = Boltz index 0.

        NOTE: These are NOT PDB resSeq numbers.  PDB chains may start at
        arbitrary resSeq values (e.g. chain B of 1K3F starts at resSeq 1001).
        The sequential-position convention is consistent with Boltz-1's
        0-indexed output regardless of PDB numbering offsets.
    n_residues : int
        Total number of residues (length of the score array from score()).
        Used to clip out-of-range positions silently.

    Returns
    -------
    list[int]  — sorted 0-based Boltz indices in the pocket region
    """
    indices: set[int] = set()
    for start, end in pocket_residue_ranges:
        for pos in range(start, end + 1):
            boltz_idx = pos - 1  # 1-based → 0-based
            if 0 <= boltz_idx < n_residues:
                indices.add(boltz_idx)
    return sorted(indices)
