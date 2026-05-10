"""Convert Boltz-1 intermediate coordinates to PDB format.

Boltz-1 captures x_hat_0 tensors of shape (batch, n_atoms, 3) alongside
atom metadata.  PocketMiner expects a PDB file with backbone atoms
(N, CA, C, O) per residue, in residue order.

This module:
  1. Filters the all-atom Boltz coordinate array to backbone atoms.
  2. Writes a minimal ATOM-record PDB that mdtraj can read.
  3. Numbering: residues are written with sequential PDB resSeq starting at
     the value from the apo PDB (via residue_mapping), or 1..N if no mapping
     is provided.

Atom metadata format (from make_timestep_capture_fn):
    atom_names  : list[str]  — e.g. ["N", "CA", "C", "O", "CB", ...]
    res_indices : list[int]  — 1-based residue index per atom
    chain_ids   : list[str]  — chain ID per atom
    res_names   : list[str]  — three-letter residue name per atom
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Backbone atom names PocketMiner requires (in mdtraj order per residue)
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def boltz_coords_to_pdb(
    coords: np.ndarray,
    atom_names: list[str],
    res_indices: list[int],
    chain_ids: list[str],
    res_names: list[str],
    output_path: str,
    resseq_offset: Optional[dict[int, int]] = None,
) -> str:
    """Write backbone atoms from Boltz coordinates to a PDB file.

    Parameters
    ----------
    coords : np.ndarray, shape (n_atoms, 3)
        Atom coordinates in Angstroms (Boltz outputs in Angstroms).
        If shape is (batch, n_atoms, 3), use coords[0].
    atom_names : list[str]
        Atom name per atom position.
    res_indices : list[int]
        1-based residue index per atom (Boltz sequential numbering).
    chain_ids : list[str]
        Chain ID per atom.
    res_names : list[str]
        Three-letter residue name per atom.
    output_path : str
        Output PDB file path.
    resseq_offset : dict[int, int] or None
        Maps 1-based Boltz res_index → PDB resSeq.
        If None, writes residues with their Boltz res_indices as resSeq.

    Returns
    -------
    str  — path to the written PDB file
    """
    if coords.ndim == 3:
        coords = coords[0]  # take first batch element

    output_path = str(output_path)
    lines = []
    atom_serial = 1

    for i, (aname, res_idx, chain, resname, xyz) in enumerate(
        zip(atom_names, res_indices, chain_ids, res_names, coords)
    ):
        if aname not in BACKBONE_ATOMS:
            continue

        pdb_resseq = resseq_offset[res_idx] if resseq_offset else res_idx

        # PDB ATOM record format (columns 1-80)
        # 1-6: record type, 7-11: serial, 13-16: atom name, 17: altLoc,
        # 18-20: resName, 22: chainID, 23-26: resSeq, 27: iCode,
        # 31-38: x, 39-46: y, 47-54: z, 55-60: occupancy, 61-66: tempFactor,
        # 77-78: element
        record = (
            f"ATOM  "
            f"{atom_serial:5d} "
            f"{_fmt_atom_name(aname)}"
            f" "  # altLoc
            f"{resname:3s} "
            f"{chain:1s}"
            f"{pdb_resseq:4d}"
            f"    "  # iCode + 3 blanks
            f"{xyz[0]:8.3f}"
            f"{xyz[1]:8.3f}"
            f"{xyz[2]:8.3f}"
            f"  1.00"
            f"  0.00"
            f"          "
            f"{_element(aname):>2s}  "
        )
        lines.append(record)
        atom_serial += 1

    lines.append("END")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def _fmt_atom_name(name: str) -> str:
    """Format atom name to PDB column 13-16 (4 chars)."""
    if len(name) == 1:
        return f" {name}  "
    if len(name) == 2:
        return f" {name} "
    if len(name) == 3:
        return f" {name}"
    return name[:4]


def _element(atom_name: str) -> str:
    """Infer element symbol from backbone atom name."""
    first = atom_name[0]
    return first  # N, C, O all work


def npz_to_pdb(
    npz_path: str,
    atom_metadata: dict,
    output_path: str,
    resseq_offset: Optional[dict[int, int]] = None,
) -> str:
    """Convert a captured .npz intermediate to a PDB file.

    Parameters
    ----------
    npz_path : str
        Path to .npz written by make_timestep_capture_fn.
        Must contain 'coords' key, shape (batch, n_atoms, 3).
    atom_metadata : dict
        From {protein_id}_atom_metadata.json.  Must have keys:
        atom_names, res_indices, chain_ids, res_names.
    output_path : str
        Output PDB path.
    resseq_offset : dict or None
        Maps Boltz 1-based res_index → PDB resSeq.

    Returns
    -------
    str — output_path
    """
    data = np.load(npz_path)
    coords = data["coords"]  # (batch, n_atoms, 3)

    return boltz_coords_to_pdb(
        coords=coords[0],  # single structure
        atom_names=atom_metadata["atom_names"],
        res_indices=atom_metadata["res_indices"],
        chain_ids=atom_metadata["chain_ids"],
        res_names=atom_metadata["res_names"],
        output_path=output_path,
        resseq_offset=resseq_offset,
    )
