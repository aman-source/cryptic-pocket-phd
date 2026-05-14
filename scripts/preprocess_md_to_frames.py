#!/usr/bin/env python3
"""
Task A0: Extract frames from MD trajectories.

For each protein trajectory:
  - Skip first 10 ns (equilibration)
  - Sample 100 frames uniformly
  - Save each frame as compressed npz: data/md_frames/{protein_id}/{frame_idx}.npz
  - Include: coords (N_atoms, 3), ca_coords (N_res, 3), sequence, residue_indices

Supports:
  - ATLAS: .xtc + .pdb topology (mdtraj)
  - mdCATH: .h5 files (h5py)

Usage:
  python scripts/preprocess_md_to_frames.py \
    --source atlas \
    --raw_dir data/md_raw/atlas \
    --out_dir data/md_frames/atlas \
    --n_frames 100 \
    --skip_ns 10
"""
import click
import json
import os
from pathlib import Path

import numpy as np

# Amino acid 3-letter to 1-letter
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYM": "C", "HID": "H", "HIE": "H", "HIP": "H",
}


def process_atlas_protein(
    protein_dir: Path,
    out_dir: Path,
    n_frames: int,
    skip_ns: float,
    dt_ps: float = 100.0,  # ATLAS saves every 100 ps
) -> dict | None:
    """Extract frames from ATLAS protein trajectory."""
    import mdtraj as md

    # Find topology (.pdb) and trajectory (.xtc) files
    pdb_files = list(protein_dir.glob("*.pdb"))
    xtc_files = list(protein_dir.glob("*.xtc"))

    if not pdb_files:
        print(f"  No PDB file in {protein_dir}")
        return None
    if not xtc_files:
        print(f"  No XTC file in {protein_dir}")
        return None

    pdb_path = pdb_files[0]
    xtc_path = xtc_files[0]

    # Load trajectory
    try:
        traj = md.load(str(xtc_path), top=str(pdb_path))
    except Exception as e:
        print(f"  Failed to load trajectory: {e}")
        return None

    total_frames = traj.n_frames
    # mdtraj time is in ps; skip first skip_ns * 1000 ps
    skip_frames = int((skip_ns * 1000) / dt_ps) if dt_ps > 0 else 0
    skip_frames = min(skip_frames, total_frames - 1)

    usable_frames = total_frames - skip_frames
    if usable_frames < n_frames:
        print(f"  Only {usable_frames} frames after skip, using all")
        frame_indices = np.arange(skip_frames, total_frames)
    else:
        frame_indices = np.linspace(skip_frames, total_frames - 1, n_frames, dtype=int)

    # Get CA atom indices and sequence
    topology = traj.topology
    ca_indices = topology.select("name CA")
    residues = list(topology.residues)
    sequence = "".join(AA3TO1.get(r.name, "X") for r in residues)

    protein_id = protein_dir.name
    frame_out_dir = out_dir / protein_id
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        frame_path = frame_out_dir / f"{idx:04d}.npz"
        if frame_path.exists():
            continue

        frame = traj[frame_idx]
        # mdtraj coords are in nm — convert to Angstroms
        all_coords = frame.xyz[0] * 10.0  # (N_atoms, 3)
        ca_coords = all_coords[ca_indices]  # (N_res, 3)

        np.savez_compressed(
            frame_path,
            coords=all_coords.astype(np.float32),
            ca_coords=ca_coords.astype(np.float32),
            sequence=sequence,
            frame_index=frame_idx,
            protein_id=protein_id,
        )

    # Save metadata
    meta = {
        "protein_id": protein_id,
        "source": "atlas",
        "total_frames": total_frames,
        "skip_frames": skip_frames,
        "n_extracted": len(frame_indices),
        "n_residues": len(ca_indices),
        "n_atoms": traj.n_atoms,
        "sequence": sequence,
    }
    with open(frame_out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def _parse_ca_indices_from_pdb(pdb_str: str) -> list[int]:
    """
    Parse CA (alpha-carbon) atom indices from a PDB-format string.

    mdCATH stores a PDB string in h5[domain_id]['pdb']. Atom serial order
    in the PDB corresponds 1:1 to axis-1 of the coords array.

    Returns 0-based atom indices where atom name (columns 13-16, stripped) == "CA".
    """
    ca_indices = []
    atom_idx = 0  # 0-based index into coords axis-1
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            ca_indices.append(atom_idx)
        atom_idx += 1
    return ca_indices


def _parse_sequence_from_pdb(pdb_str: str, ca_indices: list[int]) -> str:
    """Extract 1-letter sequence from PDB string at CA atom positions."""
    atom_records = [
        line for line in pdb_str.splitlines()
        if line.startswith("ATOM") or line.startswith("HETATM")
    ]
    sequence = ""
    for idx in ca_indices:
        if idx < len(atom_records):
            resname = atom_records[idx][17:20].strip()
            sequence += AA3TO1.get(resname, "X")
    return sequence


def process_mdcath_protein(
    protein_dir: Path,
    out_dir: Path,
    n_frames: int,
    skip_ns: float,
    temperature: int = 320,  # use lowest temperature (closest to native)
) -> dict | None:
    """
    Extract frames from mdCATH HDF5 file.

    Real mdCATH H5 structure (layout mdcath-only-protein-v1.0):
        {domain_id}/
            pdb: () bytes     — PDB string (atom serial order matches coords axis-1)
            resname: (n_atoms,) — per-atom residue names
            resid: (n_atoms,)   — per-atom residue IDs
            {temperature}/
                {replicate}/
                    coords: (n_frames, n_atoms, 3) float32  — in Angstroms
        Temperatures: 320, 348, 379, 413, 450 K
        Replicates: 0–4 (5 per temperature)
        Frames per replicate: ~440

    We use temperature=320K, replicate=0 (lowest T, most native-like).
    """
    import h5py

    h5_files = list(protein_dir.glob("*.h5"))
    if not h5_files:
        print(f"  No H5 file in {protein_dir}")
        return None

    h5_path = h5_files[0]
    domain_id = protein_dir.name
    frame_out_dir = out_dir / domain_id
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(h5_path, "r") as h5f:
            if domain_id not in h5f:
                print(f"  domain key '{domain_id}' not found in {h5_path}")
                return None

            grp = h5f[domain_id]
            temp_key = str(temperature)
            if temp_key not in grp:
                available_temps = [k for k in grp.keys() if k.isdigit()]
                if not available_temps:
                    print(f"  No temperature groups in {domain_id}")
                    return None
                temp_key = sorted(available_temps)[0]
                print(f"  Temperature {temperature}K not found, using {temp_key}K")

            # Use replicate 0
            rep_key = "0"
            if rep_key not in grp[temp_key]:
                rep_key = sorted(grp[temp_key].keys())[0]

            coords_dataset = grp[temp_key][rep_key]["coords"]  # (n_frames, n_atoms, 3)
            total_frames = coords_dataset.shape[0]

            # Parse CA indices from PDB string
            pdb_raw = grp["pdb"][()]
            pdb_str = pdb_raw.decode() if isinstance(pdb_raw, bytes) else str(pdb_raw)
            ca_indices = _parse_ca_indices_from_pdb(pdb_str)
            if not ca_indices:
                print(f"  No CA atoms found in PDB for {domain_id}")
                return None

            sequence = _parse_sequence_from_pdb(pdb_str, ca_indices)

            # mdCATH: ~1 frame/ns, skip first skip_ns frames
            skip_frames = int(skip_ns)
            skip_frames = min(skip_frames, total_frames - 1)
            usable = total_frames - skip_frames
            if usable < n_frames:
                frame_indices = np.arange(skip_frames, total_frames)
            else:
                frame_indices = np.linspace(skip_frames, total_frames - 1, n_frames, dtype=int)

            for idx, frame_idx in enumerate(frame_indices):
                frame_path = frame_out_dir / f"{idx:04d}.npz"
                if frame_path.exists():
                    continue

                coords = np.array(coords_dataset[int(frame_idx)], dtype=np.float32)  # (n_atoms, 3)
                # Sanity check: coords should already be in Å (range ~1-200 Å)
                coord_range = float(coords.max() - coords.min())
                if coord_range < 5.0:
                    coords = coords * 10.0  # nm → Å fallback

                ca_coords = coords[ca_indices]  # (n_res, 3)

                np.savez_compressed(
                    frame_path,
                    coords=coords,
                    ca_coords=ca_coords.astype(np.float32),
                    sequence=sequence,
                    frame_index=int(frame_idx),
                    protein_id=domain_id,
                )

    except Exception as e:
        print(f"  Error processing {h5_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    n_residues = len(ca_indices)
    meta = {
        "protein_id": domain_id,
        "source": "mdcath",
        "total_frames": total_frames,
        "n_extracted": len(frame_indices),
        "n_residues": n_residues,
        "n_atoms": int(coords_dataset.shape[1]),
        "sequence": sequence,
        "temperature_K": int(temp_key),
    }
    with open(frame_out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta




@click.command()
@click.option("--source", type=click.Choice(["atlas", "mdcath"]), required=True)
@click.option("--raw_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--n_frames", type=int, default=100, help="Frames to sample per trajectory")
@click.option("--skip_ns", type=float, default=10.0, help="Nanoseconds to skip (equilibration)")
def main(source: str, raw_dir: str, out_dir: str, n_frames: int, skip_ns: float):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all protein directories
    protein_dirs = sorted([
        d for d in raw_path.iterdir()
        if d.is_dir() and (d / "done.json").exists()
    ])

    print(f"Found {len(protein_dirs)} downloaded proteins in {raw_path}")

    process_fn = process_atlas_protein if source == "atlas" else process_mdcath_protein

    results = []
    for i, pdir in enumerate(protein_dirs):
        print(f"[{i+1}/{len(protein_dirs)}] {pdir.name}")
        meta = process_fn(pdir, out_path, n_frames, skip_ns)
        if meta:
            results.append(meta)
            print(f"  {meta['n_extracted']} frames, {meta.get('n_residues', '?')} residues")

    print(f"\nDone. Processed {len(results)}/{len(protein_dirs)} proteins")
    print(f"Frames saved to {out_path}")


if __name__ == "__main__":
    main()
