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


def process_mdcath_protein(
    protein_dir: Path,
    out_dir: Path,
    n_frames: int,
    skip_ns: float,
    temperature: int = 320,  # use lowest temperature (closest to native)
) -> dict | None:
    """Extract frames from mdCATH HDF5 file."""
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
            # mdCATH structure: temperatures → replicates → coords
            # Try common key patterns
            coords_key = None
            for key_pattern in [
                f"{temperature}K/coords",
                f"coords_{temperature}",
                "coords",
                "positions",
            ]:
                if key_pattern in h5f:
                    coords_key = key_pattern
                    break

            if coords_key is None:
                # List available keys for debugging
                keys = list(h5f.keys())
                print(f"  Available keys: {keys}")
                # Try navigating temperature groups
                for key in keys:
                    if str(temperature) in key or "320" in key:
                        subkeys = list(h5f[key].keys()) if isinstance(h5f[key], h5py.Group) else []
                        print(f"  {key} subkeys: {subkeys}")
                        for sk in subkeys:
                            if "coord" in sk.lower() or "pos" in sk.lower():
                                coords_key = f"{key}/{sk}"
                                break
                    if coords_key:
                        break

            if coords_key is None:
                print(f"  Cannot find coordinates in {h5_path}")
                return None

            all_coords_dataset = h5f[coords_key]
            total_frames = all_coords_dataset.shape[0]

            # Determine timestep (mdCATH typically saves every 1 ns)
            dt_ns = 1.0  # default assumption
            skip_frames_count = int(skip_ns / dt_ns)
            skip_frames_count = min(skip_frames_count, total_frames - 1)

            usable = total_frames - skip_frames_count
            if usable < n_frames:
                frame_indices = np.arange(skip_frames_count, total_frames)
            else:
                frame_indices = np.linspace(skip_frames_count, total_frames - 1, n_frames, dtype=int)

            # Extract CA coords — need topology info
            # mdCATH may store atom names
            sequence = ""
            ca_indices_list = []

            for aname_key in ["atom_names", "atom_name", "atoms"]:
                if aname_key in h5f:
                    atom_names = h5f[aname_key][:]
                    if isinstance(atom_names[0], bytes):
                        atom_names = [a.decode() for a in atom_names]
                    ca_indices_list = [i for i, n in enumerate(atom_names) if n.strip() == "CA"]
                    break

            for seq_key in ["sequence", "seq", "residue_names"]:
                if seq_key in h5f:
                    seq_data = h5f[seq_key]
                    if isinstance(seq_data, h5py.Dataset):
                        raw = seq_data[()]
                        if isinstance(raw, bytes):
                            sequence = raw.decode()
                        elif isinstance(raw, np.ndarray):
                            sequence = "".join(
                                AA3TO1.get(r.decode().strip() if isinstance(r, bytes) else r.strip(), "X")
                                for r in raw
                            )
                        else:
                            sequence = str(raw)
                    break

            for idx, frame_idx in enumerate(frame_indices):
                frame_path = frame_out_dir / f"{idx:04d}.npz"
                if frame_path.exists():
                    continue

                coords = np.array(all_coords_dataset[frame_idx], dtype=np.float32)
                # coords might be in nm — check range
                coord_range = coords.max() - coords.min()
                if coord_range < 50:  # likely nm, convert to Angstroms
                    coords = coords * 10.0

                ca_coords = coords[ca_indices_list] if ca_indices_list else coords

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
        return None

    meta = {
        "protein_id": domain_id,
        "source": "mdcath",
        "total_frames": total_frames,
        "n_extracted": len(frame_indices),
        "sequence": sequence,
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
