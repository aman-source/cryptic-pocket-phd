#!/usr/bin/env python3
"""Generate per-protein metadata for backbone extraction from Boltz all-atom coords.

For each of 120 stratified proteins:
  1. Download mdCATH H5 (stream, extract PDB, delete)
  2. Parse sequence from PDB field
  3. Compute CCD atom offsets using Boltz ref_atoms constants
  4. Save metadata JSON with backbone_indices

Usage (on pod):
    python scripts/generate_protein_metadata.py \
        --protein_list data/protein_lists/task_a1_stratified_120.txt \
        --out_dir results/task_a1/metadata \
        --tmp_dir /tmp/md_raw

Upload to HF after:
    python scripts/hf_upload_metadata.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import click
import h5py
import numpy as np
import requests

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "conformix" / "conformix_boltz" / "src"))

from boltz.data.const import ref_atoms, prot_letter_to_token

HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYM": "C", "HID": "H", "HIE": "H", "HIP": "H",
}


def download_h5(domain_id: str, tmp_dir: Path) -> Path:
    """Download one mdCATH H5 file. Returns path."""
    hf_path = f"data/mdcath_dataset_{domain_id}.h5"
    dest = tmp_dir / f"{domain_id}.h5"
    if dest.exists():
        return dest

    url = f"{HF_BASE_URL}/{hf_path}"
    tmp_path = dest.with_suffix(".tmp")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.rename(dest)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1024 / 1024
    click.echo(f"  Downloaded {domain_id}.h5 ({size_mb:.0f} MB, {elapsed:.0f}s)")
    return dest


def extract_sequence(h5_path: Path, domain_id: str) -> str:
    """Extract amino acid sequence from mdCATH H5 PDB field."""
    with h5py.File(str(h5_path), "r") as f:
        pdb_raw = f[domain_id]["pdb"][()]
        pdb_str = pdb_raw.decode() if isinstance(pdb_raw, bytes) else str(pdb_raw)

    # Parse CA indices and sequence from PDB text
    ca_indices = []
    atom_idx = 0
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            ca_indices.append(atom_idx)
        atom_idx += 1

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


def compute_metadata(domain_id: str, sequence: str) -> dict:
    """Compute CCD atom offsets and backbone indices from sequence."""
    atom_offsets = []
    atom_counts = []
    backbone_indices = []
    pos = 0

    for aa in sequence:
        token = prot_letter_to_token.get(aa, "UNK")
        n_atoms = len(ref_atoms.get(token, ref_atoms["UNK"]))

        atom_offsets.append(pos)
        atom_counts.append(n_atoms)
        backbone_indices.append([pos, pos + 1, pos + 2, pos + 3])  # N, CA, C, O
        pos += n_atoms

    n_atoms_real = pos
    n_atoms_padded = math.ceil(n_atoms_real / 32) * 32

    return {
        "domain_id": domain_id,
        "sequence": sequence,
        "n_residues": len(sequence),
        "n_atoms_real": n_atoms_real,
        "n_atoms_padded": n_atoms_padded,
        "atom_offsets": atom_offsets,
        "atom_counts": atom_counts,
        "backbone_indices": backbone_indices,
    }


def sanity_check(meta: dict) -> bool:
    """Verify metadata consistency."""
    ok = True
    # Sum of atom_counts == n_atoms_real
    if sum(meta["atom_counts"]) != meta["n_atoms_real"]:
        click.echo(f"  FAIL: sum(atom_counts)={sum(meta['atom_counts'])} != n_atoms_real={meta['n_atoms_real']}")
        ok = False
    # Padding formula
    expected_padded = math.ceil(meta["n_atoms_real"] / 32) * 32
    if meta["n_atoms_padded"] != expected_padded:
        click.echo(f"  FAIL: n_atoms_padded={meta['n_atoms_padded']} != ceil/32={expected_padded}")
        ok = False
    # Backbone indices valid
    for i, bb in enumerate(meta["backbone_indices"]):
        if bb[0] != meta["atom_offsets"][i]:
            click.echo(f"  FAIL: backbone[{i}][0]={bb[0]} != offset={meta['atom_offsets'][i]}")
            ok = False
            break
    return ok


@click.command()
@click.option("--protein_list", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--tmp_dir", type=click.Path(), default="/tmp/md_raw")
@click.option("--keep_h5", is_flag=True, default=False, help="Don't delete H5 after extraction")
def main(protein_list, out_dir, tmp_dir, keep_h5):
    out_dir = Path(out_dir)
    tmp_dir = Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(protein_list) as f:
        proteins = [line.strip() for line in f if line.strip()]

    click.echo(f"Generating metadata for {len(proteins)} proteins")
    click.echo(f"  Output: {out_dir}")
    click.echo(f"  Temp: {tmp_dir}")

    success = 0
    failed = []

    for i, domain_id in enumerate(proteins):
        out_path = out_dir / f"{domain_id}.json"
        if out_path.exists():
            click.echo(f"[{i+1}/{len(proteins)}] {domain_id} — already done")
            success += 1
            continue

        click.echo(f"[{i+1}/{len(proteins)}] {domain_id}")

        try:
            # Download H5
            h5_path = download_h5(domain_id, tmp_dir)

            # Extract sequence
            sequence = extract_sequence(h5_path, domain_id)
            if not sequence:
                click.echo(f"  SKIP: empty sequence")
                failed.append(domain_id)
                continue

            # Compute metadata
            meta = compute_metadata(domain_id, sequence)

            # Sanity check
            if not sanity_check(meta):
                click.echo(f"  FAIL: sanity check")
                failed.append(domain_id)
                continue

            # Save
            with open(out_path, "w") as f:
                json.dump(meta, f, indent=2)
            click.echo(f"  OK: {len(sequence)} residues, {meta['n_atoms_real']} atoms (padded {meta['n_atoms_padded']})")
            success += 1

        except Exception as e:
            click.echo(f"  ERROR: {e}")
            failed.append(domain_id)

        finally:
            # Delete H5 to save disk
            if not keep_h5:
                h5_path_check = tmp_dir / f"{domain_id}.h5"
                if h5_path_check.exists():
                    h5_path_check.unlink()

    click.echo(f"\nDone: {success}/{len(proteins)} succeeded, {len(failed)} failed")
    if failed:
        click.echo(f"Failed: {failed}")


if __name__ == "__main__":
    main()
