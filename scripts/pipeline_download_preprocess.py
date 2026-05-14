#!/usr/bin/env python3
"""
Streaming pipeline: download H5 → preprocess frames → delete H5, one protein at a time.

Avoids disk overflow by never holding more than `--max_on_disk` H5 files simultaneously.
Idempotent: skips proteins that already have preprocessed frames (metadata.json exists).

Usage:
    # Pod 1, GPU 0 bucket:
    python scripts/pipeline_download_preprocess.py \
        --protein_list data/protein_lists/pod1_gpu0.txt \
        --raw_dir /tmp/md_raw \
        --frames_dir data/md_frames \
        --n_frames 100 \
        --skip_ns 10

    # Pod 1, GPU 1 bucket (run concurrently in another shell):
    python scripts/pipeline_download_preprocess.py \
        --protein_list data/protein_lists/pod1_gpu1.txt \
        --raw_dir /tmp/md_raw \
        --frames_dir data/md_frames \
        --n_frames 100 \
        --skip_ns 10
"""
import json
import shutil
import sys
import time
from pathlib import Path

import click
import requests

HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main"
HF_REPO = "compsciencelab/mdCATH"

# Inline minimal versions to avoid import chain issues
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYM": "C", "HID": "H", "HIE": "H", "HIP": "H",
}


def get_hf_path(domain_id: str) -> str:
    """Construct HuggingFace file path from domain_id."""
    return f"data/mdcath_dataset_{domain_id}.h5"


def download_h5(domain_id: str, raw_dir: Path) -> Path:
    """Download one H5 file directly (no HF cache). Returns dest_h5 path."""
    hf_path = get_hf_path(domain_id)
    dest_dir = raw_dir / domain_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_h5 = dest_dir / f"{domain_id}.h5"

    if dest_h5.exists():
        return dest_h5

    url = f"{HF_BASE_URL}/{hf_path}"
    tmp_path = dest_h5.with_suffix(".tmp")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.rename(dest_h5)
    elapsed = time.time() - t0
    size_mb = dest_h5.stat().st_size / 1024 / 1024
    click.echo(f"  Downloaded {domain_id}.h5 ({size_mb:.0f} MB, {elapsed:.0f}s)")
    return dest_h5


def _parse_ca_indices_from_pdb(pdb_str: str) -> list[int]:
    ca_indices = []
    atom_idx = 0
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            ca_indices.append(atom_idx)
        atom_idx += 1
    return ca_indices


def _parse_sequence_from_pdb(pdb_str: str, ca_indices: list[int]) -> str:
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


def preprocess_mdcath(
    domain_id: str,
    h5_path: Path,
    frames_dir: Path,
    n_frames: int,
    skip_ns: float,
    temperature: int = 320,
) -> dict | None:
    """Extract frames from mdCATH H5 file. Returns metadata dict or None."""
    import h5py
    import numpy as np

    frame_out_dir = frames_dir / domain_id
    done_marker = frame_out_dir / "metadata.json"
    if done_marker.exists():
        with open(done_marker) as f:
            return json.load(f)

    frame_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(str(h5_path), "r") as h5f:
            if domain_id not in h5f:
                click.echo(f"  domain key '{domain_id}' not found in H5")
                return None

            grp = h5f[domain_id]
            temp_key = str(temperature)
            if temp_key not in grp:
                available_temps = sorted([k for k in grp.keys() if k.isdigit()])
                if not available_temps:
                    click.echo(f"  No temperature groups in {domain_id}")
                    return None
                temp_key = available_temps[0]

            rep_key = "0"
            if rep_key not in grp[temp_key]:
                rep_key = sorted(grp[temp_key].keys())[0]

            coords_dataset = grp[temp_key][rep_key]["coords"]
            total_frames = coords_dataset.shape[0]

            pdb_raw = grp["pdb"][()]
            pdb_str = pdb_raw.decode() if isinstance(pdb_raw, bytes) else str(pdb_raw)
            ca_indices = _parse_ca_indices_from_pdb(pdb_str)
            if not ca_indices:
                click.echo(f"  No CA atoms found for {domain_id}")
                return None

            sequence = _parse_sequence_from_pdb(pdb_str, ca_indices)
            skip_frames = min(int(skip_ns), total_frames - 1)
            usable = total_frames - skip_frames
            if usable < n_frames:
                frame_indices = np.arange(skip_frames, total_frames)
            else:
                frame_indices = np.linspace(skip_frames, total_frames - 1, n_frames, dtype=int)

            for idx, frame_idx in enumerate(frame_indices):
                frame_path = frame_out_dir / f"{idx:04d}.npz"
                if frame_path.exists():
                    continue
                coords = np.array(coords_dataset[int(frame_idx)], dtype=np.float32)
                coord_range = float(coords.max() - coords.min())
                if coord_range < 5.0:
                    coords = coords * 10.0
                ca_coords = coords[ca_indices]
                np.savez_compressed(
                    frame_path,
                    coords=coords,
                    ca_coords=ca_coords.astype(np.float32),
                    sequence=sequence,
                    frame_index=int(frame_idx),
                    protein_id=domain_id,
                )

        meta = {
            "protein_id": domain_id,
            "source": "mdcath",
            "total_frames": total_frames,
            "n_extracted": len(frame_indices),
            "n_residues": len(ca_indices),
            "n_atoms": int(coords_dataset.shape[1]),
            "sequence": sequence,
            "temperature_K": int(temp_key),
        }
        with open(done_marker, "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    except Exception as e:
        click.echo(f"  Error processing {domain_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


@click.command()
@click.option("--protein_list", type=click.Path(exists=True), required=True,
              help="File with one domain_id per line")
@click.option("--raw_dir", type=click.Path(), required=True,
              help="Temp directory for H5 files (deleted after preprocessing)")
@click.option("--frames_dir", type=click.Path(), required=True,
              help="Output directory for preprocessed frames")
@click.option("--n_frames", type=int, default=100)
@click.option("--skip_ns", type=float, default=10.0)
@click.option("--temperature", type=int, default=320)
def main(protein_list, raw_dir, frames_dir, n_frames, skip_ns, temperature):
    raw_path = Path(raw_dir)
    frames_path = Path(frames_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    domain_ids = [l.strip() for l in Path(protein_list).read_text().splitlines() if l.strip()]
    click.echo(f"Pipeline: {len(domain_ids)} proteins from {protein_list}")
    click.echo(f"Raw dir (temp): {raw_path}")
    click.echo(f"Frames dir: {frames_path}")

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, domain_id in enumerate(domain_ids):
        click.echo(f"\n[{i+1}/{len(domain_ids)}] {domain_id}")

        # Check if already preprocessed
        meta_path = frames_path / domain_id / "metadata.json"
        if meta_path.exists():
            click.echo(f"  Already preprocessed, skipping")
            skipped += 1
            continue

        # Step 1: Download H5
        try:
            h5_path = download_h5(domain_id, raw_path)
        except Exception as e:
            click.echo(f"  Download failed: {e}")
            failed += 1
            continue

        # Step 2: Preprocess to frames
        try:
            meta = preprocess_mdcath(
                domain_id, h5_path, frames_path, n_frames, skip_ns, temperature
            )
            if meta:
                click.echo(f"  Preprocessed: {meta['n_extracted']} frames, {meta['n_residues']} residues")
                done += 1
            else:
                click.echo(f"  Preprocessing returned None")
                failed += 1
        except Exception as e:
            click.echo(f"  Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Step 3: Delete H5 to free disk
        try:
            shutil.rmtree(raw_path / domain_id)
            click.echo(f"  Deleted H5 (freed ~{h5_path.stat().st_size // 1024 // 1024 if h5_path.exists() else '?'} MB)")
        except Exception as e:
            click.echo(f"  Warning: could not delete H5: {e}")

        elapsed = time.time() - t0
        rate = (done + skipped + failed) / elapsed * 60
        remaining_min = (len(domain_ids) - i - 1) / rate if rate > 0 else 0
        click.echo(f"  Progress: {done} done, {skipped} skipped, {failed} failed | "
                   f"ETA ~{remaining_min:.0f} min")

    click.echo(f"\n=== Done ===")
    click.echo(f"Processed: {done}, Skipped: {skipped}, Failed: {failed}")
    click.echo(f"Frames saved to: {frames_path}")
