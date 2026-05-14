#!/usr/bin/env python3
"""
Task A0: Download MD trajectory data from ATLAS and mdCATH.

ATLAS API: https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{pdb_chain}
  Returns zip with protein-only trajectory (10,000 frames).

mdCATH: HuggingFace compsciencelab/mdCATH
  HDF5 files per CATH domain. Use datasets library.

Usage:
  # Download 10-protein test subset from ATLAS
  python scripts/download_md_data.py --source atlas --subset 10 --out_dir data/md_raw/atlas

  # Download specific proteins from ATLAS
  python scripts/download_md_data.py --source atlas --proteins 1a2b_A,3xyz_B --out_dir data/md_raw/atlas

  # Download mdCATH subset
  python scripts/download_md_data.py --source mdcath --subset 10 --out_dir data/md_raw/mdcath
"""
import click
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests


# --- ATLAS ---

ATLAS_API_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
ATLAS_PARSABLE_URL = f"{ATLAS_API_BASE}/parsable"


def get_atlas_protein_list() -> list[str]:
    """Fetch list of all ATLAS protein IDs (pdb_chain format)."""
    cache_path = Path("data/md_raw/atlas_protein_list.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print("Fetching ATLAS protein list...")
    resp = requests.get(ATLAS_PARSABLE_URL, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Extract PDB_chain IDs
    protein_ids = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "pdb_chain" in entry:
                protein_ids.append(entry["pdb_chain"])
            elif isinstance(entry, str):
                protein_ids.append(entry)
    elif isinstance(data, dict):
        # Try common keys
        for key in ["data", "proteins", "entries", "results"]:
            if key in data:
                items = data[key]
                for item in items:
                    if isinstance(item, dict) and "pdb_chain" in item:
                        protein_ids.append(item["pdb_chain"])
                    elif isinstance(item, str):
                        protein_ids.append(item)
                break

    if not protein_ids:
        print(f"WARNING: Could not parse protein list. Raw response type: {type(data)}")
        print(f"First 500 chars: {str(data)[:500]}")
        sys.exit(1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(protein_ids, f)

    print(f"Found {len(protein_ids)} ATLAS proteins")
    return protein_ids


def download_atlas_protein(pdb_chain: str, out_dir: Path, retries: int = 3) -> Path:
    """Download protein trajectory zip from ATLAS API and extract."""
    protein_dir = out_dir / pdb_chain
    done_marker = protein_dir / "done.json"

    if done_marker.exists():
        return protein_dir

    protein_dir.mkdir(parents=True, exist_ok=True)
    url = f"{ATLAS_API_BASE}/ATLAS/protein/{pdb_chain}"
    zip_path = protein_dir / f"{pdb_chain}.zip"

    for attempt in range(retries):
        try:
            print(f"  Downloading {pdb_chain} (attempt {attempt + 1})...")
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(protein_dir)

            # Clean up zip
            zip_path.unlink()

            # Write done marker
            with open(done_marker, "w") as f:
                json.dump({"pdb_chain": pdb_chain, "source": "atlas"}, f)

            return protein_dir

        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  SKIPPING {pdb_chain} after {retries} failures")
                return protein_dir

    return protein_dir


# --- mdCATH ---

def download_mdcath_subset(n_proteins: int, out_dir: Path, specific_ids: list[str] | None = None):
    """Download mdCATH proteins via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets  (HuggingFace datasets library)")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mdCATH from HuggingFace (first {n_proteins} proteins)...")
    # mdCATH is streamed — we take first N
    ds = load_dataset(
        "compsciencelab/mdCATH",
        split="train",
        streaming=True,
    )

    count = 0
    for example in ds:
        if specific_ids and example.get("domain_id") not in specific_ids:
            continue

        domain_id = example.get("domain_id", f"protein_{count}")
        protein_dir = out_dir / domain_id
        done_marker = protein_dir / "done.json"

        if done_marker.exists():
            count += 1
            if count >= n_proteins:
                break
            continue

        protein_dir.mkdir(parents=True, exist_ok=True)

        # Save HDF5 or raw data
        # mdCATH provides coords, forces, etc. in the HuggingFace row
        import h5py
        import numpy as np

        h5_path = protein_dir / f"{domain_id}.h5"
        with h5py.File(h5_path, "w") as h5f:
            for key, val in example.items():
                if isinstance(val, (list, np.ndarray)):
                    h5f.create_dataset(key, data=np.array(val))
                elif isinstance(val, (int, float, str)):
                    h5f.attrs[key] = val

        with open(done_marker, "w") as f:
            json.dump({"domain_id": domain_id, "source": "mdcath"}, f)

        count += 1
        print(f"  [{count}/{n_proteins}] {domain_id}")

        if count >= n_proteins:
            break

    print(f"Downloaded {count} mdCATH proteins to {out_dir}")


# --- CLI ---

@click.command()
@click.option("--source", type=click.Choice(["atlas", "mdcath"]), required=True)
@click.option("--subset", type=int, default=10, help="Number of proteins to download")
@click.option("--proteins", type=str, default=None, help="Comma-separated protein IDs")
@click.option("--out_dir", type=click.Path(), required=True)
def main(source: str, subset: int, proteins: str | None, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    specific = proteins.split(",") if proteins else None

    if source == "atlas":
        if specific:
            protein_ids = specific
        else:
            all_ids = get_atlas_protein_list()
            protein_ids = all_ids[:subset]

        print(f"Downloading {len(protein_ids)} ATLAS proteins to {out_path}")
        for i, pid in enumerate(protein_ids):
            print(f"[{i+1}/{len(protein_ids)}] {pid}")
            download_atlas_protein(pid, out_path)

        print(f"\nDone. {len(protein_ids)} proteins in {out_path}")

    elif source == "mdcath":
        download_mdcath_subset(subset, out_path, specific)


if __name__ == "__main__":
    main()
