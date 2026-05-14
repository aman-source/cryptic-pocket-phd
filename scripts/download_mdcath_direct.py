#!/usr/bin/env python3
"""
Download mdCATH H5 files directly from HuggingFace Hub.

Uses direct HTTP download (requests) instead of hf_hub_download to avoid
the HuggingFace cache doubling disk usage. Single copy to destination only.

Idempotent: skips already-downloaded files (done.json marker).

Two modes:
  --protein_list FILE  : Download specific domain IDs (one per line).
  --n N --offset M     : Download N proteins at offset M from sorted HF list.

Usage:
    # Specific domain IDs (from select_stratified_proteins.py)
    python scripts/download_mdcath_direct.py \
        --out_dir /tmp/md_raw \
        --protein_list data/protein_lists/pod1_gpu0.txt \
        --workers 8

    # Legacy slice
    python scripts/download_mdcath_direct.py \
        --out_dir /tmp/md_raw \
        --n 60 --offset 0
"""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import requests

HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main"
HF_REPO = "compsciencelab/mdCATH"


def list_hf_h5_files() -> list[str]:
    """List all H5 files in the HuggingFace repo."""
    from huggingface_hub import list_repo_files
    return sorted([
        f for f in list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith(".h5")
    ])


def download_file_direct(url: str, dest_path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Download file via HTTP streaming directly to dest_path. No cache."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp_path.rename(dest_path)


def download_one(hf_path: str, out_path: Path) -> tuple[str, str]:
    """Download one H5 file. Returns (domain_id, local_path)."""
    domain_id = Path(hf_path).stem.replace("mdcath_dataset_", "")
    dest_dir = out_path / domain_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_h5 = dest_dir / f"{domain_id}.h5"
    done_marker = dest_dir / "done.json"

    if done_marker.exists():
        return domain_id, str(dest_h5)

    url = f"{HF_BASE_URL}/{hf_path}"
    download_file_direct(url, dest_h5)

    with open(done_marker, "w") as f:
        json.dump({"domain_id": domain_id, "source": "mdcath"}, f)

    return domain_id, str(dest_h5)


@click.command()
@click.option("--out_dir", type=click.Path(), required=True,
              help="Output directory for H5 files")
@click.option("--protein_list", type=click.Path(), default=None,
              help="File with one domain_id per line (from select_stratified_proteins.py)")
@click.option("--n", type=int, default=60,
              help="[Legacy] Number of proteins to download from sorted HF list")
@click.option("--offset", type=int, default=0,
              help="[Legacy] Start index into sorted HF list")
@click.option("--workers", type=int, default=8,
              help="Parallel download workers")
def main(out_dir, protein_list, n, offset, workers):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get full list of H5 files from HuggingFace
    click.echo("Listing mdCATH H5 files from HuggingFace...")
    all_files = list_hf_h5_files()
    click.echo(f"Total H5 files: {len(all_files)}")

    # Build domain_id → hf_path lookup
    hf_path_map = {}
    for hf_file in all_files:
        domain_id = Path(hf_file).stem.replace("mdcath_dataset_", "")
        hf_path_map[domain_id] = hf_file

    if protein_list:
        list_path = Path(protein_list)
        if not list_path.exists():
            click.echo(f"ERROR: --protein_list not found: {list_path}")
            sys.exit(1)
        domain_ids = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
        click.echo(f"Requested {len(domain_ids)} proteins from {list_path.name}")

        selected = []
        missing = []
        for domain_id in domain_ids:
            if domain_id in hf_path_map:
                selected.append(hf_path_map[domain_id])
            else:
                missing.append(domain_id)

        if missing:
            click.echo(f"WARNING: {len(missing)} domain IDs not found in HF repo:")
            for m in missing[:10]:
                click.echo(f"  {m}")
        click.echo(f"Found {len(selected)}/{len(domain_ids)} in HF repo")
    else:
        selected = all_files[offset: offset + n]
        if not selected:
            click.echo(f"ERROR: offset={offset} exceeds available files ({len(all_files)})")
            sys.exit(1)
        click.echo(f"Selected files [{offset}:{offset+n}]: {len(selected)} files")

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, hf_path, out_path): hf_path for hf_path in selected}
        for fut in as_completed(futures):
            hf_path = futures[fut]
            try:
                domain_id, local_path = fut.result()
                done += 1
                elapsed = time.time() - t0
                click.echo(f"  [{done}/{len(selected)}] {domain_id} ({elapsed:.0f}s)")
            except Exception as e:
                click.echo(f"  FAILED {hf_path}: {e}")
                failed += 1

    click.echo(f"\nDone. {done} downloaded, {failed} failed. Output: {out_path}")


if __name__ == "__main__":
    main()
