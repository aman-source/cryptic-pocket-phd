#!/usr/bin/env python3
"""
Download mdCATH H5 files directly from HuggingFace Hub.

Bypasses the datasets streaming API (which has an h5py scalar bug) by
using hf_hub_download per-file. Idempotent: skips already-downloaded files.

Two modes:
  --protein_list FILE  : Download specific domain IDs listed in FILE (one per line).
                         Use output from select_stratified_proteins.py (pod*_gpu*.txt).
  --n N --offset M     : Download N proteins starting at offset M in sorted HF list.

Usage:
    # Specific domain IDs (recommended after select_stratified_proteins.py)
    python scripts/download_mdcath_direct.py \
        --out_dir data/md_raw/mdcath \
        --protein_list data/protein_lists/pod1_gpu0.txt \
        --workers 8

    # Legacy slice-based
    python scripts/download_mdcath_direct.py \
        --out_dir data/md_raw/mdcath \
        --n 60 --offset 0
"""
import sys
import time
from pathlib import Path

import click


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
    from huggingface_hub import hf_hub_download, list_repo_files
    import concurrent.futures

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get full list of H5 files from HuggingFace
    click.echo("Listing mdCATH H5 files from HuggingFace...")
    all_files = sorted([
        f for f in list_repo_files("compsciencelab/mdCATH", repo_type="dataset")
        if f.endswith(".h5")
    ])
    click.echo(f"Total H5 files: {len(all_files)}")

    # Build domain_id → hf_path lookup
    hf_path_map = {}
    for hf_file in all_files:
        domain_id = Path(hf_file).stem.replace("mdcath_dataset_", "")
        hf_path_map[domain_id] = hf_file

    if protein_list:
        # Mode 1: specific domain IDs from stratified selection
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
        # Mode 2: legacy slice
        selected = all_files[offset: offset + n]
        if not selected:
            click.echo(f"ERROR: offset={offset} exceeds available files ({len(all_files)})")
            sys.exit(1)
        click.echo(f"Selected files [{offset}:{offset+n}]: {len(selected)} files")

    def download_one(hf_path: str) -> tuple[str, str]:
        """Download one H5 file. Returns (domain_id, local_path)."""
        domain_id = Path(hf_path).stem.replace("mdcath_dataset_", "")
        dest_dir = out_path / domain_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_h5 = dest_dir / f"{domain_id}.h5"
        done_marker = dest_dir / "done.json"

        if done_marker.exists():
            return domain_id, str(dest_h5)

        local = hf_hub_download(
            repo_id="compsciencelab/mdCATH",
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(dest_dir.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to local_dir/{filename}
        # Move to expected location if needed
        downloaded = Path(local)
        if downloaded != dest_h5:
            downloaded.rename(dest_h5)

        import json
        with open(done_marker, "w") as f:
            json.dump({"domain_id": domain_id, "source": "mdcath"}, f)

        return domain_id, str(dest_h5)

    done = 0
    skipped = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, hf_path): hf_path for hf_path in selected}
        for fut in concurrent.futures.as_completed(futures):
            hf_path = futures[fut]
            try:
                domain_id, local_path = fut.result()
                done += 1
                elapsed = time.time() - t0
                click.echo(f"  [{done}/{len(selected)}] {domain_id} ({elapsed:.0f}s)")
            except Exception as e:
                click.echo(f"  FAILED {hf_path}: {e}")
                skipped += 1

    click.echo(f"\nDone. {done} downloaded, {skipped} failed. Output: {out_path}")


if __name__ == "__main__":
    main()
