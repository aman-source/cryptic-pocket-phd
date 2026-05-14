#!/usr/bin/env python3
"""
Download mdCATH H5 files directly from HuggingFace Hub.

Bypasses the datasets streaming API (which has an h5py scalar bug) by
using hf_hub_download per-file. Idempotent: skips already-downloaded files.

Selects proteins stratified by sequence length using n_residues from H5 filenames
(length is inferred after downloading a small sample if needed).

Usage:
    # Download first 60 proteins from HF list (sorted, take slice)
    python scripts/download_mdcath_direct.py \
        --out_dir data/md_raw/mdcath \
        --n 60 \
        --offset 0     # Pod 1: proteins 0-59

    # Pod 2: proteins 60-119
    python scripts/download_mdcath_direct.py \
        --out_dir data/md_raw/mdcath \
        --n 60 \
        --offset 60
"""
import sys
import time
from pathlib import Path

import click


@click.command()
@click.option("--out_dir", type=click.Path(), required=True,
              help="Output directory for H5 files")
@click.option("--n", type=int, default=60,
              help="Number of proteins to download")
@click.option("--offset", type=int, default=0,
              help="Start index into the sorted HF file list (for pod splitting)")
@click.option("--workers", type=int, default=4,
              help="Parallel download workers")
def main(out_dir, n, offset, workers):
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

    # Select our slice
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
