#!/usr/bin/env python3
"""
Select 100 training proteins from ATLAS for Task A1.

Stratified by sequence length (100-500 residue bins) and broad CATH class
(using first digit of domain ID where available, else length proxy).

Input:
    --frames_dir: data/md_frames/atlas  (metadata.json per protein)
    --exclude: comma-separated protein IDs to exclude (Lewis 33 + homologs)
    --n: number of proteins to select (default 100)
    --out: output file path (default data/protein_lists/task_a1_100.txt)

Usage:
    python scripts/select_training_proteins.py \
        --frames_dir data/md_frames/atlas \
        --exclude data/splits/excluded_ids.txt \
        --n 100 \
        --out data/protein_lists/task_a1_100.txt
"""
import json
import sys
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True)
@click.option("--exclude", type=click.Path(), default=None,
              help="File with one protein ID per line to exclude (Lewis 33 + homologs)")
@click.option("--n", type=int, default=100)
@click.option("--out", type=click.Path(), default="data/protein_lists/task_a1_100.txt")
def main(frames_dir, exclude, n, out):
    frames_path = Path(frames_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load exclusion list
    excluded = set()
    if exclude and Path(exclude).exists():
        excluded = set(Path(exclude).read_text().splitlines())
        click.echo(f"Excluding {len(excluded)} proteins (Lewis 33 + homologs)")

    # Collect all proteins with metadata
    candidates = []
    for d in sorted(frames_path.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        if d.name in excluded:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        seq = meta.get("sequence", "")
        n_res = len(seq)
        if n_res < 50 or n_res > 800:  # skip very short/long
            continue
        candidates.append({"id": d.name, "n_res": n_res, "seq": seq})

    click.echo(f"Eligible proteins: {len(candidates)}")

    if len(candidates) <= n:
        selected = [c["id"] for c in candidates]
        click.echo(f"Fewer than {n} eligible — using all {len(selected)}")
    else:
        # Stratify by length bins: 50-150, 150-250, 250-350, 350-500, 500+
        bins = [(50, 150), (150, 250), (250, 350), (350, 500), (500, 800)]
        bin_labels = ["50-150", "150-250", "250-350", "350-500", "500-800"]
        per_bin = n // len(bins)
        remainder = n % len(bins)

        rng = np.random.default_rng(42)
        selected = []

        for i, (lo, hi) in enumerate(bins):
            bin_proteins = [c for c in candidates if lo <= c["n_res"] < hi]
            quota = per_bin + (1 if i < remainder else 0)
            if not bin_proteins:
                click.echo(f"  Bin {bin_labels[i]}: 0 proteins available")
                continue
            chosen_n = min(quota, len(bin_proteins))
            idx = rng.choice(len(bin_proteins), size=chosen_n, replace=False)
            chosen = [bin_proteins[j]["id"] for j in sorted(idx)]
            selected.extend(chosen)
            click.echo(f"  Bin {bin_labels[i]}: {len(bin_proteins)} available, {chosen_n} selected")

        # Fill remainder if any bins were short
        if len(selected) < n:
            remaining = [c["id"] for c in candidates if c["id"] not in set(selected)]
            extra_n = n - len(selected)
            if remaining:
                extra = list(rng.choice(remaining, size=min(extra_n, len(remaining)), replace=False))
                selected.extend(extra)
                click.echo(f"  Filled {len(extra)} from overflow")

    selected = sorted(set(selected))
    click.echo(f"\nSelected {len(selected)} proteins")

    out_path.write_text("\n".join(selected) + "\n")
    click.echo(f"Written to {out_path}")

    # Summary stats
    res_counts = []
    for c in candidates:
        if c["id"] in set(selected):
            res_counts.append(c["n_res"])
    if res_counts:
        click.echo(f"Length range: {min(res_counts)}-{max(res_counts)} res, "
                   f"median {int(np.median(res_counts))}")


if __name__ == "__main__":
    main()
