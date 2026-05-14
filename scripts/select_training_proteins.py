#!/usr/bin/env python3
"""
Select N training proteins from ATLAS for Task A1.

Stratified by sequence length (5 bins covering 50-800 residues).
NOTE: No CATH-family stratification — length bins only.

REQUIRED: --exclude must point to the sequence_identity_split.py output
(excluded_ids.txt). If missing or not provided, script aborts.
Lewis 33 contamination in training set is non-negotiable per Spec §3.1.

Usage:
    # Run sequence_identity_split.py first to generate excluded_ids.txt
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

# Lewis 33 IDs hardcoded as last-resort check (pdb_chain format used in ATLAS).
# sequence_identity_split.py should catch homologs too — this catches direct hits
# even if --exclude file is missing.
LEWIS_33_IDS = {
    "1awr_A", "1byb_A", "1ex6_A", "1k1e_A", "1lci_A", "1m4i_A", "1ot8_A",
    "1pz5_A", "1uyl_A", "1v48_A", "2ayh_A", "2b4g_A", "2brl_A", "2cex_A",
    "2ck3_A", "2dq7_A", "2ewk_A", "2exo_A", "2fi8_A", "2g1o_A", "2gs6_A",
    "2h7l_A", "2hda_A", "2hiw_A", "2hnl_A", "2i87_A", "2jk8_A", "2jo9_A",
    "2jv7_A", "2kzj_A", "2l2i_A", "2lgg_A", "2m1z_A",
}


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True)
@click.option("--exclude", type=click.Path(), required=True,
              help="File with one protein ID per line to exclude (Lewis 33 + homologs). "
                   "Generate with scripts/sequence_identity_split.py. REQUIRED.")
@click.option("--n", type=int, default=100)
@click.option("--out", type=click.Path(), default="data/protein_lists/task_a1_100.txt")
def main(frames_dir, exclude, n, out):
    frames_path = Path(frames_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce exclusion list — hard fail if missing
    exclude_path = Path(exclude)
    if not exclude_path.exists():
        click.echo(
            f"ERROR: --exclude file not found: {exclude_path}\n"
            "Run scripts/sequence_identity_split.py first to generate it.\n"
            "Lewis 33 contamination in training set is non-negotiable (Spec §3.1)."
        )
        sys.exit(1)

    excluded = set(exclude_path.read_text().splitlines())
    excluded = {e.strip() for e in excluded if e.strip()}
    # Always also exclude Lewis 33 hardcoded IDs (belt-and-suspenders)
    excluded |= LEWIS_33_IDS
    click.echo(f"Excluding {len(excluded)} proteins (Lewis 33 + homologs from split file)")

    # Collect all eligible proteins
    candidates = []
    n_excluded_found = 0
    for d in sorted(frames_path.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        if d.name in excluded:
            n_excluded_found += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        seq = meta.get("sequence", "")
        n_res = len(seq)
        if n_res < 50 or n_res > 800:
            continue
        candidates.append({"id": d.name, "n_res": n_res})

    click.echo(f"Proteins excluded from frames_dir: {n_excluded_found}")
    click.echo(f"Eligible candidates: {len(candidates)}")

    if not candidates:
        click.echo("ERROR: no eligible proteins found. Check frames_dir and metadata.")
        sys.exit(1)

    if len(candidates) <= n:
        selected = [c["id"] for c in candidates]
        click.echo(f"Fewer than {n} eligible — using all {len(selected)}")
    else:
        # Stratified by length bins (20 per bin × 5 bins = 100)
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
                click.echo(f"  Bin {bin_labels[i]}: 0 available")
                continue
            chosen_n = min(quota, len(bin_proteins))
            idx = rng.choice(len(bin_proteins), size=chosen_n, replace=False)
            chosen = [bin_proteins[j]["id"] for j in sorted(idx)]
            selected.extend(chosen)
            click.echo(f"  Bin {bin_labels[i]}: {len(bin_proteins)} available, {chosen_n} selected")

        # Fill from overflow if any bins were short
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

    res_counts = []
    sel_set = set(selected)
    for c in candidates:
        if c["id"] in sel_set:
            res_counts.append(c["n_res"])
    if res_counts:
        click.echo(f"Length range: {min(res_counts)}-{max(res_counts)} res, "
                   f"median {int(np.median(res_counts))}")


if __name__ == "__main__":
    main()
