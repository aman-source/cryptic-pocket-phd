#!/usr/bin/env python3
"""
Stratified selection of 120 proteins from mdCATH for Task A1 training.

Strategy:
  4 CATH classes × 4 length bins = 16 cells
  ~7-8 proteins per cell = 120 total
  Seed 42, deterministic.

Exclusions:
  - Lewis 33 proteins (by PDB ID match, belt-and-suspenders)
  - Any domain with unknown CATH class (class=0)
  - Domains with n_residues < 50 or > 500 (too short/long for Boltz)
  - If --homologs_file provided: additional excluded IDs from sequence_identity_split.py

Output:
  data/protein_lists/task_a1_stratified_120.txt  (all 120, sorted)
  data/protein_lists/pod1_gpu0.txt               (30 proteins)
  data/protein_lists/pod1_gpu1.txt               (30 proteins)
  data/protein_lists/pod2_gpu0.txt               (30 proteins)
  data/protein_lists/pod2_gpu1.txt               (30 proteins)
  data/protein_lists/selection_report.md         (distribution breakdown)

MMseqs2 dedup note:
  This script outputs 120 candidates. After downloading H5 files and extracting
  sequences, run MMseqs2 easy-linclust (--min-seq-id 0.3) on the 120 and replace
  any duplicate with the next best from the same cell (same class + length bin).

Usage:
    python scripts/select_stratified_proteins.py \
        --metadata data/mdcath_metadata.csv \
        --out_dir data/protein_lists \
        [--homologs_file data/splits/excluded_ids.txt]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np


# Lewis 33 PDB IDs (extracted from LEWIS_33_IDS in select_training_proteins.py)
# Format: lowercase 4-char PDB IDs. Domain IDs starting with these are excluded.
LEWIS_33_PDB_IDS = {
    "1awr", "1byb", "1ex6", "1k1e", "1lci", "1m4i", "1ot8",
    "1pz5", "1uyl", "1v48", "2ayh", "2b4g", "2brl", "2cex",
    "2ck3", "2dq7", "2ewk", "2exo", "2fi8", "2g1o", "2gs6",
    "2h7l", "2hda", "2hiw", "2hnl", "2i87", "2jk8", "2jo9",
    "2jv7", "2kzj", "2l2i", "2lgg", "2m1z",
}

# 4 CATH classes
CATH_CLASSES = {1: "Mainly_Alpha", 2: "Mainly_Beta", 3: "Alpha_Beta", 4: "Few_SS"}

# 4 length bins
LENGTH_BINS = [(50, 150), (150, 250), (250, 350), (350, 500)]
LENGTH_BIN_LABELS = ["50-149", "150-249", "250-349", "350-499"]


def get_length_bin(n_res: int) -> int | None:
    """Returns 0-indexed bin index, or None if out of range."""
    for i, (lo, hi) in enumerate(LENGTH_BINS):
        if lo <= n_res < hi:
            return i
    return None


def interleave_into_buckets(proteins: list[str], n_buckets: int = 4) -> list[list[str]]:
    """
    Interleave proteins into n_buckets.
    proteins should be sorted by (cath_class, length_bin) to preserve balance.
    Interleaving: proteins[0]→bucket0, proteins[1]→bucket1, etc.
    """
    buckets = [[] for _ in range(n_buckets)]
    for i, p in enumerate(proteins):
        buckets[i % n_buckets].append(p)
    return buckets


def write_report(
    out_dir: Path,
    cells: dict[tuple[int, int], list[str]],
    selected_with_meta: list[dict],
    excluded_count: int,
    total_eligible: int,
) -> None:
    """Write selection_report.md."""
    lines = [
        "# Task A1 Stratified Protein Selection Report",
        "",
        "## Summary",
        f"- Total eligible mdCATH domains: {total_eligible}",
        f"- Excluded (Lewis 33 + unknown class + out-of-range): {excluded_count}",
        f"- Final selected: {len(selected_with_meta)}",
        f"- Selection seed: 42",
        "",
        "## CATH Class Distribution",
        "",
        "| CATH Class | Name | Count |",
        "|---|---|---|",
    ]

    from collections import Counter
    class_counts = Counter(m["cath_class"] for m in selected_with_meta)
    for cls, name in CATH_CLASSES.items():
        lines.append(f"| {cls} | {name} | {class_counts.get(cls, 0)} |")

    lines += [
        "",
        "## Length Distribution",
        "",
        "| Bin | Range | Count |",
        "|---|---|---|",
    ]
    bin_counts = Counter(m["length_bin"] for m in selected_with_meta)
    for i, label in enumerate(LENGTH_BIN_LABELS):
        lines.append(f"| {i} | {label} res | {bin_counts.get(i, 0)} |")

    lines += [
        "",
        "## Cell Breakdown (CATH Class × Length Bin)",
        "",
        "| CATH Class | 50-149 | 150-249 | 250-349 | 350-499 | Total |",
        "|---|---|---|---|---|---|",
    ]
    for cls in sorted(CATH_CLASSES.keys()):
        row = [CATH_CLASSES[cls]]
        row_total = 0
        for bin_idx in range(4):
            n = len(cells.get((cls, bin_idx), []))
            row.append(str(n))
            row_total += n
        row.append(str(row_total))
        lines.append("| " + " | ".join(row) + " |")

    n_res_vals = [m["n_residues"] for m in selected_with_meta if m["n_residues"] > 0]
    if n_res_vals:
        lines += [
            "",
            "## Length Statistics",
            f"- Min: {min(n_res_vals)} residues",
            f"- Max: {max(n_res_vals)} residues",
            f"- Median: {int(np.median(n_res_vals))} residues",
        ]

    lines += [
        "",
        "## GPU Split",
        "- pod1_gpu0.txt: proteins 0,4,8,... (interleaved)",
        "- pod1_gpu1.txt: proteins 1,5,9,... (interleaved)",
        "- pod2_gpu0.txt: proteins 2,6,10,... (interleaved)",
        "- pod2_gpu1.txt: proteins 3,7,11,... (interleaved)",
        "",
        "## MMseqs2 Dedup Note",
        "This list is pre-dedup. After H5 download + sequence extraction, run:",
        "```",
        "mmseqs easy-linclust sequences.fasta clust_res /tmp/mmseqs_tmp --min-seq-id 0.3 -c 0.8",
        "```",
        "Drop any duplicate (non-representative) and replace from same cell.",
    ]

    report_path = out_dir / "selection_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    click.echo(f"Report: {report_path}")


@click.command()
@click.option("--metadata", type=click.Path(exists=True), required=True,
              help="data/mdcath_metadata.csv from get_cath_labels.py")
@click.option("--out_dir", type=click.Path(), default="data/protein_lists")
@click.option("--n", type=int, default=120, help="Total proteins to select")
@click.option("--homologs_file", type=click.Path(), default=None,
              help="Optional: excluded_ids.txt from sequence_identity_split.py")
@click.option("--seed", type=int, default=42)
def main(metadata: str, out_dir: str, n: int, homologs_file: str | None, seed: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load extra exclusions
    extra_excluded: set[str] = set()
    if homologs_file:
        hf_path = Path(homologs_file)
        if not hf_path.exists():
            click.echo(f"WARNING: --homologs_file {hf_path} not found. Using Lewis 33 only.")
        else:
            extra_excluded = {l.strip() for l in hf_path.read_text().splitlines() if l.strip()}
            click.echo(f"Loaded {len(extra_excluded)} IDs from homologs file")

    # Load metadata
    domains = []
    with open(metadata, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domains.append({
                "domain_id": row["domain_id"],
                "cath_class": int(row["cath_class"]),
                "n_residues": int(row["n_residues"]),
                "pdb_id": row.get("pdb_id", row["domain_id"][:4].lower()),
            })

    click.echo(f"Loaded {len(domains)} domains from {metadata}")

    # Filter
    n_excluded = 0
    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for d in domains:
        domain_id = d["domain_id"]
        pdb_id = d["pdb_id"].lower()

        # Exclude unknown CATH class
        if d["cath_class"] == 0 or d["cath_class"] not in CATH_CLASSES:
            n_excluded += 1
            continue

        # Exclude Lewis 33
        if pdb_id in LEWIS_33_PDB_IDS:
            n_excluded += 1
            continue

        # Exclude from homologs file
        if domain_id in extra_excluded:
            n_excluded += 1
            continue

        # Length bin
        bin_idx = get_length_bin(d["n_residues"])
        if bin_idx is None:
            n_excluded += 1
            continue

        cells[(d["cath_class"], bin_idx)].append(d)

    total_eligible = len(domains) - n_excluded
    click.echo(f"Eligible after filtering: {total_eligible} ({n_excluded} excluded)")

    # Check cell coverage
    click.echo("\nCell availability (CATH class × length bin):")
    for cls in sorted(CATH_CLASSES.keys()):
        row_parts = []
        for bin_idx in range(4):
            count = len(cells.get((cls, bin_idx), []))
            row_parts.append(f"{LENGTH_BIN_LABELS[bin_idx]}:{count}")
        click.echo(f"  Class {cls} ({CATH_CLASSES[cls]}): {', '.join(row_parts)}")

    # Check all 16 cells have at least some proteins
    empty_cells = [(cls, bin_idx) for cls in CATH_CLASSES for bin_idx in range(4)
                   if not cells.get((cls, bin_idx))]
    if empty_cells:
        click.echo(f"\nWARNING: {len(empty_cells)} empty cells:")
        for cls, bin_idx in empty_cells:
            click.echo(f"  Class {cls}, bin {LENGTH_BIN_LABELS[bin_idx]}")
        if len(empty_cells) > 4:
            click.echo("ERROR: Too many empty cells. CATH labels may be incomplete.")
            click.echo("Run get_cath_labels.py first and verify coverage.")
            sys.exit(1)

    # Stratified sampling
    n_cells = len(CATH_CLASSES) * len(LENGTH_BINS)  # 16
    per_cell = n // n_cells  # 7
    remainder = n % n_cells   # 8 cells get 8, rest get 7

    rng = np.random.default_rng(seed)
    selected_cells: dict[tuple[int, int], list[str]] = {}
    selected_with_meta: list[dict] = []

    cell_order = [(cls, bin_idx)
                  for cls in sorted(CATH_CLASSES.keys())
                  for bin_idx in range(4)]

    for i, (cls, bin_idx) in enumerate(cell_order):
        quota = per_cell + (1 if i < remainder else 0)
        pool = cells.get((cls, bin_idx), [])

        if not pool:
            click.echo(f"  Cell ({cls},{bin_idx}): EMPTY — skipping")
            selected_cells[(cls, bin_idx)] = []
            continue

        n_pick = min(quota, len(pool))
        if n_pick < quota:
            click.echo(f"  Cell ({cls},{bin_idx}): only {len(pool)} available, taking all (wanted {quota})")

        idx = rng.choice(len(pool), size=n_pick, replace=False)
        chosen = [pool[j] for j in sorted(idx)]
        selected_cells[(cls, bin_idx)] = [c["domain_id"] for c in chosen]
        for c in chosen:
            selected_with_meta.append({
                **c,
                "length_bin": bin_idx,
            })
        click.echo(f"  Class {cls} ({CATH_CLASSES[cls]}), bin {LENGTH_BIN_LABELS[bin_idx]}: {n_pick} selected")

    # Flatten in cell order (for balanced interleaving)
    all_selected_ordered: list[str] = []
    for cls, bin_idx in cell_order:
        all_selected_ordered.extend(selected_cells.get((cls, bin_idx), []))

    click.echo(f"\nTotal selected: {len(all_selected_ordered)}")

    # Write main list (sorted alphabetically)
    main_list_path = out_path / "task_a1_stratified_120.txt"
    main_list_path.write_text("\n".join(sorted(all_selected_ordered)) + "\n")
    click.echo(f"Main list: {main_list_path}")

    # Interleave into 4 GPU buckets
    buckets = interleave_into_buckets(all_selected_ordered, n_buckets=4)
    bucket_names = ["pod1_gpu0", "pod1_gpu1", "pod2_gpu0", "pod2_gpu1"]
    for bucket_name, bucket in zip(bucket_names, buckets):
        bucket_path = out_path / f"{bucket_name}.txt"
        bucket_path.write_text("\n".join(bucket) + "\n")
        click.echo(f"  {bucket_name}.txt: {len(bucket)} proteins")

    # Write report
    write_report(out_path, selected_cells, selected_with_meta, n_excluded, total_eligible)

    click.echo("\nDone. Next steps:")
    click.echo("  1. Commit data/protein_lists/ to git")
    click.echo("  2. Download H5 files per bucket:")
    click.echo("     python scripts/download_mdcath_direct.py --protein_list data/protein_lists/pod1_gpu0.txt ...")
    click.echo("  3. After H5 download: extract sequences + run MMseqs2 dedup")
    click.echo("  4. Preprocess frames: python scripts/preprocess_md_to_frames.py ...")
    click.echo("  5. LIGSITE labels: python scripts/compute_ligsite_labels.py ...")
    click.echo("  6. Launch Task A1: python scripts/generate_boltz_intermediates.py --num_gpus 2 ...")


if __name__ == "__main__":
    main()
