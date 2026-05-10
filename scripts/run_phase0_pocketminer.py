"""Phase 0, Part 2: PocketMiner + fpocket scoring — CPU only.

CUDA_VISIBLE_DEVICES="" is forced at the top of this file before any import,
so TF/XLA cannot see the GPU. Reads PDB files from phase 1, scores with
PocketMiner and fpocket, computes Spearman rho, writes phase0_rho.csv.

Usage
-----
python -u scripts/run_phase0_pocketminer.py --out_dir results/phase0_runpod
"""

from __future__ import annotations

import os
# Force CPU before any import — TF 2.18/XLA bypasses CUDA_VISIBLE_DEVICES at
# runtime but this ensures TF sees no GPU at process startup.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import yaml

from cryptic_pocket_phd.pocketminer_wrapper import score as pm_score
from cryptic_pocket_phd.fpocket_wrapper import score as fpocket_score
from cryptic_pocket_phd.correlation import build_results_table, compute_rho
from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

_REPO_ROOT = Path(__file__).resolve().parent.parent
_YAML_CONFIG = _REPO_ROOT / "configs" / "phase0_proteins.yaml"


def write_results_csv(results_table: dict, out_path: Path) -> None:
    rows = []
    for prot, ts_data in results_table["per_protein"].items():
        for t, stats in ts_data.items():
            rows.append({
                "protein": prot,
                "timestep": t,
                "rho_pocket_mean": stats["mean_pocket"],
                "rho_pocket_sd": stats["sd_pocket"],
                "rho_all_mean": stats["mean_all"],
                "rho_all_sd": stats["sd_all"],
            })
    agg = results_table.get("aggregate", {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    agg_path = out_path.with_name(out_path.stem + "_aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        agg_rows = [{"timestep": t, **v} for t, v in agg.items()]
        if agg_rows:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
    print(f"Results written to {out_path} and {agg_path}")


def run_phase0_pocketminer(out_dir: Path, proteins: list | None = None) -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "phase1_manifest.csv"
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 1 manifest not found: {manifest_path}\n"
            "Run phase 1 first: python scripts/run_phase0_boltz.py"
        )

    with open(_YAML_CONFIG) as f:
        config = yaml.safe_load(f)
    if proteins is None:
        proteins = config.get("proteins", config["experiment"].get("proteins", []))
    protein_map = {p["uniprot"]: p for p in proteins}

    with open(manifest_path) as f:
        manifest_rows = list(csv.DictReader(f))

    all_timesteps = sorted(set(float(r["timestep"]) for r in manifest_rows))

    print("=== Phase 2: PocketMiner + fpocket scoring (CPU) ===")
    print(f"Manifest rows: {len(manifest_rows)}, Timesteps: {all_timesteps}")

    # Pre-warm PocketMiner (loads TF model once)
    print("\nLoading PocketMiner model...")
    from cryptic_pocket_phd.pocketminer_wrapper import get_model
    get_model()
    print("PocketMiner loaded.")

    # Group by protein
    by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_protein[row["uniprot"]].append(row)

    all_results: dict = {}
    all_fpocket_results: dict = {}

    for uid, rows in by_protein.items():
        prot = protein_map.get(uid, {})
        short = prot.get("short_name", uid)
        apo_pdb = rows[0]["apo_pdb"]
        n_residues = int(rows[0]["n_residues"])
        # pocket_ranges stored as Python list repr in CSV
        pocket_ranges = ast.literal_eval(rows[0]["pocket_ranges"])
        pocket_idx = pocket_residue_indices(pocket_ranges, n_residues)

        print(f"\n=== {short} ({uid}) ===")
        print(f"  Scoring reference: {Path(apo_pdb).name}")
        s_0 = pm_score(apo_pdb)
        s_0_fp = fpocket_score(apo_pdb)
        print(f"  s_0: {len(s_0)} residues, {len(pocket_idx)} pocket indices")

        prot_results: dict = {}
        fpocket_prot_results: dict = {}

        for row in rows:
            sample_idx = int(row["sample_idx"])
            t = float(row["timestep"])
            pdb_path = row["pdb_path"]

            if not Path(pdb_path).exists():
                print(f"  WARNING: PDB not found: {pdb_path}")
                continue

            # PocketMiner score
            try:
                s_t = pm_score(pdb_path)
            except Exception as exc:
                print(f"  PM WARNING s{sample_idx:02d} t={t:.1f}: {exc}")
                continue

            if len(s_t) != n_residues:
                print(f"  WARNING: s_t has {len(s_t)} residues, expected {n_residues}. Skipping.")
                continue

            rho_p, rho_all = compute_rho(s_t, s_0, pocket_idx)
            print(f"  s{sample_idx:02d} t={t:.1f}  rho_pocket={rho_p:.3f}  rho_all={rho_all:.3f}")

            if t not in prot_results:
                prot_results[t] = {}
            prot_results[t][sample_idx] = (rho_p, rho_all)

            # fpocket score
            try:
                s_t_fp = fpocket_score(pdb_path)
            except Exception as exc:
                print(f"  fpocket WARNING s{sample_idx:02d} t={t:.1f}: {exc}")
                continue

            if len(s_t_fp) != n_residues:
                continue

            rho_p_fp, rho_all_fp = compute_rho(s_t_fp, s_0_fp, pocket_idx)
            print(f"  [fp] s{sample_idx:02d} t={t:.1f}  rho_pocket={rho_p_fp:.3f}")

            if t not in fpocket_prot_results:
                fpocket_prot_results[t] = {}
            fpocket_prot_results[t][sample_idx] = (rho_p_fp, rho_all_fp)

        all_results[uid] = prot_results
        all_fpocket_results[uid] = fpocket_prot_results

        # Per-protein checkpoint
        partial_table = build_results_table(all_results, all_timesteps)
        write_results_csv(partial_table, results_dir / "phase0_rho_partial.csv")
        if all_fpocket_results:
            fp_partial = build_results_table(all_fpocket_results, all_timesteps)
            write_results_csv(fp_partial, results_dir / "phase0_rho_fpocket_partial.csv")
        print(f"  [checkpoint] {uid} written")

    # Final results
    print("\n=== Building final results table ===")
    results_table = build_results_table(all_results, all_timesteps)
    write_results_csv(results_table, results_dir / "phase0_rho.csv")

    if all_fpocket_results:
        fpocket_table = build_results_table(all_fpocket_results, all_timesteps)
        write_results_csv(fpocket_table, results_dir / "phase0_rho_fpocket.csv")
        print("\n--- fpocket Aggregate rho_pocket (mean [95% CI]) ---")
        for t, stats in sorted(fpocket_table.get("aggregate", {}).items()):
            print(f"  t={t:.1f}: {stats['point']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

    print("\n--- PocketMiner Aggregate rho_pocket (mean [95% CI]) ---")
    for t, stats in sorted(results_table.get("aggregate", {}).items()):
        print(f"  t={t:.1f}: {stats['point']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

    print("\nPhase 2 complete.")
    return results_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 Part 2: PocketMiner scoring")
    parser.add_argument("--out_dir", default="results/phase0_local")
    parser.add_argument("--proteins", nargs="+", default=None,
                        help="UniProt IDs to score (default: all in manifest)")
    args = parser.parse_args()

    selected_proteins = None
    if args.proteins:
        with open(_YAML_CONFIG) as f:
            cfg = yaml.safe_load(f)
        all_prots = cfg["proteins"]
        selected_proteins = [p for p in all_prots if p["uniprot"] in args.proteins]

    run_phase0_pocketminer(
        out_dir=Path(args.out_dir),
        proteins=selected_proteins,
    )
