"""Phase 0: noise sanity check pipeline.

Runs Boltz-1 inference with intermediate x̂_0 capture, scores each captured
state with PocketMiner, computes Spearman ρ between noisy and clean scores.

Copied from boltz.main.predict (commit cb04aec) — see COPIED_FROM below.
Modified: added instrument_model() + scoring + correlation steps.

COPIED_FROM
-----------
boltz/main.py::predict() — sections: process_inputs, data_module, model loading,
trainer setup, trainer.predict call.
Pin: cb04aeccdd480fd4db707f0bbafde538397fa2ac (boltz dependency in pyproject.toml).

MODIFICATIONS
-------------
1. Removed affinity, Boltz2, DDP, multi-GPU paths.
2. Added instrument_model() to inject InstrumentedAtomDiffusion capture hook.
3. Added atom_metadata_from_sequence() (builds from const.ref_atoms, no CCD needed).
4. Added score_captured_states() — boltz_to_pdb + PocketMiner for each .npz.
5. Added compute_correlations() — Spearman via correlation.py.
6. Added write_results_csv().

CACHE
-----
Default: ~/.boltz  (ccd.pkl + boltz1_conf.ckpt)
Override: BOLTZ_CACHE env var.

USAGE
-----
From repo root:
    python scripts/run_phase0.py --out_dir results/phase0_local --samples 1 --timesteps 0.5 0.9
    python scripts/run_phase0.py  # full config: 5 samples, 5 timesteps
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
import types
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Boltz imports — mirrored from boltz.main (commit cb04aec)
# ---------------------------------------------------------------------------
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    BoltzDiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    filter_inputs_structure,
    process_inputs,
)
from boltz.model.models.boltz1 import Boltz1
from pytorch_lightning import Trainer

from boltz.data import const  # for ref_atoms

# ---------------------------------------------------------------------------
# Our modules
# ---------------------------------------------------------------------------
from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb
from cryptic_pocket_phd.correlation import build_results_table, compute_rho
from cryptic_pocket_phd.intermediate_capture import (
    InstrumentedAtomDiffusion,
    make_timestep_capture_fn,
)
from cryptic_pocket_phd.pocketminer_wrapper import score
from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_YAML_CONFIG = _REPO_ROOT / "configs" / "phase0_proteins.yaml"

# One-letter to three-letter AA mapping
_AA1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


# ---------------------------------------------------------------------------
# Injection mechanism
# ---------------------------------------------------------------------------

def instrument_model(model: Boltz1, capture_fn) -> None:
    """Patch model.structure_module.sample to inject intermediate_capture_fn.

    Python instance attribute lookup overrides class method dispatch — the
    bound method we set here will be called by Boltz1.forward() transparently.
    InstrumentedAtomDiffusion.sample() is called with model.structure_module
    as `self`; all attribute access (self.device, self.sample_schedule, etc.)
    resolves correctly because InstrumentedAtomDiffusion inherits AtomDiffusion.
    """
    def _instrumented(self, *args, **kwargs):
        kwargs["intermediate_capture_fn"] = capture_fn
        return InstrumentedAtomDiffusion.sample(self, *args, **kwargs)

    model.structure_module.sample = types.MethodType(
        _instrumented, model.structure_module
    )


def restore_model(model: Boltz1) -> None:
    """Remove instance-level sample patch (restore class-level dispatch)."""
    if "sample" in model.structure_module.__dict__:
        del model.structure_module.__dict__["sample"]


# ---------------------------------------------------------------------------
# Atom metadata from sequence
# ---------------------------------------------------------------------------

def atom_metadata_from_sequence(sequence: str, chain_id: str = "A") -> dict:
    """Build Boltz-compatible atom_metadata from a protein sequence.

    Boltz's atom ordering per residue follows const.ref_atoms[res3].
    For all standard protein AAs, backbone atoms N, CA, C, O are at
    positions 0–3. This order is what appears in x̂_0 tensors.

    Parameters
    ----------
    sequence : str   — one-letter amino acid sequence
    chain_id : str   — PDB chain ID (default 'A')

    Returns
    -------
    dict with keys: atom_names, res_indices, chain_ids, res_names
    """
    atom_names: list[str] = []
    res_indices: list[int] = []
    chain_ids: list[str] = []
    res_names: list[str] = []

    for res_1idx, aa1 in enumerate(sequence, start=1):  # 1-based
        res3 = _AA1TO3.get(aa1, "ALA")
        atoms = const.ref_atoms.get(res3, ["N", "CA", "C", "O"])
        for aname in atoms:
            atom_names.append(aname)
            res_indices.append(res_1idx)
            chain_ids.append(chain_id)
            res_names.append(res3)

    return {
        "atom_names": atom_names,
        "res_indices": res_indices,
        "chain_ids": chain_ids,
        "res_names": res_names,
    }


# ---------------------------------------------------------------------------
# Scoring captured states
# ---------------------------------------------------------------------------

def score_captured_states(
    capture_dir: Path,
    protein_id: str,
    sample_idx: int,
    timesteps: list[float],
    atom_metadata: dict,
    tmp_dir: Path,
) -> dict[float, np.ndarray]:
    """Convert captured .npz → PDB → PocketMiner score.

    Returns
    -------
    dict[float, np.ndarray]
        {timestep: scores_array (n_residues,)}
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[float, np.ndarray] = {}

    for t in timesteps:
        npz_path = capture_dir / f"{protein_id}_s{sample_idx:02d}_t{t:.1f}.npz"
        if not npz_path.exists():
            print(f"  WARNING: capture not found: {npz_path}")
            continue

        data = np.load(str(npz_path))
        coords = data["coords"]  # (batch, n_atoms, 3)

        pdb_path = tmp_dir / f"{protein_id}_s{sample_idx:02d}_t{t:.1f}.pdb"
        boltz_coords_to_pdb(
            coords=coords[0],
            atom_names=atom_metadata["atom_names"],
            res_indices=atom_metadata["res_indices"],
            chain_ids=atom_metadata["chain_ids"],
            res_names=atom_metadata["res_names"],
            output_path=str(pdb_path),
        )

        s_t = score(str(pdb_path))
        scores[t] = s_t
        print(f"  t={t:.1f} scored: shape={s_t.shape} range=[{s_t.min():.3f},{s_t.max():.3f}]")

    return scores


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def write_results_csv(results_table: dict, out_path: Path) -> None:
    """Write per-protein × timestep Spearman ρ summary to CSV."""
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
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    agg_path = out_path.with_name(out_path.stem + "_aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        agg_rows = [
            {"timestep": t, **v} for t, v in agg.items()
        ]
        if agg_rows:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    print(f"Results written to {out_path} and {agg_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase0(
    out_dir: Path,
    proteins: list[dict] | None = None,
    timesteps: list[float] | None = None,
    n_samples: int | None = None,
    cache_dir: Path | None = None,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    no_msa: bool = False,
) -> None:
    """Run phase 0 end-to-end.

    Parameters
    ----------
    out_dir : Path     — all outputs go here
    proteins : list    — from phase0_proteins.yaml; None = load all
    timesteps : list   — normalised noise levels, e.g. [0.5, 0.9]
    n_samples : int    — diffusion samples per protein
    cache_dir : Path   — Boltz cache (~/.boltz by default)
    sampling_steps : int — denoising steps (default 200; use <50 for fast CPU test)
    recycling_steps : int — recycling iterations (default 3; use 1 for fast test)
    no_msa : bool      — if True, raise if MSA file missing (don't use server)
    """
    # Set TF env var before any TF import (PocketMiner needs legacy Keras)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    # Resolve cache
    if cache_dir is None:
        env_cache = os.environ.get("BOLTZ_CACHE")
        cache_dir = Path(env_cache) if env_cache else Path.home() / ".boltz"
    ccd_path = cache_dir / "ccd.pkl"
    checkpoint = cache_dir / "boltz1_conf.ckpt"

    if not ccd_path.exists():
        raise FileNotFoundError(f"CCD not found at {ccd_path}. Run: boltz predict --help")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}.")

    # Load experiment config
    with open(_YAML_CONFIG) as f:
        config = yaml.safe_load(f)

    if proteins is None:
        proteins = config["experiment"].get("proteins", config.get("proteins", []))
        if not proteins:
            proteins = config["proteins"]

    if timesteps is None:
        timesteps = config["experiment"]["timesteps"]
    if n_samples is None:
        n_samples = config["experiment"]["n_samples"]

    out_dir = Path(out_dir)
    capture_dir = out_dir / "captures"
    pdb_dir = out_dir / "intermediate_pdbs"
    results_dir = out_dir / "results"
    boltz_out_dir = out_dir / "boltz_output"

    for d in [capture_dir, pdb_dir, results_dir, boltz_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build input file list for Boltz preprocessing
    # -----------------------------------------------------------------------
    input_dir = _REPO_ROOT / "data" / "boltz_inputs"
    input_files: list[Path] = []
    protein_map: dict[str, dict] = {}  # uniprot → config row

    for prot in proteins:
        uid = prot["uniprot"]
        protein_map[uid] = prot
        yaml_path = input_dir / uid / f"{uid}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Boltz input YAML not found: {yaml_path}. "
                "Create it with sequence + MSA path (see data/boltz_inputs/P79345/)."
            )
        input_files.append(yaml_path)

    # -----------------------------------------------------------------------
    # Preprocess (Boltz: parse sequences, build feature tensors)
    # Copied from boltz.main.predict — see module docstring COPIED_FROM.
    # -----------------------------------------------------------------------
    print("=== Preprocessing inputs ===")
    process_inputs(
        data=input_files,
        out_dir=boltz_out_dir,
        ccd_path=ccd_path,
        mol_dir=None,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,  # MSA must be in YAML (no server calls)
        preprocessing_threads=1,
    )

    manifest = Manifest.load(boltz_out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=boltz_out_dir,
        override=True,  # always re-run for experiment reproducibility
    )

    processed_dir = boltz_out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=None,
        template_dir=None,
        extra_mols_dir=None,
    )

    # -----------------------------------------------------------------------
    # Data module
    # -----------------------------------------------------------------------
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,  # 0 = main process only (avoids Windows multiprocessing issues)
        constraints_dir=processed.constraints_dir,
    )

    # -----------------------------------------------------------------------
    # Load model
    # Copied from boltz.main.predict — see module docstring COPIED_FROM.
    # -----------------------------------------------------------------------
    print("=== Loading Boltz1 model ===")
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638  # Boltz1 default

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": n_samples,
        "max_parallel_samples": n_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False

    model_module = Boltz1.load_from_checkpoint(
        str(checkpoint),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,  # no custom kernels on CPU
        pairformer_args=asdict(PairformerArgs()),
        msa_args=asdict(MSAModuleArgs(subsample_msa=False, num_subsampled_msa=2048,
                                      use_paired_feature=False)),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    # -----------------------------------------------------------------------
    # Per-protein: inject capture, run, score, correlate
    # -----------------------------------------------------------------------
    all_results: dict = {}  # protein_id → {timestep → {sample → (rho_p, rho_all)}}

    for prot in proteins:
        uid = prot["uniprot"]
        short = prot.get("short_name", uid)
        seq = prot.get("sequence")  # optional pre-loaded sequence
        pocket_ranges = prot["pocket_residue_ranges"]
        apo_pdb = _REPO_ROOT / "data" / "validation_pdbs" / f"{prot['apo_pdb']}.pdb"

        print(f"\n=== {short} ({uid}) ===")

        # Build atom_metadata from sequence (for boltz_to_pdb)
        # Sequence may not be in YAML config — extract from reference PDB
        if seq is None:
            import mdtraj as md
            aa3to1 = {v: k for k, v in _AA1TO3.items()}
            traj = md.load(str(apo_pdb))
            ca_sel = traj.top.select("protein and name CA")
            seq = "".join(
                aa3to1.get(r.name, "A")
                for r in traj.atom_slice(ca_sel).top.residues
            )
        atom_metadata = atom_metadata_from_sequence(seq, chain_id="A")

        # Score reference structure s_0
        print(f"  Scoring reference: {apo_pdb.name}")
        s_0 = score(str(apo_pdb))
        n_residues = len(s_0)
        pocket_idx = pocket_residue_indices(pocket_ranges, n_residues)
        print(f"  s_0: {n_residues} residues, {len(pocket_idx)} pocket indices")

        # Run n_samples, each capturing at specified timesteps
        prot_results: dict = {}

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx + 1}/{n_samples}")

            # Build capture callback for this sample
            capture_fn = make_timestep_capture_fn(
                target_ts=timesteps,
                output_dir=str(capture_dir),
                protein_id=uid,
                sample_idx=sample_idx,
                atom_metadata=atom_metadata,
                num_sampling_steps=sampling_steps,
            )

            # Inject InstrumentedAtomDiffusion for this sample
            instrument_model(model_module, capture_fn)

            # Boltz trainer (CPU, no GPU)
            pred_writer = BoltzWriter(
                data_dir=processed.targets_dir,
                output_dir=boltz_out_dir / "predictions",
                output_format="pdb",
                boltz2=False,
                write_embeddings=False,
            )
            trainer = Trainer(
                default_root_dir=str(boltz_out_dir),
                callbacks=[pred_writer],
                accelerator="cpu",
                devices=1,
                precision=32,
                enable_progress_bar=True,
            )

            # Run inference — capture_fn fires during sample()
            trainer.predict(
                model_module,
                datamodule=data_module,
                return_predictions=False,
            )

            restore_model(model_module)

            # Score captured intermediates → s_t
            s_t_map = score_captured_states(
                capture_dir=capture_dir,
                protein_id=uid,
                sample_idx=sample_idx,
                timesteps=timesteps,
                atom_metadata=atom_metadata,
                tmp_dir=pdb_dir,
            )

            # Compute Spearman ρ per timestep
            for t, s_t in s_t_map.items():
                if len(s_t) != n_residues:
                    print(f"  WARNING: s_t has {len(s_t)} residues, expected {n_residues}. Skipping t={t}")
                    continue
                rho_p, rho_all = compute_rho(s_t, s_0, pocket_idx)
                print(f"  t={t:.1f} rho_pocket={rho_p:.3f}  rho_all={rho_all:.3f}")
                if t not in prot_results:
                    prot_results[t] = {}
                prot_results[t][sample_idx] = (rho_p, rho_all)

        all_results[uid] = prot_results

    # -----------------------------------------------------------------------
    # Aggregate + write results
    # -----------------------------------------------------------------------
    print("\n=== Building results table ===")
    results_table = build_results_table(all_results, timesteps)

    write_results_csv(results_table, results_dir / "phase0_rho.csv")

    # Pretty print aggregate
    print("\n--- Aggregate rho_pocket (mean [95% CI]) ---")
    for t, stats in sorted(results_table.get("aggregate", {}).items()):
        print(f"  t={t:.1f}: {stats['point']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

    print("\nDone.")
    return results_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 noise sanity check")
    parser.add_argument("--out_dir", default="results/phase0_local")
    parser.add_argument("--proteins", nargs="+", default=None,
                        help="UniProt IDs to run (default: all in YAML)")
    parser.add_argument("--timesteps", type=float, nargs="+", default=None,
                        help="Normalised timesteps to capture (default: from YAML)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Diffusion samples per protein (default: from YAML)")
    parser.add_argument("--sampling_steps", type=int, default=200,
                        help="Denoising steps (default 200; use 10-20 for fast CPU test)")
    parser.add_argument("--recycling_steps", type=int, default=3,
                        help="Recycling iterations (default 3; use 1 for fast test)")
    parser.add_argument("--cache_dir", default=None,
                        help="Boltz cache dir (default: ~/.boltz)")
    args = parser.parse_args()

    # Filter protein list if specified
    selected_proteins = None
    if args.proteins:
        with open(_YAML_CONFIG) as f:
            cfg = yaml.safe_load(f)
        all_prots = cfg["proteins"]
        selected_proteins = [p for p in all_prots if p["uniprot"] in args.proteins]

    run_phase0(
        out_dir=Path(args.out_dir),
        proteins=selected_proteins,
        timesteps=args.timesteps,
        n_samples=args.samples,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
    )
