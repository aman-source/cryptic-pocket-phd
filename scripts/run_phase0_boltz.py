"""Phase 0, Part 1: Boltz-1 GPU inference — no TensorFlow imports.

Runs Boltz-1 with InstrumentedAtomDiffusion capture, saves x̂_0 states
as PDB files. No PocketMiner / TensorFlow imported anywhere.

Output
------
{out_dir}/intermediate_pdbs/{uniprot}_s{sample:02d}_t{t:.1f}.pdb
{out_dir}/phase1_manifest.csv

Usage
-----
python -u scripts/run_phase0_boltz.py --out_dir results/phase0_runpod
python -u scripts/run_phase0_boltz.py --out_dir results/local_test \\
    --samples 1 --timesteps 0.5 0.9 --sampling_steps 10 --recycling_steps 1
"""

from __future__ import annotations

import csv
import os
import sys
import types
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import yaml

from boltz.data.module.inference import BoltzInferenceDataModule
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
from boltz.data import const

# Our modules — NO pocketminer_wrapper, NO fpocket_wrapper, NO tensorflow
from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb
from cryptic_pocket_phd.intermediate_capture import (
    InstrumentedAtomDiffusion,
    make_timestep_capture_fn,
)

# Safety: verify TF never loaded transitively
assert "tensorflow" not in sys.modules, \
    "tensorflow imported in phase 1 — check transitive imports"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_YAML_CONFIG = _REPO_ROOT / "configs" / "phase0_proteins.yaml"

_AA1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def instrument_model(model, capture_fn):
    def _instrumented(self, *args, **kwargs):
        kwargs["intermediate_capture_fn"] = capture_fn
        return InstrumentedAtomDiffusion.sample(self, *args, **kwargs)
    model.structure_module.sample = types.MethodType(_instrumented, model.structure_module)


def restore_model(model):
    if "sample" in model.structure_module.__dict__:
        del model.structure_module.__dict__["sample"]


def atom_metadata_from_sequence(sequence: str, chain_id: str = "A") -> dict:
    atom_names, res_indices, chain_ids, res_names = [], [], [], []
    for res_1idx, aa1 in enumerate(sequence, start=1):
        res3 = _AA1TO3.get(aa1, "ALA")
        atoms = const.ref_atoms.get(res3, ["N", "CA", "C", "O"])
        for aname in atoms:
            atom_names.append(aname)
            res_indices.append(res_1idx)
            chain_ids.append(chain_id)
            res_names.append(res3)
    return {"atom_names": atom_names, "res_indices": res_indices,
            "chain_ids": chain_ids, "res_names": res_names}


def run_phase0_boltz(
    out_dir: Path,
    proteins: list | None = None,
    timesteps: list | None = None,
    n_samples: int | None = None,
    cache_dir: Path | None = None,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
) -> None:
    if cache_dir is None:
        env_cache = os.environ.get("BOLTZ_CACHE")
        cache_dir = Path(env_cache) if env_cache else Path.home() / ".boltz"
    ccd_path = cache_dir / "ccd.pkl"
    checkpoint = cache_dir / "boltz1_conf.ckpt"
    if not ccd_path.exists():
        raise FileNotFoundError(f"CCD not found at {ccd_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    with open(_YAML_CONFIG) as f:
        config = yaml.safe_load(f)
    if proteins is None:
        proteins = config.get("proteins", config["experiment"].get("proteins", []))
    if timesteps is None:
        timesteps = config["experiment"]["timesteps"]
    if n_samples is None:
        n_samples = config["experiment"]["n_samples"]

    out_dir = Path(out_dir)
    capture_dir = out_dir / "captures"
    pdb_dir = out_dir / "intermediate_pdbs"
    boltz_out_dir = out_dir / "boltz_output"
    manifest_path = out_dir / "phase1_manifest.csv"

    for d in [capture_dir, pdb_dir, boltz_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Boltz preprocessing
    # ------------------------------------------------------------------
    input_dir = _REPO_ROOT / "data" / "boltz_inputs"
    input_files: list[Path] = []
    for prot in proteins:
        uid = prot["uniprot"]
        yaml_path = input_dir / uid / f"{uid}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Boltz input YAML not found: {yaml_path}")
        input_files.append(yaml_path)

    print("=== Phase 1: Boltz inference (GPU, no TF) ===")
    print(f"Proteins: {len(proteins)}, Samples: {n_samples}, Timesteps: {timesteps}")
    print(f"Sampling steps: {sampling_steps}, Recycling: {recycling_steps}")

    print("\n=== Preprocessing inputs ===")
    process_inputs(
        data=input_files,
        out_dir=boltz_out_dir,
        ccd_path=ccd_path,
        mol_dir=None,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,
        preprocessing_threads=1,
    )

    manifest = Manifest.load(boltz_out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=boltz_out_dir, override=True,
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
    # Map uid → manifest record for per-protein filtering
    record_by_uid = {r.id: r for r in filtered_manifest.records}

    # ------------------------------------------------------------------
    # Load Boltz1 model
    # ------------------------------------------------------------------
    print("\n=== Loading Boltz1 model ===")
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
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
        use_kernels=False,
        pairformer_args=asdict(PairformerArgs()),
        msa_args=asdict(MSAModuleArgs(
            subsample_msa=False, num_subsampled_msa=2048, use_paired_feature=False,
        )),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    import torch
    _accel = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using accelerator: {_accel}")

    # ------------------------------------------------------------------
    # Per-protein × per-sample inference
    # ------------------------------------------------------------------
    manifest_rows: list[dict] = []

    for prot in proteins:
        uid = prot["uniprot"]
        short = prot.get("short_name", uid)
        apo_pdb = _REPO_ROOT / "data" / "validation_pdbs" / f"{prot['apo_pdb']}.pdb"

        print(f"\n=== {short} ({uid}) ===")

        seq = prot.get("sequence")
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
        n_residues = len(seq)

        # Single-protein manifest: one predict call = one protein = one sample
        single_manifest = Manifest(records=[record_by_uid[uid]])

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx + 1}/{n_samples}")

            # Fresh data module per sample so DataLoader state resets
            data_module = BoltzInferenceDataModule(
                manifest=single_manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=0,
                constraints_dir=processed.constraints_dir,
            )

            capture_fn = make_timestep_capture_fn(
                target_ts=timesteps,
                output_dir=str(capture_dir),
                protein_id=uid,
                sample_idx=sample_idx,
                atom_metadata=atom_metadata,
                num_sampling_steps=sampling_steps,
            )

            instrument_model(model_module, capture_fn)

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
                accelerator=_accel,
                devices=1,
                precision=32,
                enable_progress_bar=True,
            )
            trainer.predict(model_module, datamodule=data_module, return_predictions=False)
            restore_model(model_module)

            # Convert .npz captures → PDB files
            for t in timesteps:
                npz_path = capture_dir / f"{uid}_s{sample_idx:02d}_t{t:.1f}.npz"
                if not npz_path.exists():
                    print(f"  WARNING: capture not found: {npz_path}")
                    continue

                data = np.load(str(npz_path))
                coords = data["coords"]
                pdb_path = pdb_dir / f"{uid}_s{sample_idx:02d}_t{t:.1f}.pdb"
                boltz_coords_to_pdb(
                    coords=coords[0],
                    atom_names=atom_metadata["atom_names"],
                    res_indices=atom_metadata["res_indices"],
                    chain_ids=atom_metadata["chain_ids"],
                    res_names=atom_metadata["res_names"],
                    output_path=str(pdb_path),
                )
                npz_path.unlink()  # free disk space
                print(f"  t={t:.1f} → {pdb_path.name}")

                manifest_rows.append({
                    "uniprot": uid,
                    "apo_pdb": str(apo_pdb),
                    "n_residues": n_residues,
                    "pocket_ranges": str(prot["pocket_residue_ranges"]),
                    "sample_idx": sample_idx,
                    "timestep": t,
                    "pdb_path": str(pdb_path),
                })

    with open(manifest_path, "w", newline="") as f:
        if manifest_rows:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path} ({len(manifest_rows)} rows)")
    print("Phase 1 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 Part 1: Boltz-1 inference")
    parser.add_argument("--out_dir", default="results/phase0_local")
    parser.add_argument("--proteins", nargs="+", default=None,
                        help="UniProt IDs (default: all in YAML)")
    parser.add_argument("--timesteps", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    selected_proteins = None
    if args.proteins:
        with open(_YAML_CONFIG) as f:
            cfg = yaml.safe_load(f)
        all_prots = cfg["proteins"]
        selected_proteins = [p for p in all_prots if p["uniprot"] in args.proteins]

    run_phase0_boltz(
        out_dir=Path(args.out_dir),
        proteins=selected_proteins,
        timesteps=args.timesteps,
        n_samples=args.samples,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
    )
