#!/usr/bin/env python
"""Phase 1 sweep driver: β/α parameter search across 5 proteins.

Runs pocket_p and pocket_t configurations from configs/phase1_sweep.yaml.
Per-(protein, config) checkpointing. Captures ESS trajectories.

Usage:
  PYTHONPATH="external/conformix/conformix_boltz/src:src" \
  python scripts/run_phase1_sweep.py \
    --config configs/phase1_sweep.yaml \
    --out_dir results/phase1_sweep \
    --accelerator gpu

Local sanity test (CPU, 2 samples, 10 steps — plumbing check only):
  PYTHONPATH="external/conformix/conformix_boltz/src:src" \
  python scripts/run_phase1_sweep.py \
    --config configs/phase1_sweep.yaml \
    --out_dir results/phase1_sweep_test \
    --accelerator cpu \
    --override_samples 2 \
    --override_steps 10 \
    --proteins NPC2
"""

import sys
import os
import csv
import time
import json
import subprocess
import threading
from pathlib import Path
from dataclasses import asdict, dataclass

import click
import yaml
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "conformix" / "conformix_boltz" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from boltz.run_untwisted import download, check_inputs, process_inputs
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from boltz.model.loss.diffusion import weighted_rigid_align
from tqdm import tqdm

from cryptic_pocket_phd.pocketminer_torch import PocketMinerTorch, AA_LOOKUP, ABBREV
from cryptic_pocket_phd.pocket_potential import PocketPotential, build_bb_atom_indices
from cryptic_pocket_phd.guidance_injection import pocket_twist_fn


@dataclass
class BoltzDiffusionParams:
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = False


def parse_pocket_residues(pocket_str):
    residues = []
    for part in pocket_str.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-")
            residues.extend(range(int(s), int(e) + 1))
        else:
            residues.append(int(part))
    return residues


def load_pocketminer(device):
    weights = REPO_ROOT / "models" / "pocketminer_torch.pt"
    model = PocketMinerTorch()
    with torch.no_grad():
        model(torch.randn(1, 50, 4, 3), torch.zeros(1, 50, dtype=torch.long),
              torch.ones(1, 50))
    model.load_state_dict(torch.load(str(weights), weights_only=True,
                                     map_location=device))
    model.eval()
    model.to(device)
    return model


def prepare_fasta(protein, out_dir):
    """Create FASTA input for Boltz from apo PDB."""
    import mdtraj as md
    apo_path = REPO_ROOT / "data" / "validation_pdbs" / f"{protein['apo_pdb']}.pdb"
    traj = md.load(str(apo_path))
    prot_iis = traj.top.select("protein and (name CA)")
    abbrev = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
              "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
              "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
              "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    seq = ""
    for idx in prot_iis:
        rname = traj.top.atom(idx).residue.name
        seq += abbrev.get(rname, "X")

    fasta_dir = out_dir / "inputs"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = fasta_dir / "msa"
    msa_dir.mkdir(exist_ok=True)

    uid = protein["uniprot"]
    # YAML with MSA
    yaml_path = fasta_dir / f"{uid}.yaml"
    msa_path = msa_dir / f"{uid}_0.a3m"
    with open(msa_path, "w") as f:
        f.write(f">query\n{seq}\n")
    yaml_data = {"sequences": [{"protein": {
        "id": "A", "sequence": seq, "msa": str(msa_path)}}]}
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
    return yaml_path, seq


def load_apo_masks(apo_pdb_path):
    """Load apo coordinates and build alignment mask (all protein atoms).

    Returns coordinates in Angstroms (Boltz's native unit).
    mdtraj loads in nanometers, so we multiply by 10.
    """
    import mdtraj as md
    traj = md.load(str(apo_pdb_path))
    prot_iis = traj.top.select("protein")
    xyz = torch.from_numpy(traj.xyz[0, prot_iis]).float() * 10.0  # nm -> Angstroms
    mask = torch.ones(xyz.shape[0])
    return xyz, mask


def unit_sanity_check(protein_list):
    """Assert apo PDB coords after nm->Å conversion are in Angstrom range.

    mdtraj raw coords are nm (~0.1-2 range per axis).
    After *10 they should be ~1-200 Å range (span > 5 Å).
    Aborts with sys.exit(1) if assertion fails.
    """
    import mdtraj as md
    protein = protein_list[0]
    apo_path = REPO_ROOT / "data" / "validation_pdbs" / f"{protein['apo_pdb']}.pdb"
    traj = md.load(str(apo_path))
    xyz_nm = traj.xyz[0]
    xyz_ang = xyz_nm * 10.0
    coord_min = float(xyz_ang.min())
    coord_max = float(xyz_ang.max())
    coord_range = coord_max - coord_min
    click.echo(f"[UNIT CHECK] {protein['apo_pdb']}: "
               f"min={coord_min:.1f} max={coord_max:.1f} range={coord_range:.1f} Å")
    if coord_range < 5.0:
        click.echo(f"[UNIT CHECK] FAIL: range {coord_range:.2f} Å < 5 Å — "
                   f"coords likely still in nm (forgot *10 fix?)")
        sys.exit(1)
    if coord_range > 1000.0:
        click.echo(f"[UNIT CHECK] FAIL: range {coord_range:.2f} Å > 1000 Å — "
                   f"something wrong with PDB")
        sys.exit(1)
    click.echo("[UNIT CHECK] PASS")


def start_checkpoint_loop(out_dir, interval_sec=1200):
    """Spawn background thread: git add/commit/push results every interval_sec."""
    repo_root = REPO_ROOT

    def _loop():
        while True:
            time.sleep(interval_sec)
            try:
                # Init LFS tracking for large array files (idempotent)
                subprocess.run(
                    ["git", "lfs", "track", "*.npy", "*.npz"],
                    cwd=str(repo_root), capture_output=True, timeout=30,
                )
                subprocess.run(
                    ["git", "add", ".gitattributes"],
                    cwd=str(repo_root), capture_output=True, timeout=30,
                )
                subprocess.run(
                    ["git", "add", "results/", "-A"],
                    cwd=str(repo_root), capture_output=True, timeout=60,
                )
                commit_msg = (
                    f"checkpoint: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
                )
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=str(repo_root), capture_output=True, timeout=30,
                )
                subprocess.run(
                    ["git", "push", "origin", "master"],
                    cwd=str(repo_root), capture_output=True, timeout=120,
                )
                click.echo(
                    f"[CHECKPOINT] Pushed at "
                    f"{time.strftime('%H:%M:%S UTC', time.gmtime())}"
                )
            except Exception as e:
                click.echo(f"[CHECKPOINT] Push failed (non-fatal): {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    click.echo(f"[CHECKPOINT] Background git-push loop started "
               f"(every {interval_sec // 60} min)")


def compute_coverage(sample_coords, holo_coords, apo_coords, threshold_frac=0.5):
    """Compute worst-matched coverage at threshold_frac * ref-to-ref RMSD.

    ConforMix metric: fraction of holo-state heavy atoms within threshold
    of any generated sample, after alignment.

    Args:
        sample_coords: list of [N_atoms, 3] numpy arrays (generated samples)
        holo_coords: [N_atoms, 3] numpy array (holo reference)
        apo_coords: [N_atoms, 3] numpy array (apo reference)
        threshold_frac: fraction of ref-to-ref RMSD to use as threshold

    Returns:
        coverage: float in [0, 1]
    """
    # Ref-to-ref RMSD (apo vs holo)
    ref_rmsd = np.sqrt(((apo_coords - holo_coords) ** 2).sum(-1).mean())
    threshold = threshold_frac * ref_rmsd

    if len(sample_coords) == 0:
        return 0.0

    # For each holo atom, find minimum distance to any sample atom
    # (after alignment — samples are already aligned via Boltz output)
    n_atoms = holo_coords.shape[0]
    min_dists = np.full(n_atoms, np.inf)

    for sample in sample_coords:
        n_s = min(sample.shape[0], n_atoms)
        dists = np.sqrt(((sample[:n_s] - holo_coords[:n_s]) ** 2).sum(-1))
        min_dists[:n_s] = np.minimum(min_dists[:n_s], dists)

    coverage = (min_dists < threshold).mean()
    return float(coverage)


def run_single_config(
    protein, bias_type, param_value, alpha_val, n_samples, sampling_steps,
    seed, ess_threshold, accelerator, cache, out_dir,
    model_module, pm_model, device,
):
    """Run one (protein, bias_type, param_value, alpha_val) config. Return metrics dict."""
    import mdtraj as md

    uid = protein["uniprot"]
    short = protein["short_name"]
    apo_pdb = protein["apo_pdb"]
    pocket_str = protein["pocket_residues"]
    pocket_idx = parse_pocket_residues(pocket_str)

    if bias_type == "pocket_p":
        config_name = f"pocket_p_beta_{param_value}"
    else:
        config_name = f"pocket_t_alpha{int(alpha_val)}_target_{param_value}"
    config_dir = out_dir / uid / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    done_marker = config_dir / "done.json"
    if done_marker.exists():
        click.echo(f"  SKIP {uid}/{config_name} (already done)")
        with open(done_marker) as f:
            return json.load(f)

    click.echo(f"  RUN {uid}/{config_name}: {n_samples} samples")

    # Prepare input
    yaml_path, seq = prepare_fasta(protein, out_dir)
    apo_pdb_path = REPO_ROOT / "data" / "validation_pdbs" / f"{apo_pdb}.pdb"
    untwisted_coords, twisting_mask = load_apo_masks(apo_pdb_path)

    # Process Boltz input
    boltz_out = out_dir / uid / "boltz_processed"
    boltz_out.mkdir(parents=True, exist_ok=True)

    data_files = check_inputs(yaml_path, boltz_out, override=True)
    if not data_files:
        return {"error": "no data files"}

    ccd_path = Path(cache) / "ccd.pkl"
    process_inputs(data=data_files, out_dir=boltz_out, ccd_path=ccd_path,
                   use_msa_server=False, msa_server_url="",
                   msa_pairing_strategy="greedy")

    processed_dir = boltz_out / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")
    data_module = BoltzInferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        num_workers=0,
    )

    # alpha_val comes from caller (sweep grid); param_value is beta (pocket_p) or target (pocket_t)
    beta_val = param_value

    from pytorch_lightning import seed_everything
    seed_everything(seed)

    for batch in data_module.predict_dataloader():
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float32):
            out = model_module(
                batch, recycling_steps=3,
                num_sampling_steps=sampling_steps, diffusion_samples=1,
                run_confidence_sequentially=True, conformix=False,
                save_diff_inputs=True,
            )
            sample_inputs = {
                "s_trunk": out["s_trunk"], "z_trunk": out["z_trunk"],
                "s_inputs": out["s_inputs"], "feats": out["feats"],
                "relative_position_encoding": out["relative_position_encoding"],
                "num_sampling_steps": sampling_steps,
                "atom_mask": out["atom_mask"],
                "multiplicity": n_samples,
                "train_accumulate_token_repr": False,
            }

        # Build pocket potential
        feats = out["feats"]
        atom_to_token = feats["atom_to_token"][0]
        n_tokens = int(feats["token_pad_mask"][0].sum().item())
        bb_indices = build_bb_atom_indices(atom_to_token, n_tokens)

        traj_apo = md.load(str(apo_pdb_path))
        prot_ca = traj_apo.top.select("protein and (name CA)")
        seq_list = []
        for idx in prot_ca:
            rname = traj_apo.top.atom(idx).residue.name
            seq_list.append(AA_LOOKUP.get(ABBREV.get(rname, "A"), 0))
        seq_indices = torch.tensor(seq_list[:n_tokens], dtype=torch.long)
        N_tok_pad = atom_to_token.shape[1]
        if len(seq_indices) < N_tok_pad:
            seq_indices = torch.nn.functional.pad(
                seq_indices, (0, N_tok_pad - len(seq_indices)))

        pocket_pot = PocketPotential(
            model=pm_model,
            pocket_residue_indices=[r for r in pocket_idx if r < n_tokens],
            bb_atom_indices=bb_indices.to(device),
            seq_indices=seq_indices.to(device),
            n_tokens=n_tokens,
        )

        twist_fn_ = pocket_twist_fn(
            alpha=alpha_val, beta=beta_val,
            tstart_step=200, tstop_step=0,
            bias_type=bias_type,
            pocket_potential=pocket_pot,
            untwisted_coords=untwisted_coords,
            twisting_mask=twisting_mask,
            weighted_rigid_align_fn=weighted_rigid_align,
        )

        # Run SMC
        t0 = time.time()
        try:
            with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.float32):
                sample_out = model_module.structure_module.sample_twisted(
                    **sample_inputs, twist_fn=twist_fn_,
                    ess_threshold=ess_threshold)
        except RuntimeError as e:
            if "out of memory" in str(e):
                click.echo(f"  OOM on {uid}/{config_name}")
                torch.cuda.empty_cache()
                return {"error": "OOM", "protein": uid, "config": config_name}
            raise

        elapsed = time.time() - t0
        click.echo(f"  SMC done in {elapsed:.0f}s")

        # Extract metrics
        coords = sample_out["sample_atom_coords"].cpu()  # [n_samples, N_atoms, 3]

        # Verify Boltz output coords are in Angstrom range (same scale as apo)
        sample_range = float(coords[0].max() - coords[0].min())
        click.echo(f"  [UNIT CHECK] Boltz sample coord range: {sample_range:.1f} Å")
        if sample_range < 5.0:
            click.echo(f"  [UNIT CHECK] FAIL: Boltz output range {sample_range:.2f} < 5 Å"
                       f" — coords likely in nm, not Å!")
            sys.exit(1)

        ess_trace = sample_out["ess_trace"].cpu().numpy()  # [n_steps]

        # Save ESS
        ess_dir = out_dir / "ess"
        ess_dir.mkdir(exist_ok=True)
        np.save(str(ess_dir / f"{uid}_{config_name}.npy"), ess_trace)

        ess_min = float(ess_trace.min())
        ess_mean = float(ess_trace.mean())
        ess_final = float(ess_trace[-1])

        # Compute RMSD from apo
        padded = coords.shape[1]
        tw_pad = torch.nn.functional.pad(
            twisting_mask, (0, padded - twisting_mask.shape[0]), value=0)
        uw_pad = torch.nn.functional.pad(
            untwisted_coords, (0, 0, 0, padded - untwisted_coords.shape[0]), value=0)
        aligned = weighted_rigid_align(
            coords, uw_pad, torch.ones(n_samples, padded), tw_pad,
            keep_gradients=False)
        mse = ((aligned - uw_pad) ** 2).sum(dim=-1)
        rmsd_vals = torch.sqrt(
            torch.sum(mse * tw_pad, dim=-1) / torch.sum(tw_pad, dim=-1)
        ).numpy()
        rmsd_mean = float(rmsd_vals.mean())

        # pLDDT (if available via confidence module)
        plddt_mean = float('nan')
        try:
            with torch.no_grad():
                conf_out = model_module.confidence_module(
                    s_inputs=sample_inputs["s_inputs"].detach(),
                    s=sample_inputs["s_trunk"].detach(),
                    z=sample_inputs["z_trunk"].detach(),
                    s_diffusion=None,
                    x_pred=sample_out["sample_atom_coords"].detach(),
                    feats=sample_inputs["feats"],
                    pred_distogram_logits=out["pdistogram"].detach(),
                    multiplicity=n_samples,
                    run_sequentially=True,
                )
            plddt_mean = float(conf_out["complex_plddt"].mean().item())
        except Exception as e:
            click.echo(f"  pLDDT computation failed: {e}")

        # Coverage (if holo PDB available)
        coverage = float('nan')
        holo_pdb_path = REPO_ROOT / "data" / "validation_pdbs" / f"{protein['holo_pdb']}.pdb"
        if holo_pdb_path.exists():
            try:
                holo_traj = md.load(str(holo_pdb_path))
                apo_traj = md.load(str(apo_pdb_path))
                # mdtraj returns nm; convert to Angstroms to match Boltz coords
                holo_ca = holo_traj.xyz[0, holo_traj.top.select("protein and (name CA)")] * 10.0
                apo_ca = apo_traj.xyz[0, apo_traj.top.select("protein and (name CA)")] * 10.0
                n_min = min(len(holo_ca), len(apo_ca))
                # Extract CA from samples
                sample_cas = []
                for s in range(n_samples):
                    ca_idx = bb_indices[:n_tokens, 1].numpy()  # CA is index 1
                    ca_idx = ca_idx[ca_idx < coords.shape[1]]
                    sample_cas.append(coords[s, ca_idx].numpy())
                coverage = compute_coverage(
                    sample_cas, holo_ca[:n_min], apo_ca[:n_min])
            except Exception as e:
                click.echo(f"  Coverage computation failed: {e}")

        metrics = {
            "protein": uid,
            "short_name": short,
            "bias_type": bias_type,
            "alpha_val": alpha_val,
            "param_value": param_value,
            "n_samples": n_samples,
            "ess_min": ess_min,
            "ess_mean": ess_mean,
            "ess_final": ess_final,
            "rmsd_mean": rmsd_mean,
            "plddt_mean": plddt_mean,
            "coverage": coverage,
            "elapsed_sec": elapsed,
        }

        # Save checkpoint
        with open(done_marker, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save coords
        np.save(str(config_dir / "coords.npy"), coords.numpy())

        return metrics

    return {"error": "no batch"}


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), default="results/phase1_sweep")
@click.option("--accelerator", type=click.Choice(["gpu", "cpu"]), default="gpu")
@click.option("--cache", type=click.Path(), default="~/.boltz")
@click.option("--override_samples", type=int, default=None,
              help="Override n_samples (for local testing)")
@click.option("--override_steps", type=int, default=None,
              help="Override sampling_steps (for local testing)")
@click.option("--proteins", type=str, default=None,
              help="Comma-separated short names to run (subset)")
def main(config, out_dir, accelerator, cache, override_samples,
         override_steps, proteins):
    """Run Phase 1 β/α sweep."""

    with open(config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    n_samples = override_samples or cfg["sweep"]["n_samples"]
    sampling_steps = override_steps or cfg["sweep"]["sampling_steps"]
    seed = cfg["sweep"]["seed"]
    ess_threshold = cfg["sweep"]["ess_threshold"]

    # Filter proteins if requested
    protein_list = cfg["proteins"]
    if proteins:
        names = [n.strip() for n in proteins.split(",")]
        protein_list = [p for p in protein_list if p["short_name"] in names]
        if not protein_list:
            click.echo(f"No proteins match: {names}")
            return

    # Unit sanity check: confirm coords are in Angstroms after nm->Å conversion
    unit_sanity_check(protein_list)

    # Build config grid: (bias_type, param_value, alpha_val)
    configs = []
    pocket_p_alpha = cfg["sweep"]["pocket_p"]["alpha"]
    for beta in cfg["sweep"]["pocket_p"]["beta_values"]:
        configs.append(("pocket_p", beta, pocket_p_alpha))
    for alpha in cfg["sweep"]["pocket_t"]["alpha_values"]:
        for target in cfg["sweep"]["pocket_t"]["target_values"]:
            configs.append(("pocket_t", target, alpha))

    total = len(protein_list) * len(configs)
    click.echo(f"Sweep: {len(protein_list)} proteins × {len(configs)} configs = {total} runs")
    click.echo(f"Samples per run: {n_samples}, steps: {sampling_steps}")

    # Setup
    torch.set_float32_matmul_precision("highest")
    download(cache)

    device = torch.device("cuda" if accelerator == "gpu" else "cpu")

    # Load Boltz model once
    checkpoint = cache / "boltz1_conf.ckpt"
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": sampling_steps,
        "diffusion_samples": n_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
        "conformix": True,
    }
    click.echo("Loading Boltz model...")
    model_module = Boltz1.load_from_checkpoint(
        checkpoint, strict=True, predict_args=predict_args,
        map_location="cpu", diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False, weights_only=False,
    )
    model_module.confidence_module.use_s_diffusion = False
    model_module.accumulate_token_repr = False
    model_module.eval()
    model_module.to(device)

    # Load PocketMiner once
    click.echo("Loading PocketMiner...")
    pm_model = load_pocketminer(str(device))

    # Start background checkpoint loop (CLAUDE.md Rule 2)
    start_checkpoint_loop(out_dir, interval_sec=1200)

    # CSV for aggregate results
    csv_path = out_dir / "phase1_sweep.csv"
    partial_csv = out_dir / "phase1_sweep_partial.csv"
    fieldnames = ["protein", "short_name", "bias_type", "alpha_val", "param_value",
                  "n_samples", "ess_min", "ess_mean", "ess_final",
                  "rmsd_mean", "plddt_mean", "coverage", "elapsed_sec"]

    all_results = []
    done = 0

    for protein in protein_list:
        for bias_type, param_value, alpha_val in configs:
            done += 1
            click.echo(f"\n[{done}/{total}] {protein['short_name']} "
                        f"{bias_type} param={param_value} alpha={alpha_val}")

            metrics = run_single_config(
                protein=protein,
                bias_type=bias_type,
                param_value=param_value,
                alpha_val=alpha_val,
                n_samples=n_samples,
                sampling_steps=sampling_steps,
                seed=seed,
                ess_threshold=ess_threshold,
                accelerator=accelerator,
                cache=str(cache),
                out_dir=out_dir,
                model_module=model_module,
                pm_model=pm_model,
                device=device,
            )

            if "error" not in metrics:
                all_results.append(metrics)
                # Write partial CSV
                with open(partial_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                click.echo(f"  ESS: min={metrics['ess_min']:.2f} "
                            f"mean={metrics['ess_mean']:.2f} "
                            f"final={metrics['ess_final']:.2f}")
                click.echo(f"  RMSD={metrics['rmsd_mean']:.2f} "
                            f"pLDDT={metrics['plddt_mean']:.3f} "
                            f"cov={metrics['coverage']:.3f}")
            else:
                click.echo(f"  ERROR: {metrics}")

    # Final CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    click.echo(f"\nSweep complete. Results: {csv_path}")
    click.echo(f"ESS trajectories: {out_dir}/ess/")


if __name__ == "__main__":
    main()
