"""Phase 1 runner: ConforMix twisted SMC with pluggable guidance.

Reuses ConforMix's Boltz integration, SMC machinery, and confidence scoring.
Replaces only the twist_fn with our pocket-guided version.

Bias types:
  rmsd      — ConforMix's original RMSD guidance (regression baseline)
  pocket_p  — PocketMiner sum-based (g_p)
  pocket_t  — PocketMiner sweep-based (g_t)

Usage:
  python scripts/run_phase1.py predict <data_path> \
    --input_cif <apo.cif> --out_dir <outdir> \
    --bias_type pocket_p \
    --pocket_residues "190-200,244-263" \
    --twist_strength_values 15.0 \
    --twist_target_values 1.0 \
    --diffusion_samples 5
"""

import sys
import os
from pathlib import Path

# Add ConforMix and our src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "conformix" / "conformix_boltz" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import click
import torch
import numpy as np
from pytorch_lightning import seed_everything

from boltz.run_twisted import (
    download,
    check_inputs,
    process_inputs,
    BoltzProcessedInput,
    BoltzDiffusionParams,
    BoltzInferenceDataModule,
    get_secondary_structure_region_masks,
    get_residue_atoms_mask,
    weighted_rigid_align,
)
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from dataclasses import asdict
from tqdm import tqdm
import glob
from typing import Union, List, Optional, Literal

from cryptic_pocket_phd.pocketminer_torch import PocketMinerTorch, AA_LOOKUP, ABBREV
from cryptic_pocket_phd.pocket_potential import PocketPotential, build_bb_atom_indices
from cryptic_pocket_phd.guidance_injection import pocket_twist_fn


def parse_pocket_residues(pocket_str: str) -> list[int]:
    """Parse pocket residue string like '190-200,244-263' to list of ints."""
    residues = []
    for part in pocket_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            residues.extend(range(int(start), int(end) + 1))
        else:
            residues.append(int(part))
    return residues


def load_pocketminer_model(device: str = "cpu") -> PocketMinerTorch:
    """Load PocketMiner PyTorch model from saved state dict."""
    weights_path = REPO_ROOT / "models" / "pocketminer_torch.pt"
    model = PocketMinerTorch()
    with torch.no_grad():
        model(torch.randn(1, 50, 4, 3), torch.zeros(1, 50, dtype=torch.long),
              torch.ones(1, 50))
    model.load_state_dict(torch.load(str(weights_path), weights_only=True,
                                     map_location=device))
    model.eval()
    model.to(device)
    return model


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--input_cif", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(exists=False), default="./")
@click.option("--bias_type", type=click.Choice(["rmsd", "pocket_p", "pocket_t"]),
              default="rmsd")
@click.option("--pocket_residues", type=str, default=None,
              help="Pocket residue ranges, e.g. '190-200,244-263'. Required for pocket_p/pocket_t.")
@click.option("--twist_target_values", default="1.0",
              help="Comma-separated target values (RMSD targets or pocket strength).")
@click.option("--twist_strength_values", default="15.0",
              help="Comma-separated strength values (alpha).")
@click.option("--tstart_step", type=str, default="200")
@click.option("--tstop_step", type=str, default="0")
@click.option("--ess_threshold", type=float, default=1/3)
@click.option("--diffusion_samples", type=int, default=5)
@click.option("--sampling_steps", type=int, default=200)
@click.option("--recycling_steps", type=int, default=3)
@click.option("--accelerator", type=click.Choice(["gpu", "cpu"]), default="gpu")
@click.option("--cache", type=click.Path(exists=False), default="~/.boltz")
@click.option("--checkpoint", type=click.Path(exists=True), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--subset_residues", type=str, default=None)
@click.option("--twist_rmsd_full_sequence", is_flag=True, default=False)
@click.option("--output_format", type=click.Choice(["pdb", "mmcif"]), default="mmcif")
@click.option("--num_workers", type=int, default=2)
def predict(
    data, input_cif, out_dir, bias_type, pocket_residues,
    twist_target_values, twist_strength_values,
    tstart_step, tstop_step, ess_threshold, diffusion_samples,
    sampling_steps, recycling_steps, accelerator, cache, checkpoint,
    seed, subset_residues, twist_rmsd_full_sequence, output_format,
    num_workers,
):
    """Run ConforMix twisted SMC with pluggable guidance potential."""

    if bias_type in ("pocket_p", "pocket_t") and pocket_residues is None:
        raise click.UsageError("--pocket_residues required for pocket_p/pocket_t")

    torch.set_float32_matmul_precision("highest")
    if seed is not None:
        seed_everything(seed)

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser() / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    download(cache)

    data_files = check_inputs(data, out_dir, override=True)
    if not data_files:
        click.echo("No predictions to run.")
        return

    ccd_path = cache / "ccd.pkl"
    process_inputs(data=data_files, out_dir=out_dir, ccd_path=ccd_path,
                   use_msa_server=False, msa_server_url="", msa_pairing_strategy="greedy")

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
        "conformix": True,
    }
    model_module = Boltz1.load_from_checkpoint(
        checkpoint, strict=True, predict_args=predict_args,
        map_location="cpu", diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
    )
    model_module.confidence_module.use_s_diffusion = False
    model_module.accumulate_token_repr = False
    model_module.eval()

    # Parse sweep values
    twist_target_values = [float(x) for x in str(twist_target_values).split(",")]
    twist_strength_values = [float(x) for x in str(twist_strength_values).split(",")]
    tstart_values = [int(x) for x in tstart_step.split(",")]
    tstop_values = [int(x) for x in tstop_step.split(",")]

    # Apo structure for alignment
    untwisted_coords, region_mask, ss_atom_mask = \
        get_secondary_structure_region_masks(input_cif, subset_residues)

    twisting_mask = ss_atom_mask
    if twist_rmsd_full_sequence:
        twisting_mask = torch.ones_like(ss_atom_mask)
    if subset_residues:
        twisting_mask = twisting_mask * region_mask

    desc_string = bias_type
    existing_runs = glob.glob(str(out_dir / "predictions" / f"{desc_string}" / "run*"))
    run_num = len(existing_runs)
    desc_string = f"{desc_string}/run{run_num:02d}"

    device = torch.device("cuda" if accelerator == "gpu" else accelerator)
    model_module.to(device)

    # Load PocketMiner model if needed
    pm_model = None
    if bias_type in ("pocket_p", "pocket_t"):
        pm_model = load_pocketminer_model(device=str(device))
        pocket_idx_list = parse_pocket_residues(pocket_residues)
        click.echo(f"PocketMiner loaded. Pocket residues: {len(pocket_idx_list)}")

    for batch in tqdm(data_module.predict_dataloader()):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float32):
            out = model_module(
                batch, recycling_steps=recycling_steps,
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
                "multiplicity": diffusion_samples,
                "train_accumulate_token_repr": False,
            }

        # Build pocket potential for this protein (if needed)
        pocket_pot = None
        if bias_type in ("pocket_p", "pocket_t"):
            feats = out["feats"]
            # Build backbone atom indices from atom_to_token
            atom_to_token = feats["atom_to_token"][0]  # [N_atoms, N_tokens]
            n_tokens = int(feats["token_pad_mask"][0].sum().item())
            bb_indices = build_bb_atom_indices(atom_to_token, n_tokens)

            # Get sequence: res_type is one-hot [N_tokens, num_classes]
            # We need to map Boltz's res_type to PocketMiner's 0-19
            # For now, extract from PDB (more reliable than mapping encodings)
            import mdtraj as md
            traj = md.load(input_cif if input_cif.endswith('.pdb') else input_cif)
            prot_iis = traj.top.select("protein and (name CA)")
            n_ca = len(prot_iis)
            seq = []
            for idx in prot_iis:
                atom = traj.top.atom(idx)
                rname = atom.residue.name
                if rname in ABBREV:
                    seq.append(AA_LOOKUP[ABBREV[rname]])
                else:
                    seq.append(0)  # unknown -> ALA
            seq_indices = torch.tensor(seq[:n_tokens], dtype=torch.long)
            # Pad to N_tokens_padded
            N_tokens_padded = atom_to_token.shape[1]
            if len(seq_indices) < N_tokens_padded:
                seq_indices = torch.nn.functional.pad(
                    seq_indices, (0, N_tokens_padded - len(seq_indices)))

            pocket_pot = PocketPotential(
                model=pm_model,
                pocket_residue_indices=[r for r in pocket_idx_list if r < n_tokens],
                bb_atom_indices=bb_indices.to(device),
                seq_indices=seq_indices.to(device),
                n_tokens=n_tokens,
            )

        # Define twist_fn based on bias_type
        def make_twist_fn(alpha_val, beta_val, tstart_val, tstop_val):
            if bias_type == "rmsd":
                # Use ConforMix's original RMSD twist_fn (inlined)
                def rmsd_inner(xt, x0_hat, return_grad=True, t=None, atom_mask=None):
                    padded_atom_size = x0_hat.shape[1]
                    twisting_mask_padded = torch.nn.functional.pad(
                        twisting_mask, (0, padded_atom_size - twisting_mask.shape[0]),
                        value=0).to(x0_hat.device)
                    untwisted_padded = torch.nn.functional.pad(
                        untwisted_coords, (0, 0, 0, padded_atom_size - untwisted_coords.shape[0]),
                        value=0).to(x0_hat.device)

                    atom_pos_aligned = weighted_rigid_align(
                        x0_hat, untwisted_padded, atom_mask,
                        twisting_mask_padded, keep_gradients=True)
                    mse_loss = ((atom_pos_aligned - untwisted_padded) ** 2).sum(dim=-1)
                    rmsd = torch.sqrt(
                        torch.sum(mse_loss * twisting_mask_padded, dim=-1)
                        / torch.sum(twisting_mask_padded, dim=-1))
                    log_potential = -((rmsd - beta_val) ** 2)

                    if not return_grad:
                        return log_potential.to(device).detach()

                    if t is not None and tstart_val >= t >= tstop_val:
                        grad = torch.autograd.grad(
                            log_potential, xt,
                            grad_outputs=torch.ones_like(log_potential),
                            create_graph=False, allow_unused=True)[0]
                        if grad is None:
                            grad = torch.zeros_like(xt)
                        if t:
                            if alpha_val > 0:
                                factor = alpha_val
                            elif alpha_val < 0:
                                factor = abs(alpha_val) * 200 * (
                                    1 + np.cos(np.pi * (np.log(1 + 4 * (230 - t)) / np.log(10)))
                                ) / 2
                            else:
                                factor = 0
                            grad = grad * factor
                    else:
                        grad = torch.zeros_like(xt, device=device)

                    return log_potential.to(device).detach(), grad.to(device).detach()

                return rmsd_inner

            else:
                return pocket_twist_fn(
                    alpha=alpha_val, beta=beta_val,
                    tstart_step=tstart_val, tstop_step=tstop_val,
                    bias_type=bias_type,
                    pocket_potential=pocket_pot,
                    untwisted_coords=untwisted_coords,
                    twisting_mask=twisting_mask,
                    weighted_rigid_align_fn=weighted_rigid_align,
                )

        # Sample step (same as ConforMix's)
        def sample_step(sample_inputs, twist_fn_, pdistogram):
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float32):
                    sample_out = model_module.structure_module.sample_twisted(
                        **sample_inputs, twist_fn=twist_fn_,
                        ess_threshold=ess_threshold)
                    sample_out.update(
                        model_module.confidence_module(
                            s_inputs=sample_inputs["s_inputs"].detach(),
                            s=sample_inputs["s_trunk"].detach(),
                            z=sample_inputs["z_trunk"].detach(),
                            s_diffusion=None,
                            x_pred=sample_out["sample_atom_coords"].detach(),
                            feats=sample_inputs["feats"],
                            pred_distogram_logits=pdistogram.detach(),
                            multiplicity=diffusion_samples,
                            run_sequentially=True,
                        )
                    )
                pred_dict = {"exception": False}
                pred_dict["masks"] = batch["atom_pad_mask"]
                pred_dict["coords"] = sample_out["sample_atom_coords"]
                pred_dict["confidence_score"] = (
                    4 * sample_out["complex_plddt"] +
                    (sample_out["iptm"] if not torch.allclose(
                        sample_out["iptm"], torch.zeros_like(sample_out["iptm"]))
                     else sample_out["ptm"])
                ) / 5
                for key in ["ptm", "iptm", "ligand_iptm", "protein_iptm",
                            "pair_chains_iptm", "complex_plddt", "complex_iplddt",
                            "complex_pde", "complex_ipde", "plddt", "pae", "pde",
                            "ess_trace", "xt_trace", "grad_log_potential_xt_trace",
                            "logp_y_given_x0_trace", "log_w_trace"]:
                    pred_dict[key] = sample_out[key]
                return pred_dict
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("| WARNING: ran out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    return {"exception": True}
                raise

        for alpha_val in twist_strength_values:
            for beta_val in twist_target_values:
                for tstart_val in tstart_values:
                    for tstop_val in tstop_values:
                        click.echo(f"Running {bias_type} alpha={alpha_val} "
                                   f"beta={beta_val} tstart={tstart_val} tstop={tstop_val}")

                        full_output_dir = (out_dir / "predictions" / desc_string /
                                           f"variation_alpha_{alpha_val}_beta_{beta_val}")

                        twist_fn_ = make_twist_fn(alpha_val, beta_val, tstart_val, tstop_val)

                        sample_out = sample_step(sample_inputs, twist_fn_, out["pdistogram"])

                        if sample_out["exception"]:
                            continue

                        pred_writer = BoltzWriter(
                            data_dir=processed.targets_dir,
                            output_dir=full_output_dir,
                            output_format=output_format,
                        )

                        input_dict = {
                            "desc_string": desc_string,
                            "alpha": alpha_val,
                            "beta": beta_val,
                            "tstart": tstart_val,
                            "tstop": tstop_val,
                            "bias_type": bias_type,
                            "pocket_residues": pocket_residues,
                        }

                        # Compute RMSD for logging (always, regardless of bias type)
                        atom_pos = sample_out["coords"]
                        atom_mask_out = sample_out["masks"]
                        padded_atom_size = atom_pos.shape[1]
                        padded_tw_mask = torch.nn.functional.pad(
                            twisting_mask, (0, padded_atom_size - twisting_mask.shape[0]),
                            value=0).to(atom_pos.device)
                        untwisted_padded = torch.nn.functional.pad(
                            untwisted_coords, (0, 0, 0, padded_atom_size - untwisted_coords.shape[0]),
                            value=0).to(atom_pos.device)
                        atom_pos_aligned = weighted_rigid_align(
                            atom_pos, untwisted_padded, atom_mask_out,
                            padded_tw_mask, keep_gradients=False)
                        mse = ((atom_pos_aligned - untwisted_padded) ** 2).sum(dim=-1)
                        rmsd = torch.sqrt(
                            torch.sum(mse * padded_tw_mask, dim=-1)
                            / torch.sum(padded_tw_mask, dim=-1))
                        rmsd = rmsd.cpu().numpy().tolist()
                        input_dict["input_cif"] = input_cif

                        pred_writer.write_on_batch_end(
                            trainer=None, pl_module=None,
                            prediction=sample_out, batch_indices=None,
                            batch=batch, batch_idx=None, dataloader_idx=None,
                            input_dict=input_dict, rmsd=rmsd,
                        )

                        # Log ESS trajectory
                        if "ess_trace" in sample_out:
                            ess = sample_out["ess_trace"]
                            click.echo(f"  ESS: min={ess.min():.2f} max={ess.max():.2f} "
                                        f"final={ess[-1]:.2f}")

    click.echo(f"Done. Results in {out_dir / 'predictions' / desc_string}")


if __name__ == "__main__":
    predict()
