#!/usr/bin/env python3
"""
Task A1: Generate noisy training data for noise-aware PocketMiner.

For each (protein, frame), runs Boltz sampling and captures x̂_0(x_t) at
specified timesteps. Uses InstrumentedAtomDiffusion hook from Phase 0.

Output per (protein, frame, t):
    data/noisy_frames/{protein_id}/{frame_idx}_t{t:.1f}.npz
    Contains: noisy_coords (x̂_0), t, clean_coords (MD frame), pocket_labels

Resumable: skips if output file + done.json already exist.

Local CPU test:
    python scripts/generate_boltz_intermediates.py \
        --frames_dir data/md_frames/atlas \
        --labels_dir data/md_labels/atlas \
        --out_dir data/noisy_frames \
        --proteins 1k5n_A \
        --n_frames 5 \
        --timesteps 0.1,0.5,0.9 \
        --sampling_steps 10 \
        --accelerator cpu

GPU run (Task A1 full):
    python scripts/generate_boltz_intermediates.py \
        --frames_dir data/md_frames/atlas \
        --labels_dir data/md_labels/atlas \
        --out_dir data/noisy_frames \
        --n_frames 200 \
        --timesteps 0.1,0.3,0.5,0.7,0.9 \
        --sampling_steps 200 \
        --accelerator gpu
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "conformix" / "conformix_boltz" / "src"))

# Also honour PYTHONPATH set externally (e.g. from nohup launch command)
for _p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_DIR = Path.home() / ".boltz"


def _load_boltz(accelerator: str):
    """Load Boltz model. Same as run_phase1_sweep.py."""
    from boltz.model.model import Boltz1

    checkpoint = CACHE_DIR / "boltz1_conf.ckpt"
    if not checkpoint.exists():
        click.echo(f"Boltz checkpoint not found at {checkpoint}. Run setup_runpod.sh first.")
        sys.exit(1)

    predict_args = {
        "output_format": "pdb",
        "num_workers": 2,
        "override_method": "boltz1",
    }
    device = "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"

    click.echo(f"Loading Boltz on {device}...")
    model_module = Boltz1.load_from_checkpoint(
        checkpoint, strict=True, predict_args=predict_args,
        map_location=device,
    )
    model_module.eval()
    if device == "cuda":
        model_module = model_module.cuda()

    return model_module, device


def _setup_boltz_input(protein_id: str, sequence: str, out_dir: Path) -> tuple:
    """
    Prepare Boltz input (YAML + MSA + CCD processing) for a protein.
    Returns (data_module, batch) ready for sampling.
    Caches to out_dir/boltz_processed/{protein_id}/ so MSA is only computed once.
    """
    from boltz.run_untwisted import process_inputs
    from boltz.data.module.inference import BoltzInferenceDataModule
    from boltz.data.types import Manifest

    processed_dir = out_dir / "boltz_processed" / protein_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    ccd_path = CACHE_DIR / "ccd.pkl"

    # Write single-sequence a3m (minimal MSA — Boltz runs in single-seq mode)
    msa_dir = processed_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    msa_path = msa_dir / f"{protein_id}_0.a3m"
    if not msa_path.exists():
        msa_path.write_text(f">{protein_id}\n{sequence}\n")

    # Write YAML with MSA reference (mirrors run_phase1_sweep.py format)
    yaml_path = processed_dir / f"{protein_id}.yaml"
    if not yaml_path.exists():
        yaml_content = (
            f"sequences:\n"
            f"- protein:\n"
            f"    id: A\n"
            f"    sequence: {sequence}\n"
            f"    msa: {msa_path.as_posix()}\n"
        )
        yaml_path.write_text(yaml_content)

    # process_inputs writes into processed_dir/processed/ — same as run_phase1_sweep.py
    boltz_out = processed_dir  # process_inputs creates processed_dir/processed/manifest.json
    manifest_path = boltz_out / "processed" / "manifest.json"
    if not manifest_path.exists():
        click.echo(f"  Processing Boltz input for {protein_id}...")
        process_inputs(
            data=[yaml_path],
            out_dir=boltz_out,
            ccd_path=ccd_path,
            use_msa_server=False,
            msa_server_url="",
            msa_pairing_strategy="greedy",
        )

    # Build data module (mirrors run_phase1_sweep.py exactly)
    manifest = Manifest.load(manifest_path)
    data_module = BoltzInferenceDataModule(
        manifest=manifest,
        target_dir=boltz_out / "processed" / "structures",
        msa_dir=boltz_out / "processed" / "msa",
        num_workers=0,
    )
    data_module.setup(stage="predict")

    batches = list(data_module.predict_dataloader())
    if not batches:
        raise RuntimeError(f"No batches for {protein_id}")
    batch = batches[0]

    return batch


def _patch_sample_with_capture(atom_diffusion, capture_fn, num_steps):
    """
    Monkey-patch AtomDiffusion.sample() to call capture_fn after each
    denoising step. Works with the local conformix Boltz version.

    capture_fn(step_idx, t, x_hat_0, sigma_t) — same signature as
    intermediate_capture.py's CaptureCallback.
    """
    from math import sqrt as _sqrt
    from boltz.model.modules.utils import center_random_augmentation, default
    from boltz.model.loss.diffusion import weighted_rigid_align

    orig_sample = atom_diffusion.__class__.sample

    def patched_sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask_rep = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask_rep.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None
        token_repr = None
        token_a = None

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords, atom_mask_rep,
                augmentation=True, return_second_coords=True,
                second_coords=atom_coords_denoised,
            )
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * _sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy, t_hat, training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity, model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            # <<< CAPTURE HOOK >>>
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            capture_fn(step_idx, steering_t, atom_coords_denoised.clone(), t_hat)
            # <<< END CAPTURE HOOK >>>

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)
                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],), t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a,
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(), atom_coords_denoised.float(),
                        atom_mask_rep.float(), atom_mask_rep.float(),
                    )
                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    # Bind patched method to instance only
    import types
    atom_diffusion.sample = types.MethodType(patched_sample, atom_diffusion)
    return atom_diffusion


def _run_capture(
    model_module,
    batch: dict,
    frame_idx: int,
    target_ts: list[float],
    out_dir: Path,
    protein_id: str,
    clean_coords: np.ndarray,
    pocket_labels: np.ndarray,
    sampling_steps: int,
    device: str,
    seed: int,
) -> list[Path]:
    """
    Run Boltz sampling for one frame, capture x̂_0 at target timesteps.

    Flow:
      1. model_module(batch) → out["sample_inputs"] (conditioning features)
      2. Patch structure_module.sample() with capture hook
      3. Re-run sample() — captures x̂_0 at target timesteps

    Returns list of output paths written.
    """
    from pytorch_lightning import seed_everything

    seed_everything(seed)

    frame_out_dir = out_dir / protein_id
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    missing_ts = []
    for t in target_ts:
        if not (frame_out_dir / f"{frame_idx:04d}_t{t:.1f}.npz").exists():
            missing_ts.append(t)
    if not missing_ts:
        return []

    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_device(v) for v in obj]
        return obj

    batch = to_device(batch)

    captures: dict[float, np.ndarray] = {}
    tol = 0.5 / sampling_steps

    def capture_fn(step_idx: int, t: float, x_hat_0: torch.Tensor, sigma_t: float):
        for target_t in missing_ts:
            if target_t not in captures and abs(t - target_t) <= tol:
                captures[target_t] = x_hat_0[0].cpu().float().numpy()

    # Patch structure_module.sample BEFORE model_module(batch, ...) call.
    # model.forward() calls structure_module.sample() internally — our patch
    # injects the capture hook into that single pass.
    structure_module = model_module.structure_module
    _patch_sample_with_capture(structure_module, capture_fn, sampling_steps)

    with torch.no_grad():
        model_module(
            batch,
            num_sampling_steps=sampling_steps,
            diffusion_samples=1,
            run_confidence_sequentially=True,
        )

    # Step 4: save captures
    written = []
    for t, coords in captures.items():
        coord_range = float(coords.max() - coords.min())
        if coord_range < 5.0:
            click.echo(f"  WARNING: coords range {coord_range:.2f} — may be nm not Å")

        out_path = frame_out_dir / f"{frame_idx:04d}_t{t:.1f}.npz"
        np.savez_compressed(
            out_path,
            noisy_coords=coords.astype(np.float32),        # x̂_0(x_t), shape (N_atoms,3)
            t=np.float32(t),
            clean_coords=clean_coords.astype(np.float32),  # MD frame CA coords (N_res,3)
            pocket_labels=pocket_labels.astype(np.int32),  # LIGSITE labels (N_res,)
        )
        written.append(out_path)

    return written


@click.command()
@click.option("--frames_dir", type=click.Path(exists=True), required=True,
              help="data/md_frames/{source}")
@click.option("--labels_dir", type=click.Path(exists=True), required=True,
              help="data/md_labels/{source}")
@click.option("--out_dir", type=click.Path(), required=True,
              help="data/noisy_frames")
@click.option("--proteins", type=str, default=None,
              help="Comma-separated protein IDs. Default: all in frames_dir.")
@click.option("--n_frames", type=int, default=200,
              help="Max frames per protein to process")
@click.option("--timesteps", type=str, default="0.1,0.3,0.5,0.7,0.9",
              help="Comma-separated normalised timesteps to capture")
@click.option("--sampling_steps", type=int, default=200,
              help="Boltz denoising steps (use 10 for local CPU test)")
@click.option("--accelerator", type=click.Choice(["cpu", "gpu"]), default="cpu")
@click.option("--num_gpus", type=int, default=1,
              help="Number of GPUs to use. Splits protein list into N chunks, "
                   "each pinned to one GPU via CUDA_VISIBLE_DEVICES. "
                   "Use 1 for single-GPU or CPU runs.")
@click.option("--worker_id", type=int, default=0, hidden=True,
              help="Internal: GPU worker index for progress tracking (set by parent).")
def main(
    frames_dir, labels_dir, out_dir, proteins, n_frames,
    timesteps, sampling_steps, accelerator, num_gpus, worker_id,
):
    frames_path = Path(frames_dir)
    labels_path = Path(labels_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_ts = [float(t) for t in timesteps.split(",")]

    # Resolve protein list
    if proteins:
        protein_ids = proteins.split(",")
    else:
        protein_ids = sorted([
            d.name for d in frames_path.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])

    # Multi-GPU: spawn N subprocesses each pinned to one GPU
    if num_gpus > 1 and accelerator == "gpu":
        chunks = [protein_ids[i::num_gpus] for i in range(num_gpus)]
        logs_dir = out_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        progress_dir = out_path / ".progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        procs = []
        log_fds = []
        for gpu_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            cmd = [
                sys.executable, __file__,
                "--frames_dir", frames_dir,
                "--labels_dir", labels_dir,
                "--out_dir", out_dir,
                "--proteins", ",".join(chunk),
                "--n_frames", str(n_frames),
                "--timesteps", timesteps,
                "--sampling_steps", str(sampling_steps),
                "--accelerator", "gpu",
                "--num_gpus", "1",
                "--worker_id", str(gpu_idx),
            ]
            log_path = logs_dir / f"gpu{gpu_idx}.log"
            log_fd = open(log_path, "w", buffering=1)
            log_fds.append(log_fd)
            click.echo(f"GPU {gpu_idx}: {len(chunk)} proteins → {log_path}")
            click.echo(f"  proteins: {chunk[:3]}{'...' if len(chunk) > 3 else ''}")
            proc = subprocess.Popen(cmd, env=env, stdout=log_fd, stderr=subprocess.STDOUT)
            procs.append((gpu_idx, proc))

        total_target = len(protein_ids) * n_frames
        run_start = time.time()
        cost_per_gpu_hr = 0.40  # A40 community $/hr
        monitor_interval = 600  # 10 min

        click.echo(f"\nLaunched {len(procs)} GPU workers.")
        click.echo(f"Target: {total_target} frames ({len(protein_ids)} proteins × {n_frames} frames)")
        click.echo(f"Live logs: tail -f {logs_dir}/gpu*.log")
        click.echo(f"Progress updates every {monitor_interval // 60} min below.\n")

        def _monitor():
            while any(p.poll() is None for _, p in procs):
                time.sleep(monitor_interval)
                elapsed = time.time() - run_start
                total_done = 0
                for i in range(len(procs)):
                    pf = progress_dir / f"gpu{i}.json"
                    if pf.exists():
                        try:
                            d = json.loads(pf.read_text())
                            total_done += d.get("frames_done", 0)
                        except Exception:
                            pass
                remaining = max(0, total_target - total_done)
                rate = total_done / elapsed if elapsed > 0 else 0
                eta_s = remaining / rate if rate > 0 else float("inf")
                gpu_hrs_elapsed = (elapsed / 3600) * len(procs)
                gpu_hrs_total = (total_target / max(rate, 1e-9) / 3600) * len(procs)
                cost_so_far = gpu_hrs_elapsed * cost_per_gpu_hr
                cost_total_est = gpu_hrs_total * cost_per_gpu_hr
                pct = 100 * total_done / total_target if total_target > 0 else 0
                lines = [
                    f"=== {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} ===",
                    f"Elapsed:      {elapsed/3600:.2f}h",
                    f"Frames done:  {total_done} / {total_target} ({pct:.1f}%)",
                    f"Remaining:    {remaining}",
                    f"Rate:         {rate*3600:.0f} frames/hr",
                    f"ETA:          {eta_s/3600:.1f}h",
                    f"Cost so far:  ${cost_so_far:.2f}",
                    f"Est. total:   ${cost_total_est:.2f}  (target $46)",
                    "",
                ]
                summary = "\n".join(lines)
                print(summary, flush=True)
                plog = out_path / "progress.log"
                with open(plog, "a") as f:
                    f.write(summary)

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        failed = []
        for gpu_idx, proc in procs:
            proc.wait()
            if proc.returncode != 0:
                click.echo(f"  GPU {gpu_idx} worker exited with code {proc.returncode}")
                failed.append(gpu_idx)
        for fd in log_fds:
            fd.close()
        if failed:
            click.echo(f"WARNING: {len(failed)} GPU worker(s) failed: {failed}")
            sys.exit(1)
        click.echo("All GPU workers finished.")
        return

    click.echo(f"Proteins: {len(protein_ids)}")
    click.echo(f"Target timesteps: {target_ts}")
    click.echo(f"Sampling steps: {sampling_steps} (local test uses 10)")
    click.echo(f"Max frames per protein: {n_frames}")
    click.echo(f"Accelerator: {accelerator}")
    click.echo()

    model_module, device = _load_boltz(accelerator)

    total_written = 0
    total_skipped = 0
    frames_done_this_worker = 0
    worker_start = time.time()
    progress_dir = out_path / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    for p_idx, protein_id in enumerate(protein_ids):
        protein_frame_dir = frames_path / protein_id
        protein_label_dir = labels_path / protein_id

        if not protein_frame_dir.exists():
            click.echo(f"[{p_idx+1}/{len(protein_ids)}] {protein_id}: no frames dir, skipping")
            continue

        # Load metadata
        meta_path = protein_frame_dir / "metadata.json"
        if not meta_path.exists():
            click.echo(f"[{p_idx+1}/{len(protein_ids)}] {protein_id}: no metadata.json, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        sequence = meta.get("sequence", "")
        if not sequence:
            click.echo(f"  {protein_id}: empty sequence, skipping")
            continue

        # Setup Boltz input (once per protein — MSA is expensive)
        click.echo(f"[{p_idx+1}/{len(protein_ids)}] {protein_id} ({len(sequence)} residues)")
        try:
            batch = _setup_boltz_input(protein_id, sequence, out_path)
        except Exception as e:
            click.echo(f"  Boltz setup failed: {e}")
            continue

        # Get frame files
        frame_files = sorted(protein_frame_dir.glob("*.npz"))[:n_frames]
        click.echo(f"  {len(frame_files)} frames × {len(target_ts)} timesteps = "
                   f"{len(frame_files) * len(target_ts)} outputs")

        for f_idx, frame_file in enumerate(frame_files):
            frame_idx = int(frame_file.stem)

            # Check if all timesteps already done
            frame_out_dir = out_path / protein_id
            all_exist = all(
                (frame_out_dir / f"{frame_idx:04d}_t{t:.1f}.npz").exists()
                for t in target_ts
            )
            if all_exist:
                total_skipped += len(target_ts)
                click.echo(f"  Frame {frame_idx:04d}: already exists, skipping")
                continue

            # Load frame + labels
            frame_data = np.load(frame_file, allow_pickle=True)
            clean_coords = frame_data["ca_coords"]  # (N_res, 3) CA only

            label_file = protein_label_dir / f"{frame_file.stem}_labels.npz"
            if not label_file.exists():
                click.echo(f"  Frame {frame_idx}: no labels file, skipping")
                continue
            pocket_labels = np.load(label_file)["labels"]

            t0 = time.time()
            try:
                written = _run_capture(
                    model_module=model_module,
                    batch=batch,
                    frame_idx=frame_idx,
                    target_ts=target_ts,
                    out_dir=out_path,
                    protein_id=protein_id,
                    clean_coords=clean_coords,
                    pocket_labels=pocket_labels,
                    sampling_steps=sampling_steps,
                    device=device,
                    seed=42 + frame_idx,
                )
                elapsed = time.time() - t0
                total_written += len(written)
                frames_done_this_worker += 1
                click.echo(f"  Frame {frame_idx:04d}: {len(written)} files in {elapsed:.1f}s")

                # Write per-worker progress for parent monitor
                try:
                    pf = progress_dir / f"gpu{worker_id}.json"
                    pf.write_text(json.dumps({
                        "worker_id": worker_id,
                        "frames_done": frames_done_this_worker,
                        "start_time": worker_start,
                        "last_update": time.time(),
                        "current_protein": protein_id,
                    }))
                except Exception:
                    pass

            except Exception as e:
                click.echo(f"  Frame {frame_idx:04d}: FAILED — {e}")
                import traceback
                traceback.print_exc()
                continue

        # Per-protein done marker
        done_path = out_path / protein_id / "done.json"
        done_path.parent.mkdir(parents=True, exist_ok=True)
        with open(done_path, "w") as f:
            json.dump({
                "protein_id": protein_id,
                "n_frames": len(frame_files),
                "timesteps": target_ts,
                "sampling_steps": sampling_steps,
            }, f)

        click.echo(f"  Done. {total_written} files written, {total_skipped} skipped.")

    click.echo(f"\nTotal: {total_written} noisy frames written.")


if __name__ == "__main__":
    main()
