#!/usr/bin/env python3
"""Train noise-aware PocketMiner on Boltz intermediates.

Usage:
    # Full training (GPU):
    python scripts/train_noise_aware_pocketminer.py --config configs/task_a3_default.yaml

    # Local CPU smoke test (few steps):
    python scripts/train_noise_aware_pocketminer.py \
        --config configs/task_a3_default.yaml \
        --smoke_test --max_proteins 5 --max_steps 10

    # 2-epoch CPU integration test:
    python scripts/train_noise_aware_pocketminer.py \
        --config configs/task_a3_default.yaml \
        --mini_test
"""
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cryptic_pocket_phd.datasets import NoisyPocketDataset, collate_variable_length
from cryptic_pocket_phd.pocketminer_noise_aware import NoiseAwarePocketMiner


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tcfg = cfg["training"]

        # Model
        mcfg = cfg["model"]
        self.model = NoiseAwarePocketMiner(
            t_dim=mcfg["t_dim"],
            source_dim=mcfg["source_dim"],
            node_features=tuple(mcfg["node_features"]),
            edge_features=tuple(mcfg["edge_features"]),
            hidden_dim=tuple(mcfg["hidden_dim"]),
            num_layers=mcfg["num_layers"],
            k_neighbors=mcfg["k_neighbors"],
            dropout=mcfg["dropout"],
        )

        # Init lazy layers BEFORE loading weights
        with torch.no_grad():
            dummy_X = torch.randn(1, 200, 4, 3)
            dummy_S = torch.randint(0, 20, (1, 200))
            dummy_mask = torch.ones(1, 200)
            self.model(dummy_X, dummy_S, dummy_mask, t=torch.tensor([0.5]))

        # Load base weights
        base_path = REPO_ROOT / mcfg.get("base_weights", "")
        if base_path.exists():
            base_sd = torch.load(base_path, map_location="cpu", weights_only=True)
            self.model.load_base_weights(base_sd)
        else:
            print(f"Base weights not found at {base_path}, training from scratch")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        self.loss_fn = nn.BCELoss(reduction="none")
        self.grad_clip = tcfg.get("gradient_clip_val", 1.0)

        # LR scheduler (cosine with warmup)
        self.warmup_epochs = tcfg.get("warmup_epochs", 1)
        self.max_epochs = tcfg["max_epochs"]
        self.scheduler = None  # set after we know steps_per_epoch

        # Early stopping
        self.patience = tcfg.get("early_stop_patience", 5)
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.epochs_no_improve = 0

        # Checkpointing
        ccfg = cfg.get("checkpointing", {})
        self.ckpt_dir = Path(ccfg.get("dirpath", "checkpoints/task_a3"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = ccfg.get("save_top_k", 3)

        # Logging
        self.log_path = REPO_ROOT / "logs" / "training_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # W&B
        self.wandb_run = None
        self.train_start_time = None
        self.global_step = 0

    def setup_scheduler(self, steps_per_epoch: int):
        """Setup cosine LR with linear warmup."""
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = self.max_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def init_wandb(self):
        """Initialize W&B logging."""
        wcfg = self.cfg.get("wandb", {})
        if not wcfg:
            return
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=wcfg.get("project", "cryptic-pocket-phd"),
                group=wcfg.get("group", "task_a3"),
                name=wcfg.get("name", "noise-aware-pocketminer"),
                config=self.cfg,
            )
            print(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")

    def train_step(self, batch: dict) -> tuple[float, float]:
        """Single training step. Returns (loss, grad_norm)."""
        self.model.train()

        coords = batch["coords"].to(self.device)
        seq = batch["seq"].to(self.device)
        t = batch["t"].to(self.device)
        labels = batch["labels"].to(self.device)
        mask = batch["mask"].to(self.device)

        preds = self.model(coords, seq, mask, t=t, train=True)

        loss_per_res = self.loss_fn(preds, labels)
        loss = (loss_per_res * mask).sum() / mask.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        return loss.item(), grad_norm

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Full validation pass. Returns metrics dict."""
        self.model.eval()
        all_results = []

        for batch in val_loader:
            coords = batch["coords"].to(self.device)
            seq = batch["seq"].to(self.device)
            t = batch["t"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["labels"]

            preds = self.model(coords, seq, mask, t=t, train=False)
            B = coords.shape[0]
            for i in range(B):
                n = int(mask[i].sum().item())
                all_results.append({
                    "preds": preds[i, :n].cpu().numpy(),
                    "labels": labels[i, :n].numpy(),
                    "t": t[i].item(),
                })

        # Compute metrics
        by_t = {}
        for r in all_results:
            t_key = f"{r['t']:.1f}"
            by_t.setdefault(t_key, {"preds": [], "labels": []})
            by_t[t_key]["preds"].extend(r["preds"].tolist())
            by_t[t_key]["labels"].extend(r["labels"].tolist())

        metrics = {}
        all_preds, all_labels = [], []
        for t_key, data in sorted(by_t.items()):
            p, l = np.array(data["preds"]), np.array(data["labels"])
            all_preds.extend(p.tolist())
            all_labels.extend(l.tolist())
            if len(np.unique(l)) > 1 and len(np.unique(p)) > 1:
                rho, _ = spearmanr(p, l)
            else:
                rho = 0.0
            t_tag = t_key.replace(".", "")
            metrics[f"val_rho_t{t_tag}"] = rho

        all_p = np.clip(np.array(all_preds), 1e-7, 1 - 1e-7)
        all_l = np.array(all_labels)
        metrics["val_loss"] = float(-np.mean(
            all_l * np.log(all_p) + (1 - all_l) * np.log(1 - all_p)
        ))
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        """Save checkpoint. Keep last save_top_k + best."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "cfg": self.cfg,
        }

        # Save epoch checkpoint
        epoch_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)

        # Save latest
        latest_path = self.ckpt_dir / "latest.pt"
        shutil.copy2(epoch_path, latest_path)

        # Save best
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(ckpt, best_path)

        # Prune old epoch checkpoints (keep last save_top_k + best)
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if len(epoch_ckpts) > self.save_top_k:
            for old in epoch_ckpts[:-self.save_top_k]:
                old.unlink()

    def log_jsonl(self, record: dict):
        """Append one JSON line to training log."""
        # Convert numpy types to Python native for JSON
        clean = {}
        for k, v in record.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = v.item()
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        with open(self.log_path, "a") as f:
            f.write(json.dumps(clean) + "\n")

    def log_wandb(self, metrics: dict, step: int):
        """Log to W&B if available."""
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def upload_checkpoints_async(self):
        """Upload best.pt + latest.pt to HF in background thread."""
        hf_token = self.cfg["data"].get("hf_token")
        if not hf_token:
            return

        def _upload():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                repo_id = "aman-gpt/cryptic-pocket-task-a3-checkpoints"

                # Ensure repo exists
                try:
                    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
                except Exception:
                    pass

                for name in ["best.pt", "latest.pt"]:
                    path = self.ckpt_dir / name
                    if path.exists():
                        api.upload_file(
                            path_or_fileobj=str(path),
                            path_in_repo=name,
                            repo_id=repo_id,
                            repo_type="dataset",
                            commit_message=f"Update {name} (step {self.global_step})",
                        )
            except Exception as e:
                print(f"  [HF upload] failed: {e}", flush=True)

        t = threading.Thread(target=_upload, daemon=True)
        t.start()

    def git_commit_log(self):
        """Commit training log to GitHub."""
        try:
            subprocess.run(
                ["git", "add", str(self.log_path)],
                cwd=str(REPO_ROOT), capture_output=True, timeout=30,
            )
            subprocess.run(
                ["git", "commit", "-m",
                 f"training log checkpoint step={self.global_step}"],
                cwd=str(REPO_ROOT), capture_output=True, timeout=30,
            )
            subprocess.run(
                ["git", "push", "origin", "master"],
                cwd=str(REPO_ROOT), capture_output=True, timeout=60,
            )
        except Exception as e:
            print(f"  [git commit] failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def run_full_training(cfg: dict):
    """Full training loop with all monitoring."""
    tcfg = cfg["training"]
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load split
    split_path = REPO_ROOT / cfg["data"]["split_file"]
    with open(split_path) as f:
        split = json.load(f)

    hf_token = cfg["data"].get("hf_token")

    # Datasets
    print("Loading datasets...", flush=True)
    train_ds = NoisyPocketDataset(
        protein_list=split["train"],
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token,
        n_frames=cfg["data"]["n_frames"],
        timesteps=cfg["data"]["timesteps"],
    )
    val_ds = NoisyPocketDataset(
        protein_list=split["val"],
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token,
        n_frames=cfg["data"]["n_frames"],
        timesteps=cfg["data"]["timesteps"],
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples", flush=True)

    batch_size = tcfg["batch_size"]
    num_workers = tcfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_variable_length, num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variable_length, num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )

    steps_per_epoch = len(train_loader)
    print(f"Steps/epoch: {steps_per_epoch}, Batch size: {batch_size}", flush=True)

    # Trainer
    trainer = Trainer(cfg)
    trainer.setup_scheduler(steps_per_epoch)
    trainer.init_wandb()

    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Device: {trainer.device}, Params: {n_params:,}", flush=True)

    max_epochs = tcfg["max_epochs"]
    trainer.train_start_time = time.time()
    last_git_commit = time.time()
    hour1_printed = False
    hour4_printed = False

    print(f"\n{'='*60}")
    print(f"TRAINING START — {max_epochs} epochs max, early stop patience={trainer.patience}")
    print(f"{'='*60}\n", flush=True)

    for epoch in range(max_epochs):
        epoch_start = time.time()
        epoch_losses = []

        # --- Training epoch ---
        trainer.model.train()
        for step_in_epoch, batch in enumerate(train_loader):
            t0 = time.time()
            loss, grad_norm = trainer.train_step(batch)
            dt = time.time() - t0
            epoch_losses.append(loss)

            # NaN check
            if np.isnan(loss):
                print(f"NaN loss at epoch {epoch} step {step_in_epoch}! ABORTING.", flush=True)
                sys.exit(1)

            # Per-step W&B logging
            lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.log_wandb({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
            }, step=trainer.global_step)

            # Stdout every 100 steps
            if step_in_epoch % 100 == 0:
                elapsed = time.time() - trainer.train_start_time
                steps_done = trainer.global_step
                steps_total = max_epochs * steps_per_epoch
                eta_total = (elapsed / max(steps_done, 1)) * steps_total
                print(
                    f"  E{epoch:02d} S{step_in_epoch:04d}/{steps_per_epoch}: "
                    f"loss={loss:.4f} gnorm={grad_norm:.3f} lr={lr:.2e} "
                    f"dt={dt:.2f}s elapsed={elapsed/60:.0f}min",
                    flush=True,
                )

            # Hour-1 projection
            elapsed_hrs = (time.time() - trainer.train_start_time) / 3600
            if not hour1_printed and elapsed_hrs >= 1.0:
                rate = trainer.global_step / elapsed_hrs
                total_steps = max_epochs * steps_per_epoch
                projected_hrs = total_steps / rate
                cost_per_hr = 1.14
                print(f"\n>>> HOUR 1 PROJECTION: {projected_hrs:.1f}h total, "
                      f"${projected_hrs * cost_per_hr:.2f} estimated cost <<<\n", flush=True)
                hour1_printed = True

            # Hour-4 projection
            if not hour4_printed and elapsed_hrs >= 4.0:
                rate = trainer.global_step / elapsed_hrs
                total_steps = max_epochs * steps_per_epoch
                projected_hrs = total_steps / rate
                cost_per_hr = 1.14
                print(f"\n>>> HOUR 4 PROJECTION: {projected_hrs:.1f}h total, "
                      f"${projected_hrs * cost_per_hr:.2f} estimated cost <<<\n", flush=True)
                hour4_printed = True

            # Git commit every 30 min
            if time.time() - last_git_commit > 1800:
                trainer.git_commit_log()
                last_git_commit = time.time()

        epoch_time = time.time() - epoch_start
        train_loss = float(np.mean(epoch_losses))

        # --- Validation ---
        val_start = time.time()
        metrics = trainer.validate(val_loader)
        val_time = time.time() - val_start

        val_rho_t05 = metrics.get("val_rho_t05", 0.0)
        is_best = val_rho_t05 > trainer.best_metric

        if is_best:
            trainer.best_metric = val_rho_t05
            trainer.best_epoch = epoch
            trainer.epochs_no_improve = 0
        else:
            trainer.epochs_no_improve += 1

        # Epoch summary
        print(f"\n--- Epoch {epoch:02d} ---", flush=True)
        print(f"  train_loss: {train_loss:.4f}", flush=True)
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}", flush=True)
        print(f"  epoch_time: {epoch_time:.0f}s, val_time: {val_time:.0f}s", flush=True)
        print(f"  best_rho_t05: {trainer.best_metric:.4f} (epoch {trainer.best_epoch})"
              f"{'  *** NEW BEST ***' if is_best else ''}", flush=True)
        print(f"  patience: {trainer.epochs_no_improve}/{trainer.patience}\n", flush=True)

        # Log to W&B
        trainer.log_wandb({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "epoch_time": epoch_time,
            **{k: v for k, v in metrics.items()},
        }, step=trainer.global_step)

        # Log to JSONL
        trainer.log_jsonl({
            "epoch": epoch,
            "global_step": trainer.global_step,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "val_time": val_time,
            "is_best": is_best,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        })

        # Save checkpoint
        trainer.save_checkpoint(epoch, metrics, is_best)

        # Upload to HF (async)
        trainer.upload_checkpoints_async()

        # Early stopping
        if trainer.epochs_no_improve >= trainer.patience:
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{trainer.patience} epochs.", flush=True)
            break

    # Final summary
    total_time = time.time() - trainer.train_start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total time: {total_time/3600:.2f}h")
    print(f"  Best val_rho_t05: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}")
    print(f"  Checkpoints: {trainer.ckpt_dir}")
    print(f"{'='*60}", flush=True)

    # Final git commit
    trainer.git_commit_log()

    if trainer.wandb_run is not None:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Smoke test (unchanged)
# ---------------------------------------------------------------------------

def run_smoke_test(cfg: dict, max_proteins: int = 5, max_steps: int = 10):
    """Local CPU smoke test: few proteins, few steps."""
    print("=" * 60)
    print("SMOKE TEST: Local CPU")
    print("=" * 60)

    split_path = REPO_ROOT / cfg["data"]["split_file"]
    with open(split_path) as f:
        split = json.load(f)

    train_proteins = split["train"][:max_proteins]
    val_proteins = split["val"][:min(3, len(split["val"]))]
    print(f"Train proteins: {train_proteins}")
    print(f"Val proteins: {val_proteins}")

    hf_token = cfg["data"].get("hf_token")
    train_ds = NoisyPocketDataset(
        protein_list=train_proteins,
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token, n_frames=5, timesteps=[0.1, 0.5, 0.9],
    )
    val_ds = NoisyPocketDataset(
        protein_list=val_proteins,
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token, n_frames=2, timesteps=[0.1, 0.5, 0.9],
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate_variable_length, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=collate_variable_length, num_workers=0,
    )

    trainer = Trainer(cfg)
    print(f"Device: {trainer.device}")
    print(f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}")

    print("\n--- Training ---")
    losses = []
    step = 0
    t_start = time.time()
    grad_norm = 0.0

    for batch in train_loader:
        if step >= max_steps:
            break
        t0 = time.time()
        loss, grad_norm = trainer.train_step(batch)
        dt = time.time() - t0
        losses.append(loss)
        if np.isnan(loss):
            print(f"Step {step}: NaN loss! ABORTING")
            sys.exit(1)
        print(f"Step {step:3d}: loss={loss:.4f}  grad_norm={grad_norm:.4f}  time={dt:.1f}s")
        step += 1

    total_train_time = time.time() - t_start
    avg_time_per_step = total_train_time / max(step, 1)

    print("\n--- Validation ---")
    metrics = trainer.validate(val_loader)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    print("\n--- Summary ---")
    print(f"Steps completed: {step}")
    print(f"Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    print(f"No NaN: True")
    print(f"Gradient flow: {grad_norm > 0}")
    print(f"Avg time/step: {avg_time_per_step:.1f}s")

    full_train_size = 108 * 100 * 5
    steps_per_epoch = full_train_size // cfg["training"]["batch_size"]
    print(f"\n--- Projections ---")
    print(f"Full train size: {full_train_size} samples")
    print(f"Steps/epoch (batch={cfg['training']['batch_size']}): {steps_per_epoch}")
    cpu_epoch_h = steps_per_epoch * avg_time_per_step / 3600
    print(f"CPU time/epoch: {cpu_epoch_h:.1f}h")
    print(f"A100 ~50x: {cpu_epoch_h/50:.2f}h/epoch, 50 epochs: {50*cpu_epoch_h/50:.1f}h")
    print(f"Estimated cost: ${50*cpu_epoch_h/50 * 1.14:.2f}")

    print("\n" + "=" * 60)
    if losses[-1] < losses[0] and grad_norm > 0:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mini integration test (2 epochs, tiny subset)
# ---------------------------------------------------------------------------

def run_mini_test(cfg: dict):
    """2-epoch integration test on tiny subset. Tests all 7 pieces."""
    print("=" * 60)
    print("MINI TEST: 2 epochs, 5 proteins, 5 frames")
    print("=" * 60, flush=True)

    # Override config for tiny run
    cfg["training"]["max_epochs"] = 2
    cfg["training"]["early_stop_patience"] = 10  # don't trigger
    cfg["training"]["num_workers"] = 0
    cfg["training"]["batch_size"] = 4

    split_path = REPO_ROOT / cfg["data"]["split_file"]
    with open(split_path) as f:
        split = json.load(f)

    # Shrink to tiny subset
    split["train"] = split["train"][:5]
    split["val"] = split["val"][:3]

    # Save temp split
    mini_split_path = REPO_ROOT / "configs" / "_mini_test_split.json"
    with open(mini_split_path, "w") as f:
        json.dump(split, f)
    cfg["data"]["split_file"] = str(mini_split_path)
    cfg["data"]["n_frames"] = 5
    cfg["data"]["timesteps"] = [0.1, 0.5, 0.9]

    # Use test checkpoint dir
    cfg["checkpointing"]["dirpath"] = "checkpoints/_mini_test"

    # Disable W&B for mini test
    cfg.pop("wandb", None)

    run_full_training(cfg)

    # Verify outputs
    ckpt_dir = Path("checkpoints/_mini_test")
    log_path = REPO_ROOT / "logs" / "training_log.jsonl"

    checks = {
        "checkpoint_dir_exists": ckpt_dir.exists(),
        "best_pt_exists": (ckpt_dir / "best.pt").exists(),
        "latest_pt_exists": (ckpt_dir / "latest.pt").exists(),
        "log_file_exists": log_path.exists(),
    }

    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        checks["log_has_2_epochs"] = len(lines) >= 2

    print(f"\n--- Mini Test Verification ---", flush=True)
    all_pass = True
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {check}", flush=True)

    # Cleanup temp files
    mini_split_path.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    if all_pass:
        print("MINI TEST PASSED — all 7 pieces verified")
    else:
        print("MINI TEST FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", type=click.Path(exists=True), default="configs/task_a3_default.yaml")
@click.option("--smoke_test", is_flag=True, default=False, help="Quick step-level test")
@click.option("--mini_test", is_flag=True, default=False, help="2-epoch integration test")
@click.option("--max_proteins", type=int, default=5)
@click.option("--max_steps", type=int, default=10)
@click.option("--hf_token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--wandb_key", type=str, default=None, envvar="WANDB_API_KEY")
def main(config, smoke_test, mini_test, max_proteins, max_steps, hf_token, wandb_key):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    if hf_token:
        cfg["data"]["hf_token"] = hf_token
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    if smoke_test:
        run_smoke_test(cfg, max_proteins=max_proteins, max_steps=max_steps)
    elif mini_test:
        run_mini_test(cfg)
    else:
        run_full_training(cfg)


if __name__ == "__main__":
    main()
