#!/usr/bin/env python3
"""Train noise-aware PocketMiner on Boltz intermediates.

Usage:
    # Full training (GPU):
    python scripts/train_noise_aware_pocketminer.py --config configs/task_a3_default.yaml

    # Local CPU smoke test:
    python scripts/train_noise_aware_pocketminer.py \
        --config configs/task_a3_default.yaml \
        --smoke_test --max_proteins 5 --max_steps 10
"""
import json
import sys
import time
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


class NoiseAwarePocketMinerLit:
    """Lightweight training loop (no Lightning dependency for smoke test)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Init lazy layers BEFORE loading weights (GVP uses lazy init)
        with torch.no_grad():
            dummy_X = torch.randn(1, 200, 4, 3)
            dummy_S = torch.randint(0, 20, (1, 200))
            dummy_mask = torch.ones(1, 200)
            self.model(dummy_X, dummy_S, dummy_mask, t=torch.tensor([0.5]))

        # Load base weights after lazy init
        base_path = REPO_ROOT / mcfg.get("base_weights", "")
        if base_path.exists():
            base_sd = torch.load(base_path, map_location="cpu", weights_only=True)
            self.model.load_base_weights(base_sd)
        else:
            print(f"Base weights not found at {base_path}, training from scratch")

        self.model.to(self.device)

        # Optimizer
        tcfg = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        self.loss_fn = nn.BCELoss(reduction="none")
        self.grad_clip = tcfg.get("gradient_clip_val", 1.0)

    def train_step(self, batch: dict) -> float:
        """Single training step. Returns loss value."""
        self.model.train()

        coords = batch["coords"].to(self.device)   # (B, N, 4, 3)
        seq = batch["seq"].to(self.device)          # (B, N)
        t = batch["t"].to(self.device)              # (B,)
        labels = batch["labels"].to(self.device)    # (B, N)
        mask = batch["mask"].to(self.device)        # (B, N)

        # Forward
        preds = self.model(coords, seq, mask, t=t, train=True)  # (B, N)

        # Masked BCE loss
        loss_per_res = self.loss_fn(preds, labels)
        loss = (loss_per_res * mask).sum() / mask.sum().clamp(min=1)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clip
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        """Validation step. Returns per-sample predictions and labels."""
        self.model.eval()

        coords = batch["coords"].to(self.device)
        seq = batch["seq"].to(self.device)
        t = batch["t"].to(self.device)
        mask = batch["mask"].to(self.device)
        labels = batch["labels"]

        preds = self.model(coords, seq, mask, t=t, train=False)

        # Collect per-sample results for rho computation
        results = []
        B = coords.shape[0]
        for i in range(B):
            n = int(mask[i].sum().item())
            results.append({
                "preds": preds[i, :n].cpu().numpy(),
                "labels": labels[i, :n].numpy(),
                "t": t[i].item(),
            })
        return results

    def compute_val_metrics(self, all_results: list[dict]) -> dict:
        """Compute val loss and per-timestep Spearman rho."""
        # Group by timestep band
        by_t = {}
        for r in all_results:
            t_key = f"{r['t']:.1f}"
            by_t.setdefault(t_key, {"preds": [], "labels": []})
            by_t[t_key]["preds"].extend(r["preds"].tolist())
            by_t[t_key]["labels"].extend(r["labels"].tolist())

        metrics = {}
        all_preds, all_labels = [], []

        for t_key, data in sorted(by_t.items()):
            p = np.array(data["preds"])
            l = np.array(data["labels"])
            all_preds.extend(p.tolist())
            all_labels.extend(l.tolist())

            if len(np.unique(l)) > 1 and len(np.unique(p)) > 1:
                rho, _ = spearmanr(p, l)
            else:
                rho = 0.0

            t_tag = t_key.replace(".", "")
            metrics[f"val_rho_t{t_tag}"] = rho

        # Overall val loss
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        eps = 1e-7
        all_preds_clipped = np.clip(all_preds, eps, 1 - eps)
        val_loss = -np.mean(
            all_labels * np.log(all_preds_clipped)
            + (1 - all_labels) * np.log(1 - all_preds_clipped)
        )
        metrics["val_loss"] = val_loss

        return metrics


def run_smoke_test(cfg: dict, max_proteins: int = 5, max_steps: int = 10):
    """Local CPU smoke test: few proteins, few steps."""
    print("=" * 60)
    print("SMOKE TEST: Local CPU")
    print("=" * 60)

    # Load split
    split_path = REPO_ROOT / cfg["data"]["split_file"]
    with open(split_path) as f:
        split = json.load(f)

    train_proteins = split["train"][:max_proteins]
    val_proteins = split["val"][:min(3, len(split["val"]))]

    print(f"Train proteins: {train_proteins}")
    print(f"Val proteins: {val_proteins}")

    # Datasets (only 5 frames, 3 timesteps for speed)
    hf_token = cfg["data"].get("hf_token") or None
    train_ds = NoisyPocketDataset(
        protein_list=train_proteins,
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token,
        n_frames=5,
        timesteps=[0.1, 0.5, 0.9],
    )
    val_ds = NoisyPocketDataset(
        protein_list=val_proteins,
        hf_repo=cfg["data"]["hf_repo"],
        cache_dir=cfg["data"]["cache_dir"],
        hf_token=hf_token,
        n_frames=2,
        timesteps=[0.1, 0.5, 0.9],
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

    # Trainer
    trainer = NoiseAwarePocketMinerLit(cfg)
    print(f"Device: {trainer.device}")
    print(f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Training steps
    print("\n--- Training ---")
    losses = []
    step = 0
    t_start = time.time()

    for batch in train_loader:
        if step >= max_steps:
            break

        t0 = time.time()
        loss = trainer.train_step(batch)
        dt = time.time() - t0
        losses.append(loss)

        # Check for NaN
        if np.isnan(loss):
            print(f"Step {step}: NaN loss! ABORTING")
            sys.exit(1)

        # Check gradient flow
        grad_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        print(f"Step {step:3d}: loss={loss:.4f}  grad_norm={grad_norm:.4f}  time={dt:.1f}s")
        step += 1

    total_train_time = time.time() - t_start
    avg_time_per_step = total_train_time / max(step, 1)

    # Validation
    print("\n--- Validation ---")
    all_val_results = []
    for batch in val_loader:
        results = trainer.val_step(batch)
        all_val_results.extend(results)

    metrics = trainer.compute_val_metrics(all_val_results)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Summary
    print("\n--- Summary ---")
    print(f"Steps completed: {step}")
    print(f"Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    print(f"No NaN: True")
    print(f"Gradient flow: {grad_norm > 0}")
    print(f"Avg time/step: {avg_time_per_step:.1f}s")

    # Projections
    full_train_size = 108 * 100 * 5  # 108 proteins × 100 frames × 5 timesteps
    steps_per_epoch = full_train_size // cfg["training"]["batch_size"]
    print(f"\n--- Projections ---")
    print(f"Full train size: {full_train_size} samples")
    print(f"Steps/epoch (batch={cfg['training']['batch_size']}): {steps_per_epoch}")
    print(f"CPU time/epoch: {steps_per_epoch * avg_time_per_step / 3600:.1f}h")
    print(f"A100 speedup ~50x: {steps_per_epoch * avg_time_per_step / 3600 / 50:.2f}h/epoch")
    print(f"50 epochs on A100: {50 * steps_per_epoch * avg_time_per_step / 3600 / 50:.1f}h")
    cost_per_hour = 1.14  # A100 SXM 80GB community
    total_hours = 50 * steps_per_epoch * avg_time_per_step / 3600 / 50
    print(f"Estimated cost: ${total_hours * cost_per_hour:.2f}")

    print("\n" + "=" * 60)
    if losses[-1] < losses[0] and grad_norm > 0:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        sys.exit(1)


@click.command()
@click.option("--config", type=click.Path(exists=True), default="configs/task_a3_default.yaml")
@click.option("--smoke_test", is_flag=True, default=False)
@click.option("--max_proteins", type=int, default=5)
@click.option("--max_steps", type=int, default=10)
@click.option("--hf_token", type=str, default=None, envvar="HF_TOKEN")
def main(config, smoke_test, max_proteins, max_steps, hf_token):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    if hf_token:
        cfg["data"]["hf_token"] = hf_token

    if smoke_test:
        run_smoke_test(cfg, max_proteins=max_proteins, max_steps=max_steps)
        return

    # Full training — TODO: implement with PyTorch Lightning
    print("Full training not yet implemented. Use --smoke_test for now.")
    sys.exit(1)


if __name__ == "__main__":
    main()
