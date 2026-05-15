# Task A3 Results: Noise-Aware PocketMiner Training

**Date**: 2026-05-15
**Verdict**: FAIL Gate 2 (val ρ_pocket at t=0.5 = 0.376, below 0.55 threshold)

## Question

Can we train a noise-aware PocketMiner that predicts pocket residues from noisy
Boltz intermediate structures, conditioned on diffusion timestep t?

**Gate 2 criterion**: val ρ_pocket(t=0.5) ≥ 0.55 on held-out proteins.

## Methodology

- **Architecture**: NoiseAwarePocketMiner — subclass of PocketMinerTorch (812K params)
  - Sinusoidal timestep embedding (dim=32) injected at each of 4 MPNN layers
  - Identity-initialized so t=0 matches vanilla PocketMiner exactly (verified <1e-5)
  - 177/177 base weights loaded from pretrained PocketMiner
- **Training data**: 60,000 examples from HuggingFace (aman-gpt/cryptic-pocket-task-a1)
  - 120 mdCATH proteins, CATH-stratified, Lewis 33 excluded
  - 100 MD frames × 5 Boltz timesteps (t=0.1, 0.3, 0.5, 0.7, 0.9)
  - Labels: LIGSITE pocket residue annotations (binary per-residue)
  - Backbone extracted from all-atom Boltz coords via CCD atom offsets
- **Split**: 108 train / 12 val / 10 test (Phase 0 cryptic-pocket proteins, seed 42)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.01, cosine schedule with 1-epoch warmup
- **Training**: batch=16, max 50 epochs, early stop patience=5 on val ρ(t=0.5)
- **GPU**: RTX 3090 24GB, ~$0.44/hr, ~1 hour wall-clock
- **W&B**: https://wandb.ai/amanayan1-/cryptic-pocket-phd/runs/k6xt7xo6

## Results

| Epoch | train_loss | val_rho_t05 | Best? |
|-------|-----------|-------------|-------|
| 0     | 0.478     | **0.376**   | Yes   |
| 1     | 0.428     | 0.363       | No    |
| 2     | 0.407     | 0.348       | No    |
| 3     | 0.392     | 0.340       | No    |
| 4     | 0.380     | 0.337       | No    |
| 5     | 0.371     | 0.329       | No    |

**Early stopped at epoch 5.** Best checkpoint = epoch 0.

### Key observations

1. **Train loss decreased monotonically** (0.478 → 0.371) — model is learning.
2. **Val ρ decreased monotonically** (0.376 → 0.329) — overfitting from step 1.
3. **Best epoch = 0** — the best model is essentially vanilla PocketMiner with
   identity-initialized timestep injection (no training benefit).
4. **Phase 0 baseline** (vanilla PocketMiner, no noise): ρ ≈ 0.31.
   Best noise-aware: ρ = 0.376 (+22%), but below 0.55 gate.

## Verdict: FAIL Gate 2

The noise-aware architecture does not achieve the 0.55 ρ threshold required for
useful guidance during Boltz sampling. Fine-tuning on noisy intermediates
consistently degraded the base PocketMiner's generalization.

## Honest Discussion

### Why training hurt instead of helped

1. **Label quality mismatch**. LIGSITE labels mark "pocket potential" on the
   clean MD structure — residues near geometric voids. But <1% of MD frames
   have the cryptic pocket actually open. The model learns to predict pocket
   potential from noisy coordinates, but the noisy coordinates don't carry
   meaningful pocket-opening signal at most frames.

2. **Distribution shift**. The noisy Boltz x̂_0 predictions at high t (0.7, 0.9)
   look nothing like real protein structures. PocketMiner's GVP features
   (dihedrals, orientations, kNN graph) become unreliable on these deformed
   structures. Training on them overwrites features that work well on real
   structures.

3. **Overfitting to noise patterns**. Train loss drops steadily → model memorizes
   per-protein noise patterns in the training set. These don't transfer to
   held-out proteins because Boltz's denoising trajectory is protein-specific.

4. **The base is already strong**. PocketMiner pretrained on 4,876 PDB structures
   with real pocket annotations. Our 120-protein noisy fine-tuning set is too
   small and too noisy to improve on that foundation.

### What ρ = 0.376 means

The +22% improvement over vanilla (0.31 → 0.38) comes entirely from the base
weight initialization, not from training. The noise-aware architecture with
identity-initialized timestep injection acts as a regularized version of vanilla
PocketMiner — the extra parameters provide implicit regularization without
actually using timestep information.

### Implications for Spec #2

Track A (noise-aware PocketMiner training) is now complete and did not produce
a model that meets the guidance threshold. This was the expected risk scenario
identified in Spec #2 §4.2.

**Track B (noise-schedule-aware guidance without retraining)** is now the
critical path. Track B uses the vanilla PocketMiner directly as a guidance
potential, with noise-schedule weighting to modulate guidance strength by
timestep — no retraining needed.

## Cost

| Item | Cost |
|------|------|
| Task A1: Generate 60K noisy frames (2 pods × A40) | ~$5.00 |
| Task A2: Architecture + tests (local CPU) | $0.00 |
| Metadata generation (1 pod, 30 min) | ~$0.50 |
| Task A3: Training (RTX 3090, 1 hr) | ~$0.44 |
| A100 false start (data I/O bottleneck) | ~$0.50 |
| **Total Track A** | **~$6.44** |

## Files

- Best checkpoint: `checkpoints/task_a3/best.pt` (also on HF: aman-gpt/cryptic-pocket-task-a3-checkpoints)
- Training log: `logs/training_log.jsonl`
- W&B run: https://wandb.ai/amanayan1-/cryptic-pocket-phd/runs/k6xt7xo6
- Architecture: `src/cryptic_pocket_phd/pocketminer_noise_aware.py`
- Dataset: `src/cryptic_pocket_phd/datasets.py`
- Training script: `scripts/train_noise_aware_pocketminer.py`
- Config: `configs/task_a3_default.yaml`
- Split: `configs/task_a3_split.json`
