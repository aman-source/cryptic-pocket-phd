# cryptic-pocket-phd

PocketMiner-guided diffusion for cryptic pockets — PhD project code.

## Phase 0 — Noise sanity check

Does PocketMiner give signal on noisy diffusion states?

See `docs/SPEC_00_pocketminer_noise_sanity.md` for full spec.

## Setup

```bash
uv sync
```

### Boltz cache (one-time, ~4 GB)

Boltz-1 requires a CCD dictionary and model weights, downloaded once to `~/.boltz`.
Run from repo root (or any directory):

```bash
python -c "
from boltz.main import download_boltz1
from pathlib import Path
download_boltz1(Path.home() / '.boltz')
"
```

Files created:
- `~/.boltz/ccd.pkl`         — CCD dictionary (330 MB)
- `~/.boltz/boltz1_conf.ckpt` — model weights (3.4 GB)

To use a different cache directory:

```bash
export BOLTZ_CACHE=/path/to/your/cache
```

The `run_phase0.py` script reads `BOLTZ_CACHE` automatically, falling back to `~/.boltz`.

### PocketMiner weights

```bash
cd external/pocketminer
git checkout -- models/pocketminer.*   # if missing after Windows clone
```

## Running Phase 0

Local CPU test (NPC2 only, fast settings — verifies pipeline before RunPod):

```bash
python scripts/run_phase0.py \
  --proteins P79345 \
  --timesteps 0.5 0.9 \
  --samples 1 \
  --sampling_steps 20 \
  --recycling_steps 1 \
  --out_dir results/phase0_local_test
```

Full experiment (all 10 proteins, 5 samples, 5 timesteps — use RunPod GPU):

```bash
python scripts/run_phase0.py --out_dir results/phase0_runpod
```

Results written to `<out_dir>/results/phase0_rho.csv` and `phase0_rho_aggregate.csv`.

## W&B

Project: `cryptic-pocket-phd`, run group: `phase0_sanity`.

Set `WANDB_API_KEY` in your environment (never commit it).
