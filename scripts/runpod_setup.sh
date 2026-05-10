#!/usr/bin/env bash
# RunPod pod setup script.
# Run once after pod start, before run_phase0.py.
#
# Usage:
#   bash scripts/runpod_setup.sh
#
# Requires env vars (set in RunPod pod settings):
#   WANDB_API_KEY   — your W&B API key
#   GITHUB_TOKEN    — optional, only if repo is private
#
# Expected pod image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel (Ubuntu 22.04)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== RunPod setup starting in $REPO_ROOT ==="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/6] System packages ---"
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential git curl wget \
    2>/dev/null

# fpocket: build from source (not in apt on Ubuntu 22.04)
if ! command -v fpocket &>/dev/null; then
    echo "Building fpocket from source..."
    FPOCKET_TMP=$(mktemp -d)
    git clone --depth 1 https://github.com/Discngine/fpocket.git "$FPOCKET_TMP/fpocket"
    cd "$FPOCKET_TMP/fpocket"
    make -j"$(nproc)" 2>&1 | tail -5
    cp bin/fpocket /usr/local/bin/fpocket
    cd "$REPO_ROOT"
    rm -rf "$FPOCKET_TMP"
    echo "fpocket installed: $(fpocket --version 2>&1 | head -1)"
else
    echo "fpocket already available: $(fpocket --version 2>&1 | head -1)"
fi

# ---------------------------------------------------------------------------
# 2. Python env via uv
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/6] Python environment (uv) ---"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

cd "$REPO_ROOT"
uv sync

# ---------------------------------------------------------------------------
# 3. PocketMiner submodule + weights
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/6] PocketMiner weights ---"

# Clone pocketminer if not present
if [ ! -d "$REPO_ROOT/external/pocketminer/src" ]; then
    echo "Cloning PocketMiner (Mickdub/gvp, pocket_pred branch)..."
    git clone --branch pocket_pred --depth 1 \
        https://github.com/Mickdub/gvp.git \
        "$REPO_ROOT/external/pocketminer"
fi

bash "$REPO_ROOT/scripts/download_pocketminer_weights.sh"

# ---------------------------------------------------------------------------
# 4. Boltz-1 model cache
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/6] Boltz-1 cache (~3.7 GB, one-time download) ---"
BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/.boltz}"

if [ -f "$BOLTZ_CACHE/boltz1_conf.ckpt" ] && [ -f "$BOLTZ_CACHE/ccd.pkl" ]; then
    echo "Boltz cache already present at $BOLTZ_CACHE/"
else
    echo "Downloading Boltz cache to $BOLTZ_CACHE/ ..."
    uv run python -c "
from boltz.main import download_boltz1
from pathlib import Path
download_boltz1(Path('$BOLTZ_CACHE'))
print('Boltz cache download complete.')
"
fi

# ---------------------------------------------------------------------------
# 5. W&B login
# ---------------------------------------------------------------------------
echo ""
echo "--- [5/6] W&B login ---"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set. W&B logging will be disabled."
else
    uv run python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"
    echo "W&B login OK."
fi

# ---------------------------------------------------------------------------
# 6. Smoke test: verify imports
# ---------------------------------------------------------------------------
echo ""
echo "--- [6/6] Smoke test ---"
uv run python -c "
import sys
sys.path.insert(0, 'src')
from cryptic_pocket_phd.pocketminer_wrapper import score as pm_score
from cryptic_pocket_phd.fpocket_wrapper import score as fp_score
from cryptic_pocket_phd.correlation import compute_rho
print('All imports OK.')
"

echo ""
echo "=== Setup complete. Run: ==="
echo "  uv run python scripts/run_phase0.py --out_dir results/phase0_runpod"
