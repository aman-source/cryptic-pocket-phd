#!/usr/bin/env bash
# One-command RunPod setup. Run once after pod starts.
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

REPO=/workspace/cryptic-pocket-phd
cd "$REPO"

echo "=== Configuring git identity ==="
git config user.email "aman@runpod"
git config user.name "aman-runpod"
# For checkpoint push: set GITHUB_PAT env var before running setup
# Usage: GITHUB_PAT=ghp_xxx bash scripts/setup_runpod.sh
if [ -n "${GITHUB_PAT:-}" ]; then
  git remote set-url origin "https://aman-source:${GITHUB_PAT}@github.com/aman-source/cryptic-pocket-phd.git"
  echo "Git push auth configured via PAT"
else
  echo "WARNING: No GITHUB_PAT set — checkpoint push will fail"
  echo "Set it: export GITHUB_PAT=ghp_xxx"
fi

echo "=== Cloning ConforMix (if needed) ==="
if [ ! -d "external/conformix/conformix_boltz" ]; then
  git clone https://github.com/drorlab/conformix.git external/conformix
  cd external/conformix && git checkout d0fd34c && cd "$REPO"
  echo "ConforMix cloned at d0fd34c"
else
  echo "ConforMix already present"
fi

echo "=== Installing deps ==="
pip install -q -r requirements-runpod.txt 2>&1 | tail -5

echo "=== Verifying critical imports ==="
PYTHONPATH="external/conformix/conformix_boltz/src:src" python -c "
import numpy; print(f'numpy {numpy.__version__}')
import rdkit; print(f'rdkit {rdkit.__version__}')
import torch; print(f'torch {torch.__version__}')
import pymol; print('pymol ok')
from boltz.model.model import Boltz1; print('Boltz1 ok')
from boltz.run_twisted import predict; print('ConforMix predict ok')
from cryptic_pocket_phd.pocketminer_torch import PocketMinerTorch; print('PocketMiner ok')
from cryptic_pocket_phd.pocket_potential import PocketPotential; print('PocketPotential ok')
print('ALL IMPORTS OK')
"

echo "=== Downloading Boltz checkpoint ==="
mkdir -p ~/.boltz
PYTHONPATH="external/conformix/conformix_boltz/src" python -c "
from boltz.run_untwisted import download
from pathlib import Path
download(Path.home() / '.boltz')
"

echo "=== Verifying CCD pickle ==="
python -c "import pickle; ccd=pickle.load(open('$HOME/.boltz/ccd.pkl','rb')); print(f'CCD: {len(ccd)} entries')"

echo "=== Setup complete ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
