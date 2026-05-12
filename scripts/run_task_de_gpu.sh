#!/usr/bin/env bash
# Task D+E: GPU regression + smoke tests on RunPod A40
# Expected: ~30 min, ~$0.30
# Run: bash scripts/run_task_de_gpu.sh
set -euo pipefail

REPO=/workspace/cryptic-pocket-phd
OUT=$REPO/results/task_de
mkdir -p "$OUT"

cd "$REPO"

# ============================================================
# SETUP
# ============================================================
echo "=== Setup ==="

# Install deps
pip install -q pytorch-lightning fairscale mashumaro biopython rdkit ihm modelcif omegaconf mdtraj pymol-open-source 2>&1 | tail -3

# Ensure Boltz cache
mkdir -p ~/.boltz
if [ ! -f ~/.boltz/boltz1_conf.ckpt ]; then
    echo "Downloading Boltz checkpoint..."
    python -c "
import sys; sys.path.insert(0, 'external/conformix/conformix_boltz/src')
from boltz.run_untwisted import download
from pathlib import Path
download(Path.home() / '.boltz')
"
fi

# Build fpocket if needed
if ! command -v fpocket &>/dev/null; then
    echo "Building fpocket..."
    cd /tmp && git clone https://github.com/Discngine/fpocket.git 2>/dev/null || true
    cd /tmp/fpocket && make -j$(nproc) && cp bin/fpocket /usr/local/bin/
    cd "$REPO"
fi

echo "Setup done."

# ============================================================
# STEP 1: Pipeline regression (RMSD)
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1: Pipeline regression — ConforMix RMSD vs our RMSD"
echo "============================================================"

# 1a. Run ConforMix's original predict
echo "--- 1a: ConforMix original ---"
cd "$REPO"
PYTHONPATH="external/conformix/conformix_boltz/src:src" python -c "
import sys, os
sys.path.insert(0, 'external/conformix/conformix_boltz/src')
from boltz.run_twisted import predict
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(predict, [
    'data/inputs/P62593_TEM1.yaml',
    '--input_cif', 'data/validation_pdbs/1JWP_A.pdb',
    '--out_dir', 'results/task_de/conformix_rmsd',
    '--twist_target_values', '5.0',
    '--twist_strength_values', '15.0',
    '--diffusion_samples', '5',
    '--sampling_steps', '200',
    '--accelerator', 'gpu',
    '--seed', '42',
    '--twist_rmsd_full_sequence',
    '--output_format', 'pdb',
])
print(result.output)
if result.exit_code != 0:
    print(f'EXIT CODE: {result.exit_code}')
    if result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
" 2>&1 | tee "$OUT/step1a_conformix.log"

# 1b. Run our run_phase1.py --bias_type rmsd
echo "--- 1b: Our RMSD wrapper ---"
PYTHONPATH="external/conformix/conformix_boltz/src:src" python scripts/run_phase1.py \
    data/inputs/P62593_TEM1.yaml \
    --input_cif data/validation_pdbs/1JWP_A.pdb \
    --out_dir results/task_de/our_rmsd \
    --bias_type rmsd \
    --twist_target_values 5.0 \
    --twist_strength_values 15.0 \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --accelerator gpu \
    --seed 42 \
    --twist_rmsd_full_sequence \
    --output_format pdb \
    2>&1 | tee "$OUT/step1b_our_rmsd.log"

# 1c. Compare coordinates
echo "--- 1c: Coordinate comparison ---"
PYTHONPATH="external/conformix/conformix_boltz/src:src" python -c "
import numpy as np
import glob
import mdtraj as md

# Find output PDBs from both runs
conformix_pdbs = sorted(glob.glob('results/task_de/conformix_rmsd/**/predictions/**/*.pdb', recursive=True))
our_pdbs = sorted(glob.glob('results/task_de/our_rmsd/**/predictions/**/*.pdb', recursive=True))

print(f'ConforMix outputs: {len(conformix_pdbs)}')
print(f'Our outputs: {len(our_pdbs)}')

if len(conformix_pdbs) == 0 or len(our_pdbs) == 0:
    print('WARNING: No output PDBs found. Check logs.')
else:
    n_compare = min(len(conformix_pdbs), len(our_pdbs))
    all_pass = True
    for i in range(n_compare):
        t1 = md.load(conformix_pdbs[i])
        t2 = md.load(our_pdbs[i])
        if t1.xyz.shape != t2.xyz.shape:
            print(f'Sample {i}: shape mismatch {t1.xyz.shape} vs {t2.xyz.shape}')
            all_pass = False
            continue
        diff = np.abs(t1.xyz - t2.xyz)
        max_diff = diff.max()
        match = np.allclose(t1.xyz, t2.xyz, atol=1e-5)
        print(f'Sample {i}: max_diff={max_diff:.2e} match={match}')
        if not match:
            all_pass = False
    print(f'\\nREGRESSION: {\"PASS\" if all_pass else \"FAIL\"}')" 2>&1 | tee "$OUT/step1c_comparison.log"

# Hard gate
if grep -q "REGRESSION: FAIL" "$OUT/step1c_comparison.log"; then
    echo "REGRESSION FAILED. Stopping."
    exit 1
fi

# ============================================================
# STEP 2: pocket_p smoke test
# ============================================================
echo ""
echo "============================================================"
echo "STEP 2: pocket_p smoke test"
echo "============================================================"

PYTHONPATH="external/conformix/conformix_boltz/src:src" python scripts/run_phase1.py \
    data/inputs/P62593_TEM1.yaml \
    --input_cif data/validation_pdbs/1JWP_A.pdb \
    --out_dir results/task_de/pocket_p \
    --bias_type pocket_p \
    --pocket_residues "190-200,244-263" \
    --twist_target_values 1.0 \
    --twist_strength_values 15.0 \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --accelerator gpu \
    --seed 42 \
    --twist_rmsd_full_sequence \
    --output_format pdb \
    2>&1 | tee "$OUT/step2_pocket_p.log"

# Sanity check output
PYTHONPATH="external/conformix/conformix_boltz/src:src" python -c "
import glob, mdtraj as md, numpy as np
pdbs = sorted(glob.glob('results/task_de/pocket_p/**/predictions/**/*.pdb', recursive=True))
print(f'pocket_p outputs: {len(pdbs)}')
for p in pdbs[:5]:
    t = md.load(p)
    xyz = t.xyz[0]
    print(f'  {p.split(\"/\")[-1]}: {t.top.n_residues} res, '
          f'coord range [{xyz.min():.1f}, {xyz.max():.1f}], '
          f'NaN={np.isnan(xyz).any()}, Inf={np.isinf(xyz).any()}')" 2>&1 | tee "$OUT/step2_sanity.log"

# ============================================================
# STEP 3: pocket_t smoke test
# ============================================================
echo ""
echo "============================================================"
echo "STEP 3: pocket_t smoke test"
echo "============================================================"

PYTHONPATH="external/conformix/conformix_boltz/src:src" python scripts/run_phase1.py \
    data/inputs/P62593_TEM1.yaml \
    --input_cif data/validation_pdbs/1JWP_A.pdb \
    --out_dir results/task_de/pocket_t \
    --bias_type pocket_t \
    --pocket_residues "190-200,244-263" \
    --twist_target_values 0.5 \
    --twist_strength_values 10.0 \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --accelerator gpu \
    --seed 42 \
    --twist_rmsd_full_sequence \
    --output_format pdb \
    2>&1 | tee "$OUT/step3_pocket_t.log"

# Sanity check
PYTHONPATH="external/conformix/conformix_boltz/src:src" python -c "
import glob, mdtraj as md, numpy as np
pdbs = sorted(glob.glob('results/task_de/pocket_t/**/predictions/**/*.pdb', recursive=True))
print(f'pocket_t outputs: {len(pdbs)}')
for p in pdbs[:5]:
    t = md.load(p)
    xyz = t.xyz[0]
    print(f'  {p.split(\"/\")[-1]}: {t.top.n_residues} res, '
          f'coord range [{xyz.min():.1f}, {xyz.max():.1f}], '
          f'NaN={np.isnan(xyz).any()}, Inf={np.isinf(xyz).any()}')" 2>&1 | tee "$OUT/step3_sanity.log"

echo ""
echo "============================================================"
echo "ALL STEPS COMPLETE"
echo "============================================================"
echo "Results in: $OUT/"
echo "Logs: step1a_conformix.log, step1b_our_rmsd.log, step1c_comparison.log"
echo "       step2_pocket_p.log, step2_sanity.log"
echo "       step3_pocket_t.log, step3_sanity.log"
