#!/usr/bin/env bash
# Phase 0 two-phase pipeline: Boltz GPU inference then PocketMiner CPU scoring.
# Run from repo root.
set -euo pipefail

OUT_DIR="${1:-results/phase0_runpod}"

echo "=== Phase 0 pipeline ==="
echo "Output: $OUT_DIR"
echo ""

echo "--- Phase 1: Boltz inference (GPU, no TF) ---"
.venv/bin/python -u scripts/run_phase0_boltz.py --out_dir "$OUT_DIR"

echo ""
echo "--- Phase 2: PocketMiner + fpocket scoring (CPU) ---"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u scripts/run_phase0_pocketminer.py --out_dir "$OUT_DIR"

echo ""
echo "=== Done. Results in $OUT_DIR/results/ ==="
