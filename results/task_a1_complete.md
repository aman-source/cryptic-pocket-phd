# Task A1 Complete — Noisy Boltz Intermediates for Noise-Aware PocketMiner

## Status: COMPLETE

## What Was Built
Spec #2 Task A1: generate noisy protein structure intermediates from Boltz-1 diffusion
process, labelled with pocket proximity, for training a noise-aware PocketMiner.

## Dataset

| Item | Value |
|------|-------|
| HuggingFace repo | `aman-gpt/cryptic-pocket-task-a1` (private) |
| Total npz files | 60,240 |
| Proteins | 120 (stratified from mdCATH) |
| Frames per protein | 100 (uniformly sampled from MD trajectory, skip first 10 ns) |
| Timesteps per frame | 5 (t = 0.1, 0.3, 0.5, 0.7, 0.9) |
| Files per protein | 500 (100 frames × 5 timesteps) |
| Training examples | 60,000 (120 × 100 × 5) |

## Protein Selection

- Strategy: 4 CATH classes × 4 length bins = 16 cells, ~7-8 proteins/cell
- Source: mdCATH (compsciencelab/mdCATH on HuggingFace)
- Exclusions: Lewis 33 proteins + homologs, unknown CATH class, length <50 or >500 residues
- Seed: 42 (deterministic)
- GPU split: 4 buckets (pod1_gpu0, pod1_gpu1, pod2_gpu0, pod2_gpu1), 30 proteins each
- Full list: `data/protein_lists/task_a1_stratified_120.txt`
- Selection report: `data/protein_lists/selection_report.md`

## Per-NPZ Contents

Each `.npz` file contains:
- `coords`: full atom coordinates (n_atoms, 3), float32, Angstroms
- `ca_coords`: Calpha only coordinates (n_residues, 3), float32
- `sequence`: amino acid sequence string
- `frame_index`: original frame index in MD trajectory
- `protein_id`: domain ID string
- `noisy_ca_coords`: Calpha at Boltz diffusion timestep t
- `t`: diffusion timestep (0.1, 0.3, 0.5, 0.7, or 0.9)
- `pocket_label`: per-residue binary label (1=near cryptic pocket void, 0=not)

## Pipeline

1. **Stratified selection**: `scripts/select_stratified_proteins.py`
2. **Streaming download + preprocess**: `scripts/pipeline_download_preprocess.py`
   - Downloads mdCATH H5 (1 at a time), extracts 100 frames, deletes H5
   - `--n_frames 100 --skip_ns 10 --temperature 320`
3. **Boltz intermediate generation**: `scripts/generate_boltz_intermediates.py`
   - Monkey-patches `AtomDiffusion.sample()` to capture 5 timesteps
   - ~8s/frame on A40
4. **HF upload**: `scripts/hf_upload_task_a1.py`
   - Per-protein upload loop (avoids 25k file/commit limit)

## Compute

| Item | Value |
|------|-------|
| Hardware | 2 RunPod pods × 2 A40 GPUs each = 4 GPUs |
| Boltz rate | ~8s/frame |
| GPU time | ~6.7h per GPU for Boltz alone |
| Total cost | See OPERATIONAL_LOG.md (Task A1 row) |

## Verification

- HF repo confirmed: 60,240 npz files, 121 dirs (120 proteins + `boltz_processed` artifact)
- Per-protein spot-check: 1b5tA00=500 files, 1bifA01=500 files, 1bo7A00=500 files
- HF last modified: 2026-05-14 22:32:42 UTC

## Next Step

Task A2: Noise-aware PocketMiner architecture — train on this dataset.
Input: noisy_ca_coords + t → output: per-residue pocket probability.
