# Handoff Document — Cryptic Pocket PhD Project

**Last updated:** 2026-05-12
**Author:** Claude Opus 4.6 (via Claude Code)
**Owner:** Aman Shaik
**Repo:** github.com/aman-source/cryptic-pocket-phd
**Latest commit:** `8ac9de2` (fix: convert mdtraj nm to Angstroms)

---

## 1. Project Overview

PhD thesis: **PocketMiner-guided diffusion for cryptic pocket sampling.**

Core idea: Replace ConforMix's RMSD-based guidance in Boltz-1's twisted SMC diffusion sampler with PocketMiner's learned pocket probability as the steering signal. If PocketMiner guidance produces better cryptic pocket coverage than RMSD guidance, that's Paper 1.

### Key Papers/Tools
- **Boltz-1**: Protein structure prediction diffusion model (like AlphaFold3). github.com/jwohlwend/boltz
- **ConforMix**: Twisted Sequential Monte Carlo (SMC) on top of Boltz-1 for conformational sampling. github.com/drorlab/conformix. Uses RMSD-from-apo as guidance signal.
- **PocketMiner**: GVP-GNN that predicts per-residue cryptic pocket probability. github.com/Mickdub/gvp branch pocket_pred. Originally TensorFlow.
- **Lewis et al. benchmark**: 31 proteins with known cryptic pockets. The evaluation set.

### Two Specs
1. **Spec #0** (DONE): "Does PocketMiner give signal on noisy diffusion states?" Result: weak yes (ρ≈0.31). Located at `docs/SPEC_00_pocketminer_noise_sanity.md`.
2. **Spec #1** (IN PROGRESS): "Does PocketMiner-as-guidance beat ConforMix's 0.45 worst-matched coverage?" Located at `SPEC_01_pocketminer_guidance.md` in PhD directory (not in repo).

---

## 2. What's Been Built (Spec #1 Implementation)

### Task A: PocketMiner TF→PyTorch Port — DONE

**File:** `src/cryptic_pocket_phd/pocketminer_torch.py`

Ported the full GVP-GNN architecture (MQAModel) from TensorFlow to PyTorch. This was necessary because:
- TF 2.18 grabs all GPU memory via XLA at import time, blocking PyTorch/Boltz
- Gradient-based guidance requires `torch.autograd.grad` flowing through PocketMiner
- Subprocess-per-call would be too slow (200 calls per diffusion trajectory)

**Key details:**
- Architecture: GVP (Geometric Vector Perceptron) layers with split vector/scalar representations
- `_split(x, nv)` separates first 3*nv values as vectors [..., 3, nv], rest as scalars
- `_merge(v, s)` flattens vectors and concatenates with scalars
- Lazy layer initialization (`_build` called on first forward pass) — needed because input dims depend on runtime shapes
- **Critical bug found:** TF `LayerNormalization` defaults to `eps=1e-3`, PyTorch `LayerNorm` defaults to `eps=1e-5`. All LayerNorms in the port use `eps=1e-3`.
- **Critical bug found:** `nn.Dropout` in the dense head uses PyTorch's `.training` flag, not our custom `train` parameter. Must call `model.eval()` before inference.

**Validation:** 1e-4 per-residue match on all 9 Phase 0 proteins. Max diff 1.3e-5.

**Weight conversion:** `convert_tf_to_pytorch()` loads TF checkpoint, maps weights layer-by-layer (transposing Dense weights from (in,out) to (out,in) for PyTorch Linear). Saved state dict: `models/pocketminer_torch.pt` (2.9 MB).

**Validation script:** `scripts/validate_pocketminer_port.py` — runs 3 tests: numerical match, SO(3) equivariance, batched inference benchmark.

### Task B: Pocket Potentials g_p and g_t — DONE

**File:** `src/cryptic_pocket_phd/pocket_potential.py`

Two differentiable potentials for ConforMix's twisted SMC:

```python
# g_p: sum-based, rewards open pockets (no sweep target)
log_potential = beta * sum(PocketMiner(x)[pocket_residues])

# g_t: sweep-based, pulls toward target pocket score
log_potential = -alpha * (mean(PocketMiner(x)[pocket_residues]) - target_t)^2
```

**Key design decision: Real backbone, not pseudo-backbone.**

PocketMiner needs N/CA/C/O backbone coordinates. Initially used CA + fixed offsets for N/C/O (pseudo-backbone). This lost 13.7% accuracy on holo structures and compressed the apo-holo guidance gap from 44% to 19%. Switched to extracting real N/CA/C/O from Boltz's all-atom output.

Boltz stores atoms per residue in CCD order: N(0), CA(1), C(2), O(3), CB(4), ... Confirmed for all standard amino acids via `boltz.data.const.ref_atoms`.

`build_bb_atom_indices(atom_to_token, n_tokens)` builds a `[N_tokens, 4]` tensor mapping each residue to its 4 backbone atom indices in the global atom array.

`_extract_backbone(x_all_atom, bb_atom_indices, ...)` indexes `x_all_atom[:, bb_flat, :]` to get `[P, N_tokens, 4, 3]` backbone coords. Standard PyTorch indexing preserves autograd.

**TEM-1 validation:**
- Apo pocket mean: 0.471, Holo pocket mean: 0.678 (1.44x ratio)
- Real backbone matches direct PocketMiner within 1e-4
- Gradient norm: 34.9 (nonzero, meaningful)

### Task C: ConforMix Integration — DONE

**Files:**
- `src/cryptic_pocket_phd/guidance_injection.py` — `pocket_twist_fn` factory
- `scripts/run_phase1.py` — standalone runner

**How ConforMix's SMC works (critical to understand):**

1. `ConformixAtomDiffusion.sample_twisted()` (diffusion.py:731) creates a `TwistedDDPM` sampler
2. `TwistedDDPM` (mg_wrapper.py:41) runs Feynman-Kac particle filter via `smc_FK` (feynman_kac_pf.py:6)
3. At each diffusion step, `_compute_twisted_step` (mg_wrapper.py:354):
   - Splits particles into batches of `batch_p=25` (hardcoded at diffusion.py:806)
   - For each batch: denoises via Boltz, calls `classifier_prob_fn` (our twist_fn) to get potential + gradient
   - Gradient steers the proposal distribution mean
4. The `classifier_prob_fn` interface:
   ```python
   def inner(xt, x0_hat, return_grad=True, t=None, atom_mask=None):
       # xt: [P_batch, N_atoms, 3] — noised coords, requires_grad
       # x0_hat: [P_batch, N_atoms, 3] — Boltz's predicted clean structure
       # atom_mask: [P_FULL, N_atoms] — NOT batched (ConforMix bug, see below)
       # Returns: (log_potential [P_batch], gradient [P_batch, N_atoms, 3])
   ```

**ConforMix bug discovered:** At mg_wrapper.py:385, `self.atom_mask` (full `[P, N]`) is passed to the twist_fn, but `xt_batch` is only `[batch_p, N, 3]`. This is a latent bug in ConforMix that only triggers when `P > batch_p`. ConforMix always runs with P=5, batch_p=25, so no splitting occurs. Our sweep uses P=50, batch_p=25 → splitting → crash. Fix: `atom_mask[:P_batch]` in our twist_fn. All rows are identical (created by `repeat_interleave`), so slicing is correct.

**RMSD delegation:** ConforMix's `twist_fn` is a closure nested inside `predict()` which imports PyMOL at module level. Not importable on Windows. When PyMOL is available (RunPod), we delegate to `ConforMix.predict.callback()` directly. When not, we use a verbatim copy (provenance: d0fd34c:887-976) of the RMSD twist logic.

**RMSD regression test:** Byte-identical (diff=0.0) output coords between ConforMix's original and our wrapper. 5 samples on TEM-1, seed=42.

### Task D+E: GPU Smoke Tests — DONE

Ran on RunPod A40. Results:
- **RMSD regression:** 5/5 samples exact match (diff=0.0)
- **pocket_p** (TEM-1, β=1.0, 5 samples): ESS 1.0-5.0, no crash, coords look like proteins
- **pocket_t** (TEM-1, target=0.5, α=10, 5 samples): ESS 4.94-5.0, no crash

### Task F: β/α Sweep — PARTIALLY DONE (needs re-run)

**File:** `scripts/run_phase1_sweep.py`

Sweep grid: 5 proteins × 8 configs = 40 runs, 50 particles each.
- pocket_p: β ∈ {0.1, 0.3, 1.0, 3.0, 10.0}, α=15 fixed
- pocket_t: target_t ∈ {0.3, 0.5, 0.7}, α=10 fixed

**First run completed (all 40 configs) but results are INVALID** due to unit mismatch bug.

**Unit mismatch bug (THE BIG ONE):**
- mdtraj loads PDB coordinates in **nanometers**
- Boltz outputs coordinates in **Angstroms** (10x larger)
- `_load_apo_masks()` used raw mdtraj coords (nm) for alignment reference
- `compute_coverage()` compared Boltz-Å sample coords against mdtraj-nm holo coords
- Result: RMSD values ~16 (meaningless), coverage = 0.0 everywhere
- **Fixed:** multiply all mdtraj coords by 10.0 in loading functions

**What's valid from the first run:**
- ESS values: unit-independent, valid
- pLDDT values: Boltz-internal, valid
- RMSD from apo: INVALID (mixed units)
- Coverage: INVALID (always 0.0)
- PocketMiner alignment: INVALID (aligned against nm coords instead of Å)

**What the valid ESS data showed:**
- pocket_p β=0.1: ESS stays healthy (6-39 min across proteins)
- pocket_p β≥0.3: ESS collapses to 1.0 (single dominant particle)
- pocket_t at α=10: ESS stays >44 regardless of target → no effect (too weak)

**MUST re-run sweep** with unit fix (commit `8ac9de2`). Same script, ~2.5h on 2×A40, ~$2.

---

## 3. Repository Structure

```
cryptic-pocket-phd/
├── configs/
│   ├── phase0_proteins.yaml      # 10 Phase 0 proteins (locked)
│   └── phase1_sweep.yaml         # Sweep grid config (locked)
├── data/
│   ├── inputs/                   # FASTA/YAML for Boltz input
│   │   ├── P62593_TEM1.fasta
│   │   ├── P62593_TEM1.yaml
│   │   └── msa/                  # Dummy single-sequence MSA files
│   └── validation_pdbs/          # Apo + holo PDBs
│       ├── 1JWP_A.pdb            # TEM-1 apo
│       ├── 1PZO_A.pdb            # TEM-1 holo
│       ├── 1XCG_B.pdb            # RhoA apo
│       ├── 1OW3_B.pdb            # RhoA holo
│       ├── 1NEP_A.pdb            # NPC2 apo
│       ├── 2HKA_C.pdb            # NPC2 holo
│       ├── 2IYT_A.pdb            # AroK apo
│       ├── 2IYQ_A.pdb            # AroK holo
│       ├── 1FVR_A.pdb            # Tie-2 apo
│       └── 2OO8_A.pdb            # Tie-2 holo
├── docs/
│   └── SPEC_00_pocketminer_noise_sanity.md
├── external/                     # NOT in git — clone on each pod
│   ├── conformix/                # github.com/drorlab/conformix @ d0fd34c
│   │   ├── conformix_boltz/src/boltz/  # ConforMix's modified Boltz
│   │   └── conformix_recon.md    # Our recon document
│   └── pocketminer/              # github.com/Mickdub/gvp @ pocket_pred
│       ├── src/                  # TF GVP-GNN code
│       └── models/pocketminer.*  # TF checkpoint
├── models/
│   └── pocketminer_torch.pt      # PyTorch state dict (2.9 MB)
├── results/
│   ├── phase0_runpod/            # Phase 0 results (complete)
│   └── phase1_sweep/             # Sweep results (ESS files only, needs re-run)
│       ├── ess/                  # 40 .npy files with per-step ESS
│       └── phase1_sweep_merged.csv  # INVALID (unit bug)
├── scripts/
│   ├── run_phase0.sh             # Phase 0 two-phase runner
│   ├── run_phase0_boltz.py       # Phase 0 Boltz inference
│   ├── run_phase0_pocketminer.py # Phase 0 scoring
│   ├── run_phase1.py             # Single-config runner
│   ├── run_phase1_sweep.py       # Multi-config sweep driver
│   ├── run_task_de_gpu.sh        # Task D+E test script
│   ├── setup_runpod.sh           # One-command pod setup
│   ├── validate_pocketminer_port.py  # PyTorch port validation
│   └── monitor_sweep.py          # Mobile notification monitor
├── src/cryptic_pocket_phd/
│   ├── __init__.py
│   ├── pocketminer_torch.py      # PyTorch PocketMiner (Task A)
│   ├── pocket_potential.py       # g_p and g_t potentials (Task B)
│   ├── guidance_injection.py     # ConforMix twist_fn wrapper (Task C)
│   ├── pocketminer_wrapper.py    # OLD TF wrapper (Phase 0, deprecated)
│   ├── intermediate_capture.py   # Phase 0 Boltz capture hooks
│   ├── correlation.py            # Phase 0 Spearman computation
│   ├── fpocket_wrapper.py        # fpocket classical detector
│   └── residue_mapping.py        # PDB residue number mapping
└── requirements-runpod.txt       # Pinned deps for RunPod
```

---

## 4. RunPod Setup (Tested & Bulletproof)

```bash
cd /workspace
git clone https://github.com/aman-source/cryptic-pocket-phd.git && cd cryptic-pocket-phd
git clone https://github.com/drorlab/conformix.git external/conformix
git clone --branch pocket_pred https://github.com/Mickdub/gvp.git external/pocketminer
bash scripts/setup_runpod.sh
```

Expected output: `ALL IMPORTS OK` + `CCD: 43222 entries` + GPU info.

**Critical deps:**
- `numpy==1.26.4` (NOT numpy 2.x — breaks rdkit, scipy)
- `rdkit==2024.9.6` (NOT older — can't unpickle CCD)
- `pymol-open-source>=3.0.0a0` (alpha versions OK)
- `torch.float32` autocast (NOT bfloat16 — causes NaN in SMC)

**Boltz checkpoint:** `~/.boltz/boltz1_conf.ckpt` (3.4 GB), `~/.boltz/ccd.pkl` (330 MB). Downloaded automatically by `setup_runpod.sh`.

---

## 5. What's Next (Immediate)

### 5a. Re-run Task F sweep with unit fix

The nm→Å fix is pushed (commit `8ac9de2`). Need to re-run same sweep:

```bash
# On pod (after setup):
cd /workspace/cryptic-pocket-phd

# GPU 0: 3 proteins
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=external/conformix/conformix_boltz/src:src \
nohup python -u scripts/run_phase1_sweep.py \
  --config configs/phase1_sweep.yaml \
  --out_dir results/phase1_sweep \
  --accelerator gpu \
  --proteins TEM-1,RhoA,AroK > /tmp/sweep_gpu0.log 2>&1 &

# GPU 1: 2 proteins
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=external/conformix/conformix_boltz/src:src \
nohup python -u scripts/run_phase1_sweep.py \
  --config configs/phase1_sweep.yaml \
  --out_dir results/phase1_sweep \
  --accelerator gpu \
  --proteins NPC2,Tie-2 > /tmp/sweep_gpu1.log 2>&1 &
```

Expected: ~2.5h on 2×A40. ~$2.

**IMPORTANT: After sweep completes, DOWNLOAD coords.npy files before terminating pod:**
```bash
scp -r -P <port> root@<ip>:/workspace/cryptic-pocket-phd/results/phase1_sweep/ results/phase1_sweep_v2/
```

### 5b. Test pocket_t at higher α

Bundle with re-run. Add one extra config: pocket_t target=0.5 α=100 on TEM-1.
Either add to sweep YAML or run manually after sweep:

```bash
PYTHONPATH=external/conformix/conformix_boltz/src:src python -u scripts/run_phase1.py \
  data/inputs/P62593_TEM1.yaml \
  --input_cif data/validation_pdbs/1JWP_A.pdb \
  --out_dir results/pocket_t_alpha100 \
  --bias_type pocket_t \
  --pocket_residues "190-200,244-263" \
  --twist_target_values 0.5 \
  --twist_strength_values 100.0 \
  --diffusion_samples 50 \
  --accelerator gpu --seed 42 \
  --twist_rmsd_full_sequence
```

### 5c. After sweep: pick winning config, proceed to Task G

Per Spec #1 §2:
- Pick β (pocket_p) or (α, target) (pocket_t) with highest coverage where ESS_min > K/2
- Lock parameters
- Run full 31-protein benchmark (Task G): 200 samples each, ~52h on 2×A40, ~$42

---

## 6. Decision Rules (from Spec #1 §2, locked)

| Mean worst-matched coverage | Verdict | Next step |
|---|---|---|
| ≥ 0.55 | **Clear win** | Write workshop paper |
| 0.48 – 0.55 | **Marginal win** | Investigate per-protein, maybe Spec #2 |
| 0.42 – 0.48 | **Tie** | Spec #2 (noise-aware PocketMiner training) |
| < 0.42 | **Underperform** | Document failure, Spec #2 mandatory |

ConforMix baseline: 0.45 ± 0.18 worst-matched coverage.

---

## 7. Bugs Found & Fixed (Chronological)

### Phase 0
1. **fpocket relative path bug** — `_run_fpocket_native` changed cwd but used relative path. Fix: `Path(pdb_path).resolve()`.
2. **TF/PyTorch GPU coexistence** — TF 2.18/XLA grabs all GPU memory at import. Fix: two-process pipeline (Phase 1 GPU no TF, Phase 2 CPU no CUDA).
3. **Outer-loop labeling bug** — trainer.predict processed all proteins per call but captures named by outer loop variable. Fix: single-protein DataModule per predict call.
4. **Factor VII altloc crash** — PocketMiner crashed on extra backbone atoms. Fix: skip with guard.
5. **Seed not plumbed** — samples not truly independent. TODO added, not fixed in Phase 0.

### Task A (PocketMiner Port)
6. **TF LayerNorm eps=1e-3** — PyTorch defaults to 1e-5. Caused 0.01 max diff. Fix: explicit eps=1e-3.
7. **nn.Dropout training mode** — Dense head dropout active during eval. Caused non-deterministic output (0.015 diff between runs). Fix: `model.eval()`.

### Task B (Potentials)
8. **Pseudo-backbone 13.7% error** — Fixed offsets for N/C/O lost backbone geometry info. Fix: extract real N/CA/C/O from Boltz all-atom output.

### Task C (Integration)
9. **Click invoke vs callback** — `_CONFORMIX_PREDICT.invoke(ctx, ...)` fails. Fix: use `.callback(...)`.
10. **twist_target_values string parsing** — ConforMix's callback expects pre-parsed float lists. Fix: `[float(x) for x in str(v).split(",")]`.

### Task F (Sweep)
11. **atom_mask batch dim mismatch** — mg_wrapper passes full `[P, N]` atom_mask but twist_fn receives `[batch_p, N, 3]` xt. Fix: `atom_mask[:P_batch]`.
12. **bfloat16 NaN** — autocast with bfloat16 caused NaN in Normal distribution during SMC. Fix: use float32 (matching ConforMix).
13. **nm vs Angstroms** — mdtraj returns nm, Boltz outputs Å. All alignment and coverage computations compared mixed units. Fix: multiply mdtraj coords by 10.

---

## 8. Key Architectural Decisions

1. **PyTorch port over subprocess calls** — Eliminates TF/PyTorch coexistence, enables native autograd gradients, ~100x faster per PocketMiner call.

2. **Real backbone over pseudo** — 13.7% accuracy loss with pseudo was unacceptable. Boltz CCD atom ordering (N=0, CA=1, C=2, O=3) makes real extraction straightforward.

3. **Verbatim RMSD copy with provenance** — ConforMix's twist_fn is a non-importable closure inside a PyMOL-dependent CLI command. Verbatim copy (d0fd34c:887-976) is the only honest option. Regression tested: byte-identical output.

4. **Per-(protein, config) checkpointing** — Each config writes `done.json` on completion. Crash mid-sweep loses only the in-progress config. Enables resume.

5. **Alignment before PocketMiner** — Align x̂₀ to apo reference frame via `weighted_rigid_align` before feeding to PocketMiner. Mitigates kNN graph equivariance noise (~0.1 max diff under rotation). Implemented in `guidance_injection.py`.

---

## 9. ConforMix Internals (from Recon)

**Injection point:** `classifier_prob_fn` is a pluggable callable passed to `TwistedDDPM` constructor (mg_wrapper.py:49). Called at two points:
- `_compute_twisted_step` line 385: with `return_grad=True` (gradient for proposal)
- `G()` at t=0 line 343: with `return_grad=False` (final importance weight)

**Interface:**
```python
# Input
xt: [P_batch, N_atoms, 3]     # noised coords, requires_grad
x0_hat: [P_batch, N_atoms, 3] # Boltz denoised prediction
atom_mask: [P_FULL, N_atoms]   # WARNING: not batched (ConforMix bug)
t: int                          # diffusion step (200 down to 0)

# Output (return_grad=True)
log_potential: [P_batch]        # scalar per particle
gradient: [P_batch, N_atoms, 3] # w.r.t. xt
```

**Gradient flow:** `xt` (requires_grad) → noise added → Boltz denoise → `x0_hat` → twist_fn potential → `autograd.grad` back to `xt`. The gradient flows through Boltz's entire denoising step. This is why PocketMiner must be differentiable in PyTorch.

**SMC parameters:**
- `T=200` hardcoded (can't reduce sampling_steps for SMC)
- `batch_p=25` hardcoded (splits particles for GPU memory)
- `ess_threshold=1/3` (resample when ESS < P/3)
- Resampling: systematic (from Chopin's book)

---

## 10. Phase 0 Results Summary

**Question:** Does PocketMiner give signal on noisy diffusion intermediates?
**Answer:** Weak yes. ρ_pocket ≈ 0.31 at t=0.5 (10x over fpocket null control).

**Decision:** Proceed to Spec #1 (test pocket-as-guidance).

Results: `results/phase0_runpod/results/phase0_rho.csv`

| Protein | ρ_pocket at t=0.5 |
|---------|-------------------|
| P61586 (RhoA) | 0.500 |
| P62593 (TEM-1) | 0.517 |
| P12758 (UPase) | 0.381 |
| P9WPY3 (AroK) | 0.357 |
| P79345 (NPC2) | 0.280 |
| P26281 (FolK) | 0.270 |
| O74933 (UAP1) | 0.264 |
| P0AG16 (PurF) | 0.092 |
| Q02763 (Tie-2) | 0.146 |

---

## 11. Protein Names (DO NOT HALLUCINATE)

Always check `configs/phase0_proteins.yaml`. Never generate names from UniProt IDs by memory.

| UniProt | Name | Short |
|---------|------|-------|
| P79345 | NPC intracellular cholesterol transporter 2 | NPC2 |
| P62593 | Beta-lactamase TEM | TEM-1 |
| O74933 | UDP-GlcNAc pyrophosphorylase | UAP1 |
| P26281 | HPPK | FolK |
| P12758 | Uridine phosphorylase | UPase |
| P0AG16 | Amidophosphoribosyltransferase | PurF |
| P9WPY3 | Shikimate kinase | AroK |
| P08709 | Coagulation factor VII | Factor VII |
| Q02763 | Angiopoietin-1 receptor | Tie-2 |
| P61586 | Transforming protein RhoA | RhoA |

---

## 12. Git Log (Recent)

```
8ac9de2 fix: convert mdtraj nm to Angstroms to match Boltz coordinates
4438ab5 fix: use float32 autocast (not bfloat16) to prevent NaN
355be8b fix: slice atom_mask to batch size for SMC batch_p splitting
622c6ef fix: expand apo reference tensors to match particle batch dim
4fcbcff fix: allow pymol-open-source alpha versions on RunPod
230016b add: pinned requirements + one-command RunPod setup
9655092 feat: Task F sweep driver + holo PDBs for coverage computation
add2599 fix: parse twist values to float lists before calling ConforMix callback
a8b9a78 fix: use callback() for ConforMix predict delegation
6ae25db add: GPU test script for Task D+E (regression + smoke tests)
fd71646 refactor: use run_untwisted imports, add RMSD fallback provenance
e0d1579 feat: wire pocket potential into ConforMix SMC (Task C)
eeb4220 fix: use real backbone N/CA/C/O instead of pseudo-backbone
453bdf0 feat: PocketMiner guidance potentials g_p and g_t (Task B)
42a34cd feat: PyTorch port of PocketMiner GVP-GNN (Task A)
```

---

## 13. User Preferences

- Accuracy over speed
- Stop and ask when in doubt
- Caveman mode active (full) — terse responses, no filler
- NEVER hallucinate protein names
- Local-first development: test on CPU before GPU
- Per-run checkpointing: crash should never lose more than one config
- Kill pod immediately after run completes
- Fresh pod per experimental phase (no dirty state)

---

## 14. Cost Tracking

| Phase | GPU | Hours | Cost |
|-------|-----|-------|------|
| Phase 0 | A100 | ~4h | ~$8 |
| Task D+E validation | A40 | 0.5h | ~$0.20 |
| Setup validation | A40 | 0.1h | ~$0.05 |
| Task F sweep (invalid) | 2×A40 | 2.5h | ~$2.00 |
| **Total so far** | | | **~$10.25** |
| Task F re-run (pending) | 2×A40 | 2.5h | ~$2.00 |
| Task G full benchmark | 2×A40 | ~52h | ~$42 |

---

## 15. Known Issues / TODOs

1. **Sweep re-run needed** — Unit fix not yet validated on GPU. ESS/pLDDT from first run are valid but RMSD/coverage are wrong.
2. **pocket_t may be ineffective** — α=10 showed no effect. Need α=100 test.
3. **Seed plumbing** — `random_seed` from config not properly wired into `pl.seed_everything + per-sample offset`. TODO in `run_phase0_boltz.py`. Affects reproducibility.
4. **Coverage metric** — Our `compute_coverage()` is a simplified version. ConforMix's actual coverage uses MBAR reweighting. We skip MBAR per Spec #1 §9.
5. **coords.npy download** — Sweep script saves coords but they weren't downloaded before pod termination. Must `scp` results after re-run.
6. **monitor_sweep.py** — Uses ntfy.sh for mobile notifications. Topic: `cryptic-pocket-sweep`. Works but not tested end-to-end.
