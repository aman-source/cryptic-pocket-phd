# Phase 1 Results — PocketMiner-as-Guidance in ConforMix Twisted SMC

## Question

Can PocketMiner, used as a differentiable guidance potential in ConforMix's twisted SMC framework, steer Boltz-1 diffusion toward cryptic pocket conformations?

**Decision rule (Spec #1 §2):** If best guided coverage ≥ baseline + 0.05 on ≥3/5 proteins → proceed to Task G (31-protein validation). Otherwise → Spec #2 (noise-aware PocketMiner retraining).

## Methodology

### System

- **Diffusion model:** Boltz-1 (3.4 GB checkpoint, 200-step diffusion)
- **Guidance injection:** ConforMix twisted SMC (`classifier_prob_fn` interface)
- **Pocket predictor:** PocketMiner GVP-GNN (ported TF→PyTorch, 1e-4 agreement gate)
- **Alignment:** Kabsch superposition of sample CAs onto holo CAs before coverage computation

### Proteins

| Protein | Apo PDB | Holo PDB | Residues | Aligned ref-RMSD (Å) | Threshold (Å) |
|---------|---------|----------|----------|-----------------------|----------------|
| TEM-1 | 1JWP_A | 1PZO_A | 263 | 0.92 | 0.46 |
| RhoA | 1XCG_B | 1OW3_B | 178 | 1.90 | 0.95 |
| AroK | 2IYT_A | 2IYQ_A | 166 | 3.93 | 1.97 |
| NPC2 | 1NEP_A | 2HKA_C | 130 | 1.12 | 0.56 |
| Tie-2 | 1FVR_A | 2OO8_A | 259 | 20.05 | 10.02 |

### Conditions (4 per protein, 50 samples each)

1. **Baseline (unguided):** Standard Boltz diffusion, no twist function
2. **pocket_p β=0.1:** g_p = β × Σ PocketMiner(x)[pocket]. Gradient scaling α=15.
3. **pocket_t α=10 t=0.3:** g_t = -(mean_pocket - target)². Gradient scaling α=10.
4. **pocket_t α=100 t=0.7:** Same form, stronger gradient scaling.

### Coverage metric

Worst-matched coverage: for each holo CA atom, find minimum distance to corresponding CA atom across all 50 samples (after Kabsch alignment to holo). Coverage = fraction of atoms where min_distance < 0.5 × ref-to-ref RMSD.

## Results

| Protein | ref-RMSD (Å) | Threshold (Å) | Baseline | pp β=0.1 | pt α=10 | pt α=100 |
|---------|-------------|---------------|----------|----------|---------|----------|
| TEM-1 | 0.92 | 0.46 | 0.008 | **0.023** | 0.004 | 0.011 |
| RhoA | 1.90 | 0.95 | **0.534** | 0.523 | 0.528 | 0.489 |
| AroK | 3.93 | 1.97 | **0.241** | 0.211 | 0.241 | 0.199 |
| NPC2 | 1.12 | 0.56 | 0.008 | 0.008 | 0.000 | 0.000 |
| Tie-2 | 20.05 | 10.02 | **0.849** | 0.645 | 0.846 | 0.776 |

**Guided ≥ baseline + 0.05 on 1/5 proteins (TEM-1, marginal: +0.015).** Decision rule not met.

### Additional sweep results (pocket_p β sweep, ESS behavior)

- β=0.1 is the only ESS-safe setting. β≥0.3 collapses ESS to 1.0 on 3/4 proteins.
- β≥3.0 degrades pLDDT to ~0.30 (structurally invalid).
- pocket_t α=10 has negligible effect on all metrics (ESS stays ~48/50).
- pocket_t α=100 reduces ESS to 14–16 (healthy) but does not improve coverage.

Full sweep: 5 proteins × 11 configs (5 pocket_p + 6 pocket_t) × 50 samples = 55 runs.

## Bugs Caught and Fixed

### Bug 1: Coordinate-frame alignment missing in coverage metric

`compute_coverage()` compared sample coordinates, holo PDB coordinates, and apo PDB coordinates without alignment. All three were in different coordinate frames:
- Apo: PDB crystal frame (e.g., center at (-21, 22, 39) for RhoA)
- Holo: different PDB crystal frame (e.g., center at (18, -5, -2))
- Samples: Boltz output frame (centered near origin)

**Impact:** RhoA baseline showed coverage=1.000 (bogus — unaligned ref-RMSD was 67.6 Å, threshold 33.8 Å, everything passes). AroK guided appeared to show 2.1× improvement over baseline (0.283 vs 0.133) — entirely an artifact.

**Fix:** Added Kabsch alignment. All coordinates now superimposed onto holo frame before distance computation. Commit `[this commit]`.

### Bug 2: TEM-1 coverage=NaN

All 11 TEM-1 configs returned coverage=NaN across both sweep runs. Root cause: exception in CA index extraction caught silently by `except Exception`. Fixed by proper `atom_center` extraction from Boltz structure metadata.

### Bug 3: mdtraj nm vs Boltz Å unit mismatch

mdtraj returns coordinates in nanometers; Boltz operates in Angstroms. First sweep (May 11) produced RMSD values 10× too small and coverage=0.0 for all proteins. Fixed in commit `8ac9de2` by multiplying mdtraj coordinates by 10.

## Verdict

**PocketMiner-as-guidance does not improve cryptic pocket coverage over unguided Boltz baseline.**

Per the locked decision rule: **underperform → Spec #2 activated.**

## Why It Failed

1. **PocketMiner scores are scalar and geometrically agnostic.** The model predicts "is this residue in a cryptic pocket?" (a binary classifier output) but does not encode the direction or geometry of the open state. Many conformations can achieve high pocket scores — only one matches holo.

2. **Gradient direction is uninformative.** The gradient of PocketMiner score with respect to atom coordinates points toward "more pocket-like" in feature space, not toward the specific holo geometry. This is fundamentally different from RMSD guidance, which directly encodes WHERE to go.

3. **Phase 0 signal was weak.** ρ_pocket ≈ 0.31 between PocketMiner score and diffusion noise level indicated a marginal correlation. A stronger signal would be needed for the gradient to meaningfully steer 200 diffusion steps.

4. **Scalar potential cannot encode conformational specificity.** The twisted SMC framework is designed for potentials that reward a specific target state (e.g., RMSD to a known structure). PocketMiner rewards a property (openness) that can be achieved by many unrelated conformations.

## Cost

Total Phase 1 GPU spend: ~$5.50 across 6 pod sessions (see OPERATIONAL_LOG.md).

## Next Step

**Spec #2: Noise-aware PocketMiner training.** Train a PocketMiner variant on diffusion-noised structures so the guidance signal is meaningful at intermediate noise levels, not just on clean structures. This addresses failure mode #3 (weak Phase 0 signal) directly. Whether it can also address failure modes #1-2 (geometric agnosticism) is the research question for Phase 2.
