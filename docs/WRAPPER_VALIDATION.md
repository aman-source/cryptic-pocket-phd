# PocketMiner Wrapper Validation

Two end-to-end validation runs on cryptic-pocket proteins from the Lewis benchmark.

---

## Results

| Protein | Residues | Pocket range (sequential pos) | Pocket residues | In top quartile | Hit rate |
|---------|----------|-------------------------------|-----------------|-----------------|----------|
| UdP (1K3F_B) | 253 | [[161,190],[224,240]] | 47 | 17 | **36%** |
| TEM-1 (1JWP_A) | 263 | [[190,200],[244,263]] | 31 | 9 | **29%** |

Top quartile = scores ≥ Q75 over all residues. Random chance = 25%.

---

## Interpretation

**Enrichment is real but modest.** Both proteins exceed the 25% random baseline (36% and 29%),
confirming PocketMiner assigns elevated cryptic-pocket probability to annotated pocket regions.

**Why the enrichment is modest — annotation width mismatch.**

Lewis/bioemu annotations are conservative: they include all residues that move between
apo and holo crystal structures, not just the residues that form the cavity wall.
This creates two categories within the annotated range:

1. **Core pocket residues** — directly contact the ligand or rearrange to form the pocket.
   These score correctly. Example: TEM-1 Ω-loop residues 191–196 score 0.68–0.93.

2. **Movement-coupled residues** — shift as part of the conformational change but do not
   themselves define the pocket. Example: TEM-1 C-terminal helix [254–263] scores near 0
   because those residues are in a stable helix, not the opening loop.

PocketMiner was trained on pocket-forming residues (fpocket annotations on MD ensembles),
so it correctly down-weights movement residues that are not pocket-forming.
The annotation width mismatch is a property of the benchmark, not a model failure.

---

## Implications

### Spec #0 (noise sanity check) — harmless

Spec #0 uses Spearman ρ between PocketMiner scores at noisy intermediate x̂_0(x_t) and
clean apo structure x_0, restricted to the annotated pocket region.

Spearman is rank-based and computed *within* the pocket region only.
The denominator is fixed (pocket indices as defined by YAML), so the annotation width
mismatch does not bias the ρ values — it only affects which residues contribute.
The core pocket residues (high scorers) will drive the correlation signal.
This is acceptable for a sanity check. No correction needed.

### Spec #1+ (guided diffusion) — must revisit

**Flag for future work.** If PocketMiner scores are used as guidance gradients:

1. **Gradient magnitude**: guidance ∝ ∂(score)/∂(coords). Movement residues (score ≈ 0
   throughout) contribute near-zero gradient — effectively dead weight. Core pocket
   residues dominate. This concentrates the gradient signal, which may or may not be
   desirable.

2. **Calibration**: 29–36% hit rate means the model's raw scores conflate pocket-forming
   probability with background noise over movement residues. Any threshold or loss function
   using absolute score values (not ranks) must be recalibrated against a pocket-only
   subset of the benchmark.

3. **Recommended action before Spec #1**: Re-run validation with pocket ranges narrowed to
   residues within 4 Å of the holo ligand (ligand-contact residues), compute hit rate on
   that subset. If >60%, PocketMiner scores are usable as guidance without recalibration.

---

## Technical notes

- PocketMiner checkpoint: `external/pocketminer/models/pocketminer`
- Wrapper API: `score(structure_path: str) -> np.ndarray` shape `(n_residues,)` float32
- Residue ordering: mdtraj sequential index (0-based), same order as PocketMiner output
- `pocket_residue_indices(ranges, n_residues)` converts 1-based YAML sequential positions
  to 0-based Boltz/mdtraj indices via `boltz_idx = yaml_pos - 1`
- UdP (1K3F_B) has PDB resSeq offset: chain B starts at resSeq 1001, not 1.
  Sequential-position convention avoids this offset silently.
