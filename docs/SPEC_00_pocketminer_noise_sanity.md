# SPEC #0 — Does PocketMiner give signal on noisy diffusion states?

**Status:** ready to execute
**Owner:** Aman (runner) + Claude Code (implementer)
**Author of spec:** Claude (brain)
**Estimated cost:** 1–2 GPU-hours on A100 (~$2.50)
**Estimated wall-clock:** 2 days including local dev + GPU run + analysis

---

## 1. Why this exists

The entire thesis direction (PocketMiner-guided diffusion for cryptic pockets) rests on one untested assumption: **vanilla PocketMiner, trained on clean MD frames, produces meaningful pocket scores when fed half-denoised intermediate structures from Boltz-1's diffusion sampler.**

If yes → guidance is plug-and-play, Phase 1 paper writes itself fast.
If no → we have to train a noise-aware version first, which becomes its own contribution.

**Both outcomes are publishable.** The point of this spec is to find out which, cheaply, before we commit weeks to either path.

---

## 2. The single question

> When we feed PocketMiner the intermediate structure `x_t` produced midway through Boltz-1's reverse diffusion, are its per-residue pocket scores meaningfully correlated with what PocketMiner would say about the final clean structure `x_0`?

Specifically: at each diffusion timestep `t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, compute Spearman correlation between `PocketMiner(x_t)` and `PocketMiner(x_0)`, restricted to the residues annotated as cryptic-pocket-region.

---

## 3. Decision rule (locked before running)

We commit these thresholds **now**, before seeing any results, to prevent post-hoc rationalisation.

| Mean Spearman ρ at `t = 0.5`, on cryptic-pocket residues | Verdict | Next step |
|---|---|---|
| ρ ≥ 0.5 | **Strong yes** | Skip noise-aware training. Go straight to Spec #1: pocket-as-guidance in ConforMix. |
| 0.2 ≤ ρ < 0.5 | **Weak yes** | Conditional. Test pocket-as-guidance with clean PocketMiner; if it underperforms ConforMix-RMSD, fall back to noise-aware training. |
| ρ < 0.2 | **No** | Noise-aware PocketMiner becomes Paper 1. Reshape thesis. |

`t = 0.5` is the commit point because guidance during early/mid diffusion is where it matters most. Late-step guidance (t close to 0) is too late to steer the trajectory.

**Per-protein outliers matter.** If the mean ρ is 0.4 but it's 0.7 on 6 proteins and –0.1 on 4 proteins, the answer is "no, it's bimodal" — we report this honestly, not by mean alone.

---

## 4. Scope decisions (locked)

These are calls I'm making now. If something downstream forces a revision, we revise the spec, not silently change scope.

- **Proteins: 10, not 5, not 31.** Subset of the Lewis et al. cryptic-pocket benchmark. Ten is the smallest number that gives statistical signal while staying in 1–2 GPU-hours.
- **Selection rule for the 10:** stratified — pick proteins across a range of size (≤200, 200–400, ≥400 residues) and across a range of expected difficulty (some that ConforMix-RMSD recovered easily; some it struggled with). **Not** "the 10 smallest proteins," because that's the cherry-picking failure mode from LogicDiff.
- **Base model: Boltz-1.** Not BioEmu. Boltz-1 is the model ConforMix uses; it's the one we'll most likely build on; ConforMix's code already plugs into it. Apples-to-apples.
- **PocketMiner: clean weights only.** No fine-tuning, no modifications. The point is to test it as-is.
- **Samples per protein: 5.** Not 1 (no noise estimate), not 100 (wastes GPU for a diagnostic). Five gives us an SD per protein per timestep.
- **Denoising scheme: capture intermediate `x_t` at the 5 timesteps above. Use Boltz-1's existing diffusion sampler — do not modify the dynamics.**
- **PocketMiner input format:** Boltz-1 outputs all-atom; PocketMiner takes Cα + dihedral features. Conversion needs care (more in §6).

---

## 5. What "ready to run" means

The implementation is done when:

1. There is a single Python script `scripts/run_phase0.py` that takes a config file, runs the full pipeline on one protein, and outputs a CSV of per-residue scores at each timestep.
2. The script runs end-to-end **locally on CPU** with a tiny dummy protein (50 residues, 1 sample, 2 timesteps) without errors. **No RunPod until this passes.**
3. There is a separate analysis script `scripts/analyze_phase0.py` that loads CSVs from all 10 proteins and produces:
   - Spearman ρ per protein per timestep (table)
   - Per-protein ρ distribution at `t = 0.5` (strip plot)
   - Mean ρ vs timestep with error bars (line plot)
   - One-line verdict according to the decision rule in §3
4. All outputs are logged to W&B project `cryptic-pocket-phd`, run group `phase0_sanity`.
5. The final report is a single `phase0_results.md` in the repo, written by Claude Code from the analysis output, with the verdict at the top.

---

## 6. Implementation plan (for Claude Code)

This section is what Claude Code reads. Aman: pass this section to Claude Code in your IDE.

### 6.1 Repo structure to create

```
cryptic-pocket-phd/
├── pyproject.toml             # uv-managed
├── README.md                  # one-line for now
├── .gitignore                 # standard Python + checkpoints
├── configs/
│   └── phase0.yaml            # protein list, timesteps, sample count
├── data/
│   └── lewis_subset/          # PDB files for the 10 chosen proteins
├── external/                  # third-party code clones
│   ├── boltz/
│   ├── pocketminer/           # Mickdub/gvp branch pocket_pred
│   └── conformix/             # only for ref, not run yet
├── scripts/
│   ├── prepare_data.py        # downloads & subsets the Lewis set
│   ├── run_phase0.py          # main experiment loop
│   └── analyze_phase0.py      # produces the report
├── src/
│   └── cryptic_pocket_phd/
│       ├── __init__.py
│       ├── intermediate_capture.py  # hooks into Boltz-1 diffusion
│       ├── pocketminer_wrapper.py   # input format conversion
│       └── correlation.py            # Spearman + bootstraps
└── results/
    └── phase0/
        ├── per_protein/       # CSVs
        └── phase0_results.md  # final report
```

### 6.2 Core technical tasks, in order

**Task A: Data prep**
- Identify the 10 proteins. Lewis et al. cryptic-pocket set has 31–34 proteins (the report flags the size discrepancy). Source: BioEmu paper SI or ConforMix SI. Cross-reference with which proteins ConforMix Table 1 reports on.
- For each protein: download the apo PDB structure (the *reference* default-state structure ConforMix uses as `x_d`).
- For each protein: extract the cryptic-pocket region annotation (residue indices). This is in Lewis et al. SI tables.
- **Stratification check:** before running anything, print the size + ConforMix-RMSD-Boltz worst-matched coverage for each of the 10. Confirm we have spread across both axes.

**Task B: Boltz-1 setup**
- Use `external/boltz/` from `github.com/jwohlwend/boltz`.
- Get default sampling working on one protein end-to-end. Save the output `x_0`.
- Then add a hook: at user-specified timesteps `t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, capture the *current* `x_t` (the noised state at that step) AND the model's prediction of `x̂_0` at that step (the model's guess of the final clean structure given current `x_t`). Save both.
  - **Note:** PocketMiner needs a coherent structure. Raw `x_t` mid-diffusion is partially Gaussian noise — PocketMiner will choke on it. **It's `x̂_0(x_t)` that we should feed to PocketMiner**, because that's what classifier guidance actually evaluates (per ConforMix algorithm — `g_r` is computed at `x̂_0`, not at `x_t`). Update the spec terminology: when we say "PocketMiner score at timestep t", we mean `PocketMiner(x̂_0(x_t))`.

**Task C: PocketMiner setup**
- Use `external/pocketminer/` from `github.com/Mickdub/gvp` branch `pocket_pred`.
- Get inference working on one clean PDB. Confirm output shape = (n_residues,) of floats in [0, 1].
- Wrap it: `pocketminer_wrapper.score(structure_path) → np.array of shape (n_residues,)`.
- **Validation check:** on a known cryptic-pocket protein (e.g., one the PocketMiner paper highlights), confirm the wrapper reproduces a known per-residue distribution roughly. If the score on a known cryptic residue is near zero, something is wrong with the wrapper, not with PocketMiner.

**Task D: Format conversion**
- Boltz-1 produces all-atom mmCIF/PDB. PocketMiner takes a PDB and processes Cα + sidechain features. Confirm the conversion is lossless for what PocketMiner needs.
- **Trap to avoid:** residue numbering mismatches. If Boltz-1 outputs residues numbered 1–N but the Lewis annotation uses PDB residue numbers (which may have insertions, gaps, alt locs), the cryptic-pocket-region indices won't match. Build an explicit mapping table per protein.

**Task E: Correlation computation**
- For each protein, for each of 5 samples, for each of 5 timesteps:
  - Load `x̂_0(x_t)` and `x_0` (the final).
  - Compute `s_t = PocketMiner(x̂_0(x_t))` and `s_0 = PocketMiner(x_0)`.
  - Compute Spearman ρ between `s_t` and `s_0`, **restricted to cryptic-pocket-region residues**. Also compute it on all residues for comparison.
- Aggregate: 10 proteins × 5 samples × 5 timesteps = 250 ρ values. Report mean and SD per protein × timestep.
- Bootstrap: 1000 resamples over proteins for confidence intervals on the aggregate ρ at each timestep.

**Task F: Sanity baseline**
- Run the same correlation, but instead of `PocketMiner(x̂_0(x_t))`, use **fpocket** scores on `x̂_0(x_t)`. fpocket is a classical detector — if its correlation with `fpocket(x_0)` is similar to PocketMiner's correlation, the result is "any pocket detector handles noisy states OK", not specifically "PocketMiner does."
- This protects us from the "cherry-picked baseline" failure mode.

**Task G: Report**
- Generate plots described in §5.3.
- Write the verdict at the top of `phase0_results.md` according to §3.
- Commit everything to git.

### 6.3 Local-first development rule

Before any RunPod usage:

1. Get Tasks B and C working with a single protein on CPU. Boltz-1 can run on CPU for tiny inputs.
2. Run the full pipeline on a 50-residue dummy protein, 1 sample, 2 timesteps. Confirm CSV + plot generation works end-to-end.
3. Only after that passes locally — spin up RunPod and run the real experiment.

This is the rule from the conversation: no idle GPU time spent on debugging missing imports.

---

## 7. Risks I'm flagging now

**Risk 1: Boltz-1's diffusion module isn't easily hookable.** It might require modifying internal sampling code rather than a clean callback. If hooks are ugly, prefer modifying a fork over fighting the upstream API. Don't waste 2 days on a clean abstraction.

**Risk 2: PocketMiner repo is research-quality.** The `Mickdub/gvp` branch may have undocumented preprocessing assumptions. Budget half a day for understanding its data pipeline before assuming the wrapper "just works."

**Risk 3: Lewis cryptic-pocket residue annotations may not be cleanly available.** If the SI table is buried or in a weird format, fall back to using fpocket on the holo structure to define the pocket region — but document this fallback explicitly.

**Risk 4: We discover that `x̂_0(x_t)` at high noise (`t = 0.9`) is itself nonsense — Boltz-1's denoiser might produce wildly bad guesses early in the schedule.** That's still a real result: it tells us PocketMiner isn't the bottleneck, the early-`x̂_0` is. The spec doesn't fail; the conclusion shifts.

**Risk 5: I'm wrong about something I haven't anticipated.** Aman, your job during execution is to stop me when results look weird. "The mean ρ is 0.6, ship it" might hide a bug. Spot-check 2–3 individual proteins manually before I write the verdict.

---

## 8. What I will *not* do in this spec

To prevent scope creep:

- **No noise-aware PocketMiner training.** That's Spec #2 if needed.
- **No actual guidance during sampling.** That's Spec #1.
- **No comparison to ConforMix on the cryptic-pocket benchmark.** That's Phase 1.
- **No fine-tuning of Boltz-1.** That's Phase 2.
- **No more than 10 proteins.** This is a diagnostic, not a benchmark.

If during execution we want to add anything from this list — stop, write Spec #0.5, then continue.

---

## 9. Success criteria for this spec itself

This spec is "good" if:

- A reviewer reading only this document understands what we're testing, why, and what each outcome means for the next 6 months of work.
- Aman can hand sections 6.1–6.3 to Claude Code and get a working repo without further interpretation.
- The decision rule in §3 cannot be argued with after the fact — the thresholds are locked before any data is seen.

If any of these fail, the spec is bad. Tell me before executing.

---

**End of Spec #0.**
