# Lewis et al. Cryptic-Pocket Benchmark — Research Notes

**Status:** protein selection PENDING (8 of 10 locked, 2 replacements needed)
**Last updated:** 2026-05-10

---

## Source paper

- Lewis et al. 2024–2025, "Scalable emulation of protein equilibrium ensembles with generative deep learning."
- bioRxiv: 2024.12.05.626885 (v2 = Feb 2025, latest)
- Published: Science 389, eadv9817 (2025) — the **BioEmu** paper
- GitHub: github.com/microsoft/bioemu and github.com/microsoft/bioemu-benchmarks
- HuggingFace: microsoft/bioemu

## Canonical data source

Cryptic-pocket residue annotations sourced from:
**github.com/microsoft/bioemu-benchmarks** (confirmed, cleanest format)

Under `cryptic_pockets/` → per-protein directories with `local_residinfo/` files
containing pocket-region residue annotations.

## Set size

- Raw set: 34 UniProt IDs
- Q16539: excluded — no annotation file found in repo
- **Working pool: 33 proteins with annotations**
- ConforMix paper evaluated 31 (filtered 3 — identities unknown; see below)
- Discrepancy across literature: 31 / 33 / 34. Use bioemu-benchmarks count as ground truth.

## ConforMix exclusions

- ConforMix (github.com/drorlab/conformix) evaluated 31 of 34 proteins
- 3 were filtered; the repo does not publish an explicit exclusion list
- **Action taken:** searched ConforMix repo configs — exclusion list not found
- **Decision:** pick from all 33 annotated; document that up to 3 chosen proteins
  may overlap with ConforMix's exclusion set. Cannot fix what we cannot see.

## Stratification axes

**Axis 1 — ref-to-ref RMSD** (apo ↔ holo Cα RMSD from bioemu-benchmarks)
- Range: 1.08 Å – 14.73 Å
- Bins: low ≤4 Å, mid 4–8 Å, high >8 Å
- Rationale: proxies how hard the diffusion trajectory has to work; wider than
  sequence length for our diagnostic question

**Axis 2 — sequence length**
- Within each RMSD bin: pick smallest / median / largest

## Exclusion criteria (pocket-region ratio)

Proteins excluded from selection pool because pocket-region / total-residue > 25%
indicate whole-domain motion, not a localized cryptic pocket:

| UniProt | Name | Length | Pocket res | Ratio | Reason |
|---------|------|--------|-----------|-------|--------|
| P02787 | Serotransferrin | 337 | 324 | 96.1% | Whole protein |
| P12823 | ? | 394 | 320 | 81.2% | Whole domain |
| P0DP23 | Calmodulin | 148 | 147 | 99.3% | Whole protein |
| P69441 | Adenylate kinase | 214 | 95 | 44.4% | LID/NMP hinge — domain motion |
| P44542 | ? | 306 | 136 | 44.4% | Domain motion suspected |

Note: P0DP23 (calmodulin) was also flagged as RMSD=14.73 Å outlier — huge RMSD
from whole-domain reorientation, not pocket opening.

## VP35 check

VP35 (UniProt Q05318) — used in Bowman 2026 benchmark. **NOT in Lewis 33-protein set.**
Cannot use as wildcard. Wildcard slot still open.

## Proposed protein selection (DRAFT — awaiting approval)

**Status: 8 of 10 confirmed. P69441 and P44542 dropped; 2 replacements needed.**

### Low-RMSD bin (≤4 Å) — CONFIRMED

| UniProt | Name | Length | Pocket res | Ratio | RMSD |
|---------|------|--------|-----------|-------|------|
| TBD-small | — | — | — | — | — |
| TBD-med | — | — | — | — | — |
| TBD-large | — | — | — | — | — |

### Mid-RMSD bin (4–8 Å) — 1 REPLACEMENT NEEDED (for P44542)

| UniProt | Name | Length | Pocket res | Ratio | RMSD | Status |
|---------|------|--------|-----------|-------|------|--------|
| P26281 | — | 158 | 18 | 11.4% | 4.10 | confirmed |
| TBD-replacement | — | — | ≤25% | — | 4–8 Å | **PENDING** |
| P0AG16 | — | 504 | 36 | 7.1% | 4.73 | confirmed |

### High-RMSD bin (>8 Å) — CONFIRMED

| UniProt | Name | Length | Pocket res | Ratio | RMSD | Status |
|---------|------|--------|-----------|-------|------|--------|
| P9WPY3 | — | 184 | 41 | 22.3% | 5.26 | confirmed |
| P08709 | — | 254 | 63 | 24.8% | 5.14 | confirmed |
| Q02763 | — | 327 | 42 | 12.8% | 5.29 | confirmed |

### Wildcard — REPLACEMENT NEEDED (for P69441)

VP35 not available. Need: recognizable protein from PocketMiner paper or Bowman 2026,
ratio ≤25%, preferably kinase / beta-lactamase / GPCR.

TEM-1 beta-lactamase (P62593, 263 res, 31 pocket, 11.8%) already in confirmed set.

**PENDING:** identify second wildcard candidate.

---

## Candidates for 2 replacement slots

Mid-bin (4–8 Å), ratio ≤25%, not yet selected:

| UniProt | Length | Pocket | Ratio | RMSD |
|---------|--------|--------|-------|------|
| P45568 | 398 | 24 | 6.0% | 2.79* |
| P61586 | 178 | 37 | 20.8% | 2.86* |
| P12758 | 253 | 47 | 18.6% | 3.16* |
| P0A7D4 | 431 | 92 | 21.3% | 3.16* |
| P84080 | 181 | 45 | 24.9% | 3.84* |
| P08037 | 286 | 57 | 19.9% | 4.86* |

*RMSD values approximate — verify against bioemu-benchmarks before committing.

**Recommendation for mid-bin replacement:** P12758 (253 res, 18.6%, ~3.16 Å)
— median size, clean ratio, no domain-motion flag.

**Recommendation for wildcard:** P08037 (286 res, factor Xa inhibitor? — need UniProt
name lookup) OR any protein named in PocketMiner paper supplementary.
Names for P84080 (ARF1?), P12758 needed — pull from UniProt REST before final commit.

---

## Next actions

1. Fetch UniProt names for: P26281, P0AG16, P9WPY3, P08709, Q02763, P12758, P08037, P84080, P0A7D4
2. Verify RMSD values against bioemu-benchmarks CSV
3. User selects final 2 from candidates above
4. Lock into `configs/phase0_proteins.yaml`
5. No changes to protein list after first commit of config
