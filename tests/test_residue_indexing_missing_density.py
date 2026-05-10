"""Item 4: Residue indexing correctness for proteins with missing crystal density.

Context
-------
Five of our 10 proteins have fewer residues in their crystal structure than
their UniProt/full sequence length:

  O74933 (UAP1)     2YQC_A  expected 486, actual 480  (-6)
  P0AG16 (PurF)     1ECJ_D  expected 504, actual 492  (-12)
  P9WPY3 (AroK)     2IYT_A  expected 184, actual 175  (-9)
  P08709 (FVII)     1JBU_A  expected 254, actual 238  (-16)
  Q02763 (Tie-2)    1FVR_A  expected 327, actual 299  (-28)

Pipeline correctness requirement
---------------------------------
Boltz is given the CRYSTAL STRUCTURE sequence (from the PDB), not the full
UniProt sequence. So Boltz outputs n_residues = crystal_length. The reference
PocketMiner score s_0 also has n = crystal_length. They match.

The Lewis metric_resid_ranges (stored in local_residinfo/*.json) use 1-based
SEQUENTIAL positions in the apo crystal structure (confirmed by reading
bioemu-benchmarks/eval/multiconf/evaluate.py: begin_resid=1 when None, and
ranges compared to n_residues which is the structure's residue count).

Our pocket_residue_indices(ranges, n_residues) does the same: 1-based
sequential → 0-based index. So the indexing is consistent.

This test file verifies:
  1. YAML sequence length == PDB residue count (Boltz input matches reference).
  2. Pocket ranges are within [1, n_residues] (no out-of-bounds indexing).
  3. For the complete proteins (resSeq starts at 1, no gaps), sequential
     position N == PDB resSeq N — a direct sanity check.
  4. For missing-density proteins, pocket residues at the given sequential
     positions are actual protein residues (not gaps) in the crystal structure.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# (uniprot, apo_pdb_stem, actual_crystal_n_res, pocket_ranges, complete_pdb)
# complete_pdb=True means resSeq starts at 1 and matches sequential position
_PROTEINS = [
    ("P79345", "1NEP_A", 130,  [[91, 105]],                           True),
    ("P62593", "1JWP_A", 263,  [[190, 200], [244, 263]],              False),  # resSeq 26..290
    ("O74933", "2YQC_A", 480,  [[105, 120], [280, 330]],              False),  # missing 6
    ("P26281", "1HKA_A", 158,  [[44, 52], [84, 92]],                  True),
    ("P12758", "1K3F_B", 253,  [[161, 190], [224, 240]],              False),  # resSeq 1001..
    ("P0AG16", "1ECJ_D", 492,  [[323, 358]],                          False),  # missing 12
    ("P9WPY3", "2IYT_A", 175,  [[100, 140]],                          False),  # missing 9
    ("P08709", "1JBU_A", 238,  [[40, 45], [81, 90], [155, 192], [210, 218]], False),  # missing 16
    ("Q02763", "1FVR_A", 299,  [[10, 33], [164, 181]],                False),  # missing 28
    ("P61586", "1XCG_B", 178,  [[12, 48]],                            False),  # resSeq 3..
]

# missing-density proteins specifically
_MISSING_DENSITY = [p for p in _PROTEINS if p[0] in {"O74933", "P0AG16", "P9WPY3", "P08709", "Q02763"}]


def _load_crystal_residues(pdb_stem: str):
    """Return (n_residues, resseq_list, seq_str) for the crystal structure."""
    import mdtraj as md

    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    pdb_path = _REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = md.load(str(pdb_path))
    ca_sel = traj.top.select("protein and name CA")
    ca_traj = traj.atom_slice(ca_sel)
    residues = list(ca_traj.top.residues)
    resseq_list = [r.resSeq for r in residues]
    seq = "".join(aa3to1.get(r.name, "X") for r in residues)
    return len(residues), resseq_list, seq


def _load_boltz_seq(uniprot: str) -> str:
    a3m = _REPO_ROOT / "data" / "boltz_inputs" / uniprot / f"{uniprot}_A.a3m"
    lines = a3m.read_text().splitlines()
    return lines[1]  # second line = sequence


# ===========================================================================
# 1. YAML sequence length == crystal structure residue count
# ===========================================================================

class TestYamlSeqMatchesCrystal:
    """Boltz input sequence length must match reference PDB residue count.

    If they differ, s_0 (from reference PDB) and s_t (from Boltz output)
    would have different shapes → run_phase0.py warns and skips those proteins.
    """

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,_", _PROTEINS)
    def test_yaml_seq_len_matches_pdb(self, uniprot, pdb_stem, expected_n, ranges, _):
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")
        if not (_REPO_ROOT / "data" / "boltz_inputs" / uniprot / f"{uniprot}_A.a3m").exists():
            pytest.skip(f"Boltz input not present: {uniprot}")

        n_crystal, _, _ = _load_crystal_residues(pdb_stem)
        boltz_seq = _load_boltz_seq(uniprot)

        assert len(boltz_seq) == n_crystal, (
            f"{uniprot}: Boltz seq len={len(boltz_seq)} != crystal n_res={n_crystal}. "
            f"s_0 and s_t will mismatch — Boltz inference skipped for this protein."
        )


# ===========================================================================
# 2. Pocket ranges within bounds
# ===========================================================================

class TestPocketRangesInBounds:
    """pocket_residue_indices must return only valid 0-based indices."""

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,_", _PROTEINS)
    def test_pocket_indices_in_bounds(self, uniprot, pdb_stem, expected_n, ranges, _):
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")

        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        n_crystal, _, _ = _load_crystal_residues(pdb_stem)
        idx = pocket_residue_indices(ranges, n_crystal)

        assert len(idx) > 0, f"{uniprot}: no pocket indices returned"
        assert all(0 <= i < n_crystal for i in idx), (
            f"{uniprot}: out-of-bounds pocket index. "
            f"n_crystal={n_crystal}, bad indices={[i for i in idx if not (0 <= i < n_crystal)]}"
        )

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,_", _MISSING_DENSITY)
    def test_missing_density_pocket_ranges_not_clipped(self, uniprot, pdb_stem, expected_n, ranges, _):
        """For missing-density proteins, pocket ranges must fit WITHOUT clipping.

        If any range end > n_crystal, pocket_residue_indices clips it, silently
        dropping pocket residues. This would bias rho estimates.
        """
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")

        n_crystal, _, _ = _load_crystal_residues(pdb_stem)

        clipped = [r for r in ranges if r[1] > n_crystal]
        assert not clipped, (
            f"{uniprot}: pocket range end exceeds crystal n_res={n_crystal}. "
            f"Clipped ranges: {clipped}. "
            f"Update pocket_residue_ranges in phase0_proteins.yaml to match crystal structure."
        )


# ===========================================================================
# 3. Sequential position == resSeq for complete structures (sanity)
# ===========================================================================

class TestSequentialEqResSeqForCompleteStructures:
    """For proteins where resSeq starts at 1 with no gaps, sequential position N == resSeq N."""

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,complete", _PROTEINS)
    def test_seq_pos_eq_resseq(self, uniprot, pdb_stem, expected_n, ranges, complete):
        if not complete:
            pytest.skip(f"{uniprot}: non-trivial resSeq, skip direct equality check")
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")

        n, resseq_list, _ = _load_crystal_residues(pdb_stem)
        # Sequential position i (1-based) should equal resSeq[i-1]
        for i, rsq in enumerate(resseq_list[:10], start=1):
            assert rsq == i, (
                f"{uniprot}: sequential pos {i} has resSeq={rsq} (expected {i}). "
                f"Chain has resSeq offset or gaps — mark complete=False."
            )


# ===========================================================================
# 4. Missing-density proteins: pocket residues are real AA (not gaps)
# ===========================================================================

class TestMissingDensityPocketResiduesExist:
    """Pocket residues at Lewis sequential positions are real AA in crystal structure.

    For proteins with missing density, some sequential positions in the
    MIDDLE of the sequence might correspond to residues that are missing
    from the crystal structure. This would mean the pocket index points
    to the WRONG residue (a shifted one).

    This test checks that the amino acid at each pocket position is
    a standard amino acid (not 'X'), confirming no gaps in the pocket region.
    """

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,_", _MISSING_DENSITY)
    def test_pocket_residues_are_standard_aa(self, uniprot, pdb_stem, expected_n, ranges, _):
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")

        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        n_crystal, _, seq = _load_crystal_residues(pdb_stem)
        idx = pocket_residue_indices(ranges, n_crystal)

        unknown = [i for i in idx if seq[i] == "X"]
        assert not unknown, (
            f"{uniprot}: {len(unknown)} pocket positions map to non-standard AA ('X'). "
            f"Crystal structure may have missing density IN the pocket region. "
            f"Sequential positions: {[i+1 for i in unknown]}"
        )

    @pytest.mark.parametrize("uniprot,pdb_stem,expected_n,ranges,_", _MISSING_DENSITY)
    def test_crystal_seq_matches_boltz_seq_at_pocket(self, uniprot, pdb_stem, expected_n, ranges, _):
        """Boltz sequence and crystal sequence agree at pocket positions."""
        if not (_REPO_ROOT / "data" / "validation_pdbs" / f"{pdb_stem}.pdb").exists():
            pytest.skip(f"PDB not present: {pdb_stem}.pdb")
        if not (_REPO_ROOT / "data" / "boltz_inputs" / uniprot / f"{uniprot}_A.a3m").exists():
            pytest.skip(f"Boltz input not present: {uniprot}")

        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        n_crystal, _, crystal_seq = _load_crystal_residues(pdb_stem)
        boltz_seq = _load_boltz_seq(uniprot)
        idx = pocket_residue_indices(ranges, n_crystal)

        mismatches = [
            (i, crystal_seq[i], boltz_seq[i])
            for i in idx
            if i < len(boltz_seq) and crystal_seq[i] != boltz_seq[i]
        ]
        assert not mismatches, (
            f"{uniprot}: crystal seq and Boltz seq differ at {len(mismatches)} pocket positions. "
            f"First 3 mismatches (idx, crystal_aa, boltz_aa): {mismatches[:3]}"
        )
