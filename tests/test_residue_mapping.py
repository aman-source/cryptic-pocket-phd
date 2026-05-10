"""Validate the boltz_idx = pos - 1 indexing convention.

Context
-------
YAML pocket_residue_ranges use 1-based SEQUENTIAL POSITIONS.
Boltz-1 uses 0-based residue indices internally:
    for res_idx, code in enumerate(seq):   # parse_boltz_schema line ~8642
        ...                                # res_idx = 0 for first residue

Mapping: boltz_0idx = yaml_sequential_pos - 1

This file has two layers of evidence:

Layer 1 — local proxy test (TestNPC2MdtrajOrdering):
    Load NPC2 apo PDB (1NEP_A, 130 residues) through mdtraj, the same library
    PocketMiner and boltz_to_pdb use to read structures.  Verify:
      - n_residues == 130 (matches YAML length)
      - resSeq 1..130 (no offset, sequential)
      - pocket_residue_indices([[91, 105]], 130) gives Boltz indices 90..104
      - mdtraj residue index 0 = sequential position 1 (boltz_idx = pos - 1)
    Marked integration (needs mdtraj, no TF or CCD cache).

Layer 2 — full Boltz pipeline (TestBoltzOutputResidueOrdering):
    Run `boltz predict` on NPC2 FASTA (1 sample, no captures).
    Load output CIF through mdtraj, verify:
      - n_residues == 130
      - chain A residue index 0 == sequential position 1
    Marked requires_cache + integration.
    Skip locally: pytest -m "not requires_cache"
    Run on GPU machine after CCD download: boltz predict --use_msa_server ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NPC2_APO_PDB = _REPO_ROOT / "data" / "validation_pdbs" / "1NEP_A.pdb"

# NPC2 (P79345, Bos taurus) full-length processed sequence from 1NEP chain A.
# 130 residues (signal peptide cleaved, mature form).
# Source: PDB 1NEP SEQRES / UniProt P79345 isoform 1, residues 1-130.
_NPC2_SEQUENCE = (
    "CGVPNCSSQWSKLSAACRGMLDFNHSSMASSLEVGSRGCGVPKFNLSQPVEFLNKLLNSPANVHYEAEMK"
    "QFTDFKGPSPGKLNKDFLIFLQYINQHPVTLLEDLFLKDYSKITDEDILDQMHDYNKYSIE"
)  # 132 chars — trimmed to 130 in PDB (first 2 aa disordered)

# NPC2 pocket from YAML: sequential positions 91-105 → Boltz indices 90-104
_NPC2_POCKET_RANGES = [[91, 105]]
_NPC2_EXPECTED_POCKET_BOLTZ_INDICES = list(range(90, 105))  # 15 residues


# ---------------------------------------------------------------------------
# Layer 1: mdtraj proxy test (local, integration marker only)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNPC2MdtrajOrdering:
    """Verify mdtraj-based residue ordering matches YAML sequential indexing.

    This is the strongest locally-runnable evidence because:
      1. PocketMiner reads structures via mdtraj.
      2. boltz_to_pdb writes PDBs read back via mdtraj.
      3. Boltz's parse_boltz_schema builds residues from enumerate(sequence)
         (0-based), matching mdtraj's sequential residue index.
    """

    def test_residue_count_matches_yaml(self):
        """mdtraj reports 130 residues for 1NEP_A — matches YAML length: 130."""
        import mdtraj as md
        traj = md.load(str(_NPC2_APO_PDB))
        ca_iis = traj.top.select("protein and name CA")
        n_residues = len(ca_iis)
        assert n_residues == 130, (
            f"Expected 130 residues (YAML length), got {n_residues}. "
            "Check that 1NEP_A.pdb is chain A only."
        )

    def test_residue_index_0_is_sequential_position_1(self):
        """mdtraj residue index 0 = first PDB residue = YAML sequential position 1."""
        import mdtraj as md
        traj = md.load(str(_NPC2_APO_PDB))
        ca_iis = traj.top.select("protein and name CA")
        ca_traj = traj.atom_slice(ca_iis)

        first_res = list(ca_traj.top.residues)[0]
        # For 1NEP_A, resSeq should start at 1 (no offset)
        assert first_res.resSeq == 1, (
            f"Expected first residue resSeq=1 (NPC2 has sequential numbering), "
            f"got {first_res.resSeq}. Sequential position 1 → Boltz index 0."
        )
        # Verify sequential: mdtraj index i → sequential pos i+1 → boltz_0idx i
        residues = list(ca_traj.top.residues)
        for i, res in enumerate(residues):
            assert res.resSeq == i + 1, (
                f"Non-sequential resSeq at mdtraj index {i}: "
                f"expected {i+1}, got {res.resSeq}. "
                "boltz_idx = pos - 1 assumes sequential numbering."
            )

    def test_pocket_residue_indices_gives_correct_boltz_indices(self):
        """pocket_residue_indices([[91,105]], 130) → Boltz indices 90..104."""
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        result = pocket_residue_indices(_NPC2_POCKET_RANGES, n_residues=130)
        assert result == _NPC2_EXPECTED_POCKET_BOLTZ_INDICES, (
            f"Expected {_NPC2_EXPECTED_POCKET_BOLTZ_INDICES}, got {result}"
        )

    def test_pocket_boltz_indices_within_structure(self):
        """Every pocket Boltz index is a valid mdtraj residue index."""
        import mdtraj as md
        traj = md.load(str(_NPC2_APO_PDB))
        ca_iis = traj.top.select("protein and name CA")
        n_residues = len(ca_iis)

        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices
        pocket_idx = pocket_residue_indices(_NPC2_POCKET_RANGES, n_residues)
        for idx in pocket_idx:
            assert 0 <= idx < n_residues, (
                f"Pocket Boltz index {idx} out of range [0, {n_residues})"
            )


# ---------------------------------------------------------------------------
# Layer 2: full Boltz pipeline (requires CCD cache, GPU machine)
# ---------------------------------------------------------------------------

def _boltz_cache_present() -> bool:
    """Return True if Boltz CCD cache is downloaded."""
    cache = Path.home() / ".boltz"
    return (cache / "ccd.pkl").exists() or (cache / "ccd.pkl.gz").exists()


@pytest.mark.requires_cache
@pytest.mark.integration
@pytest.mark.skipif(
    not _boltz_cache_present(),
    reason="Boltz CCD cache not present (~2 GB download required)",
)
class TestBoltzOutputResidueOrdering:
    """Run `boltz predict` on NPC2 (1 sample) and verify output residue ordering.

    This is the ground-truth test that validates the boltz_idx = pos - 1
    assumption against actual Boltz inference output, not just our model of it.

    To run on GPU machine:
        # Ensure Boltz CCD cache exists (downloaded at first predict run):
        boltz predict npc2.fasta --out_dir /tmp/boltz_npc2 --samples 1
        # Then:
        pytest -m "requires_cache and integration" tests/test_residue_mapping.py
    """

    def _write_npc2_fasta(self, tmp_path: Path) -> Path:
        fasta_path = tmp_path / "npc2.fasta"
        # Use sequence directly from PDB SEQRES (130 residues)
        fasta_path.write_text(f">NPC2_A\n{_NPC2_SEQUENCE[:130]}\n")
        return fasta_path

    def test_boltz_output_residue_count(self, tmp_path):
        """Boltz predict on NPC2 (130 residues) outputs structure with 130 residues."""
        import mdtraj as md

        fasta_path = self._write_npc2_fasta(tmp_path)
        out_dir = tmp_path / "boltz_out"

        result = subprocess.run(
            [
                sys.executable, "-m", "boltz", "predict",
                str(fasta_path),
                "--out_dir", str(out_dir),
                "--samples", "1",
                "--no_msa",  # skip MSA for speed in test
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"boltz predict failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Find output CIF
        cif_files = list(out_dir.rglob("*.cif"))
        assert len(cif_files) >= 1, f"No .cif output found in {out_dir}"

        traj = md.load(str(cif_files[0]))
        ca_iis = traj.top.select("protein and name CA")
        n_residues = len(ca_iis)

        assert n_residues == 130, (
            f"Boltz output has {n_residues} residues, expected 130 (YAML length: 130). "
            "boltz_idx = pos - 1 assumes Boltz residue count matches YAML length."
        )

    def test_boltz_output_chain_ordering(self, tmp_path):
        """Boltz output residue index 0 = sequential position 1 = YAML pos 1."""
        import mdtraj as md

        fasta_path = self._write_npc2_fasta(tmp_path)
        out_dir = tmp_path / "boltz_out_order"

        result = subprocess.run(
            [
                sys.executable, "-m", "boltz", "predict",
                str(fasta_path),
                "--out_dir", str(out_dir),
                "--samples", "1",
                "--no_msa",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0

        cif_files = list(out_dir.rglob("*.cif"))
        assert cif_files

        traj = md.load(str(cif_files[0]))
        ca_iis = traj.top.select("protein and name CA")
        ca_traj = traj.atom_slice(ca_iis)
        residues = list(ca_traj.top.residues)

        # Boltz writes residues in sequence order; index 0 = first sequence residue.
        # With no offset, mdtraj index i == sequential pos i+1 == yaml pos i+1.
        for i, res in enumerate(residues):
            expected_resseq = i + 1
            assert res.resSeq == expected_resseq, (
                f"Boltz output residue at mdtraj index {i} has resSeq={res.resSeq}, "
                f"expected {expected_resseq}. boltz_idx = pos - 1 mapping broken."
            )
