"""Tests for fpocket_wrapper (Task F).

Test structure
--------------
Unit tests (no WSL/fpocket needed):
  - _parse_out_pdb: mock PDB content → resSeq→score mapping
  - score(): mock subprocess → shape, dtype, values in [0, 1]

Integration test (mark: integration):
  - Real fpocket on 1NEP_A.pdb (NPC2 apo structure).
  - Pocket residues 91-105 have at least one non-zero score.
  - Run with: pytest -m integration tests/test_fpocket_wrapper.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NPC2_APO = _REPO_ROOT / "data" / "validation_pdbs" / "1NEP_A.pdb"

# NPC2 pocket: sequential positions 91-105 → 0-based indices 90-104
_NPC2_POCKET_IDX = list(range(90, 105))


# ===========================================================================
# Helper: build a minimal fpocket *_out.pdb content string
# ===========================================================================

def _make_out_pdb_lines(records: list[tuple[int, float]]) -> str:
    """Build a minimal fpocket output PDB with ATOM records.

    records: list of (resSeq, occupancy).
    Each record produces one ATOM line with given occupancy in cols 55-60.
    """
    lines = []
    for i, (resseq, occ) in enumerate(records, start=1):
        # PDB ATOM record, fixed-width
        # cols: 1-6 record, 7-11 serial, 13-16 name, 17 altloc,
        #       18-20 resName, 22 chainID, 23-26 resSeq, ...
        # occupancy at cols 55-60 (1-indexed)
        line = (
            f"ATOM  {i:5d}  CA  ALA A{resseq:4d}    "
            f"   0.000   0.000   0.000"
            f"{occ:6.2f}"
            f"  0.00           C"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


# ===========================================================================
# 1. _parse_out_pdb — unit tests (no I/O, just logic via tmp_path)
# ===========================================================================

class TestParseOutPdb:
    """_parse_out_pdb reads occupancy column, returns max per resSeq."""

    def test_single_residue(self, tmp_path):
        pdb_content = _make_out_pdb_lines([(42, 0.75)])
        p = tmp_path / "out.pdb"
        p.write_text(pdb_content)

        from cryptic_pocket_phd.fpocket_wrapper import _parse_out_pdb
        result = _parse_out_pdb(p)
        assert result == {42: pytest.approx(0.75, abs=1e-3)}

    def test_max_over_atoms_same_residue(self, tmp_path):
        """Multiple atoms for residue 5 — max occupancy returned."""
        pdb_content = _make_out_pdb_lines([(5, 0.30), (5, 0.70), (5, 0.50)])
        p = tmp_path / "out.pdb"
        p.write_text(pdb_content)

        from cryptic_pocket_phd.fpocket_wrapper import _parse_out_pdb
        result = _parse_out_pdb(p)
        assert result[5] == pytest.approx(0.70, abs=1e-3)

    def test_zero_occupancy_residue_included(self, tmp_path):
        """Residue with occ=0.0 is included (max is 0.0 — not in pocket)."""
        pdb_content = _make_out_pdb_lines([(1, 0.0), (2, 0.5)])
        p = tmp_path / "out.pdb"
        p.write_text(pdb_content)

        from cryptic_pocket_phd.fpocket_wrapper import _parse_out_pdb
        result = _parse_out_pdb(p)
        assert 1 in result
        assert result[1] == pytest.approx(0.0, abs=1e-3)
        assert result[2] == pytest.approx(0.5, abs=1e-3)

    def test_non_atom_lines_skipped(self, tmp_path):
        """REMARK/HETATM/TER lines must not be parsed."""
        content = (
            "REMARK  1 fpocket output\n"
            + _make_out_pdb_lines([(10, 0.88)])
            + "TER\nEND\n"
        )
        p = tmp_path / "out.pdb"
        p.write_text(content)

        from cryptic_pocket_phd.fpocket_wrapper import _parse_out_pdb
        result = _parse_out_pdb(p)
        assert list(result.keys()) == [10]


# ===========================================================================
# 2. score() — shape, dtype via mock (no WSL needed)
# ===========================================================================

def _build_mock_mdtraj(n_residues: int, start_resseq: int = 1):
    """Return a mock mdtraj trajectory with n_residues CA atoms."""
    mock_traj = MagicMock()
    mock_traj.top.select.return_value = list(range(n_residues))

    mock_ca = MagicMock()
    residues = []
    for i in range(n_residues):
        r = MagicMock()
        r.resSeq = start_resseq + i
        residues.append(r)
    mock_ca.top.residues = residues
    mock_traj.atom_slice.return_value = mock_ca
    return mock_traj


class TestScoreShape:
    """score() returns (n_residues,) float32 without running real fpocket."""

    def _mock_run_wsl(self, tmp_path, n_residues=10):
        """Return a mock for _run_fpocket_wsl that writes a temp out.pdb."""
        records = [(i + 1, 0.5) for i in range(n_residues)]
        content = _make_out_pdb_lines(records)
        out_pdb = tmp_path / "mock_out.pdb"
        out_pdb.write_text(content)
        return out_pdb

    def test_shape(self, tmp_path):
        n = 10
        out_pdb = self._mock_run_wsl(tmp_path, n)
        fake_input = tmp_path / "in.pdb"
        fake_input.write_text("REMARK mock\n")

        mock_traj = _build_mock_mdtraj(n)

        with (
            patch("cryptic_pocket_phd.fpocket_wrapper._run_fpocket_wsl", return_value=out_pdb),
            patch("cryptic_pocket_phd.fpocket_wrapper._is_windows", return_value=True),
            patch("mdtraj.load", return_value=mock_traj),
        ):
            from cryptic_pocket_phd.fpocket_wrapper import score
            result = score(str(fake_input))

        assert result.shape == (n,)

    def test_dtype_float32(self, tmp_path):
        n = 10
        out_pdb = self._mock_run_wsl(tmp_path, n)
        fake_input = tmp_path / "in.pdb"
        fake_input.write_text("REMARK mock\n")

        mock_traj = _build_mock_mdtraj(n)

        with (
            patch("cryptic_pocket_phd.fpocket_wrapper._run_fpocket_wsl", return_value=out_pdb),
            patch("cryptic_pocket_phd.fpocket_wrapper._is_windows", return_value=True),
            patch("mdtraj.load", return_value=mock_traj),
        ):
            from cryptic_pocket_phd.fpocket_wrapper import score
            result = score(str(fake_input))

        assert result.dtype == np.float32

    def test_values_in_range(self, tmp_path):
        """Occupancy values 0.0–1.0 must stay in [0, 1]."""
        records = [(i + 1, i / 9.0) for i in range(10)]
        content = _make_out_pdb_lines(records)
        out_pdb = tmp_path / "mock_out.pdb"
        out_pdb.write_text(content)
        fake_input = tmp_path / "in.pdb"
        fake_input.write_text("REMARK mock\n")

        mock_traj = _build_mock_mdtraj(10)

        with (
            patch("cryptic_pocket_phd.fpocket_wrapper._run_fpocket_wsl", return_value=out_pdb),
            patch("cryptic_pocket_phd.fpocket_wrapper._is_windows", return_value=True),
            patch("mdtraj.load", return_value=mock_traj),
        ):
            from cryptic_pocket_phd.fpocket_wrapper import score
            result = score(str(fake_input))

        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0 + 1e-6

    def test_residue_not_in_pocket_scores_zero(self, tmp_path):
        """Residues absent from fpocket output → score 0.0."""
        # Only residue 5 in output (resSeq 5 → sequential index 4)
        content = _make_out_pdb_lines([(5, 0.80)])
        out_pdb = tmp_path / "mock_out.pdb"
        out_pdb.write_text(content)
        fake_input = tmp_path / "in.pdb"
        fake_input.write_text("REMARK mock\n")

        mock_traj = _build_mock_mdtraj(10, start_resseq=1)

        with (
            patch("cryptic_pocket_phd.fpocket_wrapper._run_fpocket_wsl", return_value=out_pdb),
            patch("cryptic_pocket_phd.fpocket_wrapper._is_windows", return_value=True),
            patch("mdtraj.load", return_value=mock_traj),
        ):
            from cryptic_pocket_phd.fpocket_wrapper import score
            result = score(str(fake_input))

        # index 4 (resSeq 5) → 0.80; all others → 0.0
        assert result[4] == pytest.approx(0.80, abs=1e-3)
        assert result[0] == pytest.approx(0.0, abs=1e-3)
        assert result[9] == pytest.approx(0.0, abs=1e-3)


# ===========================================================================
# 3. Integration test — real fpocket via WSL on NPC2
# ===========================================================================

@pytest.mark.integration
class TestFpocketNPC2Integration:
    """Real fpocket on 1NEP_A.pdb (NPC2, 130 residues).

    Requirements:
      - Windows + WSL with fpocket at ~/tools/fpocket/bin/fpocket
      - data/validation_pdbs/1NEP_A.pdb present

    Done criterion (Spec §6.2 Task F):
      At least one pocket residue (positions 91-105) has a non-zero fpocket score.
    """

    def test_reference_pdb_exists(self):
        assert _NPC2_APO.exists(), f"NPC2 apo PDB not found: {_NPC2_APO}"

    def test_score_shape(self):
        from cryptic_pocket_phd.fpocket_wrapper import score
        scores = score(str(_NPC2_APO))
        assert scores.shape == (130,), f"Expected (130,), got {scores.shape}"

    def test_score_dtype(self):
        from cryptic_pocket_phd.fpocket_wrapper import score
        scores = score(str(_NPC2_APO))
        assert scores.dtype == np.float32

    def test_pocket_residues_detected(self):
        """At least one NPC2 pocket residue (91-105) must have score > 0."""
        from cryptic_pocket_phd.fpocket_wrapper import score
        scores = score(str(_NPC2_APO))
        pocket_scores = scores[_NPC2_POCKET_IDX]
        n_detected = int((pocket_scores > 0.0).sum())
        print(
            f"\n[integration] pocket_scores={pocket_scores.round(3).tolist()} "
            f"n_detected={n_detected}"
        )
        assert n_detected >= 1, (
            f"No pocket residue detected. pocket_scores={pocket_scores.tolist()}"
        )

    def test_global_nonzero_coverage(self):
        """fpocket must detect at least 1 pocket site (non-zero residues > 0)."""
        from cryptic_pocket_phd.fpocket_wrapper import score
        scores = score(str(_NPC2_APO))
        n_nonzero = int((scores > 0.0).sum())
        assert n_nonzero > 0, "fpocket found no pockets at all"
