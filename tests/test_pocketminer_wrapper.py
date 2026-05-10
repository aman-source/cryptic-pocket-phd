"""Tests for Task C (PocketMiner wrapper) and Task D (residue mapping + format conversion).

Test structure
--------------
All classes EXCEPT TestTopQuartileIntegration run without TF or mdtraj:
  - score() is called with mock preprocess_fn + mock model → no TF needed.
  - residue_mapping tested with mocked mdtraj.load.
  - boltz_to_pdb tested purely with numpy + file I/O.

Integration test (requires TF + mdtraj, marks: integration):
  - Real PocketMiner on 1K3FB_clean_h.pdb (UdP apo, in pocketminer repo).
  - Verifies ≥1 pocket residue (sequential positions 161-190, 224-240) in top quartile.
  - Run with: pytest -m integration tests/test_pocketminer_wrapper.py

Residue numbering note
----------------------
pocket_residue_ranges use 1-based SEQUENTIAL POSITIONS (= Boltz index + 1),
NOT PDB resSeq.  1K3F chain B has PDB resSeq starting at 1001; sequential
positions 161-190 → Boltz indices 160-189.  See residue_mapping.pocket_residue_indices().
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PM_APO = _REPO_ROOT / "external" / "pocketminer" / "data" / "pm-dataset" / "apo-structures"
_PM_K3F_APO = _PM_APO / "1K3FB_clean_h.pdb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
N_RESIDUES = 60  # synthetic protein size for unit tests
L_MAX = 64       # padded length (L_max ≥ N_RESIDUES)


def _fake_preprocess(structure_path: str):
    """Mock preprocess_fn: returns (X, S, mask) for N_RESIDUES protein residues."""
    X = np.zeros([1, L_MAX, 4, 3], dtype=np.float32)
    S = np.zeros([1, L_MAX], dtype=np.int32)
    mask = np.zeros([1, L_MAX], dtype=np.float32)
    mask[0, :N_RESIDUES] = 1.0
    return X, S, mask


class _FakeModel:
    """Mock model: returns deterministic sigmoid-like values in (0, 1)."""

    def __call__(self, X, S, mask, train=False, res_level=False):
        n = X.shape[1]
        # Linspace 0.05..0.95, high scores toward the end to test top-quartile
        vals = np.linspace(0.05, 0.95, n).astype(np.float32)
        return vals[np.newaxis, :]  # shape [1, n]


def _make_mock_mdtraj(resSeqs: list[int]):
    """Build a mock mdtraj.Trajectory whose CA residues have given resSeqs."""
    mock_traj = MagicMock()

    # select("protein and name CA") → dummy atom indices
    mock_traj.top.select.return_value = list(range(len(resSeqs)))

    # atom_slice → sliced trajectory with residues
    mock_ca_traj = MagicMock()
    mock_residues = []
    for resSeq in resSeqs:
        res = MagicMock()
        res.resSeq = resSeq
        mock_residues.append(res)
    mock_ca_traj.top.residues = mock_residues
    mock_traj.atom_slice.return_value = mock_ca_traj

    return mock_traj


# ===========================================================================
# 1. score() — shape and dtype
# ===========================================================================

class TestScoreShape:
    """score() returns ndarray of shape (n_residues,) and dtype float32."""

    def test_shape(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        result = score(
            str(fake_pdb),
            model=_FakeModel(),
            preprocess_fn=_fake_preprocess,
        )
        assert result.shape == (N_RESIDUES,), f"Expected ({N_RESIDUES},), got {result.shape}"

    def test_dtype_float32(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        result = score(str(fake_pdb), model=_FakeModel(), preprocess_fn=_fake_preprocess)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_str_path_accepted(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        # Should not raise even if path is a str (not Path)
        result = score(str(fake_pdb), model=_FakeModel(), preprocess_fn=_fake_preprocess)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# 2. score() — value range [0, 1]
# ===========================================================================

class TestScoreValueRange:
    """score() values must be in [0, 1]."""

    def test_min_ge_0(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        result = score(str(fake_pdb), model=_FakeModel(), preprocess_fn=_fake_preprocess)
        assert result.min() >= 0.0, f"min={result.min()} < 0"

    def test_max_le_1(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        result = score(str(fake_pdb), model=_FakeModel(), preprocess_fn=_fake_preprocess)
        assert result.max() <= 1.0, f"max={result.max()} > 1"

    def test_no_nan_or_inf(self, tmp_path):
        fake_pdb = tmp_path / "fake.pdb"
        fake_pdb.write_text("REMARK mock\n")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        result = score(str(fake_pdb), model=_FakeModel(), preprocess_fn=_fake_preprocess)
        assert np.isfinite(result).all(), "NaN or Inf in scores"


# ===========================================================================
# 3. residue_mapping.build_residue_mapping — gaps and offsets
# ===========================================================================

class TestResidueMappingBuild:
    """build_residue_mapping maps 0-based sequential index → PDB resSeq.
    Handles chains with non-trivial numbering (gaps, offsets).
    """

    def test_sequential_mapping_standard(self):
        """Chain A, resSeq 1..5 → {0:1, 1:2, 2:3, 3:4, 4:5}."""
        resSeqs = [1, 2, 3, 4, 5]
        mock_traj = _make_mock_mdtraj(resSeqs)

        with patch("mdtraj.load", return_value=mock_traj):
            from cryptic_pocket_phd.residue_mapping import build_residue_mapping

            mapping = build_residue_mapping("fake.pdb")

        assert mapping == {i: resSeqs[i] for i in range(len(resSeqs))}

    def test_chain_b_offset(self):
        """Chain B, resSeq 1001..1005 → {0:1001, 1:1002, ..., 4:1005}."""
        resSeqs = [1001, 1002, 1003, 1004, 1005]
        mock_traj = _make_mock_mdtraj(resSeqs)

        with patch("mdtraj.load", return_value=mock_traj):
            from cryptic_pocket_phd.residue_mapping import build_residue_mapping

            mapping = build_residue_mapping("fake.pdb")

        assert mapping == {0: 1001, 1: 1002, 2: 1003, 3: 1004, 4: 1005}

    def test_gaps_in_numbering(self):
        """resSeq with gap [10, 12, 13, 15] → correct sequential indices."""
        resSeqs = [10, 12, 13, 15]
        mock_traj = _make_mock_mdtraj(resSeqs)

        with patch("mdtraj.load", return_value=mock_traj):
            from cryptic_pocket_phd.residue_mapping import build_residue_mapping

            mapping = build_residue_mapping("fake.pdb")

        assert mapping == {0: 10, 1: 12, 2: 13, 3: 15}


# ===========================================================================
# 4. residue_mapping.save/load round-trip
# ===========================================================================

class TestResidueMappingSaveLoad:
    """Mapping written to JSON and loaded back, keys int → int."""

    def test_roundtrip(self, tmp_path):
        resSeqs = [10, 12, 13, 15]
        mock_traj = _make_mock_mdtraj(resSeqs)

        with patch("mdtraj.load", return_value=mock_traj):
            from cryptic_pocket_phd.residue_mapping import load_residue_mapping, save_residue_mapping

            mapping = save_residue_mapping("P99999", "fake.pdb", str(tmp_path))

        # Keys and values must be int (not str)
        assert all(isinstance(k, int) for k in mapping)
        assert all(isinstance(v, int) for v in mapping.values())

        # Round-trip via JSON
        loaded = load_residue_mapping("P99999", str(tmp_path))
        assert loaded == mapping

    def test_json_keys_are_strings_on_disk(self, tmp_path):
        """JSON spec: keys must be strings.  We store {str(int): int}."""
        resSeqs = [1, 2, 3]
        mock_traj = _make_mock_mdtraj(resSeqs)

        with patch("mdtraj.load", return_value=mock_traj):
            from cryptic_pocket_phd.residue_mapping import save_residue_mapping

            save_residue_mapping("P11111", "fake.pdb", str(tmp_path))

        json_path = tmp_path / "P11111" / "residue_mapping.json"
        raw = json.loads(json_path.read_text())
        assert all(isinstance(k, str) for k in raw)


# ===========================================================================
# 5. residue_mapping.boltz_idx_to_pdb_resid
# ===========================================================================

class TestBoltzIdxToPdbResid:
    """boltz_idx_to_pdb_resid returns correct PDB resSeq."""

    def _write_mapping(self, tmp_path, resSeqs):
        mapping_dir = tmp_path / "P12345"
        mapping_dir.mkdir()
        data = {str(i): r for i, r in enumerate(resSeqs)}
        (mapping_dir / "residue_mapping.json").write_text(json.dumps(data))
        return str(tmp_path)

    def test_index_0(self, tmp_path):
        data_dir = self._write_mapping(tmp_path, [10, 12, 13, 15])

        from cryptic_pocket_phd.residue_mapping import boltz_idx_to_pdb_resid

        assert boltz_idx_to_pdb_resid("P12345", 0, data_dir) == 10

    def test_index_with_gap(self, tmp_path):
        data_dir = self._write_mapping(tmp_path, [10, 12, 13, 15])

        from cryptic_pocket_phd.residue_mapping import boltz_idx_to_pdb_resid

        # boltz 1 → resSeq 12 (gap at 11)
        assert boltz_idx_to_pdb_resid("P12345", 1, data_dir) == 12

    def test_out_of_range_raises(self, tmp_path):
        data_dir = self._write_mapping(tmp_path, [10, 12])

        from cryptic_pocket_phd.residue_mapping import boltz_idx_to_pdb_resid

        with pytest.raises(KeyError):
            boltz_idx_to_pdb_resid("P12345", 99, data_dir)


# ===========================================================================
# 6. residue_mapping.pocket_residue_indices — sequential positions
# ===========================================================================

class TestPocketResidueIndices:
    """pocket_residue_indices uses 1-based sequential positions (NOT PDB resSeq).

    This design handles chains with resSeq offsets (e.g. chain B starting at 1001).
    """

    def test_simple_range(self):
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        # n_residues=10, pocket [3, 5] → boltz {2, 3, 4}
        result = pocket_residue_indices([[3, 5]], n_residues=10)
        assert result == [2, 3, 4]

    def test_multiple_ranges(self):
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        # pocket [2,3] and [7,8] → boltz {1,2,6,7}
        result = pocket_residue_indices([[2, 3], [7, 8]], n_residues=10)
        assert result == [1, 2, 6, 7]

    def test_out_of_range_clipped(self):
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        # n_residues=5, range [4, 8] → only {3, 4}
        result = pocket_residue_indices([[4, 8]], n_residues=5)
        assert result == [3, 4]

    def test_udp_pocket_ranges_vs_k3fb_length(self):
        """UdP (1K3F_B): length 253, pockets [[161,190],[224,240]].
        Boltz indices 160..189 and 223..239 must all be within [0, 252].
        """
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        result = pocket_residue_indices([[161, 190], [224, 240]], n_residues=253)
        assert len(result) == 30 + 17  # 30 + 17 = 47 pocket residues
        assert min(result) == 160
        assert max(result) == 239
        assert all(0 <= idx < 253 for idx in result)


# ===========================================================================
# 7. boltz_to_pdb — format conversion (no TF/mdtraj)
# ===========================================================================

class TestBoltzToPdb:
    """boltz_coords_to_pdb writes correct PDB ATOM records."""

    def _make_inputs(self, n_residues=2):
        """2 residues × {N, CA, C, O, CB} = 10 atoms per residue."""
        atom_names = ["N", "CA", "C", "O", "CB"] * n_residues
        res_indices = []
        for r in range(1, n_residues + 1):
            res_indices.extend([r] * 5)
        chain_ids = ["A"] * (5 * n_residues)
        res_names = ["ALA"] * (5 * n_residues)
        coords = np.random.default_rng(42).random((5 * n_residues, 3)).astype(np.float32)
        return atom_names, res_indices, chain_ids, res_names, coords

    def test_only_backbone_atoms_written(self, tmp_path):
        """CB atoms must be excluded; only N, CA, C, O written."""
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb

        atom_names, res_indices, chain_ids, res_names, coords = self._make_inputs(2)
        out = str(tmp_path / "out.pdb")
        boltz_coords_to_pdb(coords, atom_names, res_indices, chain_ids, res_names, out)

        lines = [l for l in Path(out).read_text().splitlines() if l.startswith("ATOM")]
        assert len(lines) == 8, f"Expected 8 backbone ATOM lines (2 res × 4 atoms), got {len(lines)}"
        atom_names_written = {l[12:16].strip() for l in lines}
        assert atom_names_written == {"N", "CA", "C", "O"}

    def test_resseq_offset_applied(self, tmp_path):
        """resseq_offset={1:100, 2:102} should appear in PDB resSeq column."""
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb

        atom_names, res_indices, chain_ids, res_names, coords = self._make_inputs(2)
        out = str(tmp_path / "out.pdb")
        boltz_coords_to_pdb(
            coords, atom_names, res_indices, chain_ids, res_names, out,
            resseq_offset={1: 100, 2: 102},
        )

        lines = [l for l in Path(out).read_text().splitlines() if l.startswith("ATOM")]
        resseqs = {int(l[22:26]) for l in lines}
        assert resseqs == {100, 102}

    def test_batch_dim_squeezed(self, tmp_path):
        """coords shape (batch, n_atoms, 3) — first batch element taken."""
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb

        atom_names, res_indices, chain_ids, res_names, coords = self._make_inputs(2)
        batched = coords[np.newaxis, :, :]  # (1, 10, 3)

        out = str(tmp_path / "out.pdb")
        boltz_coords_to_pdb(batched, atom_names, res_indices, chain_ids, res_names, out)

        lines = [l for l in Path(out).read_text().splitlines() if l.startswith("ATOM")]
        assert len(lines) == 8

    def test_npz_to_pdb(self, tmp_path):
        """npz_to_pdb: load .npz, extract coords, write PDB."""
        from cryptic_pocket_phd.boltz_to_pdb import npz_to_pdb

        atom_names, res_indices, chain_ids, res_names, coords = self._make_inputs(2)
        npz_path = str(tmp_path / "capture.npz")
        np.savez(npz_path, coords=coords[np.newaxis, :, :])  # (1, n_atoms, 3)

        meta = {
            "atom_names": atom_names,
            "res_indices": res_indices,
            "chain_ids": chain_ids,
            "res_names": res_names,
        }
        out = str(tmp_path / "from_npz.pdb")
        result_path = npz_to_pdb(npz_path, meta, out)

        lines = [l for l in Path(result_path).read_text().splitlines() if l.startswith("ATOM")]
        assert len(lines) == 8


# ===========================================================================
# 8. Integration test — real PocketMiner inference (mark: integration)
# ===========================================================================

@pytest.mark.integration
class TestTopQuartileIntegration:
    """End-to-end PocketMiner inference on UdP apo structure (1K3F_B).

    Requirements:
      - TF installed with TF_USE_LEGACY_KERAS=1 or legacy Keras backend
      - mdtraj installed
      - external/pocketminer/models/pocketminer.* checkpoint present
      - external/pocketminer/data/pm-dataset/apo-structures/1K3FB_clean_h.pdb present

    Done criterion (Spec §6.2 Task C):
      At least one residue in the annotated cryptic pocket region (sequential
      positions 161-190 and 224-240 for UdP/P12758) scores in the top quartile
      of all residues in that structure.
    """

    def test_apo_pdb_exists(self):
        assert _PM_K3F_APO.exists(), (
            f"PocketMiner apo PDB not found: {_PM_K3F_APO}. "
            "Restore with: cd external/pocketminer && git checkout -- data/"
        )

    def test_score_shape_and_range(self):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score

        reset_model()
        scores = score(str(_PM_K3F_APO))

        # UdP (P12758) length 253
        assert scores.shape == (253,), f"Expected (253,), got {scores.shape}"
        assert scores.dtype == np.float32
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_pocket_residues_in_top_quartile(self):
        """At least 1 annotated pocket residue (pos 161-190, 224-240) in Q75+."""
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        from cryptic_pocket_phd.pocketminer_wrapper import reset_model, score
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        reset_model()
        scores = score(str(_PM_K3F_APO))

        n_residues = len(scores)
        pocket_idx = pocket_residue_indices([[161, 190], [224, 240]], n_residues=n_residues)
        assert len(pocket_idx) > 0, "No pocket indices found — check ranges vs n_residues"

        q75 = float(np.percentile(scores, 75))
        pocket_scores = scores[pocket_idx]
        n_top = int((pocket_scores >= q75).sum())

        print(
            f"\n[integration] n_residues={n_residues}, n_pocket={len(pocket_idx)}, "
            f"Q75={q75:.4f}, pocket_max={pocket_scores.max():.4f}, "
            f"n_pocket_in_top_quartile={n_top}"
        )

        assert n_top >= 1, (
            f"No pocket residue scored in top quartile (Q75={q75:.4f}). "
            f"Pocket score range: [{pocket_scores.min():.4f}, {pocket_scores.max():.4f}]. "
            "Check that annotations match sequential (not PDB resSeq) positions."
        )
