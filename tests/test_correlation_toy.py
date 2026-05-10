"""Toy pipeline test for Task E correlation computation.

Uses synthetic noisy intermediates from the NPC2 apo PDB (1NEP_A) to exercise
the full pipeline WITHOUT requiring Boltz inference or CCD cache:

  x_0 coords (from 1NEP_A.pdb)
    + Gaussian noise (σ ∝ t)
    → synthetic x̂_0
    → boltz_coords_to_pdb()
    → PocketMiner score()
    → Spearman ρ vs s_0

Expected behaviour: ρ_pocket near 1.0 for small σ, decreases as σ grows.
Exact values are not asserted (random noise); we test that the pipeline runs
end-to-end without error and that ρ is in [-1, 1].

Run: pytest tests/test_correlation_toy.py -m integration -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NPC2_APO = _REPO_ROOT / "data" / "validation_pdbs" / "1NEP_A.pdb"
_NPC2_POCKET_RANGES = [[91, 105]]
_SIGMAS = [0.5, 2.0, 5.0]  # Å — proxy for low/mid/high noise (t ≈ 0.1/0.5/0.9)


def _extract_backbone_metadata(pdb_path: str) -> tuple[np.ndarray, dict]:
    """Load backbone atoms from PDB; return (coords_angstrom, atom_metadata).

    coords : np.ndarray, shape (n_backbone_atoms, 3)  — Angstroms
    atom_metadata : dict with keys atom_names, res_indices, chain_ids, res_names
    """
    import mdtraj as md

    traj = md.load(str(pdb_path))
    sel = traj.top.select("protein and (name N or name CA or name C or name O)")
    bb = traj.atom_slice(sel)

    coords_nm = bb.xyz[0]  # (n_atoms, 3)  — mdtraj uses nanometres
    coords_ang = coords_nm * 10.0  # → Angstroms

    atom_names = [a.name for a in bb.top.atoms]
    res_indices = [a.residue.index + 1 for a in bb.top.atoms]  # 1-based sequential
    chain_ids = [a.residue.chain.chain_id for a in bb.top.atoms]
    res_names = [a.residue.name for a in bb.top.atoms]

    metadata = {
        "atom_names": atom_names,
        "res_indices": res_indices,
        "chain_ids": chain_ids,
        "res_names": res_names,
    }
    return coords_ang, metadata


@pytest.mark.integration
class TestCorrelationToyPipeline:
    """End-to-end: synthetic noisy PDB → PocketMiner → Spearman ρ."""

    def test_reference_score_runs(self):
        """PocketMiner scores 1NEP_A with shape (130,) and values in [0,1]."""
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from cryptic_pocket_phd.pocketminer_wrapper import score

        s_0 = score(str(_NPC2_APO))
        assert s_0.shape == (130,), f"Expected (130,), got {s_0.shape}"
        assert s_0.min() >= 0.0 and s_0.max() <= 1.0

    def test_noisy_pdb_scoreable(self, tmp_path):
        """Noisy backbone-only PDB is parseable and scoreable by PocketMiner."""
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb
        from cryptic_pocket_phd.pocketminer_wrapper import score

        coords_0, meta = _extract_backbone_metadata(str(_NPC2_APO))
        rng = np.random.default_rng(42)

        noisy_coords = coords_0 + 0.5 * rng.standard_normal(coords_0.shape)
        pdb_path = str(tmp_path / "noisy_sigma0.5.pdb")
        boltz_coords_to_pdb(
            coords=noisy_coords,
            atom_names=meta["atom_names"],
            res_indices=meta["res_indices"],
            chain_ids=meta["chain_ids"],
            res_names=meta["res_names"],
            output_path=pdb_path,
        )
        s_t = score(pdb_path)
        assert s_t.shape == (130,), f"Expected (130,), got {s_t.shape}"
        assert s_t.min() >= 0.0 and s_t.max() <= 1.0

    def test_rho_in_range(self, tmp_path):
        """Spearman ρ is in [-1, 1] for all noise levels."""
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb
        from cryptic_pocket_phd.correlation import compute_rho
        from cryptic_pocket_phd.pocketminer_wrapper import score
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        s_0 = score(str(_NPC2_APO))
        pocket_idx = pocket_residue_indices(_NPC2_POCKET_RANGES, n_residues=len(s_0))
        coords_0, meta = _extract_backbone_metadata(str(_NPC2_APO))
        rng = np.random.default_rng(7)

        for sigma in _SIGMAS:
            noisy_coords = coords_0 + sigma * rng.standard_normal(coords_0.shape)
            pdb_path = str(tmp_path / f"noisy_sigma{sigma}.pdb")
            boltz_coords_to_pdb(
                coords=noisy_coords,
                atom_names=meta["atom_names"],
                res_indices=meta["res_indices"],
                chain_ids=meta["chain_ids"],
                res_names=meta["res_names"],
                output_path=pdb_path,
            )
            s_t = score(pdb_path)
            rho_p, rho_all = compute_rho(s_t, s_0, pocket_idx)
            assert -1.0 <= rho_p <= 1.0, f"sigma={sigma}: rho_pocket={rho_p} out of range"
            assert -1.0 <= rho_all <= 1.0, f"sigma={sigma}: rho_all={rho_all} out of range"

    def test_rho_decreases_with_noise(self, tmp_path):
        """ρ_pocket at σ=0.5 Å > ρ_pocket at σ=5 Å (monotone degradation sanity)."""
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from cryptic_pocket_phd.boltz_to_pdb import boltz_coords_to_pdb
        from cryptic_pocket_phd.correlation import compute_rho
        from cryptic_pocket_phd.pocketminer_wrapper import score
        from cryptic_pocket_phd.residue_mapping import pocket_residue_indices

        s_0 = score(str(_NPC2_APO))
        pocket_idx = pocket_residue_indices(_NPC2_POCKET_RANGES, n_residues=len(s_0))
        coords_0, meta = _extract_backbone_metadata(str(_NPC2_APO))
        rng = np.random.default_rng(99)

        rhos = {}
        for sigma in [0.5, 5.0]:
            noisy_coords = coords_0 + sigma * rng.standard_normal(coords_0.shape)
            pdb_path = str(tmp_path / f"noisy_sigma{sigma}.pdb")
            boltz_coords_to_pdb(
                coords=noisy_coords,
                atom_names=meta["atom_names"],
                res_indices=meta["res_indices"],
                chain_ids=meta["chain_ids"],
                res_names=meta["res_names"],
                output_path=pdb_path,
            )
            s_t = score(pdb_path)
            rho_p, _ = compute_rho(s_t, s_0, pocket_idx)
            rhos[sigma] = rho_p

        assert rhos[0.5] > rhos[5.0], (
            f"Expected ρ to degrade with noise: σ=0.5 ρ={rhos[0.5]:.3f}, "
            f"σ=5.0 ρ={rhos[5.0]:.3f}"
        )
