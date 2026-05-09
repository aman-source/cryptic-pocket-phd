"""Regression test: InstrumentedDiffusionModule produces identical output to
upstream DiffusionModule when intermediate_capture_fn=None.

This test does NOT require Boltz weights.  It constructs a minimal
DiffusionModule with random weights and verifies that the subclass does not
alter the sampling trajectory.

Run with: pytest tests/test_intermediate_capture.py -v
"""

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Skip the whole module if boltz is not installed.
# This prevents CI failure before we install boltz on the RunPod.
# ---------------------------------------------------------------------------
boltz = pytest.importorskip("boltz", reason="boltz not installed")

from boltz.model.modules.diffusion import DiffusionModule
from cryptic_pocket_phd.intermediate_capture import (
    InstrumentedDiffusionModule,
    make_timestep_capture_fn,
)


# ---------------------------------------------------------------------------
# Minimal config for a tiny DiffusionModule (CPU, small dims)
# ---------------------------------------------------------------------------
SMALL_CONFIG = dict(
    token_s=16,
    token_z=8,
    atom_s=16,
    atom_z=8,
    atoms_per_window_queries=4,
    atoms_per_window_keys=8,
    sigma_data=16,
    dim_fourier=16,
    atom_encoder_depth=1,
    atom_encoder_heads=2,
    token_transformer_depth=1,
    token_transformer_heads=2,
    atom_decoder_depth=1,
    atom_decoder_heads=2,
    atom_feature_dim=16,
    conditioning_transition_layers=1,
)

N_ATOMS = 20       # tiny "protein"
N_TOKENS = 5
BATCH = 1
SEED = 42


def _make_dummy_feats(n_atoms=N_ATOMS, n_tokens=N_TOKENS):
    """Minimal feats dict matching what Boltz sample() accesses."""
    return {
        "token_index": torch.zeros(BATCH, n_tokens, dtype=torch.long),
        "atom_pad_mask": torch.ones(BATCH, n_atoms, dtype=torch.bool),
        # Add any other keys the network forward actually needs.
        # For this regression test we only care that the LOOP body
        # executes without crashing and that atom_coords_next is identical.
    }


def _build_module(cls, seed=SEED):
    """Instantiate a module with fixed random weights."""
    torch.manual_seed(seed)
    return cls(**SMALL_CONFIG).eval()


# ---------------------------------------------------------------------------
# Test 1: subclass output == upstream output when capture fn is None
# ---------------------------------------------------------------------------

class TestRegressionNoCaptureCallback:
    """Verify that adding intermediate_capture_fn=None changes nothing."""

    def _run_sample(self, cls, seed=SEED, n_steps=4):
        """Run sample() and return the final atom coordinates."""
        torch.manual_seed(seed)
        module = _build_module(cls, seed=seed)
        feats = _make_dummy_feats()
        atom_mask = feats["atom_pad_mask"]

        with torch.no_grad():
            if cls is InstrumentedDiffusionModule:
                out = module.sample(
                    atom_mask,
                    num_sampling_steps=n_steps,
                    multiplicity=1,
                    max_parallel_samples=1,
                    intermediate_capture_fn=None,
                    feats=feats,
                )
            else:
                out = module.sample(
                    atom_mask,
                    num_sampling_steps=n_steps,
                    multiplicity=1,
                    max_parallel_samples=1,
                    feats=feats,
                )
        return out["sample_atom_coords"]

    def test_identical_output_no_callback(self):
        """InstrumentedDiffusionModule must reproduce upstream coords exactly."""
        coords_upstream = self._run_sample(DiffusionModule)
        coords_instrumented = self._run_sample(InstrumentedDiffusionModule)

        max_diff = (coords_upstream - coords_instrumented).abs().max().item()
        assert max_diff < 1e-5, (
            f"Subclass diverges from upstream when capture fn is None. "
            f"Max absolute diff = {max_diff:.2e}. "
            f"Check that sample() body was copied verbatim."
        )


# ---------------------------------------------------------------------------
# Test 2: callback is called at every step
# ---------------------------------------------------------------------------

class TestCallbackInvocation:
    """Verify callback fires and receives correct arguments."""

    def test_callback_called_every_step(self):
        n_steps = 4
        torch.manual_seed(SEED)
        module = _build_module(InstrumentedDiffusionModule)
        feats = _make_dummy_feats()
        atom_mask = feats["atom_pad_mask"]

        calls = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            calls.append({
                "step_idx": step_idx,
                "t": t,
                "x_hat_0_shape": x_hat_0.shape,
                "sigma_t": sigma_t,
            })

        with torch.no_grad():
            module.sample(
                atom_mask,
                num_sampling_steps=n_steps,
                multiplicity=1,
                max_parallel_samples=1,
                intermediate_capture_fn=capture,
                feats=feats,
            )

        assert len(calls) == n_steps, (
            f"Expected {n_steps} callback calls, got {len(calls)}"
        )

        # step_idx must be 0, 1, 2, 3
        assert [c["step_idx"] for c in calls] == list(range(n_steps))

        # t = 1 - step_idx/n_steps; first call t=1.0, last call t=0.25 for 4 steps
        expected_ts = [1.0 - i / n_steps for i in range(n_steps)]
        for call, expected_t in zip(calls, expected_ts):
            assert abs(call["t"] - expected_t) < 1e-6, (
                f"step {call['step_idx']}: t={call['t']:.6f}, expected {expected_t:.6f}"
            )

        # x_hat_0 must have shape (batch, n_atoms, 3)
        for call in calls:
            assert call["x_hat_0_shape"] == (BATCH, N_ATOMS, 3)


# ---------------------------------------------------------------------------
# Test 3: make_timestep_capture_fn writes .npz files at correct timesteps
# ---------------------------------------------------------------------------

class TestMakeTimestepCaptureFn:
    """Verify that the factory callback writes files at requested t values."""

    def test_files_written_at_target_ts(self, tmp_path):
        target_ts = [0.9, 0.5]  # using 2-step schedule: steps are t=1.0, t=0.5
        n_steps = 2
        protein_id = "TEST"
        sample_idx = 0

        atom_metadata = {
            "atom_names": ["CA"] * N_ATOMS,
            "res_indices": list(range(1, N_ATOMS + 1)),
            "chain_ids": ["A"] * N_ATOMS,
            "res_names": ["ALA"] * N_ATOMS,
        }

        callback = make_timestep_capture_fn(
            target_ts=target_ts,
            output_dir=str(tmp_path),
            protein_id=protein_id,
            sample_idx=sample_idx,
            atom_metadata=atom_metadata,
            num_sampling_steps=n_steps,
        )

        torch.manual_seed(SEED)
        module = _build_module(InstrumentedDiffusionModule)
        feats = _make_dummy_feats()
        atom_mask = feats["atom_pad_mask"]

        with torch.no_grad():
            module.sample(
                atom_mask,
                num_sampling_steps=n_steps,
                multiplicity=1,
                max_parallel_samples=1,
                intermediate_capture_fn=callback,
                feats=feats,
            )

        # Metadata file must exist
        meta_path = tmp_path / f"{protein_id}_atom_metadata.json"
        assert meta_path.exists(), "atom_metadata.json not written"

        # Check .npz files for each target t that falls within a step
        # With 2 steps: steering_t values are 1.0 and 0.5.
        # Target 0.9 → closest step is t=1.0 (within tol=0.5/n_steps=0.25) ✓
        # Target 0.5 → closest step is t=0.5 (exact) ✓
        for t in target_ts:
            out_path = tmp_path / f"{protein_id}_s{sample_idx:02d}_t{t:.1f}.npz"
            assert out_path.exists(), f"Expected output file for t={t}: {out_path}"

            data = np.load(out_path)
            assert "coords" in data
            assert data["coords"].shape == (BATCH, N_ATOMS, 3)
            assert abs(float(data["t"]) - t) < 1e-6
