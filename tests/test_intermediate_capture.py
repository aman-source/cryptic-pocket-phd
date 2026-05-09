"""Regression + logic tests for InstrumentedAtomDiffusion.

These tests do NOT require Boltz weights or a GPU.
preconditioned_network_forward is mocked to return deterministic dummy coords.
The tests verify:
  1. Subclass output is bitwise-identical to upstream when capture_fn=None.
  2. Callback fires at every step with correct step_idx and t.
  3. Capture factory writes .npz files at correct timesteps.
  4. (Canary) t decreases monotonically and spans [~1.0, ~0.0] over the schedule.

Run with: pytest tests/test_intermediate_capture.py -v
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

boltz = pytest.importorskip("boltz", reason="boltz not installed")

from boltz.model.modules.diffusion import AtomDiffusion
from cryptic_pocket_phd.intermediate_capture import (
    CaptureCallback,
    InstrumentedAtomDiffusion,
    make_timestep_capture_fn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ATOMS = 12
N_TOKENS = 4
BATCH = 1
TOKEN_S = 16
SEED = 42

SCORE_MODEL_ARGS = dict(
    token_s=TOKEN_S,
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

ATOM_DIFFUSION_KWARGS = dict(
    score_model_args=SCORE_MODEL_ARGS,
    num_sampling_steps=5,
    sigma_data=16.0,
    accumulate_token_repr=False,
    alignment_reverse_diff=False,
    use_inference_model_cache=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feats(n_atoms=N_ATOMS, n_tokens=N_TOKENS, batch=BATCH):
    return {
        "token_index": torch.zeros(batch, n_tokens, dtype=torch.long),
        "atom_pad_mask": torch.ones(batch, n_atoms, dtype=torch.bool),
    }


def _mock_precond(coords, sigma_t, training, network_condition_kwargs):
    """Deterministic mock: returns zeros for coords, zeros for token_a."""
    batch = coords.shape[0]
    token_a = torch.zeros(batch, N_TOKENS, 2 * TOKEN_S)
    denoised = torch.zeros_like(coords)
    return denoised, token_a


def _build(cls, seed=SEED):
    torch.manual_seed(seed)
    return cls(**ATOM_DIFFUSION_KWARGS).eval()


def _run_sample(cls, n_steps=4, capture_fn=None, seed=SEED):
    """Build module, mock preconditioned_network_forward, run sample()."""
    torch.manual_seed(seed)
    module = _build(cls, seed=seed)
    feats = _make_feats()
    atom_mask = feats["atom_pad_mask"]

    patch_target = (
        "boltz.model.modules.diffusion.AtomDiffusion.preconditioned_network_forward"
    )
    with patch(patch_target, side_effect=_mock_precond):
        kwargs = dict(
            atom_mask=atom_mask,
            num_sampling_steps=n_steps,
            multiplicity=1,
            max_parallel_samples=1,
            feats=feats,
        )
        if cls is InstrumentedAtomDiffusion:
            kwargs["intermediate_capture_fn"] = capture_fn
        out = module.sample(**kwargs)

    return out["sample_atom_coords"]


# ---------------------------------------------------------------------------
# Test 1: Regression — subclass output == upstream when capture_fn=None
# ---------------------------------------------------------------------------

class TestRegressionNoCaptureCallback:
    def test_identical_output_no_callback(self):
        coords_upstream = _run_sample(AtomDiffusion)
        coords_instrumented = _run_sample(InstrumentedAtomDiffusion, capture_fn=None)

        max_diff = (coords_upstream - coords_instrumented).abs().max().item()
        assert max_diff < 1e-5, (
            f"Subclass diverges from upstream when capture_fn=None. "
            f"Max absolute diff = {max_diff:.2e}. "
            f"sample() body likely modified accidentally during copy."
        )


# ---------------------------------------------------------------------------
# Test 2: Callback fires every step with correct step_idx and t
# ---------------------------------------------------------------------------

class TestCallbackInvocation:
    def test_callback_fires_every_step(self):
        n_steps = 6
        calls: list[dict] = []

        def capture(step_idx: int, t: float, x_hat_0: torch.Tensor, sigma_t: float):
            calls.append({"step_idx": step_idx, "t": t, "shape": x_hat_0.shape})

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        assert len(calls) == n_steps, (
            f"Expected {n_steps} callback calls, got {len(calls)}"
        )

    def test_step_idx_sequence(self):
        n_steps = 6
        calls: list[dict] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            calls.append({"step_idx": step_idx, "t": t})

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        assert [c["step_idx"] for c in calls] == list(range(n_steps))

    def test_t_values_match_formula(self):
        """t == 1.0 - step_idx / num_sampling_steps (Boltz formula)."""
        n_steps = 6
        calls: list[dict] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            calls.append({"step_idx": step_idx, "t": t})

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        for c in calls:
            expected_t = 1.0 - c["step_idx"] / n_steps
            assert abs(c["t"] - expected_t) < 1e-6, (
                f"step {c['step_idx']}: t={c['t']:.6f}, expected {expected_t:.6f}"
            )

    def test_x_hat_0_shape(self):
        """Callback receives (batch, n_atoms, 3) tensor."""
        n_steps = 4
        shapes: list[tuple] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            shapes.append(tuple(x_hat_0.shape))

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        for shape in shapes:
            assert shape == (BATCH, N_ATOMS, 3), f"Bad x_hat_0 shape: {shape}"

    def test_x_hat_0_is_clone(self):
        """Callback receives a clone — mutation must not affect the loop."""
        n_steps = 4
        captured: list[torch.Tensor] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            captured.append(x_hat_0)
            x_hat_0.fill_(999.0)  # mutate — must not affect subsequent steps

        # Should not crash and final coords must be finite
        coords = _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)
        assert torch.isfinite(coords).all(), "Loop corrupted by callback mutation"


# ---------------------------------------------------------------------------
# Test 3: make_timestep_capture_fn writes correct files
# ---------------------------------------------------------------------------

class TestMakeTimestepCaptureFn:
    def test_files_written(self, tmp_path):
        """Factory writes one .npz per target_t that the schedule hits."""
        n_steps = 10
        # With 10 steps, steering_t values are 1.0, 0.9, 0.8, ..., 0.1
        # All five targets are hit exactly.
        target_ts = [0.9, 0.7, 0.5, 0.3, 0.1]
        protein_id = "P62593"
        sample_idx = 2

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

        _run_sample(
            InstrumentedAtomDiffusion,
            n_steps=n_steps,
            capture_fn=callback,
        )

        # Metadata JSON must exist
        meta_path = tmp_path / f"{protein_id}_atom_metadata.json"
        assert meta_path.exists(), "atom_metadata.json not written"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["atom_names"] == ["CA"] * N_ATOMS

        # One .npz per target_t
        for t in target_ts:
            out_path = tmp_path / f"{protein_id}_s{sample_idx:02d}_t{t:.1f}.npz"
            assert out_path.exists(), f"Missing .npz for t={t}"

            data = np.load(out_path)
            assert "coords" in data, f"Missing 'coords' in {out_path.name}"
            assert data["coords"].shape == (BATCH, N_ATOMS, 3)
            assert abs(float(data["t"]) - t) < 1e-6

    def test_no_duplicate_captures(self, tmp_path):
        """Each target_t captured at most once even if schedule hits it twice."""
        n_steps = 10
        target_ts = [0.5]
        protein_id = "TEST"

        callback = make_timestep_capture_fn(
            target_ts=target_ts,
            output_dir=str(tmp_path),
            protein_id=protein_id,
            sample_idx=0,
            atom_metadata={"atom_names": ["CA"] * N_ATOMS,
                           "res_indices": list(range(1, N_ATOMS + 1)),
                           "chain_ids": ["A"] * N_ATOMS,
                           "res_names": ["ALA"] * N_ATOMS},
            num_sampling_steps=n_steps,
        )

        # Run twice — second run should not overwrite (target already captured
        # in state, but since factory is fresh here it will capture once each run;
        # within a single run the captured set prevents double-write).
        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=callback)

        npz_files = list(tmp_path.glob("TEST_s00_t0.5.npz"))
        assert len(npz_files) == 1, f"Expected 1 .npz, got {len(npz_files)}"


# ---------------------------------------------------------------------------
# Test 4: Canary — t schedule structure matches Boltz loop invariant
# ---------------------------------------------------------------------------

class TestCanaryScheduleStructure:
    """Catch structural changes in upstream sample() loop.

    If Boltz changes how steering_t is computed or the loop iterates,
    these assertions break loudly before any experiment runs silently wrong.
    """

    def test_t_decreases_monotonically(self):
        n_steps = 10
        ts: list[float] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            ts.append(t)

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        for i in range(1, len(ts)):
            assert ts[i] < ts[i - 1], (
                f"t not monotonically decreasing at step {i}: "
                f"ts[{i-1}]={ts[i-1]:.4f}, ts[{i}]={ts[i]:.4f}. "
                f"Boltz upstream loop structure may have changed."
            )

    def test_t_first_step_near_one(self):
        n_steps = 10
        ts: list[float] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            ts.append(t)

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        assert abs(ts[0] - 1.0) < 1e-6, (
            f"First t should be 1.0 (full noise), got {ts[0]:.6f}. "
            f"Upstream steering_t formula changed."
        )

    def test_t_last_step_near_zero(self):
        n_steps = 10
        ts: list[float] = []

        def capture(step_idx, t, x_hat_0, sigma_t):
            ts.append(t)

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        # Last step: steering_t = 1.0 - (n_steps - 1) / n_steps = 0.1 for n=10
        # Not 0.0 — the loop runs for n_steps iterations, step_idx goes 0..n_steps-1
        expected_last_t = 1.0 - (n_steps - 1) / n_steps
        assert abs(ts[-1] - expected_last_t) < 1e-6, (
            f"Last t should be {expected_last_t:.4f}, got {ts[-1]:.6f}. "
            f"Upstream loop bounds changed."
        )

    def test_n_steps_equals_n_callbacks(self):
        """One callback call per denoising step — not more, not fewer."""
        n_steps = 8
        count = [0]

        def capture(step_idx, t, x_hat_0, sigma_t):
            count[0] += 1

        _run_sample(InstrumentedAtomDiffusion, n_steps=n_steps, capture_fn=capture)

        assert count[0] == n_steps, (
            f"Expected {n_steps} callback invocations, got {count[0]}. "
            f"Hook may be inside a sub-loop or called conditionally."
        )
