# Copied from boltz commit cb04aeccdd480fd4db707f0bbafde538397fa2ac,
# src/boltz/model/modules/diffusion.py, DiffusionModule.sample().
# Modified: added `intermediate_capture_fn` callback parameter, called after
# preconditioned_network_forward (and after any guidance/steering update) at
# each denoising step, before the Euler update.
# Callback is injected at the location marked "# <<< CAPTURE HOOK >>>".
# All other logic is identical to upstream.

from __future__ import annotations

from math import sqrt
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from boltz.model.modules.diffusion import DiffusionModule
from boltz.model.modules.utils import compute_random_augmentation, default
from boltz.model.loss.diffusion import weighted_rigid_align
from boltz.model.potentials.potentials import get_potentials


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------

# Called once per denoising step (always — filtering by desired timestep is
# the caller's responsibility, not the sampler's).
#
# Args:
#   step_idx  : int   — 0-indexed step position in the schedule
#   t         : float — normalised time ∈ (0, 1]; 1 = full noise, 0 = clean.
#                       Computed as steering_t = 1.0 - step_idx / num_sampling_steps
#                       (same variable Boltz uses for guidance).
#   x_hat_0   : Tensor[batch, n_atoms, 3] — denoiser's prediction of x_0
#                       at the current step.  This is atom_coords_denoised
#                       *after* any guidance/steering update, *before* the
#                       Euler step.  Cloned — safe to keep a reference.
#   sigma_t   : float — effective sigma (t_hat) at this step.
CaptureCallback = Callable[[int, float, Tensor, float], None]


# ---------------------------------------------------------------------------
# Subclass
# ---------------------------------------------------------------------------

class InstrumentedDiffusionModule(DiffusionModule):
    """DiffusionModule with an intermediate-state capture hook.

    Identical to the upstream ``DiffusionModule.sample()`` in every way
    except that it accepts an optional ``intermediate_capture_fn`` callback
    that is invoked after each denoising step.

    Usage
    -----
    >>> def my_callback(step_idx, t, x_hat_0, sigma_t):
    ...     if abs(t - 0.5) < 0.02:
    ...         np.save(f"x_hat_t{t:.1f}.npy", x_hat_0.cpu().numpy())
    ...
    >>> module = InstrumentedDiffusionModule.from_pretrained(...)
    >>> module.sample(..., intermediate_capture_fn=my_callback)
    """

    def sample(  # noqa: C901  (complexity is inherited from upstream)
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        train_accumulate_token_repr=False,
        steering_args=None,
        intermediate_capture_fn: Optional[CaptureCallback] = None,
        **network_condition_kwargs,
    ):
        # ------------------------------------------------------------------ #
        # Everything below is copied verbatim from boltz sample() except the  #
        # two lines marked <<< CAPTURE HOOK >>>.                               #
        # ------------------------------------------------------------------ #

        if steering_args is not None and (
            steering_args["fk_steering"] or steering_args["physical_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=False)
        if steering_args is not None and steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if steering_args is not None and steering_args["physical_guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)
        token_repr_shape = (
            multiplicity,
            network_condition_kwargs["feats"]["token_index"].shape[1],
            2 * self.token_s,
        )

        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if (
                steering_args is not None
                and steering_args["physical_guidance_update"]
                and scaled_guidance_update is not None
            ):
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                token_a = torch.zeros(token_repr_shape).to(atom_coords_noisy)

                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )
                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk, token_a_chunk = (
                        self.preconditioned_network_forward(
                            atom_coords_noisy[sample_ids_chunk],
                            t_hat,
                            training=False,
                            network_condition_kwargs=dict(
                                multiplicity=sample_ids_chunk.numel(),
                                model_cache=model_cache,
                                **network_condition_kwargs,
                            ),
                        )
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk
                    token_a[sample_ids_chunk] = token_a_chunk

                if (
                    steering_args is not None
                    and steering_args["fk_steering"]
                    and (
                        (
                            step_idx % steering_args["fk_resampling_interval"] == 0
                            and noise_var > 0
                        )
                        or step_idx == num_sampling_steps - 1
                    )
                ):
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    if steering_args["physical_guidance_update"] and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                if (
                    steering_args is not None
                    and steering_args["physical_guidance_update"]
                    and step_idx < num_sampling_steps - 1
                ):
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if (
                    steering_args is not None
                    and steering_args["fk_steering"]
                    and (
                        (
                            step_idx % steering_args["fk_resampling_interval"] == 0
                            and noise_var > 0
                        )
                        or step_idx == num_sampling_steps - 1
                    )
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if steering_args["physical_guidance_update"]:
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]
                    if token_a is not None:
                        token_a = token_a[resample_indices]

            # <<< CAPTURE HOOK >>>
            # Called after all steering/guidance has been applied to
            # atom_coords_denoised (= x̂_0(x_t)), before the Euler step.
            # intermediate_capture_fn decides which steps to record.
            if intermediate_capture_fn is not None:
                intermediate_capture_fn(
                    step_idx,
                    steering_t,
                    atom_coords_denoised.clone(),
                    t_hat,
                )
            # <<< END CAPTURE HOOK >>>

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )
                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)


# ---------------------------------------------------------------------------
# Capture callback factory
# ---------------------------------------------------------------------------

def make_timestep_capture_fn(
    target_ts: list[float],
    output_dir: str,
    protein_id: str,
    sample_idx: int,
    atom_metadata: dict,
    num_sampling_steps: int,
    tol: float = 0.5 / 200,  # half a step at 200-step schedule
) -> CaptureCallback:
    """Return a callback that saves x̂_0 at requested normalised timesteps.

    Parameters
    ----------
    target_ts : list of float
        Normalised timesteps to capture, e.g. [0.1, 0.3, 0.5, 0.7, 0.9].
        Each t ∈ (0, 1], where 1 = full noise, 0 = clean.
    output_dir : str
        Directory to write files into.  Must exist.
    protein_id : str
        UniProt ID, used in output filenames.
    sample_idx : int
        Sample index (0-based), used in filenames.
    atom_metadata : dict
        Must contain:
          - "atom_names":   list[str]  e.g. ["N", "CA", "C", ...]
          - "res_indices":  list[int]  1-based residue numbers per atom
          - "chain_ids":    list[str]  per atom
          - "res_names":    list[str]  three-letter codes per atom
    num_sampling_steps : int
        Total number of denoising steps (used to set tolerance).
    tol : float
        Tolerance for matching t to a target. Default = half a step at 200
        steps = 0.0025. Increase if your schedule has fewer steps.

    Returns
    -------
    CaptureCallback
        A function with signature (step_idx, t, x_hat_0, sigma_t) -> None.

    Output files
    ------------
    One .npz per captured (protein_id, sample_idx, t), containing:
      - "coords":      float32 array (batch, n_atoms, 3)
      - "t":           float scalar
      - "sigma_t":     float scalar
      - "step_idx":    int scalar
    One metadata JSON written once (skipped if already exists):
      - {output_dir}/{protein_id}_atom_metadata.json
    """
    import json
    import os
    import numpy as np

    tol = max(tol, 1.0 / (2 * num_sampling_steps))

    # Write atom metadata once
    meta_path = os.path.join(output_dir, f"{protein_id}_atom_metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(atom_metadata, f)

    captured = set()

    def callback(step_idx: int, t: float, x_hat_0: Tensor, sigma_t: float) -> None:
        for target_t in target_ts:
            if target_t in captured:
                continue
            if abs(t - target_t) <= tol:
                out_path = os.path.join(
                    output_dir,
                    f"{protein_id}_s{sample_idx:02d}_t{target_t:.1f}.npz",
                )
                import numpy as np
                np.savez(
                    out_path,
                    coords=x_hat_0.cpu().float().numpy(),
                    t=float(target_t),
                    sigma_t=float(sigma_t),
                    step_idx=int(step_idx),
                )
                captured.add(target_t)

    return callback
