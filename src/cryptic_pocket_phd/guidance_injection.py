"""Pocket-guided twist functions for ConforMix's twisted SMC.

Drop-in replacement for ConforMix's twist_fn (run_twisted.py:887-976).
Same interface: inner(xt, x0_hat, return_grad, t, atom_mask) -> (log_pot, grad).

Bias types:
  - 'pocket_p': PocketMiner sum-based (g_p) — rewards open pockets
  - 'pocket_t': PocketMiner sweep-based (g_t) — pulls toward target score
"""

from __future__ import annotations

import numpy as np
import torch

from .pocket_potential import PocketPotential


def pocket_twist_fn(
    alpha: float,
    beta: float,
    tstart_step: int,
    tstop_step: int,
    bias_type: str,
    pocket_potential: PocketPotential,
    untwisted_coords: torch.Tensor,
    twisting_mask: torch.Tensor,
    weighted_rigid_align_fn=None,
):
    """Factory returning a callable matching ConforMix's classifier_prob_fn.

    Args:
        alpha: gradient strength multiplier (same role as in RMSD twist_fn)
        beta: for pocket_p = sum strength; for pocket_t = target pocket score
        tstart_step: start twisting at this step (200 = first)
        tstop_step: stop twisting at this step (0 = last)
        bias_type: 'pocket_p' or 'pocket_t'
        pocket_potential: PocketPotential with loaded model + pocket indices
        untwisted_coords: [N_atoms_real, 3] apo coords for alignment
        twisting_mask: [N_atoms_real] binary mask for alignment region
        weighted_rigid_align_fn: ConforMix's weighted_rigid_align
    """

    def inner_twist_fn(xt, x0_hat, return_grad=True, t=None, atom_mask=None):
        P = x0_hat.shape[0]
        padded_atom_size = x0_hat.shape[1]
        device = x0_hat.device

        # Align x0_hat to apo reference frame (mitigates kNN equivariance noise)
        if weighted_rigid_align_fn is not None:
            twisting_mask_padded = torch.nn.functional.pad(
                twisting_mask,
                (0, padded_atom_size - twisting_mask.shape[0]),
                value=0,
            ).to(device)
            untwisted_padded = torch.nn.functional.pad(
                untwisted_coords,
                (0, 0, 0, padded_atom_size - untwisted_coords.shape[0]),
                value=0,
            ).to(device)
            # Expand to match particle batch dim P (may be smaller than
            # atom_mask due to SMC batch_p splitting in mg_wrapper.py:380)
            untwisted_expanded = untwisted_padded.unsqueeze(0).expand(P, -1, -1)
            mask_expanded = twisting_mask_padded.unsqueeze(0).expand(P, -1)
            # atom_mask may have more rows than P (full particle count vs batch)
            atom_mask_batch = atom_mask[:P] if atom_mask.shape[0] > P else atom_mask
            x0_hat_aligned = weighted_rigid_align_fn(
                x0_hat, untwisted_expanded, atom_mask_batch,
                mask_expanded, keep_gradients=True,
            )
        else:
            x0_hat_aligned = x0_hat

        # Compute pocket scores
        pocket_idx = pocket_potential.pocket_idx.to(device)
        scores = pocket_potential._score(x0_hat_aligned)  # [P, N_tokens]
        pocket_scores = scores[:, pocket_idx]  # [P, n_pocket]

        # Compute log potential (no alpha scaling — that's done below via factor)
        if bias_type == 'pocket_p':
            # g_p = beta * sum(pocket_scores). Higher = more pocket-like.
            log_potential = beta * pocket_scores.sum(dim=-1)  # [P]
        elif bias_type == 'pocket_t':
            # g_t = -(pocket_mean - beta)^2. Pulls toward target=beta.
            # Note: alpha scaling on gradient is handled below, not here.
            pocket_mean = pocket_scores.mean(dim=-1)  # [P]
            log_potential = -((pocket_mean - beta) ** 2)  # [P]
        else:
            raise ValueError(f"Unknown bias_type: {bias_type}")

        if not return_grad:
            return log_potential.to(device).detach()

        # Gradient w.r.t. xt
        if t is not None and tstart_step >= t >= tstop_step:
            grad_log_potential = torch.autograd.grad(
                log_potential,
                xt,
                grad_outputs=torch.ones_like(log_potential),
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_log_potential is None:
                grad_log_potential = torch.zeros_like(xt)

            # Time-dependent scaling (same as ConforMix's RMSD twist_fn)
            if t:
                if alpha > 0:
                    factor = alpha
                elif alpha < 0:
                    factor = abs(alpha) * 200 * (
                        1 + np.cos(np.pi * (np.log(1 + 4 * (230 - t)) / np.log(10)))
                    ) / 2
                else:
                    factor = 0
                grad_log_potential = grad_log_potential * factor
        else:
            grad_log_potential = torch.zeros_like(xt, device=device)

        return (
            log_potential.to(device).detach(),
            grad_log_potential.to(device).detach(),
        )

    return inner_twist_fn
