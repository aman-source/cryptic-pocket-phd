"""PocketMiner-based guidance potentials for ConforMix twisted SMC.

Two potentials:
  g_p: sum-based — rewards open pockets (no sweep target)
  g_t: sweep-based — pulls toward a target pocket score level

Both are differentiable: gradient flows from scalar output back
to all-atom coordinates via PocketMiner's PyTorch forward pass.

Interface matches ConforMix's classifier_prob_fn contract:
  Input:  (xt, x0_hat, return_grad, t, atom_mask)
  Output: (log_potential [P], gradient [P, N_atoms, 3])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from .pocketminer_torch import (
    PocketMinerTorch,
    preprocess_pdb,
    AA_LOOKUP,
    ABBREV,
)


def _extract_calpha_and_sequence(
    x_all_atom: torch.Tensor,
    token_to_rep_atom: torch.Tensor,
    atom_to_token: torch.Tensor,
    seq_indices: torch.Tensor,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Calpha positions, sequence, and mask from Boltz all-atom coords.

    Args:
        x_all_atom: [P, N_atoms, 3] all-atom coords, requires_grad
        token_to_rep_atom: [N_tokens, N_atoms] one-hot, maps token -> rep atom
        atom_to_token: [N_atoms, N_tokens] one-hot, maps atom -> token
        seq_indices: [N_tokens] amino acid type indices (0-19)
        n_tokens: number of real (non-padding) tokens

    Returns:
        X_bb: [P, N_tokens, 4, 3] backbone coords (N, CA, C, O format)
              Only CA is real; N/C/O filled with CA offset for valid dihedrals
        S: [P, N_tokens] sequence indices
        mask: [P, N_tokens] 1 for real residues
    """
    P = x_all_atom.shape[0]
    N_tokens_padded = token_to_rep_atom.shape[0]

    # Get representative atom (Calpha) positions: [P, N_tokens, 3]
    # token_to_rep_atom: [N_tokens, N_atoms] one-hot float
    # x_all_atom: [P, N_atoms, 3]
    ca_pos = torch.matmul(
        token_to_rep_atom.float().unsqueeze(0).expand(P, -1, -1),
        x_all_atom,
    )  # [P, N_tokens, 3]

    # Build pseudo-backbone: PocketMiner needs [B, N, 4, 3] for N, CA, C, O
    # We only have CA. Create pseudo-backbone by small offsets for valid dihedrals.
    # This is the same approach used in Phase 0's boltz_coords_to_pdb.
    N_pos = ca_pos + torch.tensor([1.458, 0.0, 0.0], device=ca_pos.device)
    C_pos = ca_pos + torch.tensor([-0.553, 1.419, 0.0], device=ca_pos.device)
    O_pos = ca_pos + torch.tensor([-0.553, 2.163, 0.780], device=ca_pos.device)

    X_bb = torch.stack([N_pos, ca_pos, C_pos, O_pos], dim=2)  # [P, N_tokens, 4, 3]

    S = seq_indices.unsqueeze(0).expand(P, -1).long()  # [P, N_tokens]
    mask = torch.zeros(P, N_tokens_padded, device=x_all_atom.device)
    mask[:, :n_tokens] = 1.0

    return X_bb, S, mask


class PocketPotential:
    """Wraps PocketMiner PyTorch model as a guidance potential.

    Usage:
        pot = PocketPotential(model, pocket_residues, token_to_rep_atom,
                              atom_to_token, seq_indices, n_tokens)
        # As g_p:
        log_pot, grad = pot.g_p(xt, x0_hat, beta=1.0)
        # As g_t:
        log_pot, grad = pot.g_t(xt, x0_hat, target_t=0.5, alpha=10.0)
    """

    def __init__(
        self,
        model: PocketMinerTorch,
        pocket_residue_indices: list[int],
        token_to_rep_atom: torch.Tensor,
        atom_to_token: torch.Tensor,
        seq_indices: torch.Tensor,
        n_tokens: int,
    ):
        self.model = model
        self.pocket_idx = torch.tensor(pocket_residue_indices, dtype=torch.long)
        self.token_to_rep_atom = token_to_rep_atom
        self.atom_to_token = atom_to_token
        self.seq_indices = seq_indices
        self.n_tokens = n_tokens

    def _score(self, x_all_atom: torch.Tensor) -> torch.Tensor:
        """Run PocketMiner on all-atom coords, return per-residue scores.

        Args:
            x_all_atom: [P, N_atoms, 3]

        Returns:
            scores: [P, N_tokens] pocket probabilities in [0, 1]
        """
        X_bb, S, mask = _extract_calpha_and_sequence(
            x_all_atom,
            self.token_to_rep_atom.to(x_all_atom.device),
            self.atom_to_token.to(x_all_atom.device),
            self.seq_indices.to(x_all_atom.device),
            self.n_tokens,
        )
        scores = self.model(X_bb, S, mask, train=False, res_level=True)
        return scores  # [P, N_tokens]

    def g_p(
        self,
        xt: torch.Tensor,
        x0_hat: torch.Tensor,
        beta: float = 1.0,
        return_grad: bool = True,
        t: Optional[int] = None,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Sum-based pocket potential.

        log g_p(x) = beta * sum(PocketMiner(x)[pocket_residues])

        Higher score = more pocket-like. No sweep target.

        Args:
            xt: [P, N_atoms, 3] noised coords (gradient anchor)
            x0_hat: [P, N_atoms, 3] denoised prediction
            beta: strength multiplier
            return_grad: if True, compute gradient w.r.t. xt

        Returns:
            log_potential: [P] scalar per particle
            gradient: [P, N_atoms, 3] if return_grad
        """
        pocket_idx = self.pocket_idx.to(x0_hat.device)

        if return_grad:
            scores = self._score(x0_hat)  # [P, N_tokens]
            pocket_scores = scores[:, pocket_idx]  # [P, n_pocket]
            log_potential = beta * pocket_scores.sum(dim=-1)  # [P]

            grad = torch.autograd.grad(
                log_potential,
                xt,
                grad_outputs=torch.ones_like(log_potential),
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                grad = torch.zeros_like(xt)

            return log_potential.detach(), grad.detach()
        else:
            with torch.no_grad():
                scores = self._score(x0_hat)
                pocket_scores = scores[:, pocket_idx]
                log_potential = beta * pocket_scores.sum(dim=-1)
            return log_potential

    def g_t(
        self,
        xt: torch.Tensor,
        x0_hat: torch.Tensor,
        target_t: float = 0.5,
        alpha: float = 10.0,
        return_grad: bool = True,
        t: Optional[int] = None,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Sweep-based pocket potential.

        log g_t(x) = -alpha * (mean(PocketMiner(x)[pocket_residues]) - target_t)^2

        Pulls toward target_t pocket score level. Matches ConforMix's sweep structure.

        Args:
            xt: [P, N_atoms, 3] noised coords (gradient anchor)
            x0_hat: [P, N_atoms, 3] denoised prediction
            target_t: target mean pocket probability (sweep 0 -> 1)
            alpha: strength multiplier
            return_grad: if True, compute gradient w.r.t. xt

        Returns:
            log_potential: [P] scalar per particle
            gradient: [P, N_atoms, 3] if return_grad
        """
        pocket_idx = self.pocket_idx.to(x0_hat.device)

        if return_grad:
            scores = self._score(x0_hat)  # [P, N_tokens]
            pocket_scores = scores[:, pocket_idx]  # [P, n_pocket]
            pocket_mean = pocket_scores.mean(dim=-1)  # [P]
            log_potential = -alpha * (pocket_mean - target_t) ** 2  # [P]

            grad = torch.autograd.grad(
                log_potential,
                xt,
                grad_outputs=torch.ones_like(log_potential),
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                grad = torch.zeros_like(xt)

            return log_potential.detach(), grad.detach()
        else:
            with torch.no_grad():
                scores = self._score(x0_hat)
                pocket_scores = scores[:, pocket_idx]
                pocket_mean = pocket_scores.mean(dim=-1)
                log_potential = -alpha * (pocket_mean - target_t) ** 2
            return log_potential


# ---------------------------------------------------------------------------
# Convenience: build from PDB paths (for testing / standalone use)
# ---------------------------------------------------------------------------

def load_pocket_potential(
    model_weights: str,
    pocket_residue_indices: list[int],
    pdb_path: str,
    device: str = "cpu",
) -> PocketPotential:
    """Create a PocketPotential from a PDB and pre-saved PyTorch weights.

    For testing. In the full pipeline, Boltz provides atom_to_token etc.

    Args:
        model_weights: path to pocketminer_torch.pt state dict
        pocket_residue_indices: list of 0-indexed residue indices
        pdb_path: path to PDB file (for sequence extraction)
        device: 'cpu' or 'cuda'

    Returns:
        PocketPotential ready for g_p / g_t calls
    """
    import mdtraj as md

    # Load model
    model = PocketMinerTorch()
    with torch.no_grad():
        model(torch.randn(1, 50, 4, 3), torch.zeros(1, 50, dtype=torch.long),
              torch.ones(1, 50))
    model.load_state_dict(torch.load(model_weights, weights_only=True,
                                     map_location=device))
    model.eval()
    model.to(device)

    # Extract sequence from PDB
    traj = md.load(pdb_path)
    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)
    n_res = prot_bb.top.n_residues

    seq = [r.name for r in prot_bb.top.residues]
    seq_indices = torch.tensor([AA_LOOKUP[ABBREV[a]] for a in seq], dtype=torch.long)

    # Build fake atom_to_token and token_to_rep_atom for standalone testing
    # In real pipeline, Boltz provides these. Here we treat each residue's
    # CA as the only atom (simplification for testing).
    N_atoms = n_res  # 1 atom per residue
    token_to_rep_atom = torch.eye(n_res)  # [N_tokens, N_atoms] identity
    atom_to_token = torch.eye(n_res)  # [N_atoms, N_tokens] identity

    return PocketPotential(
        model=model,
        pocket_residue_indices=pocket_residue_indices,
        token_to_rep_atom=token_to_rep_atom,
        atom_to_token=atom_to_token,
        seq_indices=seq_indices,
        n_tokens=n_res,
    )
