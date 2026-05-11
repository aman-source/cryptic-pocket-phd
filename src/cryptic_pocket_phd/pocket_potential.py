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


def _extract_backbone(
    x_all_atom: torch.Tensor,
    bb_atom_indices: torch.Tensor,
    seq_indices: torch.Tensor,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract real N/CA/C/O backbone from Boltz all-atom coordinates.

    Boltz stores atoms per residue in CCD order: N(0), CA(1), C(2), O(3), ...
    bb_atom_indices provides the global atom indices for these 4 atoms per residue.

    Args:
        x_all_atom: [P, N_atoms, 3] all-atom coords, requires_grad
        bb_atom_indices: [N_tokens_padded, 4] global atom indices for N/CA/C/O
            per residue. Padding tokens should have index 0 (safe — masked out).
        seq_indices: [N_tokens_padded] amino acid type indices (0-19)
        n_tokens: number of real (non-padding) tokens

    Returns:
        X_bb: [P, N_tokens_padded, 4, 3] backbone coords
        S: [P, N_tokens_padded] sequence indices
        mask: [P, N_tokens_padded] 1 for real residues
    """
    P = x_all_atom.shape[0]
    N_tokens_padded = bb_atom_indices.shape[0]

    # Index backbone atoms: bb_atom_indices[t, k] is the global atom index
    # for backbone atom k (0=N, 1=CA, 2=C, 3=O) of token t.
    # x_all_atom[:, idx, :] preserves autograd.
    bb_flat = bb_atom_indices.reshape(-1)  # [N_tokens_padded * 4]
    X_bb = x_all_atom[:, bb_flat, :]  # [P, N_tokens_padded*4, 3]
    X_bb = X_bb.reshape(P, N_tokens_padded, 4, 3)  # [P, N_tokens_padded, 4, 3]

    S = seq_indices.unsqueeze(0).expand(P, -1).long()  # [P, N_tokens_padded]
    mask = torch.zeros(P, N_tokens_padded, device=x_all_atom.device)
    mask[:, :n_tokens] = 1.0

    return X_bb, S, mask


def build_bb_atom_indices(atom_to_token: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Build backbone atom index table from Boltz's atom_to_token mapping.

    For each token (residue), finds the first 4 atoms belonging to it.
    Boltz stores atoms in CCD order: N(0), CA(1), C(2), O(3) for proteins.

    Args:
        atom_to_token: [N_atoms, N_tokens] one-hot matrix
        n_tokens: number of real tokens (non-padding)

    Returns:
        bb_indices: [N_tokens_padded, 4] int64 tensor of global atom indices
    """
    N_tokens_padded = atom_to_token.shape[1]
    bb_indices = torch.zeros(N_tokens_padded, 4, dtype=torch.long)

    # atom_to_token[a, t] = 1 means atom a belongs to token t
    token_assignment = atom_to_token.argmax(dim=1)  # [N_atoms] -> token index

    for t in range(n_tokens):
        atom_ids = (token_assignment == t).nonzero(as_tuple=True)[0]
        if len(atom_ids) >= 4:
            bb_indices[t] = atom_ids[:4]  # N, CA, C, O
        elif len(atom_ids) > 0:
            # Fewer than 4 atoms (e.g., GLY with missing O) — pad with first atom
            for k in range(4):
                bb_indices[t, k] = atom_ids[min(k, len(atom_ids) - 1)]

    return bb_indices


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
        bb_atom_indices: torch.Tensor,
        seq_indices: torch.Tensor,
        n_tokens: int,
    ):
        """
        Args:
            model: PyTorch PocketMiner model (eval mode)
            pocket_residue_indices: 0-indexed residue indices for pocket region
            bb_atom_indices: [N_tokens_padded, 4] global atom indices for N/CA/C/O
            seq_indices: [N_tokens_padded] amino acid type indices (0-19)
            n_tokens: number of real (non-padding) tokens
        """
        self.model = model
        self.pocket_idx = torch.tensor(pocket_residue_indices, dtype=torch.long)
        self.bb_atom_indices = bb_atom_indices
        self.seq_indices = seq_indices
        self.n_tokens = n_tokens

    def _score(self, x_all_atom: torch.Tensor) -> torch.Tensor:
        """Run PocketMiner on all-atom coords, return per-residue scores.

        Args:
            x_all_atom: [P, N_atoms, 3]

        Returns:
            scores: [P, N_tokens] pocket probabilities in [0, 1]
        """
        X_bb, S, mask = _extract_backbone(
            x_all_atom,
            self.bb_atom_indices.to(x_all_atom.device),
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

    # Extract backbone from PDB: each residue has N, CA, C, O as atoms 0-3
    traj = md.load(pdb_path)
    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)
    n_res = prot_bb.top.n_residues

    seq = [r.name for r in prot_bb.top.residues]
    seq_indices = torch.tensor([AA_LOOKUP[ABBREV[a]] for a in seq], dtype=torch.long)

    # For standalone testing: each residue has exactly 4 atoms (N, CA, C, O)
    # after the backbone selection above. So bb_atom_indices[t] = [4t, 4t+1, 4t+2, 4t+3].
    bb_atom_indices = torch.arange(n_res * 4).reshape(n_res, 4)

    return PocketPotential(
        model=model,
        pocket_residue_indices=pocket_residue_indices,
        bb_atom_indices=bb_atom_indices,
        seq_indices=seq_indices,
        n_tokens=n_res,
    )
