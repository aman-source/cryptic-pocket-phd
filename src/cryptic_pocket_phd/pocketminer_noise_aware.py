"""Noise-aware PocketMiner: timestep-conditioned GVP-GNN for cryptic pockets.

Subclasses PocketMinerTorch (Phase 1) and adds:
  1. Sinusoidal timestep embedding (t in [0,1]) injected at each MPNN layer
  2. Optional dataset-source one-hot flag (ATLAS / mdCATH / PocketMiner)

Output unchanged: per-residue pocket probability [B, N].
Timestep injection preserves SO(3) equivariance because t is a scalar (invariant).

Spec #2 Task A2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .pocketminer_torch import (
    PocketMinerTorch, _split, _merge, _vs_concat,
    _gather_nodes, GVPLayerNorm, GVPDropout,
)


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for diffusion timesteps.

    Maps scalar t in [0, 1] to a vector of dimension `dim`.
    Same formulation as Stable Diffusion / DDPM.
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or scalar, timestep values in [0, 1]
        Returns:
            [B, dim] embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


# Number of dataset sources for one-hot encoding
SOURCE_NAMES = ["ATLAS", "mdCATH", "PocketMiner"]
NUM_SOURCES = len(SOURCE_NAMES)


class NoiseAwarePocketMiner(PocketMinerTorch):
    """PocketMiner conditioned on diffusion timestep t.

    Architecture changes vs base:
      - SinusoidalTimestepEmbedding(t_dim) produces t_emb [B, t_dim]
      - At each MPNN layer: scalar part of h_V gets t_emb concatenated,
        then a learned linear projects back to hs. This is done via
        per-layer `t_inject` modules.
      - Optional source one-hot [B, 3] concatenated to initial node scalars
        before W_v projection, with a learned linear to absorb extra dims.

    At t=0 with default source (zeros), output should match vanilla PocketMiner
    up to floating-point noise, because:
      - sin(0*freq)=0, cos(0*freq)=1 → t_emb is a fixed pattern
      - t_inject linear is initialized to pass-through + zero for t dims
    """

    def __init__(
        self,
        t_dim: int = 32,
        source_dim: int = NUM_SOURCES,
        node_features: Tuple[int, int] = (8, 50),
        edge_features: Tuple[int, int] = (1, 32),
        hidden_dim: Tuple[int, int] = (16, 100),
        num_layers: int = 4,
        k_neighbors: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )

        self.t_dim = t_dim
        self.source_dim = source_dim
        _, hs = hidden_dim
        hv, _ = hidden_dim

        # Timestep embedding
        self.t_embed = SinusoidalTimestepEmbedding(t_dim)

        # Per-layer timestep injection: concat t_emb to scalar part, project back
        # Initialized so that at t=0 the output ~= identity on original scalars
        self.t_inject = nn.ModuleList()
        for _ in range(num_layers):
            proj = nn.Linear(hs + t_dim, hs)
            # Init: pass-through for first hs dims, zero for t_dim dims
            with torch.no_grad():
                proj.weight.zero_()
                proj.weight[:hs, :hs] = torch.eye(hs)
                proj.bias.zero_()
            self.t_inject.append(proj)

        # Source embedding: project (ns + source_dim) → ns before W_v
        # ns = node_features[1] = 50 (scalar part of node embedding output)
        # But actually, after StructuralFeatures + seq concat, scalar dim = ns + hs
        # Let me trace: V from features has nv=8 vectors, ns=50 scalars
        # h_S from W_s has hs=100 scalars
        # _vs_concat(V, h_S, nv, 0) → nv vectors, (ns + hs) scalars
        # Then W_v projects to hidden_dim
        # Source one-hot should be added to the scalar part before W_v
        nv, ns = node_features
        self._src_proj = nn.Linear(ns + hs + source_dim, ns + hs)
        # Init: pass-through for first (ns+hs) dims, zero for source dims
        with torch.no_grad():
            self._src_proj.weight.zero_()
            self._src_proj.weight[: ns + hs, : ns + hs] = torch.eye(ns + hs)
            self._src_proj.bias.zero_()

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        source: Optional[torch.Tensor] = None,
        train: bool = False,
        res_level: bool = True,
    ) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Args:
            X: [B, N, 4, 3] backbone coords
            S: [B, N] amino acid indices
            mask: [B, N] padding mask
            t: [B] or scalar, diffusion timestep in [0, 1]. Default 0.
            source: [B, source_dim] one-hot dataset source. Default zeros.
            train: enable dropout
            res_level: per-residue (True) or protein-level (False)

        Returns:
            [B, N] pocket probabilities if res_level, else [B]
        """
        B = X.shape[0]
        device = X.device

        # Default t=0 (clean structure)
        if t is None:
            t = torch.zeros(B, device=device)
        elif t.dim() == 0:
            t = t.expand(B)

        # Default source = zeros (no dataset info)
        if source is None:
            source = torch.zeros(B, self.source_dim, device=device)

        # Timestep embedding [B, t_dim]
        t_emb = self.t_embed(t)

        # Structural features
        V, E, E_idx = self.features(X, mask)

        # Sequence embedding
        h_S = self.W_s(S)
        V = _vs_concat(V, h_S, self.nv, 0)

        # Inject source: split V into vectors and scalars, concat source, project
        v_part, s_part = _split(V, self.nv)
        s_with_src = torch.cat([s_part, source.unsqueeze(1).expand(-1, s_part.shape[1], -1)], dim=-1)
        s_part = self._src_proj(s_with_src)
        V = _merge(v_part, s_part)

        # Project to hidden dim
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Run encoder layers manually (instead of self.encoder) to inject t_emb
        mask_attend = _gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for i, layer in enumerate(self.encoder.layers):
            # Inject timestep into scalar part of h_V
            hv_vec, hv_sca = _split(h_V, self.hv)
            t_expanded = t_emb.unsqueeze(1).expand(-1, hv_sca.shape[1], -1)
            hv_sca = self.t_inject[i](torch.cat([hv_sca, t_expanded], dim=-1))
            h_V = _merge(hv_vec, hv_sca)

            # Standard MPNN layer
            from .pocketminer_torch import _cat_neighbors_nodes
            h_M = _cat_neighbors_nodes(h_V, h_E, E_idx, self.encoder.nv, self.encoder.ev)
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, train=train)

        # Output head (same as base)
        h_V_out = self.W_V_out(h_V)

        if not res_level:
            mask_expanded = mask.unsqueeze(-1).float()
            h_V_out = (h_V_out * mask_expanded).sum(dim=-2)
            h_V_out = h_V_out / mask_expanded.sum(dim=-2).clamp(min=1)

        out = self.dense(h_V_out).squeeze(-1)

        if res_level:
            out = torch.sigmoid(out)

        return out

    def load_base_weights(self, base_state_dict: dict):
        """Load weights from vanilla PocketMinerTorch, ignoring new params.

        New params (t_inject, _src_proj, t_embed) keep their identity init.
        """
        own_state = self.state_dict()
        loaded = 0
        for name, param in base_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"Loaded {loaded}/{len(base_state_dict)} base weights "
              f"({len(own_state) - loaded} new params kept at init)")
