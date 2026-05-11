"""PyTorch port of PocketMiner's GVP-GNN (MQAModel).

Architecture: Meller et al. 2023, Nat Comms.
Original: TensorFlow, github.com/Mickdub/gvp branch pocket_pred.

This port is numerically validated against TF at 1e-4 per-residue tolerance.
Designed for batched inference and differentiable (autograd-compatible) for
use as a guidance potential in ConforMix's twisted SMC.

Input: X [B, N, 4, 3], S [B, N], mask [B, N]
Output: [B, N] pocket probabilities in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# GVP primitives
# ---------------------------------------------------------------------------

def _norm_no_nan(x: torch.Tensor, axis: int = -1,
                 keepdims: bool = False, eps: float = 1e-8,
                 sqrt: bool = True) -> torch.Tensor:
    out = torch.clamp((x * x).sum(axis, keepdim=keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x: torch.Tensor, nv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split GVP representation into vector and scalar parts.
    x: [..., 3*nv + ns]
    Returns: v [..., 3, nv], s [..., ns]
    """
    v = x[..., :3 * nv].reshape(*x.shape[:-1], 3, nv)
    s = x[..., 3 * nv:]
    return v, s


def _merge(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Merge vector and scalar parts into GVP representation.
    v: [..., 3, nv], s: [..., ns]
    Returns: [..., 3*nv + ns]
    """
    v_flat = v.reshape(*v.shape[:-2], 3 * v.shape[-1])
    return torch.cat([v_flat, s], dim=-1)


def _vs_concat(x1: torch.Tensor, x2: torch.Tensor,
               nv1: int, nv2: int) -> torch.Tensor:
    """Concatenate two GVP representations keeping vectors at top."""
    v1, s1 = _split(x1, nv1)
    v2, s2 = _split(x2, nv2)
    v = torch.cat([v1, v2], dim=-1)
    s = torch.cat([s1, s2], dim=-1)
    return _merge(v, s)


def _normalize(tensor: torch.Tensor, axis: int = -1) -> torch.Tensor:
    norm = torch.linalg.norm(tensor, dim=axis, keepdim=True)
    return tensor / (norm + 1e-8)


def _gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices.
    nodes: [B, N, C], neighbor_idx: [B, N, K] -> [B, N, K, C]
    """
    B, N, K = neighbor_idx.shape
    C = nodes.shape[-1]
    # Flatten K into N dimension for gather
    idx_flat = neighbor_idx.reshape(B, N * K)  # [B, N*K]
    idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C)  # [B, N*K, C]
    gathered = torch.gather(nodes, 1, idx_expanded)  # [B, N*K, C]
    return gathered.reshape(B, N, K, C)


def _cat_neighbors_nodes(h_nodes: torch.Tensor, h_neighbors: torch.Tensor,
                         E_idx: torch.Tensor, nv_nodes: int,
                         nv_neighbors: int) -> torch.Tensor:
    h_nodes_gathered = _gather_nodes(h_nodes, E_idx)
    return _vs_concat(h_neighbors, h_nodes_gathered, nv_neighbors, nv_nodes)


# ---------------------------------------------------------------------------
# GVP Layer
# ---------------------------------------------------------------------------

class GVP(nn.Module):
    def __init__(self, vi: int, vo: int, so: int,
                 nlv=torch.sigmoid, nls=F.relu):
        super().__init__()
        self.vi = vi
        self.vo = vo
        self.so = so
        self.nlv = nlv
        self.nls = nls

        # si is computed from input at call time, but for weight init we need
        # to know the full input dim. We'll use lazy init via _build.
        self._wh = None
        self._ws = None
        self._wv = None
        self._built = False

    def _build(self, si: int):
        if self._built:
            return
        ws_in = si + (max(self.vi, self.vo) if self.vi else 0)
        self._ws = nn.Linear(ws_in, self.so)
        if self.vi:
            self._wh = nn.Linear(self.vi, max(self.vi, self.vo), bias=True)
        if self.vo:
            self._wv = nn.Linear(max(self.vi, self.vo), self.vo, bias=True)
        self._built = True

    def forward(self, x: torch.Tensor,
                return_split: bool = False) -> torch.Tensor:
        v, s = _split(x, self.vi)

        if not self._built:
            self._build(s.shape[-1])
            # Move newly created layers to same device as input
            device = x.device
            if self._ws is not None:
                self._ws = self._ws.to(device)
            if self._wh is not None:
                self._wh = self._wh.to(device)
            if self._wv is not None:
                self._wv = self._wv.to(device)

        if self.vi:
            vh = self._wh(v)  # [..., 3, max(vi, vo)]
            vn = _norm_no_nan(vh, axis=-2)  # [..., max(vi, vo)]
            cat_input = torch.cat([s, vn], dim=-1)
            out = self._ws(cat_input)
        else:
            out = self._ws(s)

        if self.nls is not None:
            out = self.nls(out)

        if self.vo:
            vo = self._wv(vh)  # [..., 3, vo]
            if self.nlv is not None:
                vo = vo * self.nlv(_norm_no_nan(vo, axis=-2, keepdims=True))
            if return_split:
                return vo, out
            return _merge(vo, out)

        return out


class GVPDropout(nn.Module):
    def __init__(self, rate: float, nv: int):
        super().__init__()
        self.nv = nv
        self.rate = rate

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if not training:
            return x
        v, s = _split(x, self.nv)
        # Vector dropout: same mask across spatial dim
        if self.rate > 0:
            vmask = torch.ones(v.shape[:-2] + (v.shape[-1],),
                               device=v.device).bernoulli_(1 - self.rate) / (1 - self.rate)
            v = v * vmask.unsqueeze(-2)
            s = F.dropout(s, p=self.rate, training=True)
        return _merge(v, s)


class GVPLayerNorm(nn.Module):
    def __init__(self, nv: int):
        super().__init__()
        self.nv = nv
        self.snorm = None  # lazy init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, s = _split(x, self.nv)
        if self.snorm is None:
            # TF LayerNormalization defaults to eps=1e-3; match it exactly.
            self.snorm = nn.LayerNorm(s.shape[-1], eps=1e-3, device=s.device)
        # Vector norm: RMS over spatial dim, then mean over channels
        vn = _norm_no_nan(v, axis=-2, keepdims=True, sqrt=False)  # [..., 1, nv]
        vn = torch.sqrt(vn.mean(dim=-1, keepdim=True))  # [..., 1, 1]
        return _merge(v / vn, self.snorm(s))


# ---------------------------------------------------------------------------
# MPNN Layer
# ---------------------------------------------------------------------------

class MPNNLayer(nn.Module):
    def __init__(self, vec_in: int, num_hidden: Tuple[int, int],
                 dropout: float = 0.1):
        super().__init__()
        self.vo, self.so = num_hidden
        self.vec_in = vec_in

        self.norm = nn.ModuleList([GVPLayerNorm(self.vo) for _ in range(2)])
        self.dropout = GVPDropout(dropout, self.vo)

        # Message network: receives vec_in + receiver node
        self.W_EV = nn.Sequential(
            GVP(vi=vec_in + self.vo, vo=self.vo, so=self.so),
            GVP(vi=self.vo, vo=self.vo, so=self.so),
            GVP(vi=self.vo, vo=self.vo, so=self.so, nls=None, nlv=None),
        )

        # Feedforward
        self.W_dh = nn.Sequential(
            GVP(vi=self.vo, vo=2 * self.vo, so=4 * self.so),
            GVP(vi=2 * self.vo, vo=self.vo, so=self.so, nls=None, nlv=None),
        )

    def forward(self, h_V: torch.Tensor, h_M: torch.Tensor,
                mask_V: Optional[torch.Tensor] = None,
                mask_attend: Optional[torch.Tensor] = None,
                train: bool = False) -> torch.Tensor:
        # h_V: [B, N, D], h_M: [B, N, K, D]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_M.shape[-2], -1)
        h_EV = _vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self.W_EV(h_EV)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1).float() * h_message

        dh = h_message.mean(dim=-2)
        h_V = self.norm[0](h_V + self.dropout(dh, training=train))

        # Position-wise feedforward
        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh, training=train))

        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1).float() * h_V

        return h_V


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, node_features: Tuple[int, int],
                 edge_features: Tuple[int, int],
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        nv, _ = node_features
        ev, _ = edge_features
        self.nv = nv
        self.ev = ev

        self.layers = nn.ModuleList([
            MPNNLayer(nv + ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h_V: torch.Tensor, h_E: torch.Tensor,
                E_idx: torch.Tensor, mask: torch.Tensor,
                train: bool = False) -> torch.Tensor:
        mask_attend = _gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.layers:
            h_M = _cat_neighbors_nodes(h_V, h_E, E_idx, self.nv, self.ev)
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, train=train)

        return h_V


# ---------------------------------------------------------------------------
# Structural Features
# ---------------------------------------------------------------------------

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int = 16):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx: torch.Tensor) -> torch.Tensor:
        # E_idx: [B, N, K]
        N_nodes = E_idx.shape[1]
        ii = torch.arange(N_nodes, device=E_idx.device, dtype=torch.float32)
        ii = ii.reshape(1, -1, 1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, device=E_idx.device, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.reshape(1, 1, 1, -1)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class StructuralFeatures(nn.Module):
    def __init__(self, node_features: Tuple[int, int],
                 edge_features: Tuple[int, int],
                 num_positional_embeddings: int = 16,
                 num_rbf: int = 16, top_k: int = 30,
                 ablate_sidechain_vectors: bool = True,
                 ablate_rbf: bool = False):
        super().__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.ablate_sidechain_vectors = ablate_sidechain_vectors
        self.ablate_rbf = ablate_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        vo, so = node_features
        ve, se = edge_features

        vi_v = 3 if ablate_sidechain_vectors else 4
        self.node_embedding = GVP(vi=vi_v, vo=vo, so=so, nlv=None, nls=None)

        vi_e = 1 if ablate_sidechain_vectors else 2
        self.edge_embedding = GVP(vi=vi_e, vo=ve, so=se, nlv=None, nls=None)

        # Lazy init for LayerNorm (need to know dims)
        self.norm_nodes = None
        self.norm_edges = None

    def _dist(self, X: torch.Tensor, mask: torch.Tensor,
              eps: float = 1e-6):
        """Pairwise distances and kNN."""
        mask_f = mask.float()
        mask_2D = mask_f.unsqueeze(1) * mask_f.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt((dX ** 2).sum(-1) + eps)

        D_max = D.max(dim=-1, keepdim=True).values
        D_adjust = D + (1.0 - mask_2D) * D_max

        k = min(self.top_k, X.shape[1])
        # top_k on negated distances = k nearest
        D_neighbors, E_idx = torch.topk(-D_adjust, k=k, dim=-1)
        D_neighbors = -D_neighbors

        return D_neighbors, E_idx

    def _directions(self, X: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        X_neighbors = _gather_nodes(X, E_idx)
        dX = X_neighbors - X.unsqueeze(-2)
        return _normalize(dX, axis=-1)

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.reshape(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _orientations(self, X: torch.Tensor) -> torch.Tensor:
        forward = _normalize(X[:, 1:] - X[:, :-1])
        backward = _normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, (0, 0, 0, 1))  # pad N dim
        backward = F.pad(backward, (0, 0, 1, 0))
        return torch.stack([forward, backward], dim=-1)  # [B, N, 3, 2]

    def _sidechains(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, N, 4, 3], atoms = N, CA, C, O
        n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        c_norm = _normalize(c - origin)
        n_norm = _normalize(n - origin)
        bisector = _normalize(c_norm + n_norm)
        perp = _normalize(torch.linalg.cross(c_norm, n_norm))
        vec = -bisector * np.sqrt(1 / 3) - perp * np.sqrt(2 / 3)
        return vec  # [B, N, 3]

    def _dihedrals(self, X: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        # First 3 atoms: N, CA, C
        B = X.shape[0]
        N = X.shape[1]
        X_flat = X[:, :, :3, :].reshape(B, 3 * N, 3)

        dX = X_flat[:, 1:, :] - X_flat[:, :-1, :]
        U = _normalize(dX, axis=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]

        n_2 = _normalize(torch.linalg.cross(u_2, u_1), axis=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0), axis=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        D = F.pad(D, (1, 2))  # pad to 3*N - 2 -> 3*N
        D = D.reshape(B, N, 3)

        return torch.cat([torch.cos(D), torch.sin(D)], dim=2)  # [B, N, 6]

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        """Featurize coordinates as attributed graph.
        X: [B, N, 4, 3], mask: [B, N]
        Returns: V, E, E_idx
        """
        X_ca = X[:, :, 1, :]  # Calpha
        D_neighbors, E_idx = self._dist(X_ca, mask)

        E_directions = self._directions(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)
        E_positional = self.embeddings(E_idx)

        V_dihedrals = self._dihedrals(X)
        V_orientations = self._orientations(X_ca)
        V_sidechains = self._sidechains(X)

        V_vec = torch.cat([V_sidechains.unsqueeze(-1), V_orientations], dim=-1)
        V = _merge(V_vec, V_dihedrals)

        if self.ablate_rbf:
            E = torch.cat([E_directions, E_positional], dim=-1)
        else:
            E = torch.cat([E_directions, RBF, E_positional], dim=-1)

        # Embed nodes
        Vv, Vs = self.node_embedding(V, return_split=True)
        if self.norm_nodes is None:
            self.norm_nodes = nn.LayerNorm(Vs.shape[-1], eps=1e-3, device=Vs.device)
        V = _merge(Vv, self.norm_nodes(Vs))

        # Embed edges
        Ev, Es = self.edge_embedding(E, return_split=True)
        if self.norm_edges is None:
            self.norm_edges = nn.LayerNorm(Es.shape[-1], eps=1e-3, device=Es.device)
        E = _merge(Ev, self.norm_edges(Es))

        return V, E, E_idx


# ---------------------------------------------------------------------------
# MQAModel (PocketMiner main model)
# ---------------------------------------------------------------------------

class PocketMinerTorch(nn.Module):
    """PyTorch port of PocketMiner's MQAModel for pocket prediction.

    Designed for batched inference: accepts [B, N, 4, 3] backbone coords.
    All operations are differentiable for gradient-based guidance.
    """

    def __init__(self, node_features: Tuple[int, int] = (8, 50),
                 edge_features: Tuple[int, int] = (1, 32),
                 hidden_dim: Tuple[int, int] = (16, 100),
                 num_layers: int = 4, k_neighbors: int = 30,
                 dropout: float = 0.1):
        super().__init__()

        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        self.features = StructuralFeatures(
            node_features, edge_features, top_k=k_neighbors,
            ablate_sidechain_vectors=True, ablate_rbf=False,
        )

        self.W_s = nn.Embedding(20, self.hs)

        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)

        self.encoder = Encoder(hidden_dim, edge_features,
                               num_layers=num_layers, dropout=dropout)

        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs, nls=None, nlv=None)

        self.dense = nn.Sequential(
            nn.Linear(self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.hs, 2 * self.hs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * self.hs, eps=1e-3),
            nn.Linear(2 * self.hs, 1),
        )

    def forward(self, X: torch.Tensor, S: torch.Tensor,
                mask: torch.Tensor, train: bool = False,
                res_level: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            X: [B, N, 4, 3] backbone atom coords (N, CA, C, O)
            S: [B, N] amino acid type indices (0-19)
            mask: [B, N] 1 for real residues, 0 for padding
            train: dropout active if True
            res_level: if True, return per-residue scores [B, N]

        Returns:
            [B, N] pocket probabilities (sigmoid) if res_level
            [B] protein-level scores otherwise
        """
        V, E, E_idx = self.features(X, mask)

        h_S = self.W_s(S)
        V = _vs_concat(V, h_S, self.nv, 0)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)

        h_V_out = self.W_V_out(h_V)

        if not res_level:
            mask_expanded = mask.unsqueeze(-1).float()
            h_V_out = (h_V_out * mask_expanded).sum(dim=-2)
            h_V_out = h_V_out / mask_expanded.sum(dim=-2).clamp(min=1)

        out = self.dense(h_V_out).squeeze(-1)

        if res_level:
            out = torch.sigmoid(out)

        return out


# ---------------------------------------------------------------------------
# Weight conversion: TF checkpoint -> PyTorch state dict
# ---------------------------------------------------------------------------

def convert_tf_to_pytorch(tf_checkpoint_path: str,
                          model: Optional[PocketMinerTorch] = None,
                          ) -> PocketMinerTorch:
    """Convert PocketMiner TF checkpoint to PyTorch model.

    This function:
    1. Loads the TF checkpoint
    2. Creates a PocketMinerTorch model
    3. Runs a dummy forward pass (to trigger lazy layer init)
    4. Maps TF weights to PyTorch parameters

    Args:
        tf_checkpoint_path: Path to TF checkpoint (without .index/.data suffix)
        model: Optional pre-created model

    Returns:
        PocketMinerTorch with loaded weights
    """
    import os
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf

    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')

    if model is None:
        model = PocketMinerTorch()

    # Dummy forward pass to initialize all lazy layers
    dummy_X = torch.randn(1, 50, 4, 3)
    dummy_S = torch.zeros(1, 50, dtype=torch.long)
    dummy_mask = torch.ones(1, 50)
    with torch.no_grad():
        model(dummy_X, dummy_S, dummy_mask, train=False, res_level=True)

    # Load TF model
    import sys
    pm_src = str(os.path.join(os.path.dirname(tf_checkpoint_path), '..', 'src'))
    if pm_src not in sys.path:
        sys.path.insert(0, pm_src)

    from models import MQAModel as TFMQAModel
    from util import load_checkpoint

    tf_model = TFMQAModel(
        node_features=(8, 50), edge_features=(1, 32),
        hidden_dim=(16, 100), num_layers=4, dropout=0.1,
    )
    opt = tf.keras.optimizers.legacy.Adam()

    # Build TF model with dummy input
    tf_X = np.random.randn(1, 50, 4, 3).astype(np.float32)
    tf_S = np.zeros((1, 50), dtype=np.int32)
    tf_mask = np.ones((1, 50), dtype=np.float32)
    with tf.device('/CPU:0'):
        tf_model(tf_X, tf_S, tf_mask, train=False, res_level=True)
        load_checkpoint(tf_model, opt, tf_checkpoint_path)

    # Now map weights
    _map_weights(tf_model, model)
    model.eval()  # nn.Dropout must be in eval mode for inference

    return model


def _map_weights(tf_model, pt_model: PocketMinerTorch):
    """Map TF MQAModel weights to PyTorch PocketMinerTorch.

    Weight mapping strategy:
    - TF Dense(in, out) weight shape: (in, out), bias: (out,)
    - PyTorch Linear(in, out) weight shape: (out, in), bias: (out,)
    - So we transpose the weight matrix.
    - TF Embedding weight: (vocab, dim) same as PyTorch
    - TF LayerNorm: gamma (scale), beta (offset) -> PyTorch weight, bias
    """
    # Helper to set a PyTorch param from numpy
    def _set(pt_param, np_val):
        pt_param.data = torch.from_numpy(np_val).float()

    def _set_dense(pt_linear, tf_dense):
        w, b = tf_dense.get_weights()
        _set(pt_linear.weight, w.T)
        _set(pt_linear.bias, b)

    def _set_layernorm(pt_ln, tf_ln):
        gamma, beta = tf_ln.get_weights()
        _set(pt_ln.weight, gamma)
        _set(pt_ln.bias, beta)

    def _set_gvp(pt_gvp, tf_gvp):
        """Map a TF GVP to PyTorch GVP."""
        if pt_gvp.vi and pt_gvp._wh is not None:
            _set_dense(pt_gvp._wh, tf_gvp.wh)
        _set_dense(pt_gvp._ws, tf_gvp.ws)
        if pt_gvp.vo and pt_gvp._wv is not None:
            _set_dense(pt_gvp._wv, tf_gvp.wv)

    def _set_gvp_layernorm(pt_gvpln, tf_gvpln):
        _set_layernorm(pt_gvpln.snorm, tf_gvpln.snorm)

    # 1. Embedding
    _set(pt_model.W_s.weight, tf_model.W_s.get_weights()[0])

    # 2. StructuralFeatures
    _set_gvp(pt_model.features.node_embedding, tf_model.features.node_embedding)
    _set_gvp(pt_model.features.edge_embedding, tf_model.features.edge_embedding)
    _set_layernorm(pt_model.features.norm_nodes, tf_model.features.norm_nodes)
    _set_layernorm(pt_model.features.norm_edges, tf_model.features.norm_edges)

    # 3. W_v, W_e (initial GVP projections)
    _set_gvp(pt_model.W_v, tf_model.W_v)
    _set_gvp(pt_model.W_e, tf_model.W_e)

    # 4. Encoder layers
    for i, (pt_layer, tf_layer) in enumerate(
            zip(pt_model.encoder.layers, tf_model.encoder.vglayers)):
        # MPNNLayer has W_EV (3 GVPs), W_dh (2 GVPs), 2 GVPLayerNorms
        for j, (pt_gvp, tf_gvp) in enumerate(
                zip(pt_layer.W_EV, tf_layer.W_EV.layers)):
            _set_gvp(pt_gvp, tf_gvp)

        for j, (pt_gvp, tf_gvp) in enumerate(
                zip(pt_layer.W_dh, tf_layer.W_dh.layers)):
            _set_gvp(pt_gvp, tf_gvp)

        for k in range(2):
            _set_gvp_layernorm(pt_layer.norm[k], tf_layer.norm[k])

    # 5. W_V_out
    _set_gvp(pt_model.W_V_out, tf_model.W_V_out)

    # 6. Dense head
    tf_dense_layers = [l for l in tf_model.dense.layers
                       if hasattr(l, 'get_weights') and len(l.get_weights()) > 0]
    pt_dense_layers = [m for m in pt_model.dense if isinstance(m, (nn.Linear, nn.LayerNorm))]

    for pt_l, tf_l in zip(pt_dense_layers, tf_dense_layers):
        if isinstance(pt_l, nn.Linear):
            _set_dense(pt_l, tf_l)
        elif isinstance(pt_l, nn.LayerNorm):
            _set_layernorm(pt_l, tf_l)


# ---------------------------------------------------------------------------
# Preprocessing (standalone, no TF dependency)
# ---------------------------------------------------------------------------

# Amino acid lookup tables
ABBREV = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "CYM": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}

AA_LOOKUP = {
    'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14,
    'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10,
    'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12,
}


def preprocess_pdb(pdb_path: str):
    """Preprocess PDB to (X, S, mask) tensors for PocketMinerTorch.

    Returns:
        X: [1, N, 4, 3] float32 tensor
        S: [1, N] long tensor
        mask: [1, N] float32 tensor
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)

    n_residues = prot_bb.top.n_residues
    xyz = prot_bb.xyz.reshape(n_residues, 4, 3).astype(np.float32)

    seq = [r.name for r in prot_bb.top.residues]
    S_np = np.array([AA_LOOKUP[ABBREV[a]] for a in seq], dtype=np.int64)

    X = torch.from_numpy(xyz).unsqueeze(0)  # [1, N, 4, 3]
    S = torch.from_numpy(S_np).unsqueeze(0)  # [1, N]
    mask = torch.ones(1, n_residues, dtype=torch.float32)

    return X, S, mask


def preprocess_pdb_batch(pdb_paths: list[str]):
    """Preprocess multiple PDBs into a padded batch.

    Returns:
        X: [B, N_max, 4, 3] float32 tensor
        S: [B, N_max] long tensor
        mask: [B, N_max] float32 tensor
    """
    import mdtraj as md

    all_xyz = []
    all_S = []
    all_n = []

    for path in pdb_paths:
        traj = md.load(path)
        prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = traj.atom_slice(prot_iis)
        n = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(n, 4, 3).astype(np.float32)
        seq = [r.name for r in prot_bb.top.residues]
        S_np = np.array([AA_LOOKUP[ABBREV[a]] for a in seq], dtype=np.int64)
        all_xyz.append(xyz)
        all_S.append(S_np)
        all_n.append(n)

    B = len(pdb_paths)
    N_max = max(all_n)

    X = torch.zeros(B, N_max, 4, 3)
    S = torch.zeros(B, N_max, dtype=torch.long)
    mask = torch.zeros(B, N_max)

    for i in range(B):
        n = all_n[i]
        X[i, :n] = torch.from_numpy(all_xyz[i])
        S[i, :n] = torch.from_numpy(all_S[i])
        mask[i, :n] = 1.0

    return X, S, mask
