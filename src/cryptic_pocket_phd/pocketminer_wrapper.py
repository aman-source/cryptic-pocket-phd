"""Wrapper for PocketMiner inference.

PocketMiner (Meller et al. 2023, Nat Comms) predicts cryptic pocket
probability per residue from a single protein structure.

Repo: github.com/Mickdub/gvp, branch pocket_pred.
Checkpoint: external/pocketminer/models/pocketminer

Architecture (from xtal_predict.py):
    MQAModel(node_features=(8,50), edge_features=(1,32),
             hidden_dim=(16,100), num_layers=4, dropout=0.1)

Input:
    Any standard PDB file with at least one protein chain containing
    backbone atoms N, CA, C, O.  Hydrogens and HETATM records are
    ignored — no pre-cleaning required.

Output:
    np.ndarray of shape (n_residues,), dtype float32, values in [0, 1].
    Index i = i-th residue in mdtraj topology order (preserves PDB
    chain/residue order).  The model uses sigmoid activation internally;
    no additional transformation is needed.

Residue ordering:
    Scores are indexed 0..N-1 in the order mdtraj reads the PDB.
    Use residue_mapping.build_residue_mapping() to get the corresponding
    PDB resSeq numbers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Must be set before any tensorflow import (even transitive).
# tf_keras provides legacy Keras 2 API required by PocketMiner on TF 2.16+.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ---------------------------------------------------------------------------
# PocketMiner source path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PM_SRC = _REPO_ROOT / "external" / "pocketminer" / "src"
_MODEL_CHECKPOINT = _REPO_ROOT / "external" / "pocketminer" / "models" / "pocketminer"

_DROPOUT_RATE = 0.1
_NUM_LAYERS = 4
_HIDDEN_DIM = 100


def _ensure_pm_on_path() -> None:
    p = str(_PM_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None


def _load_model():
    """Instantiate MQAModel and restore PocketMiner weights."""
    _ensure_pm_on_path()

    import tensorflow as tf
    # Restrict TF to CPU — leaves all GPU memory for PyTorch/Boltz.
    # Must be called before any TF GPU op. PocketMiner GVP is small; CPU adds <1s.
    tf.config.set_visible_devices([], "GPU")
    from models import MQAModel  # noqa: PL — PocketMiner's models.py
    from util import load_checkpoint  # noqa: PL — PocketMiner's util.py

    if not _MODEL_CHECKPOINT.with_suffix(".index").exists():
        raise FileNotFoundError(
            f"PocketMiner checkpoint not found at {_MODEL_CHECKPOINT}. "
            "Run: git clone --branch pocket_pred https://github.com/Mickdub/gvp.git "
            "external/pocketminer"
        )

    model = MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=(16, _HIDDEN_DIM),
        num_layers=_NUM_LAYERS,
        dropout=_DROPOUT_RATE,
    )
    # Use legacy optimizer — required to restore TF 2.x checkpoints in TF 2.11+.
    opt = tf.keras.optimizers.legacy.Adam()
    load_checkpoint(model, opt, str(_MODEL_CHECKPOINT))
    return model


def get_model():
    """Return (and cache) the singleton PocketMiner model."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def reset_model() -> None:
    """Clear the singleton (useful in tests)."""
    global _model
    _model = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(
    structure_path: str,
    model=None,
    preprocess_fn=None,
) -> np.ndarray:
    """Score residues for cryptic pocket probability.

    Parameters
    ----------
    structure_path : str
        Path to a PDB file.
    model : optional MQAModel
        Pre-loaded model (for testing).  If None, the singleton is used.
    preprocess_fn : optional callable
        Callable(structure_path: str) -> (X, S, mask).  If None, uses
        mdtraj + PocketMiner's process_strucs.  Pass a mock in unit tests
        to avoid importing TF/mdtraj.

    Returns
    -------
    np.ndarray, shape (n_residues,), float32, values in [0, 1].
    """
    structure_path = str(structure_path)

    if preprocess_fn is not None:
        X, S, mask = preprocess_fn(structure_path)
    else:
        _ensure_pm_on_path()
        import mdtraj as md
        from validate_performance_on_xtals import process_strucs  # PocketMiner

        traj = md.load(structure_path)
        X, S, mask = process_strucs([traj])

    m = model if model is not None else get_model()
    preds = m(X, S, mask, train=False, res_level=True)  # [1, L_max]

    # Support both TF Tensors (.numpy()) and plain numpy arrays (unit tests).
    preds_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)

    n_residues = int(mask[0].sum())
    scores = preds_np[0, :n_residues].astype(np.float32)
    return scores
