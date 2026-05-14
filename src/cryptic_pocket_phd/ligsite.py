"""
LIGSITEcs-style pocket detection.

Implements the grid-based pocket scanning algorithm from:
  Huang & Schroeder (2006) "LIGSITEcsc: predicting ligand binding sites
  using the Connolly surface and degree of conservation"
  BMC Structural Biology 6:19

Settings matched to PocketMiner training pipeline:
  min_rank=7, grid_spacing=1.0 Å, probe_length=7 directions
"""
import numpy as np
from scipy.spatial import cKDTree


def compute_ligsite_grid(
    coords: np.ndarray,
    grid_spacing: float = 1.0,
    padding: float = 10.0,
    min_rank: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LIGSITEcs pocket grid points.

    Args:
        coords: (N_atoms, 3) atom coordinates in Angstroms
        grid_spacing: grid resolution in Angstroms
        padding: extra space around protein bounding box
        min_rank: minimum number of directions showing PSP pattern
                  (max 7: x, y, z, xy, xz, yz, xyz diagonals)

    Returns:
        pocket_points: (M, 3) coordinates of pocket grid points
        pocket_ranks: (M,) rank of each pocket point (how many directions show PSP)
    """
    # Build grid around protein
    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding
    axes = [np.arange(lo, hi + grid_spacing, grid_spacing) for lo, hi in zip(mins, maxs)]
    nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])

    # Build atom KD-tree for fast distance queries
    atom_tree = cKDTree(coords)

    # Mark grid points that are "inside" protein (within vdW radius ~1.8 Å)
    # Use a uniform probe radius for simplicity (matching LIGSITEcs default)
    probe_radius = 1.6  # Å, roughly carbon vdW radius

    # Generate all grid points
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Query which grid points are "protein" (within probe_radius of any atom)
    dists, _ = atom_tree.query(grid_points)
    protein_mask = (dists <= probe_radius).reshape(nx, ny, nz)

    # Scan 7 directions for PSP (protein-solvent-protein) pattern
    directions = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),   # x, y, z
        (1, 1, 0), (1, 0, 1), (0, 1, 1),    # xy, xz, yz diagonals
        (1, 1, 1),                            # xyz diagonal
    ]

    rank_grid = np.zeros((nx, ny, nz), dtype=np.int32)

    for di, dj, dk in directions:
        psp = _scan_direction(protein_mask, di, dj, dk)
        rank_grid += psp

    # Extract pocket points where rank >= min_rank
    pocket_mask = (rank_grid >= min_rank) & (~protein_mask)
    pocket_indices = np.argwhere(pocket_mask)

    if len(pocket_indices) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)

    pocket_points = np.column_stack([
        axes[0][pocket_indices[:, 0]],
        axes[1][pocket_indices[:, 1]],
        axes[2][pocket_indices[:, 2]],
    ])
    pocket_ranks = rank_grid[pocket_mask]

    return pocket_points, pocket_ranks


def _scan_direction(
    protein_mask: np.ndarray, di: int, dj: int, dk: int
) -> np.ndarray:
    """
    Scan along a direction for PSP (protein-solvent-protein) pattern.

    Vectorized: accumulate OR over max_scan shift steps instead of
    looping per grid point. ~50-100x faster than pure Python loop.

    Returns binary array: 1 if PSP detected at that point, 0 otherwise.
    """
    nx, ny, nz = protein_mask.shape
    max_scan = 8

    fwd = np.zeros((nx, ny, nz), dtype=bool)
    bwd = np.zeros((nx, ny, nz), dtype=bool)

    for s in range(1, max_scan + 1):
        si, sj, sk = s * di, s * dj, s * dk

        # Forward: point (i,j,k) sees protein at (i+si, j+sj, k+sk)
        i_dst = slice(None, nx - si if si > 0 else None)
        j_dst = slice(None, ny - sj if sj > 0 else None)
        k_dst = slice(None, nz - sk if sk > 0 else None)
        i_src = slice(si, None)
        j_src = slice(sj, None)
        k_src = slice(sk, None)
        fwd[i_dst, j_dst, k_dst] |= protein_mask[i_src, j_src, k_src]

        # Backward: point (i,j,k) sees protein at (i-si, j-sj, k-sk)
        i_dst2 = slice(si, None)
        j_dst2 = slice(sj, None)
        k_dst2 = slice(sk, None)
        i_src2 = slice(None, nx - si if si > 0 else None)
        j_src2 = slice(None, ny - sj if sj > 0 else None)
        k_src2 = slice(None, nz - sk if sk > 0 else None)
        bwd[i_dst2, j_dst2, k_dst2] |= protein_mask[i_src2, j_src2, k_src2]

    return (fwd & bwd & ~protein_mask).astype(np.int32)


def assign_pocket_to_residues(
    pocket_points: np.ndarray,
    pocket_ranks: np.ndarray,
    ca_coords: np.ndarray,
    n_residues: int,
) -> np.ndarray:
    """
    Assign pocket grid points to nearest CA atom (residue).

    PocketMiner's "grid point to nearest residue" procedure.

    Args:
        pocket_points: (M, 3) pocket grid point coordinates
        pocket_ranks: (M,) rank of each pocket point
        ca_coords: (N_residues, 3) CA atom coordinates
        n_residues: number of residues

    Returns:
        per_residue_score: (N_residues,) pocket score per residue
            (sum of ranks of assigned grid points)
    """
    if len(pocket_points) == 0:
        return np.zeros(n_residues, dtype=np.float32)

    ca_tree = cKDTree(ca_coords)
    _, nearest_residue = ca_tree.query(pocket_points)

    per_residue_score = np.zeros(n_residues, dtype=np.float32)
    for idx, rank in zip(nearest_residue, pocket_ranks):
        per_residue_score[idx] += rank

    return per_residue_score


def ligsite_labels(
    coords: np.ndarray,
    ca_coords: np.ndarray,
    n_residues: int,
    pos_thresh: int = 20,
    min_rank: int = 7,
    grid_spacing: float = 1.0,
) -> np.ndarray:
    """
    Full LIGSITE labeling pipeline: coords → binary per-residue labels.

    Matches PocketMiner training settings.

    Args:
        coords: (N_atoms, 3) all-atom coordinates in Angstroms
        ca_coords: (N_residues, 3) CA coordinates in Angstroms
        n_residues: number of residues
        pos_thresh: grid-point count threshold for positive label
        min_rank: min directions for PSP
        grid_spacing: grid spacing in Angstroms

    Returns:
        labels: (N_residues,) binary labels (1 = pocket residue)
    """
    pocket_points, pocket_ranks = compute_ligsite_grid(
        coords, grid_spacing=grid_spacing, min_rank=min_rank
    )
    scores = assign_pocket_to_residues(
        pocket_points, pocket_ranks, ca_coords, n_residues
    )
    labels = (scores >= pos_thresh).astype(np.int32)
    return labels
