"""Spearman correlation between PocketMiner scores at noisy intermediates vs x_0.

Task E — Spec #0 §6.2
----------------------
For each protein × sample × timestep:
  s_t  = PocketMiner(x̂_0(x_t))   — score at noisy intermediate
  s_0  = PocketMiner(x_0)          — score at reference apo structure

Compute Spearman ρ(s_t, s_0):
  - restricted to cryptic-pocket residues (primary metric)
  - on all residues (diagnostic)

Aggregate per protein × timestep: mean ± SD over samples.
Bootstrap (1000 resamples over proteins) for CI on aggregate ρ at each timestep.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_rho(
    s_t: np.ndarray,
    s_0: np.ndarray,
    pocket_indices: list[int],
) -> tuple[float, float]:
    """Spearman ρ between noisy-intermediate and reference scores.

    Parameters
    ----------
    s_t : np.ndarray, shape (n_residues,)
        PocketMiner scores at intermediate x̂_0(x_t).
    s_0 : np.ndarray, shape (n_residues,)
        PocketMiner scores at reference x_0.
    pocket_indices : list[int]
        0-based residue indices for the cryptic-pocket region.

    Returns
    -------
    (rho_pocket, rho_all) : tuple[float, float]
        Spearman ρ restricted to pocket residues, then over all residues.
    """
    if len(s_t) != len(s_0):
        raise ValueError(
            f"Score arrays must be same length: len(s_t)={len(s_t)}, len(s_0)={len(s_0)}"
        )
    if len(pocket_indices) < 2:
        raise ValueError(
            f"Need ≥2 pocket residues for Spearman; got {len(pocket_indices)}"
        )

    idx = np.array(pocket_indices, dtype=int)
    rho_pocket = float(spearmanr(s_t[idx], s_0[idx]).statistic)
    rho_all = float(spearmanr(s_t, s_0).statistic)
    return rho_pocket, rho_all


def aggregate_rho(
    rho_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± SD of ρ over samples for a single protein × timestep.

    Parameters
    ----------
    rho_matrix : np.ndarray, shape (n_samples,)
        Spearman ρ values from each sample for one (protein, timestep) pair.

    Returns
    -------
    (mean, sd) : tuple[np.ndarray, np.ndarray]
        Scalars (float64).
    """
    return float(np.mean(rho_matrix)), float(np.std(rho_matrix, ddof=1))


def bootstrap_ci(
    protein_means: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI on the mean of per-protein ρ values at one timestep.

    Parameters
    ----------
    protein_means : np.ndarray, shape (n_proteins,)
        Per-protein mean ρ at one timestep.
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    ci : float
        Confidence level (default 0.95 → 95% CI).
    rng : np.random.Generator or None
        RNG for reproducibility.

    Returns
    -------
    (point_estimate, lower, upper) : tuple[float, float, float]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(protein_means)
    if n == 0:
        raise ValueError("protein_means is empty")

    point = float(np.mean(protein_means))

    boot_means = np.array([
        np.mean(rng.choice(protein_means, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lower, upper


def build_results_table(
    results: dict,
    timesteps: list[float],
) -> dict:
    """Aggregate raw per-sample ρ into summary per protein × timestep.

    Parameters
    ----------
    results : dict
        Nested: results[protein_id][timestep][sample_idx] = (rho_pocket, rho_all)
    timesteps : list[float]
        Ordered list of timesteps, e.g. [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns
    -------
    dict with structure:
        {
          "per_protein": {
            protein_id: {
              t: {"mean_pocket": float, "sd_pocket": float,
                  "mean_all": float, "sd_all": float}
            }
          },
          "aggregate": {
            t: {"point": float, "ci_lower": float, "ci_upper": float,
                "point_all": float, "ci_lower_all": float, "ci_upper_all": float}
          }
        }
    """
    per_protein: dict = {}
    for prot, ts_data in results.items():
        per_protein[prot] = {}
        for t in timesteps:
            samples = ts_data.get(t, {})
            if not samples:
                continue
            rho_pocket_arr = np.array([v[0] for v in samples.values()])
            rho_all_arr = np.array([v[1] for v in samples.values()])
            m_p, sd_p = aggregate_rho(rho_pocket_arr)
            m_a, sd_a = aggregate_rho(rho_all_arr)
            per_protein[prot][t] = {
                "mean_pocket": m_p, "sd_pocket": sd_p,
                "mean_all": m_a, "sd_all": sd_a,
            }

    aggregate: dict = {}
    rng = np.random.default_rng(42)
    for t in timesteps:
        pocket_means = np.array([
            per_protein[p][t]["mean_pocket"]
            for p in per_protein if t in per_protein[p]
        ])
        all_means = np.array([
            per_protein[p][t]["mean_all"]
            for p in per_protein if t in per_protein[p]
        ])
        if len(pocket_means) == 0:
            continue
        pt, lo, hi = bootstrap_ci(pocket_means, rng=rng)
        pt_a, lo_a, hi_a = bootstrap_ci(all_means, rng=rng)
        aggregate[t] = {
            "point": pt, "ci_lower": lo, "ci_upper": hi,
            "point_all": pt_a, "ci_lower_all": lo_a, "ci_upper_all": hi_a,
        }

    return {"per_protein": per_protein, "aggregate": aggregate}
