"""Wrapper for fpocket classical pocket detector.

fpocket (Le Guilloux et al. 2009) detects pockets via Voronoi-based alpha
spheres.  Per-atom annotation is in the output PDB occupancy column.

Output per residue
------------------
Per-atom score = occupancy in {stem}_out/{stem}_out.pdb.
Non-zero = atom contacts at least one detected alpha sphere cluster.
The value is approximately proportional to the score of the strongest pocket
the atom participates in (monotone but not exactly equal to pocket score).

Per-residue score = max occupancy across all heavy atoms in that residue.
Value in [0, 1].  0 = residue in no detected pocket.

Platform
--------
fpocket is a C binary; not available natively on Windows.
On Linux/Mac: FPOCKET_BIN env var or auto-detected on PATH.
On Windows: invoked via WSL (Ubuntu) using the binary at ~/tools/fpocket/bin/fpocket.

Usage
-----
>>> scores = score("data/validation_pdbs/1NEP_A.pdb")
>>> scores.shape
(130,)
>>> scores[90]   # residue 91 (1-based) → index 90
0.58
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

_WSL_FPOCKET = "~/tools/fpocket/bin/fpocket"  # path inside WSL Ubuntu

def _get_fpocket_cmd() -> list[str]:
    """Return the command prefix to invoke fpocket.

    On Linux/Mac: ['fpocket'] or [$FPOCKET_BIN].
    On Windows: ['wsl', 'bash', '-c', '...'] — calls WSL binary.
    """
    env_bin = os.environ.get("FPOCKET_BIN")

    if platform.system() in ("Linux", "Darwin"):
        if env_bin:
            return [env_bin]
        found = shutil.which("fpocket")
        if found:
            return [found]
        raise FileNotFoundError(
            "fpocket binary not found. Install via apt/conda or set FPOCKET_BIN."
        )

    # Windows — use WSL
    return None  # sentinel: use _run_fpocket_wsl


def _is_windows() -> bool:
    return platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Run fpocket
# ---------------------------------------------------------------------------

def _run_fpocket_native(pdb_path: str, out_dir: str) -> Path:
    """Run fpocket natively (Linux/Mac).  Returns path to *_out.pdb."""
    cmd = _get_fpocket_cmd()
    pdb_path = Path(pdb_path).resolve()  # absolute — safe with cwd change
    result = subprocess.run(
        cmd + ["-f", pdb_path.name],     # just filename; cwd = parent dir
        capture_output=True,
        text=True,
        cwd=str(pdb_path.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"fpocket failed:\n{result.stderr}")
    stem = pdb_path.stem
    out_pdb = pdb_path.parent / f"{stem}_out" / f"{stem}_out.pdb"
    return out_pdb


def _run_fpocket_wsl(pdb_path: str) -> Path:
    """Run fpocket via WSL on Windows.

    Copies PDB to /tmp in WSL (avoids spaces-in-path issues), runs fpocket,
    copies output PDB back to a Windows temp directory.

    Returns
    -------
    Path  — path to the *_out.pdb on the Windows side.
    """
    pdb_path = Path(pdb_path).resolve()
    stem = pdb_path.stem

    # Copy input PDB to WSL /tmp (avoids Windows path spaces)
    wsl_src = f"/tmp/{stem}_input.pdb"
    wsl_out_dir = f"/tmp/{stem}_input_out"
    wsl_out_pdb = f"{wsl_out_dir}/{stem}_input_out.pdb"

    # Write PDB content via wsl
    script = (
        f"cp /dev/stdin {wsl_src} && "
        f"cd /tmp && {_WSL_FPOCKET} -f {wsl_src} > /dev/null 2>&1 && "
        f"cat {wsl_out_pdb}"
    )

    with open(str(pdb_path), "rb") as f:
        pdb_content = f.read()

    result = subprocess.run(
        ["wsl", "bash", "-c", script],
        input=pdb_content,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0 or not result.stdout:
        # Fallback: copy file first then run
        wsl_win_path_result = subprocess.run(
            ["wsl", "wslpath", "-u", str(pdb_path)],
            capture_output=True, text=True,
        )
        wsl_win_path = wsl_win_path_result.stdout.strip()
        script2 = (
            f"cp '{wsl_win_path}' {wsl_src} && "
            f"cd /tmp && {_WSL_FPOCKET} -f {wsl_src} > /dev/null 2>&1 && "
            f"cat {wsl_out_pdb}"
        )
        result = subprocess.run(
            ["wsl", "bash", "-c", script2],
            capture_output=True, timeout=120,
        )

    if not result.stdout:
        raise RuntimeError(
            f"fpocket via WSL produced no output for {pdb_path}.\n"
            f"stderr: {result.stderr.decode(errors='replace')[:500]}"
        )

    # Write output PDB to Windows temp
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="wb") as tmp:
        tmp.write(result.stdout)
        return Path(tmp.name)


# ---------------------------------------------------------------------------
# Parse fpocket output PDB
# ---------------------------------------------------------------------------

def _parse_out_pdb(out_pdb: Path) -> dict[int, float]:
    """Parse fpocket *_out.pdb → {resSeq: max_occupancy}.

    The occupancy column (1-indexed cols 55-60) is non-zero for atoms that
    contact alpha spheres.  We take max over all heavy atoms per residue.
    """
    res_score: dict[int, float] = defaultdict(float)

    with open(str(out_pdb)) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                resseq = int(line[22:26])
                occ = float(line[54:60])
            except ValueError:
                continue
            if occ > res_score[resseq]:
                res_score[resseq] = occ

    return dict(res_score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(structure_path: str) -> np.ndarray:
    """Score residues for classical pocket probability via fpocket.

    Parameters
    ----------
    structure_path : str
        Path to a PDB file (backbone-only or all-atom; must have ATOM records).

    Returns
    -------
    np.ndarray, shape (n_residues,), float32.
        Per-residue score: max occupancy of any heavy atom in the residue.
        0.0 = residue contacts no alpha sphere (not in any pocket).
        Values in [0, 1] approximately.

    Residue ordering
    ----------------
    Indexed 0..N-1 in mdtraj sequential order (= Boltz-1 output order).
    Residue 0 = first residue in PDB chain order.
    """
    import mdtraj as md

    structure_path = str(structure_path)

    # Run fpocket
    if _is_windows():
        out_pdb = _run_fpocket_wsl(structure_path)
    else:
        out_pdb = _run_fpocket_native(structure_path, "")

    try:
        # Parse per-resSeq scores
        res_score = _parse_out_pdb(out_pdb)

        # Build sequential index → resSeq mapping via mdtraj
        traj = md.load(structure_path)
        ca_sel = traj.top.select("protein and name CA")
        ca_traj = traj.atom_slice(ca_sel)
        residues = list(ca_traj.top.residues)

        scores = np.zeros(len(residues), dtype=np.float32)
        for i, res in enumerate(residues):
            scores[i] = res_score.get(res.resSeq, 0.0)

        return scores

    finally:
        # Clean up WSL temp file
        if _is_windows():
            try:
                out_pdb.unlink()
            except Exception:
                pass


def score_no_structure_required(
    structure_path: str,
    n_residues: int,
    resseq_list: list[int],
) -> np.ndarray:
    """Like score() but caller provides residue ordering (avoids double mdtraj load).

    Parameters
    ----------
    structure_path : str
    n_residues : int
    resseq_list : list[int]  — resSeq values in sequential order (from mdtraj)

    Returns
    -------
    np.ndarray, shape (n_residues,)
    """
    structure_path = str(structure_path)

    if _is_windows():
        out_pdb = _run_fpocket_wsl(structure_path)
    else:
        out_pdb = _run_fpocket_native(structure_path, "")

    try:
        res_score = _parse_out_pdb(out_pdb)
        scores = np.array(
            [res_score.get(seq, 0.0) for seq in resseq_list],
            dtype=np.float32,
        )
        return scores
    finally:
        if _is_windows():
            try:
                out_pdb.unlink()
            except Exception:
                pass
