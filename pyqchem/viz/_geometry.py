"""Geometry utilities: XYZ/cube readers, bond detection, 2D projection."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._elements import ELEMENTS, Z_TO_SYMBOL, get_element


# ---------------------------------------------------------------------------
# XYZ reader
# ---------------------------------------------------------------------------

def read_xyz(path: Path) -> list[dict]:
    """Read an XYZ file and return a list of atom dicts.

    Each dict has keys: atom_index (1-based), element, x, y, z (Å).
    """
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        natoms = int(lines[0])
    except ValueError:
        raise ValueError(f"First line of XYZ must be atom count, got: {lines[0]!r}")
    if natoms <= 0:
        raise ValueError(f"Atom count must be positive, got {natoms}")
    if len(lines) < natoms + 2:
        raise ValueError(f"XYZ file too short: expected {natoms + 2} lines, got {len(lines)}")

    atoms: list[dict] = []
    for i, line in enumerate(lines[2 : 2 + natoms], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {i + 2}: {line!r}")
        sym = parts[0]
        # Normalise symbol capitalisation
        if sym.isdigit():
            sym = Z_TO_SYMBOL.get(int(sym), sym)
        else:
            elem = get_element(sym)
            sym = elem.symbol
        atoms.append(
            {
                "atom_index": i,
                "element": sym,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            }
        )
    return atoms


# ---------------------------------------------------------------------------
# Gaussian cube reader
# ---------------------------------------------------------------------------

def read_cube(path: Path) -> dict:
    """Read a Gaussian cube file.

    Returns dict with keys: origin, vx, vy, vz, shape, atoms, grid.
    Coordinates are in bohr (as stored in cube).
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Line 3: natoms, origin
    parts2 = lines[2].split()
    natoms = int(parts2[0])
    origin = np.array([float(x) for x in parts2[1:4]], dtype=float)

    # Lines 4-6: grid vectors
    def _parse_grid_line(line: str):
        p = line.split()
        return int(p[0]), np.array([float(p[1]), float(p[2]), float(p[3])], dtype=float)

    nx, vx = _parse_grid_line(lines[3])
    ny, vy = _parse_grid_line(lines[4])
    nz, vz = _parse_grid_line(lines[5])

    # Atom block
    atoms: list[dict] = []
    for i, line in enumerate(lines[6 : 6 + natoms], start=1):
        p = line.split()
        z_num = int(float(p[0]))
        sym = Z_TO_SYMBOL.get(z_num, str(z_num))
        atoms.append(
            {
                "atom_index": i,
                "element": sym,
                "x": float(p[2]),
                "y": float(p[3]),
                "z": float(p[4]),
            }
        )

    # Cube volumetric data is whitespace-separated but line-wrapped arbitrarily,
    # so parse it as a flat stream rather than assuming a fixed column count.
    data_text = "".join(lines[6 + natoms :])
    data = np.fromstring(data_text, sep=" ")
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(
            f"Cube grid size mismatch in {path}: expected {expected} values, got {data.size}"
        )
    grid = data.reshape((nx, ny, nz))

    return {
        "origin": origin,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "shape": (nx, ny, nz),
        "atoms": atoms,
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def build_bonds(
    atoms: list[dict],
    tolerance_heavy: float = 1.25,
    tolerance_h: float = 1.15,
) -> list[tuple[int, int]]:
    """Detect covalent bonds based on inter-atomic distances and covalent radii.

    Returns list of (atom_index_i, atom_index_j) tuples.
    """
    bonds: list[tuple[int, int]] = []
    n = len(atoms)
    for i in range(n):
        ai = atoms[i]
        ri = ELEMENTS.get(ai["element"])
        cov_i = ri.covalent_radius if ri else 0.75
        for j in range(i + 1, n):
            aj = atoms[j]
            rj = ELEMENTS.get(aj["element"])
            cov_j = rj.covalent_radius if rj else 0.75
            is_h = ai["element"] == "H" or aj["element"] == "H"
            factor = tolerance_h if is_h else tolerance_heavy
            dx = ai["x"] - aj["x"]
            dy = ai["y"] - aj["y"]
            dz = ai["z"] - aj["z"]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= factor * (cov_i + cov_j):
                bonds.append((ai["atom_index"], aj["atom_index"]))
    return bonds


def build_bonds_from_coords(
    coords: np.ndarray,
    elements: list[str],
    tolerance_heavy: float = 1.25,
    tolerance_h: float = 1.15,
) -> list[tuple[int, int]]:
    """Bond detection using raw coordinate array (0-based indices).

    Used by ESP renderer where atoms are already in array form.
    """
    bonds: list[tuple[int, int]] = []
    n = len(elements)
    for i in range(n):
        ri = ELEMENTS.get(elements[i])
        cov_i = ri.covalent_radius if ri else 0.75
        for j in range(i + 1, n):
            rj = ELEMENTS.get(elements[j])
            cov_j = rj.covalent_radius if rj else 0.75
            is_h = elements[i] == "H" or elements[j] == "H"
            factor = tolerance_h if is_h else tolerance_heavy
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= factor * (cov_i + cov_j):
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# 2D projection via SVD
# ---------------------------------------------------------------------------

def project_2d(atoms: list[dict]) -> np.ndarray:
    """Project 3D atom coordinates onto the best-fit 2D plane (SVD).

    Returns (N, 2) array.  Also mutates atom dicts adding 'px', 'py' keys.
    """
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh[:2].T
    for atom, (px, py) in zip(atoms, proj):
        atom["px"] = float(px)
        atom["py"] = float(py)
    return proj
