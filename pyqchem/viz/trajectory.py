"""Geometry-optimization trajectory parsing and animation.

Reads a multi-frame XYZ file (e.g. ORCA's ``*_trj.xyz``, the trajectories
extracted from a Gaussian opt log, or any concatenated XYZ) and writes an
animated 2D ball-and-stick view of the optimization path.

Output format is determined from the suffix of ``output``:
- ``.gif``  → matplotlib's PillowWriter (no external tool needed)
- ``.mp4``  → ffmpeg (must be on PATH)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle

from ._elements import Z_TO_SYMBOL, get_element
from ._geometry import build_bonds_from_coords
from ._style import (
    DEFAULT_DPI,
    BOND_COLOR,
    H_FILL,
    H_EDGE,
    HEAVY_EDGE,
    HEAVY_ALPHA,
    H_ALPHA,
    get_display_radius,
    get_atom_color,
)


def read_xyz_trajectory(path: Path) -> tuple[list[str], list[np.ndarray]]:
    """Read a multi-frame XYZ file.

    Returns
    -------
    elements
        List of element symbols (length N) — assumed constant across frames.
    frames
        List of (N, 3) numpy arrays of Cartesian coordinates (Å).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    frames: list[np.ndarray] = []
    elements: list[str] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            natoms = int(line)
        except ValueError:
            i += 1
            continue
        if i + 1 + natoms >= n_lines:
            break
        block = lines[i + 2: i + 2 + natoms]
        coords = np.zeros((natoms, 3), dtype=float)
        block_elements: list[str] = []
        for j, atom_line in enumerate(block):
            parts = atom_line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line in {path}: {atom_line!r}")
            sym = parts[0]
            if sym.isdigit():
                sym = Z_TO_SYMBOL.get(int(sym), sym)
            else:
                sym = get_element(sym).symbol
            block_elements.append(sym)
            coords[j] = (float(parts[1]), float(parts[2]), float(parts[3]))
        if not elements:
            elements = block_elements
        elif block_elements != elements:
            raise ValueError(
                f"Frame {len(frames)+1} in {path} has different elements than the first frame."
            )
        frames.append(coords)
        i += 2 + natoms
    if not frames:
        raise ValueError(f"No frames parsed from {path}")
    return elements, frames


def _project_2d_consistent(frames: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Project all frames into a single 2D plane using SVD on the first frame.

    Returns the list of projected (N,2) arrays and the global axis bounds.
    """
    ref = frames[0]
    centered = ref - ref.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T  # (3,2)
    projected = []
    for f in frames:
        c = f - f.mean(axis=0)
        projected.append(c @ basis)
    all_pts = np.vstack(projected)
    pad = 0.6
    bounds = np.array([
        all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad,
        all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad,
    ])
    return projected, bounds


def render_trajectory_animation(
    trajectory_path: Path,
    output: Path,
    fps: int = 8,
    dpi: int = DEFAULT_DPI,
    title: str = "",
    stride: int = 1,
) -> None:
    """Render a multi-frame XYZ as a 2D ball-and-stick animation.

    Parameters
    ----------
    trajectory_path
        Multi-frame XYZ (e.g. ORCA ``*_trj.xyz``).
    output
        Output ``.gif`` or ``.mp4`` path.
    fps
        Frames per second.
    dpi
        Output resolution.
    title
        Optional figure title; the frame index is appended automatically.
    stride
        Take every Nth frame (1 = all frames).
    """
    elements, all_frames = read_xyz_trajectory(Path(trajectory_path))
    if stride > 1:
        all_frames = all_frames[::stride]
    if len(all_frames) < 2:
        raise ValueError(
            f"Trajectory {trajectory_path} only has {len(all_frames)} frame(s); "
            "need at least 2 to animate."
        )

    projected, bounds = _project_2d_consistent(all_frames)
    bonds = build_bonds_from_coords(all_frames[0], elements)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # Persistent artists we update each frame
    bond_lines = []
    for _ in bonds:
        line, = ax.plot([], [], color=BOND_COLOR, lw=2.4, zorder=1, solid_capstyle="round")
        bond_lines.append(line)

    atom_patches: list[Circle] = []
    for elem in elements:
        radius = get_display_radius(elem)
        if elem == "H":
            patch = Circle((0, 0), radius * 0.75,
                           facecolor=H_FILL, edgecolor=H_EDGE,
                           lw=1.0, alpha=H_ALPHA, zorder=3)
        else:
            patch = Circle((0, 0), radius,
                           facecolor=get_atom_color(elem),
                           edgecolor=HEAVY_EDGE,
                           lw=1.0, alpha=HEAVY_ALPHA, zorder=4)
        ax.add_patch(patch)
        atom_patches.append(patch)

    title_text = ax.set_title(title or "Trajectory", fontsize=13)

    def _update(frame_idx: int):
        proj = projected[frame_idx]
        for (i, j), line in zip(bonds, bond_lines):
            line.set_data([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]])
        for patch, (px, py) in zip(atom_patches, proj):
            patch.center = (px, py)
        title_text.set_text(
            f"{title or 'Trajectory'}  —  frame {frame_idx + 1}/{len(projected)}"
        )
        return [*bond_lines, *atom_patches, title_text]

    anim = FuncAnimation(
        fig, _update, frames=len(projected), interval=1000 / fps, blit=False,
    )

    out = Path(output)
    suffix = out.suffix.lower()
    if suffix == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
