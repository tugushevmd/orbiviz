"""2D contour slice through a molecular-orbital cube file.

Unlike :mod:`pyqchem.viz.mo_surface` (which extracts a 3D isosurface using
scikit-image's marching cubes), this module produces a flat 2D plot of the
orbital amplitude on an axis-aligned plane (xy/xz/yz). It only depends on
NumPy + Matplotlib, so it works in the lightweight install profile.

Atoms close to the slicing plane are overlaid as small CPK markers so the
viewer keeps a sense of where the orbital sits in the molecule.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._geometry import read_cube
from ._style import DEFAULT_DPI, get_atom_color, get_atom_edge_color


_BOHR_TO_ANG = 0.52917721067
_PLANE_AXES = {
    "xy": (0, 1, 2),  # vary X, Y; sliced normal = Z
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
_PLANE_LABELS = {
    "xy": ("x, Å", "y, Å", "z"),
    "xz": ("x, Å", "z, Å", "y"),
    "yz": ("y, Å", "z, Å", "x"),
}


def render_mo_slice_2d(
    cube_path: Path,
    output: Path,
    plane: str = "xy",
    offset: float = 0.0,
    n_levels: int = 18,
    title: str = "",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 2D contour slice of an MO cube file.

    Parameters
    ----------
    cube_path
        Gaussian cube file containing MO amplitudes.
    output
        Output image path. Format is chosen from the suffix.
    plane
        ``"xy"``, ``"xz"`` or ``"yz"``.
    offset
        Offset (in Å) along the plane normal from the molecular geometric
        centre. ``0`` slices through the centre.
    n_levels
        Number of contour levels (split symmetrically around 0).
    title
        Optional figure title.
    dpi
        Output resolution.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {list(_PLANE_AXES)}, got {plane!r}")

    cube = read_cube(cube_path)
    grid = cube["grid"]                # values
    origin = cube["origin"] * _BOHR_TO_ANG
    nx, ny, nz = cube["shape"]
    dx = cube["vx"][0] * _BOHR_TO_ANG
    dy = cube["vy"][1] * _BOHR_TO_ANG
    dz = cube["vz"][2] * _BOHR_TO_ANG

    xs = origin[0] + np.arange(nx) * dx
    ys = origin[1] + np.arange(ny) * dy
    zs = origin[2] + np.arange(nz) * dz

    atom_xyz_ang = np.array(
        [[a["x"] * _BOHR_TO_ANG, a["y"] * _BOHR_TO_ANG, a["z"] * _BOHR_TO_ANG]
         for a in cube["atoms"]],
        dtype=float,
    )
    elements = [a["element"] for a in cube["atoms"]]
    center = atom_xyz_ang.mean(axis=0)

    a, b, n = _PLANE_AXES[plane]
    axis_arrays = [xs, ys, zs]
    plane_a = axis_arrays[a]
    plane_b = axis_arrays[b]
    normal_axis = axis_arrays[n]
    normal_value = center[n] + offset

    # Closest grid index along the normal axis
    k = int(np.argmin(np.abs(normal_axis - normal_value)))
    if plane == "xy":
        slab = grid[:, :, k].T   # shape (ny, nx) for proper imshow orientation
    elif plane == "xz":
        slab = grid[:, k, :].T
    else:  # yz
        slab = grid[k, :, :].T

    vmax = float(np.nanmax(np.abs(slab))) or 1.0
    levels = np.linspace(-vmax, vmax, n_levels + 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=dpi)
    extent = (plane_a[0], plane_a[-1], plane_b[0], plane_b[-1])

    cf = ax.contourf(plane_a, plane_b, slab, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(plane_a, plane_b, slab, levels=levels, colors="k", linewidths=0.25, alpha=0.5)
    ax.contour(plane_a, plane_b, slab, levels=[0.0], colors="k", linewidths=0.6)

    # Overlay atoms close to the slicing plane (within ±0.6 Å)
    for elem, xyz in zip(elements, atom_xyz_ang):
        if abs(xyz[n] - normal_value) > 0.6:
            continue
        ax.plot(
            xyz[a], xyz[b],
            marker="o", markersize=9 if elem != "H" else 6,
            markerfacecolor=get_atom_color(elem),
            markeredgecolor=get_atom_edge_color(elem),
            markeredgewidth=0.7, linestyle="none", zorder=5,
        )
        if elem != "H":
            ax.text(
                xyz[a], xyz[b] + 0.12, elem,
                ha="center", va="bottom", fontsize=8, weight="bold",
                color="black", zorder=6,
            )

    xl, yl, nl = _PLANE_LABELS[plane]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(
        title or f"MO slice ({plane}, {nl}={normal_value:+.2f} Å)",
        fontsize=13,
    )

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Orbital amplitude (a.u.)")

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
