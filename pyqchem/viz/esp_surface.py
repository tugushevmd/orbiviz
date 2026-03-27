"""3D ESP-colored electron density isosurface visualization."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

from ._elements import ELEMENTS
from ._geometry import read_cube, build_bonds_from_coords
from ._style import DEFAULT_DPI, get_atom_color, get_atom_edge_color


def render_esp_surface(
    density_cube_path: Path,
    esp_cube_path: Path,
    output: Path,
    isovalue: float = 0.001,
    alpha: float = 0.22,
    elev: float = 22.0,
    azim: float = 58.0,
    max_faces: int = 55000,
    labels: str = "elements",
    title: str = "",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render an ESP-colored density isosurface and save to PNG.

    Parameters
    ----------
    density_cube_path : Path
        Path to electron density Gaussian cube file.
    esp_cube_path : Path
        Path to ESP Gaussian cube file (compatible grid).
    output : Path
        Output PNG path.
    isovalue : float
        Density isovalue for the isosurface (a.u.).
    alpha : float
        Surface transparency (0–1).
    elev, azim : float
        Camera elevation and azimuth angles.
    max_faces : int
        Maximum triangular faces kept for rendering.
    labels : str
        'none', 'elements', or 'indices'.
    title : str
        Optional figure title.
    dpi : int
        Output resolution.
    """
    density = read_cube(density_cube_path)
    esp = read_cube(esp_cube_path)

    atom_xyz = np.array(
        [[a["x"], a["y"], a["z"]] for a in density["atoms"]], dtype=float
    )
    elements = [a["element"] for a in density["atoms"]]
    center = atom_xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(atom_xyz - center, full_matrices=False)
    rot = vh.T

    # ESP interpolator
    x_esp = esp["origin"][0] + np.arange(esp["shape"][0]) * esp["vx"][0]
    y_esp = esp["origin"][1] + np.arange(esp["shape"][1]) * esp["vy"][1]
    z_esp = esp["origin"][2] + np.arange(esp["shape"][2]) * esp["vz"][2]
    interp = RegularGridInterpolator(
        (x_esp, y_esp, z_esp), esp["grid"],
        bounds_error=False, fill_value=np.nan,
    )

    # Isosurface extraction
    verts, faces, _, _ = marching_cubes(
        density["grid"],
        level=isovalue,
        spacing=(density["vx"][0], density["vy"][1], density["vz"][2]),
        step_size=2,
        allow_degenerate=False,
    )
    verts = verts + density["origin"]
    esp_vals = interp(verts)

    # Filter non-finite ESP values
    finite = np.isfinite(esp_vals)
    rot_verts = (verts[finite] - center) @ rot
    rot_atoms = (atom_xyz - center) @ rot
    esp_vals = esp_vals[finite]

    index_map = -np.ones(len(finite), dtype=int)
    index_map[np.where(finite)[0]] = np.arange(np.count_nonzero(finite))
    mask = finite[faces].all(axis=1)
    faces = index_map[faces[mask]]

    if len(faces) > max_faces:
        faces = faces[:: int(math.ceil(len(faces) / max_faces))]

    face_vals = esp_vals[faces].mean(axis=1)

    # Color mapping — use RdBu_r (perceptually uniform diverging, better than jet)
    surface_min = float(np.nanmin(esp_vals))
    surface_max = float(np.nanmax(esp_vals))
    lim = max(abs(surface_min), abs(surface_max))
    cmap = colormaps["RdBu_r"]
    norm = Normalize(vmin=-lim, vmax=lim)
    face_colors = cmap(norm(face_vals))
    face_colors[:, 3] = alpha

    # --- Figure ---
    fig = plt.figure(figsize=(8.2, 7.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            rot_verts[faces],
            facecolors=face_colors,
            edgecolors=(0.25, 0.25, 0.25, 0.02),
            linewidths=0.03,
        )
    )

    # Bonds
    bonds = build_bonds_from_coords(atom_xyz, elements)
    for i, j in bonds:
        p1, p2 = rot_atoms[i], rot_atoms[j]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=(0.92, 0.76, 0.44, 0.75), lw=1.25,
        )

    # Atoms as 3D scatter with element-specific CPK colors
    _size_map = {"H": 42}
    _default_size = 150
    for atom, xyz in zip(density["atoms"], rot_atoms):
        elem = atom["element"]
        ax.scatter(
            [xyz[0]], [xyz[1]], [xyz[2]],
            s=_size_map.get(elem, _default_size),
            color=get_atom_color(elem),
            edgecolors=get_atom_edge_color(elem),
            linewidths=0.75,
            alpha=0.90 if elem != "H" else 0.75,
            depthshade=False,
        )

    # Axes limits
    mins = rot_verts.min(axis=0)
    maxs = rot_verts.max(axis=0)
    span = (maxs - mins).max() * 0.34
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span * 0.80, mid[2] + span * 0.80)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.78))
    ax.set_axis_off()

    fig.subplots_adjust(top=0.92, right=0.96, left=0.02, bottom=0.10)
    if title:
        fig.suptitle(title, y=0.965, fontsize=15)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.08, 0.08, 0.24, 0.035])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("ESP, a.u.", labelpad=2)

    # Atom labels
    if labels != "none":
        fig.canvas.draw()
        for atom, xyz in zip(density["atoms"], rot_atoms):
            if atom["element"] == "H":
                continue
            x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], ax.get_proj())
            xdisp, ydisp = ax.transData.transform((x2, y2))
            xfig, yfig = fig.transFigure.inverted().transform((xdisp, ydisp))
            label = atom["element"] if labels == "elements" else str(atom["atom_index"])
            txt = fig.text(
                xfig, yfig, label,
                ha="center", va="center", fontsize=8.5, color="black",
            )
            txt.set_path_effects(
                [pe.withStroke(linewidth=1.9, foreground="white", alpha=0.92)]
            )

    fig.savefig(output, pad_inches=0.14)
    plt.close(fig)
