"""3D molecular orbital isosurface visualization.

Renders positive and negative MO lobes in contrasting colors
on top of a ball-and-stick molecular model. Inspired by IBOview.
"""
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
from skimage.measure import marching_cubes

from ._elements import ELEMENTS
from ._geometry import read_cube, build_bonds_from_coords
from ._style import DEFAULT_DPI, get_atom_color, get_atom_edge_color


# IBOview-inspired orbital lobe colours
POSITIVE_LOBE_COLOR = (0.15, 0.45, 0.82, 0.55)   # blue, translucent
NEGATIVE_LOBE_COLOR = (0.90, 0.25, 0.20, 0.55)   # red, translucent
POSITIVE_EDGE = (0.08, 0.28, 0.58, 0.10)
NEGATIVE_EDGE = (0.60, 0.15, 0.10, 0.10)


def render_mo_surface(
    cube_path: Path,
    output: Path,
    isovalue: float = 0.02,
    elev: float = 20.0,
    azim: float = 55.0,
    max_faces: int = 60000,
    labels: str = "elements",
    title: str = "",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a molecular orbital isosurface with positive/negative lobes.

    Parameters
    ----------
    cube_path : Path
        Gaussian cube file containing MO data on a grid.
    output : Path
        Output PNG path.
    isovalue : float
        Isosurface value for the orbital lobes. Both +iso and -iso are drawn.
    elev, azim : float
        Camera elevation and azimuth.
    max_faces : int
        Max triangular faces per lobe.
    labels : str
        'none', 'elements', or 'indices'.
    title : str
        Optional figure title.
    dpi : int
        Output resolution.
    """
    cube = read_cube(cube_path)
    grid = cube["grid"]
    origin = cube["origin"]
    spacing = (cube["vx"][0], cube["vy"][1], cube["vz"][2])

    atom_xyz = np.array(
        [[a["x"], a["y"], a["z"]] for a in cube["atoms"]], dtype=float
    )
    elements = [a["element"] for a in cube["atoms"]]

    # SVD rotation for best viewing angle
    center = atom_xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(atom_xyz - center, full_matrices=False)
    rot = vh.T

    rot_atoms = (atom_xyz - center) @ rot

    # --- Figure ---
    fig = plt.figure(figsize=(8.5, 7.5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Draw both lobes
    for sign, lobe_color, edge_color in [
        (+1, POSITIVE_LOBE_COLOR, POSITIVE_EDGE),
        (-1, NEGATIVE_LOBE_COLOR, NEGATIVE_EDGE),
    ]:
        level = sign * isovalue
        try:
            verts, faces, _, _ = marching_cubes(
                grid, level=level, spacing=spacing,
                step_size=2, allow_degenerate=False,
            )
        except ValueError:
            continue  # no surface at this level

        verts = verts + origin
        rot_verts = (verts - center) @ rot

        if len(faces) > max_faces:
            faces = faces[:: int(math.ceil(len(faces) / max_faces))]

        face_colors = np.tile(lobe_color, (len(faces), 1))
        ax.add_collection3d(
            Poly3DCollection(
                rot_verts[faces],
                facecolors=face_colors,
                edgecolors=edge_color,
                linewidths=0.02,
            )
        )

    # Bonds
    bonds = build_bonds_from_coords(atom_xyz, elements)
    for i, j in bonds:
        p1, p2 = rot_atoms[i], rot_atoms[j]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=(0.4, 0.4, 0.4, 0.85), lw=2.5, solid_capstyle="round",
        )

    # Atoms
    _size_map = {"H": 50}
    _default_size = 180
    for atom, xyz in zip(cube["atoms"], rot_atoms):
        elem = atom["element"]
        ax.scatter(
            [xyz[0]], [xyz[1]], [xyz[2]],
            s=_size_map.get(elem, _default_size),
            color=get_atom_color(elem),
            edgecolors=get_atom_edge_color(elem),
            linewidths=0.9,
            alpha=0.95 if elem != "H" else 0.80,
            depthshade=False,
        )

    # Limits
    all_rot = rot_atoms
    pad = 3.5  # bohr padding
    mins = all_rot.min(axis=0) - pad
    maxs = all_rot.max(axis=0) + pad
    span = (maxs - mins).max() * 0.50
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span * 0.85, mid[2] + span * 0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.82))
    ax.set_axis_off()

    if title:
        fig.suptitle(title, y=0.96, fontsize=15)

    # Legend for lobe colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POSITIVE_LOBE_COLOR[:3], alpha=0.7, edgecolor="gray", label="+ lobe"),
        Patch(facecolor=NEGATIVE_LOBE_COLOR[:3], alpha=0.7, edgecolor="gray", label="− lobe"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=9, framealpha=0.85, edgecolor="#cccccc",
    )

    # Atom labels
    if labels != "none":
        fig.canvas.draw()
        for atom, xyz in zip(cube["atoms"], rot_atoms):
            if atom["element"] == "H":
                continue
            x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], ax.get_proj())
            xdisp, ydisp = ax.transData.transform((x2, y2))
            xfig, yfig = fig.transFigure.inverted().transform((xdisp, ydisp))
            label = atom["element"] if labels == "elements" else str(atom["atom_index"])
            txt = fig.text(
                xfig, yfig, label,
                ha="center", va="center", fontsize=9, color="black", weight="bold",
            )
            txt.set_path_effects(
                [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.92)]
            )

    fig.subplots_adjust(top=0.93, right=0.96, left=0.02, bottom=0.04)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
