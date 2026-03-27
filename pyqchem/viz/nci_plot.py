"""Non-Covalent Interaction (NCI) analysis and visualization.

Implements the NCI index (Johnson et al., JACS 2010):
- Computes the reduced density gradient (RDG) from an electron density cube.
- Plots RDG vs sign(λ₂)·ρ scatter (the classic NCI fingerprint).
- Renders 3D isosurfaces of RDG colored by sign(λ₂)·ρ.

Requires density cube; optionally a pre-computed RDG cube.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import uniform_filter
from skimage.measure import marching_cubes

from ._elements import ELEMENTS
from ._geometry import read_cube, build_bonds_from_coords
from ._style import DEFAULT_DPI, get_atom_color, get_atom_edge_color


def _compute_rdg_and_sign_lambda2(
    density: np.ndarray,
    spacing: tuple[float, float, float],
    rho_cutoff: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reduced density gradient and sign(λ₂)·ρ from density grid.

    RDG = (1/(2(3π²)^(1/3))) * |∇ρ| / ρ^(4/3)

    sign(λ₂) is the sign of the second eigenvalue of the electron
    density Hessian, indicating the nature of the interaction:
    - negative → attractive (H-bonds, van der Waals)
    - positive → repulsive (steric)
    """
    rho = density.copy()
    # Avoid division by zero
    rho[rho < 1e-10] = 1e-10

    # Gradient via central differences
    gx, gy, gz = np.gradient(rho, *spacing)
    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    # RDG = 1/(2(3π²)^(1/3)) * |∇ρ| / ρ^(4/3)
    prefactor = 1.0 / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0))
    rdg = prefactor * grad_norm / (rho ** (4.0 / 3.0))

    # Hessian — second derivatives for sign(λ₂)
    hxx = np.gradient(gx, spacing[0], axis=0)
    hxy = np.gradient(gx, spacing[1], axis=1)
    hxz = np.gradient(gx, spacing[2], axis=2)
    hyy = np.gradient(gy, spacing[1], axis=1)
    hyz = np.gradient(gy, spacing[2], axis=2)
    hzz = np.gradient(gz, spacing[2], axis=2)

    # Eigenvalues of 3x3 symmetric Hessian at each point
    # λ₂ is the middle eigenvalue
    sign_l2_rho = np.zeros_like(rho)
    nx, ny, nz = rho.shape
    for ix in range(nx):
        for iy in range(ny):
            H = np.array([
                [hxx[ix, iy], hxy[ix, iy], hxz[ix, iy]],
                [hxy[ix, iy], hyy[ix, iy], hyz[ix, iy]],
                [hxz[ix, iy], hyz[ix, iy], hzz[ix, iy]],
            ])
            eigvals = np.linalg.eigvalsh(H)
            # eigvals sorted ascending: λ₁ ≤ λ₂ ≤ λ₃
            sign_l2_rho[ix, iy] = np.sign(eigvals[1]) * density[ix, iy]

    # Mask high-density regions
    rdg[density > rho_cutoff] = 100.0

    return rdg, sign_l2_rho


def _compute_rdg_and_sign_lambda2_fast(
    density: np.ndarray,
    spacing: tuple[float, float, float],
    rho_cutoff: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast vectorized NCI computation (approximate Hessian eigenvalues).

    Uses the analytical formula for eigenvalues of a 3x3 symmetric matrix
    computed at each grid point. Much faster than the loop version.
    """
    rho = density.copy()
    rho[rho < 1e-10] = 1e-10

    sx, sy, sz = spacing

    # Gradient
    gx, gy, gz = np.gradient(rho, sx, sy, sz)
    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    # RDG
    prefactor = 1.0 / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0))
    rdg = prefactor * grad_norm / (rho ** (4.0 / 3.0))

    # Hessian components
    hxx = np.gradient(gx, sx, axis=0)
    hxy = np.gradient(gx, sy, axis=1)
    hxz = np.gradient(gx, sz, axis=2)
    hyy = np.gradient(gy, sy, axis=1)
    hyz = np.gradient(gy, sz, axis=2)
    hzz = np.gradient(gz, sz, axis=2)

    # Eigenvalues of symmetric 3x3 via Cardano's formula (vectorized)
    # Characteristic polynomial: λ³ - p·λ² + q·λ - r = 0
    p = hxx + hyy + hzz  # trace
    q = hxx*hyy + hxx*hzz + hyy*hzz - hxy**2 - hxz**2 - hyz**2
    r = (hxx*hyy*hzz + 2*hxy*hxz*hyz
         - hxx*hyz**2 - hyy*hxz**2 - hzz*hxy**2)  # determinant

    # Use numpy eigenvalue approach on flattened Hessians for robustness
    # Reshape to (N, 3, 3) and compute eigenvalues in batches
    shape = rho.shape
    N = rho.size
    H_flat = np.zeros((N, 3, 3), dtype=np.float32)
    H_flat[:, 0, 0] = hxx.ravel()
    H_flat[:, 0, 1] = hxy.ravel()
    H_flat[:, 0, 2] = hxz.ravel()
    H_flat[:, 1, 0] = hxy.ravel()
    H_flat[:, 1, 1] = hyy.ravel()
    H_flat[:, 1, 2] = hyz.ravel()
    H_flat[:, 2, 0] = hxz.ravel()
    H_flat[:, 2, 1] = hyz.ravel()
    H_flat[:, 2, 2] = hzz.ravel()

    eigvals = np.linalg.eigvalsh(H_flat)  # (N, 3), sorted ascending
    lambda2 = eigvals[:, 1].reshape(shape)

    sign_l2_rho = np.sign(lambda2) * density

    # Mask high-density regions
    rdg[density > rho_cutoff] = 100.0

    return rdg, sign_l2_rho


def render_nci_scatter(
    density_cube_path: Path,
    output: Path,
    rho_cutoff: float = 0.05,
    rdg_cutoff: float = 2.0,
    title: str = "NCI Scatter Plot",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render the classic NCI scatter plot: RDG vs sign(λ₂)·ρ.

    Parameters
    ----------
    density_cube_path : Path
        Electron density cube file.
    output : Path
        Output PNG path.
    rho_cutoff : float
        Maximum density to include in analysis (a.u.).
    rdg_cutoff : float
        Maximum RDG value to plot.
    title : str
        Figure title.
    dpi : int
        Output resolution.
    """
    cube = read_cube(density_cube_path)
    spacing = (cube["vx"][0], cube["vy"][1], cube["vz"][2])

    rdg, sign_l2_rho = _compute_rdg_and_sign_lambda2_fast(
        cube["grid"], spacing, rho_cutoff=rho_cutoff,
    )

    # Filter points
    mask = (rdg < rdg_cutoff) & (cube["grid"] > 1e-6) & (cube["grid"] < rho_cutoff)
    x = sign_l2_rho[mask].ravel()
    y = rdg[mask].ravel()

    # Subsample if too many points
    max_pts = 200000
    if len(x) > max_pts:
        idx = np.random.default_rng(42).choice(len(x), max_pts, replace=False)
        x, y = x[idx], y[idx]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    # Color by sign(λ₂)·ρ: blue=attractive, green=vdW, red=repulsive
    cmap = colormaps["RdYlGn_r"]
    norm = Normalize(vmin=-0.04, vmax=0.04)
    ax.scatter(x, y, c=x, cmap=cmap, norm=norm, s=0.3, alpha=0.4, rasterized=True)

    ax.set_xlabel("sign(λ₂) · ρ (a.u.)", fontsize=12)
    ax.set_ylabel("Reduced Density Gradient", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.06, 0.06)
    ax.set_ylim(0, rdg_cutoff)
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    # Annotation regions
    ax.annotate("Attractive\n(H-bonds)", xy=(-0.03, 0.3), fontsize=9,
                color="#2166ac", ha="center", weight="bold")
    ax.annotate("van der Waals", xy=(0.0, 0.3), fontsize=9,
                color="#1a9850", ha="center", weight="bold")
    ax.annotate("Repulsive\n(steric)", xy=(0.03, 0.3), fontsize=9,
                color="#b2182b", ha="center", weight="bold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("sign(λ₂) · ρ", rotation=90)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_nci_3d(
    density_cube_path: Path,
    output: Path,
    rho_cutoff: float = 0.05,
    rdg_isovalue: float = 0.5,
    elev: float = 20.0,
    azim: float = 55.0,
    max_faces: int = 80000,
    labels: str = "elements",
    title: str = "NCI Isosurface",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 3D NCI isosurface colored by sign(λ₂)·ρ.

    Green = van der Waals, Blue = H-bonds, Red = steric repulsion.
    """
    cube = read_cube(density_cube_path)
    spacing = (cube["vx"][0], cube["vy"][1], cube["vz"][2])

    rdg, sign_l2_rho = _compute_rdg_and_sign_lambda2_fast(
        cube["grid"], spacing, rho_cutoff=rho_cutoff,
    )

    atom_xyz = np.array(
        [[a["x"], a["y"], a["z"]] for a in cube["atoms"]], dtype=float
    )
    elements = [a["element"] for a in cube["atoms"]]
    center = atom_xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(atom_xyz - center, full_matrices=False)
    rot = vh.T
    rot_atoms = (atom_xyz - center) @ rot

    # Isosurface of RDG
    try:
        verts, faces, _, _ = marching_cubes(
            rdg, level=rdg_isovalue, spacing=spacing,
            step_size=2, allow_degenerate=False,
        )
    except ValueError:
        raise ValueError(f"No RDG isosurface at level {rdg_isovalue}")

    verts = verts + cube["origin"]
    rot_verts = (verts - center) @ rot

    if len(faces) > max_faces:
        faces = faces[:: int(math.ceil(len(faces) / max_faces))]

    # Color by sign(λ₂)·ρ at each face
    from scipy.interpolate import RegularGridInterpolator
    x_g = cube["origin"][0] + np.arange(cube["shape"][0]) * spacing[0]
    y_g = cube["origin"][1] + np.arange(cube["shape"][1]) * spacing[1]
    z_g = cube["origin"][2] + np.arange(cube["shape"][2]) * spacing[2]
    interp_sl2 = RegularGridInterpolator(
        (x_g, y_g, z_g), sign_l2_rho,
        bounds_error=False, fill_value=0.0,
    )
    vert_sl2 = interp_sl2(verts)
    face_sl2 = vert_sl2[faces].mean(axis=1)

    cmap = colormaps["RdYlGn_r"]
    norm = Normalize(vmin=-0.04, vmax=0.04)
    face_colors = cmap(norm(face_sl2))
    face_colors[:, 3] = 0.65

    # --- Figure ---
    fig = plt.figure(figsize=(8.5, 7.5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.add_collection3d(
        Poly3DCollection(
            rot_verts[faces],
            facecolors=face_colors,
            edgecolors=(0.3, 0.3, 0.3, 0.02),
            linewidths=0.02,
        )
    )

    # Bonds + atoms
    bonds = build_bonds_from_coords(atom_xyz, elements)
    for i, j in bonds:
        p1, p2 = rot_atoms[i], rot_atoms[j]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=(0.4, 0.4, 0.4, 0.85), lw=2.5, solid_capstyle="round",
        )

    for atom, xyz in zip(cube["atoms"], rot_atoms):
        elem = atom["element"]
        ax.scatter(
            [xyz[0]], [xyz[1]], [xyz[2]],
            s=50 if elem == "H" else 180,
            color=get_atom_color(elem),
            edgecolors=get_atom_edge_color(elem),
            linewidths=0.9,
            alpha=0.95 if elem != "H" else 0.80,
            depthshade=False,
        )

    mins = rot_atoms.min(axis=0) - 3.5
    maxs = rot_atoms.max(axis=0) + 3.5
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

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.08, 0.06, 0.24, 0.035])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("sign(λ₂) · ρ (a.u.)", labelpad=2, fontsize=9)

    fig.subplots_adjust(top=0.93, right=0.96, left=0.02, bottom=0.12)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
