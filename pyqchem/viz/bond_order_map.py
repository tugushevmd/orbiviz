"""2D bond order visualization (Wiberg, Mayer, etc.).

Renders a molecular diagram with bonds colored and sized by their bond order,
plus a ranked list of the strongest/weakest bonds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch

from ._geometry import build_bonds, project_2d
from ._style import (
    DEFAULT_DPI,
    get_display_radius,
    get_atom_color,
    get_atom_edge_color,
    HEAVY_ALPHA,
    H_FILL,
    H_EDGE,
    H_ALPHA,
    LABEL_FONT_SIZE,
)


def render_bond_order_map(
    atoms: list[dict],
    bond_orders: list[dict],
    output: Path,
    title: str = "Bond Order Map",
    min_bo: float = 0.3,
    top_count: int = 12,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 2D bond order map with bonds colored by order.

    Parameters
    ----------
    atoms : list[dict]
        Atom dicts with keys: atom_index, element, x, y, z.
    bond_orders : list[dict]
        Bond order dicts with keys: atom_i, atom_j, element_i, element_j, bond_order.
    output : Path
        Output PNG path.
    title : str
        Figure title.
    min_bo : float
        Minimum bond order to display.
    top_count : int
        Number of bonds to show in the bar chart.
    dpi : int
        Output resolution.
    """
    proj = project_2d(atoms)
    idx_to_atom = {a["atom_index"]: a for a in atoms}

    # Filter significant bonds
    sig_bonds = [bo for bo in bond_orders if bo["bond_order"] >= min_bo]

    # Bond order range for colormap
    bo_values = [bo["bond_order"] for bo in sig_bonds] if sig_bonds else [1.0]
    bo_min_val = min(bo_values)
    bo_max_val = max(bo_values)
    norm = Normalize(vmin=0.5, vmax=max(bo_max_val, 3.0))
    cmap = colormaps["YlOrRd"]

    # --- Figure ---
    fig = plt.figure(figsize=(10, 11), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[5.5, 2.5], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    # Draw bonds with width & color proportional to bond order
    for bo in sig_bonds:
        ai = idx_to_atom.get(bo["atom_i"])
        aj = idx_to_atom.get(bo["atom_j"])
        if ai is None or aj is None:
            continue
        order = bo["bond_order"]
        rgba = cmap(norm(order))
        lw = 1.0 + order * 1.8  # thicker for higher BO
        ax.plot(
            [ai["px"], aj["px"]], [ai["py"], aj["py"]],
            color=rgba, lw=lw, solid_capstyle="round", zorder=1,
        )
        # Label bond order at midpoint for significant bonds
        if order >= 1.5:
            mx = (ai["px"] + aj["px"]) / 2
            my = (ai["py"] + aj["py"]) / 2
            ax.text(
                mx, my, f"{order:.2f}",
                ha="center", va="center", fontsize=7,
                color="#333333", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )

    # Draw atoms
    for atom in atoms:
        px, py = atom["px"], atom["py"]
        elem = atom["element"]
        radius = get_display_radius(elem)
        if elem == "H":
            ax.add_patch(
                Circle((px, py), radius * 0.75,
                       facecolor=H_FILL, edgecolor=H_EDGE, lw=1.0, zorder=3)
            )
        else:
            ax.add_patch(
                Circle((px, py), radius,
                       facecolor=get_atom_color(elem),
                       edgecolor=get_atom_edge_color(elem),
                       lw=1.0, alpha=HEAVY_ALPHA, zorder=4)
            )
            ax.text(
                px, py, f"{atom['atom_index']}{elem}",
                ha="center", va="center", fontsize=7.5, weight="bold",
                color="white", zorder=5,
            )

    # Axes setup
    xs = proj[:, 0]
    ys = proj[:, 1]
    xpad = (xs.max() - xs.min()) * 0.12 + 0.5
    ypad = (ys.max() - ys.min()) * 0.14 + 0.6
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar for bond order
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Bond Order", rotation=90)

    # --- Bar chart: top bonds ---
    top_bonds = sorted(sig_bonds, key=lambda b: b["bond_order"], reverse=True)[:top_count]
    top_bonds = list(reversed(top_bonds))

    if top_bonds:
        labels = [
            f'{b["atom_i"]}{b["element_i"]}–{b["atom_j"]}{b["element_j"]}'
            for b in top_bonds
        ]
        vals = [b["bond_order"] for b in top_bonds]
        y = np.arange(len(top_bonds))
        colors = [cmap(norm(v)) for v in vals]

        ax_bar.barh(y, vals, color=colors, edgecolor="#777777", linewidth=0.8)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels, fontsize=9)
        ax_bar.set_title("Strongest Bonds", fontsize=12, pad=6)
        ax_bar.set_xlabel("Bond Order", fontsize=10)
        ax_bar.grid(axis="x", alpha=0.25, linestyle=":")
        for yi, v in zip(y, vals):
            ax_bar.text(v + 0.02, yi, f"{v:.3f}", va="center", ha="left", fontsize=8)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)

    fig.suptitle(title, y=0.975, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
