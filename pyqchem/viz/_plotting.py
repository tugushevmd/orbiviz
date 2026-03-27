"""Shared matplotlib drawing helpers for 2D molecular visualizations."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle

from ._style import (
    BOND_COLOR,
    BOND_WIDTH,
    H_FILL,
    H_EDGE,
    HEAVY_EDGE,
    LABEL_FONT_SIZE,
    BAR_FONT_SIZE,
    get_display_radius,
    get_fill_alpha,
    text_color_from_rgba,
)


def draw_bonds_2d(ax, atoms: list[dict], bonds: list[tuple[int, int]]) -> None:
    """Draw chemical bonds as gray lines on a 2D axes."""
    idx_to_atom = {a["atom_index"]: a for a in atoms}
    for i, j in bonds:
        ai = idx_to_atom[i]
        aj = idx_to_atom[j]
        ax.plot(
            [ai["px"], aj["px"]],
            [ai["py"], aj["py"]],
            color=BOND_COLOR,
            lw=BOND_WIDTH,
            zorder=1,
        )


def draw_atoms_2d(
    ax,
    atoms: list[dict],
    value_by_idx: dict[int, float],
    norm: Normalize,
    cmap,
    top_idx: set[int] | None = None,
) -> None:
    """Draw atom circles colored by a scalar value. Label atoms in top_idx."""
    for atom in atoms:
        px, py = atom["px"], atom["py"]
        elem = atom["element"]
        radius = get_display_radius(elem)
        alpha = get_fill_alpha(elem)

        if elem == "H":
            ax.add_patch(
                Circle(
                    (px, py),
                    radius * 0.75,
                    facecolor=H_FILL,
                    edgecolor=H_EDGE,
                    lw=1.0,
                    zorder=3,
                )
            )
            continue

        val = value_by_idx.get(atom["atom_index"], 0.0)
        rgba = cmap(norm(val))
        ax.add_patch(
            Circle(
                (px, py),
                radius,
                facecolor=rgba,
                edgecolor=HEAVY_EDGE,
                lw=1.0,
                alpha=alpha,
                zorder=4,
            )
        )
        if top_idx and atom["atom_index"] in top_idx:
            ax.text(
                px,
                py,
                str(atom["atom_index"]),
                ha="center",
                va="center",
                fontsize=LABEL_FONT_SIZE,
                weight="bold",
                color=text_color_from_rgba(rgba),
                zorder=5,
            )


def setup_mol_axes(ax, proj: np.ndarray) -> None:
    """Configure axes limits, aspect, and turn off frame for a molecule view."""
    xs = proj[:, 0]
    ys = proj[:, 1]
    xpad = (xs.max() - xs.min()) * 0.11 + 0.5
    ypad = (ys.max() - ys.min()) * 0.14 + 0.65
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    ax.set_aspect("equal")
    ax.axis("off")


def add_colorbar(fig, ax, norm, cmap, label: str = "Charge, e"):
    """Add a vertical colorbar next to an axes."""
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label, rotation=90)
    return cbar


def draw_bar_chart(
    ax,
    rows: list[dict],
    value_key: str,
    norm: Normalize,
    cmap,
    title: str = "Most Charged Atoms",
) -> None:
    """Draw a horizontal bar chart of top atoms by value."""
    if not rows:
        return
    labels = [f'{r["atom_index"]}{r["element"]}' for r in rows]
    vals = [r[value_key] for r in rows]
    y = np.arange(len(rows))
    colors = [cmap(norm(v)) for v in vals]

    ax.barh(y, vals, color=colors, edgecolor="#777777", linewidth=0.8)
    ax.axvline(0, color="#888888", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=12, pad=6)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    lim = max(abs(min(vals)), abs(max(vals))) * 1.22 if vals else 1.0
    ax.set_xlim(-lim, lim)
    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        offset = lim * 0.02 if v >= 0 else -lim * 0.02
        ax.text(
            v + offset, yi, f"{v:+.3f}",
            va="center", ha=ha, fontsize=BAR_FONT_SIZE,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
