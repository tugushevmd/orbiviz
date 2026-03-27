"""2D ranked atomic charge map with bar chart (ADCH, CHELPG, NBO, etc.)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from ._geometry import build_bonds, project_2d
from ._style import DEFAULT_DPI
from ._plotting import (
    draw_bonds_2d,
    draw_atoms_2d,
    setup_mol_axes,
    add_colorbar,
    draw_bar_chart,
)


def render_ranked_charge_map(
    atoms: list[dict],
    charges: list[dict],
    output: Path,
    title: str = "Charge Map",
    subtitle: str = "Only the most charged atoms are numbered",
    top_count: int = 8,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 2D charge map with ranked bar chart and save to PNG.

    Parameters
    ----------
    atoms : list[dict]
        Atom dicts with keys: atom_index, element, x, y, z.
    charges : list[dict]
        Charge dicts with keys: atom_index, element, charge.
    output : Path
        Output PNG path.
    title : str
        Figure title.
    subtitle : str
        Smaller subtitle below title.
    top_count : int
        Number of most positive / most negative atoms to label and chart.
    dpi : int
        Output resolution.
    """
    bonds = build_bonds(atoms)
    proj = project_2d(atoms)

    values = [r["charge"] for r in charges]
    vmax = max(abs(min(values)), abs(max(values))) if values else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = colormaps["coolwarm"]

    value_by_idx = {r["atom_index"]: r["charge"] for r in charges}
    top_idx = {
        r["atom_index"]
        for r in sorted(charges, key=lambda r: abs(r["charge"]), reverse=True)[
            : top_count * 2
        ]
    }

    # --- Figure layout ---
    fig = plt.figure(figsize=(10, 11), dpi=dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[6.4, 0.3, 2.9], hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[2, 0])

    draw_bonds_2d(ax, atoms, bonds)
    draw_atoms_2d(ax, atoms, value_by_idx, norm, cmap, top_idx)
    setup_mol_axes(ax, proj)
    add_colorbar(fig, ax, norm, cmap, label="Charge, e")

    # Bar chart: top positive (reversed) + top negative, deduplicated
    pos_rows = sorted(charges, key=lambda r: r["charge"], reverse=True)[:top_count]
    neg_rows = sorted(charges, key=lambda r: r["charge"])[:top_count]
    seen = set()
    order = []
    for r in neg_rows:
        if r["atom_index"] not in seen:
            order.append(r)
            seen.add(r["atom_index"])
    for r in reversed(pos_rows):
        if r["atom_index"] not in seen:
            order.append(r)
            seen.add(r["atom_index"])
    draw_bar_chart(ax_bar, order, "charge", norm, cmap, title="Most Charged Atoms")

    fig.suptitle(title, y=0.975, fontsize=18)
    fig.text(
        0.5, 0.952, subtitle,
        ha="center", va="center", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
