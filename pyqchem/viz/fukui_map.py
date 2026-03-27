"""2D condensed Fukui index visualization with bar chart."""
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


METRIC_PRETTY = {
    "f_plus": "f⁺",
    "f_minus": "f⁻",
    "dual_descriptor": "dual descriptor",
}


def render_condensed_fukui(
    atoms: list[dict],
    fukui_rows: list[dict],
    metric: str,
    output: Path,
    title: str = "",
    label_count: int = 8,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a condensed Fukui index map and save to PNG.

    Parameters
    ----------
    atoms : list[dict]
        Atom dicts with keys: atom_index, element, x, y, z.
    fukui_rows : list[dict]
        Fukui dicts with keys: atom_index, element, f_plus, f_minus, dual_descriptor.
    metric : str
        One of 'f_plus', 'f_minus', 'dual_descriptor'.
    output : Path
        Output PNG path.
    title : str
        Figure title. Defaults to the pretty metric name.
    label_count : int
        How many heavy atoms to label in the molecule view.
    dpi : int
        Output resolution.
    """
    if metric not in METRIC_PRETTY:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {list(METRIC_PRETTY)}")

    pretty_name = METRIC_PRETTY[metric]
    if not title:
        title = pretty_name

    bonds = build_bonds(atoms)
    proj = project_2d(atoms)

    values = [r[metric] for r in fukui_rows]
    vmax = max(abs(min(values)), abs(max(values))) if values else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = colormaps["coolwarm"]

    value_by_idx = {r["atom_index"]: r[metric] for r in fukui_rows}
    top_idx = {
        r["atom_index"]
        for r in sorted(fukui_rows, key=lambda r: abs(r[metric]), reverse=True)[
            :label_count
        ]
    }

    # --- Figure layout ---
    fig = plt.figure(figsize=(6.2, 7.8), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.5, 2.2], hspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0])

    draw_bonds_2d(ax, atoms, bonds)
    draw_atoms_2d(ax, atoms, value_by_idx, norm, cmap, top_idx)
    setup_mol_axes(ax, proj)
    ax.set_title(title, fontsize=14)
    add_colorbar(fig, ax, norm, cmap, label="Index value")

    # Bar chart
    if metric == "dual_descriptor":
        bar_rows = sorted(
            fukui_rows, key=lambda r: abs(r[metric]), reverse=True
        )[:10]
        bar_title = "Largest |dual| values"
    else:
        bar_rows = sorted(fukui_rows, key=lambda r: r[metric], reverse=True)[:10]
        bar_title = f"Top {pretty_name} values"
    bar_rows = list(reversed(bar_rows))
    draw_bar_chart(axb, bar_rows, metric, norm, cmap, title=bar_title)

    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.96)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
