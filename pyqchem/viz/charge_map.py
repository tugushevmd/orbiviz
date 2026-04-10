"""2D ranked atomic charge maps and related variants.

This module exposes three renderers, all of which write to a Path whose
suffix determines the output format (matplotlib will dispatch to png/svg/pdf
automatically):

- ``render_ranked_charge_map``       — generic atomic charges (Mulliken, ADCH, …)
- ``render_spin_density_map``        — open-shell α−β spin populations
- ``render_charge_difference_map``   — Δq between two output files
"""
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


def _ranked_value_map(
    atoms: list[dict],
    rows: list[dict],
    value_key: str,
    output: Path,
    *,
    cmap_name: str,
    title: str,
    subtitle: str,
    bar_title: str,
    colorbar_label: str,
    top_count: int,
    dpi: int,
    symmetric: bool = True,
) -> None:
    """Shared rendering pipeline for any ranked-value 2D map."""
    bonds = build_bonds(atoms)
    proj = project_2d(atoms)

    values = [r[value_key] for r in rows]
    if not values:
        raise ValueError("No data rows provided.")
    if symmetric:
        vmax = max(abs(min(values)), abs(max(values))) or 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        lo, hi = min(values), max(values)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
    cmap = colormaps[cmap_name]

    value_by_idx = {r["atom_index"]: r[value_key] for r in rows}
    top_idx = {
        r["atom_index"]
        for r in sorted(rows, key=lambda r: abs(r[value_key]), reverse=True)[
            : top_count * 2
        ]
    }

    fig = plt.figure(figsize=(10, 11), dpi=dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[6.4, 0.3, 2.9], hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[2, 0])

    draw_bonds_2d(ax, atoms, bonds)
    draw_atoms_2d(ax, atoms, value_by_idx, norm, cmap, top_idx)
    setup_mol_axes(ax, proj)
    add_colorbar(fig, ax, norm, cmap, label=colorbar_label)

    pos_rows = sorted(rows, key=lambda r: r[value_key], reverse=True)[:top_count]
    neg_rows = sorted(rows, key=lambda r: r[value_key])[:top_count]
    seen: set[int] = set()
    order: list[dict] = []
    for r in neg_rows:
        if r["atom_index"] not in seen:
            order.append(r)
            seen.add(r["atom_index"])
    for r in reversed(pos_rows):
        if r["atom_index"] not in seen:
            order.append(r)
            seen.add(r["atom_index"])
    draw_bar_chart(ax_bar, order, value_key, norm, cmap, title=bar_title)

    fig.suptitle(title, y=0.975, fontsize=18)
    if subtitle:
        fig.text(
            0.5, 0.952, subtitle,
            ha="center", va="center", fontsize=9, color="#444444",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_ranked_charge_map(
    atoms: list[dict],
    charges: list[dict],
    output: Path,
    title: str = "Charge Map",
    subtitle: str = "Only the most charged atoms are numbered",
    top_count: int = 8,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 2D ranked atomic charge map (PNG/SVG/PDF, by suffix).

    Parameters
    ----------
    atoms
        Atom dicts with keys: atom_index, element, x, y, z.
    charges
        Charge dicts with keys: atom_index, element, charge.
    output
        Output image path. The format is chosen from the suffix
        (``.png``, ``.svg``, ``.pdf``, …).
    title, subtitle
        Figure headings.
    top_count
        Number of most-positive and most-negative atoms to label and chart.
    dpi
        Output resolution.
    """
    _ranked_value_map(
        atoms, charges, "charge", output,
        cmap_name="coolwarm",
        title=title, subtitle=subtitle,
        bar_title="Most Charged Atoms",
        colorbar_label="Charge, e",
        top_count=top_count, dpi=dpi,
    )


def render_spin_density_map(
    atoms: list[dict],
    spin_rows: list[dict],
    output: Path,
    title: str = "Spin density",
    subtitle: str = "α − β spin populations",
    top_count: int = 8,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render a 2D map of α−β spin populations for an open-shell calculation.

    ``spin_rows`` must contain dicts with keys ``atom_index``, ``element`` and
    ``spin`` (the α−β population). The same colormap as for charges is used,
    so positive (excess α) is red, negative (excess β) is blue.
    """
    rows = [
        {"atom_index": r["atom_index"], "element": r["element"], "charge": r["spin"]}
        for r in spin_rows
    ]
    _ranked_value_map(
        atoms, rows, "charge", output,
        cmap_name="coolwarm",
        title=title, subtitle=subtitle,
        bar_title="Largest spin populations",
        colorbar_label="α − β population",
        top_count=top_count, dpi=dpi,
    )


def render_charge_difference_map(
    atoms: list[dict],
    charges_a: list[dict],
    charges_b: list[dict],
    output: Path,
    title: str = "Δq",
    subtitle: str = "B − A",
    top_count: int = 8,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render Δq = q_B − q_A as a 2D ranked map.

    The two charge tables must have matching atom indices.
    """
    by_a = {r["atom_index"]: r for r in charges_a}
    by_b = {r["atom_index"]: r for r in charges_b}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        raise ValueError("Charge tables A and B share no atom indices.")
    rows = [
        {
            "atom_index": idx,
            "element": by_a[idx]["element"],
            "charge": by_b[idx]["charge"] - by_a[idx]["charge"],
        }
        for idx in common
    ]
    _ranked_value_map(
        atoms, rows, "charge", output,
        cmap_name="coolwarm",
        title=title, subtitle=subtitle,
        bar_title="Largest Δq",
        colorbar_label="Δq, e",
        top_count=top_count, dpi=dpi,
    )
