"""HOMO/LUMO energy analysis and MO energy level diagram.

Provides:
- compute_homo_lumo_descriptors(): HOMO/LUMO energies + chemical reactivity indices
- render_homo_lumo_diagram(): publication-quality MO energy level figure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._style import DEFAULT_DPI

HARTREE_TO_EV = 27.211386245988

HOMO_COLOR = "#d64e2a"
LUMO_COLOR = "#2a6cb8"
OCC_COLOR  = "#555555"
VIRT_COLOR = "#aaaaaa"
GAP_COLOR  = "#6a3d8f"


def compute_homo_lumo_descriptors(
    energies_hartree,
    n_occ: int,
) -> dict:
    """Compute HOMO/LUMO energies and chemical reactivity descriptors.

    Parameters
    ----------
    energies_hartree : array-like
        Orbital energies in Hartree. Index n_occ-1 is HOMO, n_occ is LUMO.
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    dict with keys: homo_ev, lumo_ev, gap_ev, homo_hartree, lumo_hartree,
    gap_hartree, ip_ev, ea_ev, mu_ev, eta_ev, softness_per_ev, omega_ev.
    """
    e = np.asarray(energies_hartree, dtype=float)
    n = len(e)
    if n_occ <= 0 or n_occ >= n:
        raise ValueError(
            f"n_occ={n_occ} is out of range for {n} orbital energies "
            f"(need 0 < n_occ < {n})."
        )
    homo_h = float(e[n_occ - 1])
    lumo_h = float(e[n_occ])
    homo_ev = homo_h * HARTREE_TO_EV
    lumo_ev = lumo_h * HARTREE_TO_EV
    gap_ev  = lumo_ev - homo_ev
    ip_ev   = -homo_ev
    ea_ev   = -lumo_ev
    mu_ev   = (homo_ev + lumo_ev) / 2.0
    eta_ev  = gap_ev / 2.0
    softness = 1.0 / (2.0 * eta_ev) if eta_ev != 0.0 else float("inf")
    omega_ev = mu_ev**2 / (2.0 * eta_ev) if eta_ev != 0.0 else float("inf")
    return {
        "homo_ev":          homo_ev,
        "lumo_ev":          lumo_ev,
        "gap_ev":           gap_ev,
        "homo_hartree":     homo_h,
        "lumo_hartree":     lumo_h,
        "gap_hartree":      lumo_h - homo_h,
        "ip_ev":            ip_ev,
        "ea_ev":            ea_ev,
        "mu_ev":            mu_ev,
        "eta_ev":           eta_ev,
        "softness_per_ev":  softness,
        "omega_ev":         omega_ev,
    }


def render_homo_lumo_diagram(
    energies_hartree,
    n_occ: int,
    output,
    n_window: int = 5,
    title: str = "",
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Render a publication-quality MO energy level diagram.

    Shows a window of orbitals around the HOMO/LUMO pair with a panel of
    computed chemical reactivity descriptors (Koopmans' IP/EA, hardness,
    softness, chemical potential, electrophilicity index).

    Parameters
    ----------
    energies_hartree : array-like
        All orbital energies in Hartree.
    n_occ : int
        Number of occupied orbitals (HOMO = index n_occ-1, LUMO = n_occ).
    output : str or Path
        Output file (PNG/SVG/PDF — format inferred from suffix).
    n_window : int
        Additional orbitals to show below HOMO and above LUMO (default 5).
    title : str
        Optional figure title (molecule name, method/basis, etc.).
    dpi : int
        Output resolution (default 200 dpi).

    Returns
    -------
    dict
        Chemical descriptors in eV (see compute_homo_lumo_descriptors).
    """
    e    = np.asarray(energies_hartree, dtype=float)
    desc = compute_homo_lumo_descriptors(e, n_occ)
    e_ev = e * HARTREE_TO_EV

    homo_idx = n_occ - 1
    lumo_idx = n_occ

    # Orbital window
    i_start = max(0, homo_idx - n_window + 1)
    i_end   = min(len(e) - 1, lumo_idx + n_window - 1)
    window  = list(range(i_start, i_end + 1))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 7.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[3, 2.2],
        left=0.07, right=0.97, top=0.91, bottom=0.06, wspace=0.06,
    )
    ax   = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])

    # ── Left: MO energy levels ───────────────────────────────────────────────
    ax.set_facecolor("#f9f9f9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_ylabel("Energy (eV)", fontsize=12, labelpad=8)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.grid(axis="y", color="white", lw=1.0, zorder=0)

    x_l      = 0.22   # left end of orbital lines
    x_r      = 0.68   # right end of orbital lines
    x_arrow  = x_l - 0.09  # position of the gap arrow
    x_rlabel = x_r + 0.025
    x_llabel = x_l - 0.025

    for idx in window:
        ev       = e_ev[idx]
        occupied = idx <= homo_idx

        if idx == homo_idx:
            color, lw, ls, alpha = HOMO_COLOR, 3.2, "-",  1.0
        elif idx == lumo_idx:
            color, lw, ls, alpha = LUMO_COLOR, 3.2, "--", 1.0
        elif occupied:
            dist  = homo_idx - idx
            grey  = min(0.45 + dist * 0.08, 0.72)
            color, lw, ls, alpha = (grey, grey, grey), 1.8, "-",  0.9
        else:
            dist  = idx - lumo_idx
            grey  = min(0.60 + dist * 0.07, 0.82)
            color, lw, ls, alpha = (grey, grey, grey), 1.6, "--", 0.8

        ax.plot(
            [x_l, x_r], [ev, ev],
            color=color, lw=lw, ls=ls, alpha=alpha,
            solid_capstyle="round", zorder=5,
        )

        # Energy label (left)
        label_color = color if idx in (homo_idx, lumo_idx) else "#999999"
        ax.text(x_llabel, ev, f"{ev:.3f}", va="center", ha="right",
                fontsize=8, color=label_color)

        # MO index / name label (right)
        if idx == homo_idx:
            mo_label, weight = f"HOMO  ({idx + 1})", "bold"
        elif idx == lumo_idx:
            mo_label, weight = f"LUMO  ({idx + 1})", "bold"
        else:
            mo_label, weight = str(idx + 1), "normal"
        ax.text(x_rlabel, ev, mo_label, va="center", ha="left",
                fontsize=8.5, color=label_color, weight=weight)

    # Shaded HOMO-LUMO gap region
    ax.fill_betweenx(
        [desc["homo_ev"], desc["lumo_ev"]],
        x_arrow - 0.03, x_l - 0.01,
        alpha=0.10, color=GAP_COLOR, zorder=1,
    )

    # Gap double arrow
    ax.annotate(
        "", xy=(x_arrow, desc["lumo_ev"]), xytext=(x_arrow, desc["homo_ev"]),
        arrowprops=dict(arrowstyle="<->", color=GAP_COLOR, lw=1.8),
        zorder=6,
    )
    mid_gap = (desc["homo_ev"] + desc["lumo_ev"]) / 2.0
    ax.text(
        x_arrow - 0.02, mid_gap,
        f"{desc['gap_ev']:.3f} eV",
        ha="right", va="center", fontsize=10.5,
        color=GAP_COLOR, weight="bold", rotation=90,
    )

    # Axis limits
    vis_ev = e_ev[window]
    pad = max(0.6, desc["gap_ev"] * 0.35)
    ax.set_ylim(vis_ev.min() - pad, vis_ev.max() + pad)
    ax.set_xlim(x_arrow - 0.18, x_rlabel + 0.30)
    ax.set_title("Molecular Orbital Energy Levels", fontsize=12, pad=10)

    # ── Right: Descriptors panel ─────────────────────────────────────────────
    ax_d.set_facecolor("white")
    ax_d.set_axis_off()
    t = ax_d.transAxes

    from matplotlib.patches import FancyBboxPatch
    ax_d.add_patch(FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.94,
        boxstyle="round,pad=0.015",
        linewidth=1.0, edgecolor="#cccccc", facecolor="#fafafa",
        transform=t, zorder=0,
    ))

    ax_d.text(0.50, 0.945, "Electronic Properties",
              ha="center", va="top", fontsize=12, weight="bold",
              color="#222222", transform=t)

    y  = 0.88
    dy = 0.065

    def _row(label, val, color="#333333", bold=False, small=False):
        nonlocal y
        fs = 9.5 if not small else 8.5
        ax_d.text(0.08, y, label, ha="left", va="top", fontsize=fs,
                  color=color, weight="bold" if bold else "normal", transform=t)
        if val:
            ax_d.text(0.92, y, val, ha="right", va="top", fontsize=fs,
                      color=color, weight="bold" if bold else "normal", transform=t)
        y -= dy

    def _sep(gap: float = 0.45):
        nonlocal y
        y_line = y + dy * 0.35
        ax_d.plot([0.08, 0.92], [y_line, y_line],
                  color="#dddddd", lw=0.8, transform=t, solid_capstyle="round")
        y -= dy * gap

    _row("HOMO energy",  f"{desc['homo_ev']:+.4f} eV",  color=HOMO_COLOR, bold=True)
    _row("LUMO energy",  f"{desc['lumo_ev']:+.4f} eV",  color=LUMO_COLOR, bold=True)
    _row("HOMO–LUMO gap", f"{desc['gap_ev']:.4f} eV",   color=GAP_COLOR,  bold=True)
    _row("HOMO",         f"{desc['homo_hartree']:+.6f} Eh", color="#888888", small=True)
    _row("LUMO",         f"{desc['lumo_hartree']:+.6f} Eh", color="#888888", small=True)
    _sep()
    _row("Koopmans' theorem", "", color="#444444", bold=True)
    _row("  IP  (≈ −ε_HOMO)",    f"{desc['ip_ev']:.4f} eV")
    _row("  EA  (≈ −ε_LUMO)",    f"{desc['ea_ev']:.4f} eV")
    _sep()
    _row("Chemical reactivity", "", color="#444444", bold=True)
    _row("  μ  chem. potential",  f"{desc['mu_ev']:+.4f} eV")
    _row("  η  hardness",         f"{desc['eta_ev']:.4f} eV")
    _row("  S  softness",         f"{desc['softness_per_ev']:.4f} eV⁻¹")
    _row("  ω  electrophilicity", f"{desc['omega_ev']:.4f} eV")

    # ── Title ────────────────────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=14, y=0.97)

    fig.savefig(str(output), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    return desc
