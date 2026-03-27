"""Visual theming: colors, sizes, helper functions for publication-quality output."""
from __future__ import annotations

from ._elements import ELEMENTS


# ---------------------------------------------------------------------------
# Default theme constants
# ---------------------------------------------------------------------------

DEFAULT_DPI = 200
BOND_COLOR = "#9a9a9a"
BOND_WIDTH = 2.0
H_FILL = "#f3dfd4"
H_EDGE = "#b8aaa1"
H_ALPHA = 0.55
HEAVY_EDGE = "#555555"
HEAVY_ALPHA = 0.92
LABEL_FONT_SIZE = 8.5
BAR_FONT_SIZE = 8.0
LUMINANCE_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Atom display properties derived from element data
# ---------------------------------------------------------------------------

def get_display_radius(element: str) -> float:
    """Return a display radius (arbitrary units) for 2D molecule maps."""
    elem = ELEMENTS.get(element)
    if elem is None:
        return 0.28
    if element == "H":
        return 0.18
    # Scale from covalent radius: heavier atoms slightly larger circles
    return min(max(elem.covalent_radius * 0.38, 0.22), 0.38)


def get_fill_alpha(element: str) -> float:
    """Return fill transparency for atom circles."""
    if element == "H":
        return H_ALPHA
    return HEAVY_ALPHA


def get_atom_color(element: str) -> str:
    """Return CPK/Jmol color for an element."""
    elem = ELEMENTS.get(element)
    return elem.cpk_color if elem else "#777777"


def get_atom_edge_color(element: str) -> str:
    """Return edge/outline color for an element."""
    elem = ELEMENTS.get(element)
    return elem.edge_color if elem else "#333333"


# ---------------------------------------------------------------------------
# Text contrast helper
# ---------------------------------------------------------------------------

def text_color_from_rgba(rgba) -> str:
    """Choose black or white text depending on background luminance."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < LUMINANCE_THRESHOLD else "#222222"
