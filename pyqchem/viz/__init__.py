"""pyqchem.viz — Publication-quality visualizations for computational chemistry.

Supported visualizations:
- Atomic charge maps (ADCH, CHELPG, NBO, Mulliken, etc.)
- Condensed Fukui index maps (f+, f-, dual descriptor)
- ESP-colored electron density isosurfaces
- Molecular orbital isosurfaces (positive/negative lobes)
- NCI analysis (scatter plots and 3D isosurfaces)
- Bond order maps (Wiberg, Mayer)

Supported input formats:
- XYZ geometry files
- Gaussian cube files
- Gaussian .log / .out files
- Gaussian .fchk files
- ORCA output files
- Molden files
- CSV data files
"""

from .charge_map import render_ranked_charge_map
from .fukui_map import render_condensed_fukui
from .esp_surface import render_esp_surface
from .mo_surface import render_mo_surface
from .nci_plot import render_nci_scatter, render_nci_3d
from .bond_order_map import render_bond_order_map
from ._parsers import auto_parse

__all__ = [
    "render_ranked_charge_map",
    "render_condensed_fukui",
    "render_esp_surface",
    "render_mo_surface",
    "render_nci_scatter",
    "render_nci_3d",
    "render_bond_order_map",
    "auto_parse",
]
