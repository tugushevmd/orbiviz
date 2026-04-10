"""pyqchem.viz — Publication-quality visualizations for computational chemistry.

Supported visualizations
------------------------
- Atomic charge maps (ADCH, CHELPG, NBO, NPA, Mulliken, Loewdin, APT)
- Spin density maps (alpha − beta) for open-shell systems
- Charge difference (Δq) maps between two outputs
- Condensed Fukui index maps (f+, f-, dual descriptor)
- ESP-colored electron density isosurfaces (3D)
- Molecular orbital isosurfaces (3D, IBOview-style)
- 2D molecular orbital contour slices (no SciPy needed)
- NCI analysis (scatter plots and 3D isosurfaces)
- Bond order maps (Mayer, Wiberg)
- Geometry-optimization trajectory animations

Supported input formats
-----------------------
- XYZ geometry files (single & multi-frame trajectories)
- Gaussian cube files
- Gaussian .log / .out files
- Gaussian .fchk files
- ORCA output files
- NWChem output files
- Q-Chem output files
- NBO 6/7 standalone output files
- Molden files
- CSV data files

The 3D isosurface renderers (`render_esp_surface`, `render_mo_surface`,
`render_nci_*`) require the optional dependencies SciPy and scikit-image.
The rest of the package works with NumPy + Matplotlib only.

All public renderers are exposed lazily, so importing this package never
fails because of missing optional dependencies — the import error is raised
only when you actually call a renderer that needs them.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Mapping of public name -> (submodule, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    # Always available (NumPy + Matplotlib only)
    "render_ranked_charge_map": (".charge_map", "render_ranked_charge_map"),
    "render_spin_density_map": (".charge_map", "render_spin_density_map"),
    "render_charge_difference_map": (".charge_map", "render_charge_difference_map"),
    "render_condensed_fukui": (".fukui_map", "render_condensed_fukui"),
    "render_bond_order_map": (".bond_order_map", "render_bond_order_map"),
    "render_mo_slice_2d": (".mo_slice", "render_mo_slice_2d"),
    "render_trajectory_animation": (".trajectory", "render_trajectory_animation"),
    "auto_parse": ("._parsers", "auto_parse"),
    # Require SciPy / scikit-image
    "render_esp_surface": (".esp_surface", "render_esp_surface"),
    "render_mo_surface": (".mo_surface", "render_mo_surface"),
    "render_nci_scatter": (".nci_plot", "render_nci_scatter"),
    "render_nci_3d": (".nci_plot", "render_nci_3d"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module 'pyqchem.viz' has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # pragma: no cover
    from .charge_map import (
        render_ranked_charge_map,
        render_spin_density_map,
        render_charge_difference_map,
    )
    from .fukui_map import render_condensed_fukui
    from .bond_order_map import render_bond_order_map
    from .mo_slice import render_mo_slice_2d
    from .trajectory import render_trajectory_animation
    from ._parsers import auto_parse
    from .esp_surface import render_esp_surface
    from .mo_surface import render_mo_surface
    from .nci_plot import render_nci_scatter, render_nci_3d
