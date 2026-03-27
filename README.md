# Orbiviz

**Lightweight quantum chemistry toolkit with publication-quality visualization.**

Orbiviz combines a minimal HF/STO-3G engine with a powerful visualization module (`Orbiviz.viz`) designed for computational chemists who need beautiful, informative, and publication-ready figures from standard quantum chemistry output files.

Inspired by [IBOview](https://www.iboview.org/) (visual quality) and [Multiwfn](http://sobereva.com/multiwfn/) (analytical breadth).

## Features

### Visualizations

| Visualization | Description | Input |
|---|---|---|
| **Charge Map** | 2D molecular diagram with atoms colored by charge + ranked bar chart | CSV or Gaussian/ORCA output |
| **Fukui Map** | Condensed Fukui indices (f+, f-, dual descriptor) | CSV + XYZ |
| **ESP Surface** | 3D electrostatic potential mapped onto electron density isosurface | Gaussian cube files |
| **MO Surface** | Molecular orbital isosurfaces with positive/negative lobes (IBOview-style) | Gaussian cube file |
| **NCI Plot** | Non-covalent interaction analysis (scatter + 3D isosurface) | Electron density cube |
| **Bond Order Map** | 2D diagram with bonds colored/sized by Mayer bond order | ORCA output |

### File Format Support

| Format | Extension | What's parsed |
|---|---|---|
| XYZ | `.xyz` | Geometry |
| Gaussian cube | `.cube` | Geometry + volumetric data |
| Gaussian log | `.log`, `.out` | Geometry, charges (Mulliken, NBO, CHELPG, APT), energy, dipole |
| Gaussian fchk | `.fchk` | Geometry, MO coefficients, orbital energies, density matrix |
| ORCA output | `.out` | Geometry, charges (Mulliken, Lowdin), Mayer bond orders, energy |
| Molden | `.molden` | Geometry, MO energies, occupations |

### Design Principles

- **Drop a file, get a figure** — `auto_parse()` detects format automatically
- **Publication quality** — 200 DPI, proper colormaps, clean typography
- **Full periodic table** — 86 elements (H–Rn) with correct covalent radii and CPK colors
- **Perceptually uniform colormaps** — `coolwarm` for charges, `RdBu_r` for ESP (no `jet`!)
- **Modular** — use from CLI, Python scripts, or Jupyter notebooks

## Installation

```bash
# Basic (charges, Fukui maps)
pip install -e .

# Full (ESP, MO, NCI — requires scipy + scikit-image)
pip install -e ".[full]"

# Development
pip install -e ".[dev]"
```

### Dependencies

- **Core:** Python >= 3.10, NumPy, Matplotlib
- **3D visualizations:** SciPy, scikit-image

## Quick Start

### Command Line

```bash
to do
```

### Python API

```python
to do
```


## Project Structure

```
Orbiviz/
├── viz/                    # Visualization module
│   ├── __init__.py         # Public API
│   ├── cli.py              # CLI: Orbiviz {charge,fukui,esp,mo,nci,bond-order,auto}
│   ├── _elements.py        # Periodic table (86 elements, radii, CPK colors)
│   ├── _geometry.py        # XYZ/cube readers, bond detection, 2D projection
│   ├── _io.py              # CSV data readers
│   ├── _parsers.py         # Gaussian/ORCA/Molden parsers
│   ├── _style.py           # Visual theme constants
│   ├── _plotting.py        # Shared matplotlib helpers
│   ├── charge_map.py       # 2D atomic charge visualization
│   ├── fukui_map.py        # Condensed Fukui index visualization
│   ├── esp_surface.py      # 3D ESP isosurface
│   ├── mo_surface.py       # 3D MO isosurface (IBOview-style)
│   ├── nci_plot.py         # NCI scatter + 3D isosurface
│   └── bond_order_map.py   # Bond order diagram
├── mol/                    # Molecular data structures
├── scf/                    # Hartree-Fock SCF solver
├── integrals/              # One- and two-electron integrals
├── basis/                  # Basis set data (STO-3G)
├── grad/                   # Analytical gradients
├── optimizer/              # Geometry optimization
└── io/                     # Input file parsing
```

## Testing

```bash
pytest tests/ -v
```

## License

Apache-2.0
