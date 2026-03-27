# PyQChem

**Lightweight quantum chemistry toolkit with publication-quality visualization.**

PyQChem combines a minimal HF/STO-3G engine with a powerful visualization module (`pyqchem.viz`) designed for computational chemists who need beautiful, informative, and publication-ready figures from standard quantum chemistry output files.

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
# Auto-detect file format and show what's inside
pyqchem-viz auto molecule.log

# Charge map from CSV
pyqchem-viz charge --charges-csv charges.csv --charge-column mulliken_charge --output charges.png

# Fukui indices
pyqchem-viz fukui --xyz mol.xyz --indices-csv fukui.csv --metric f_plus --output fukui.png

# ESP surface from cube files
pyqchem-viz esp --density-cube density.cube --esp-cube esp.cube --output esp.png

# Molecular orbital
pyqchem-viz mo --cube homo.cube --output homo.png --title "HOMO"

# NCI analysis
pyqchem-viz nci --density-cube density.cube --output nci_scatter.png --mode scatter
pyqchem-viz nci --density-cube density.cube --output nci_3d.png --mode 3d

# Bond orders from ORCA
pyqchem-viz bond-order --orca-out calc.out --output bonds.png
```

### Python API

```python
from pyqchem.viz import auto_parse, render_ranked_charge_map

# Auto-parse any supported file
data = auto_parse("calculation.log")
print(f"Format: {data['format']}, Atoms: {len(data['atoms'])}")

# Render a charge map
from pyqchem.viz._io import read_charge_csv
from pyqchem.viz._geometry import read_xyz

atoms = read_xyz("molecule.xyz")
charges = read_charge_csv("charges.csv", "mulliken_charge")
render_ranked_charge_map(atoms, charges, "output.png", title="Mulliken Charges")
```

```python
# Parse Gaussian log and visualize charges
from pyqchem.viz._parsers import GaussianLogParser

parser = GaussianLogParser("calculation.log")
atoms = parser.get_geometry()
charges = parser.get_mulliken_charges()
energy = parser.get_energy()
print(f"E = {energy:.6f} Hartree")
```

```python
# Parse ORCA output for bond orders
from pyqchem.viz._parsers import OrcaOutputParser
from pyqchem.viz import render_bond_order_map

parser = OrcaOutputParser("orca_calc.out")
atoms = parser.get_geometry()
bond_orders = parser.get_mayer_bond_orders()
render_bond_order_map(atoms, bond_orders, "bonds.png")
```

## Project Structure

```
pyqchem/
├── viz/                    # Visualization module
│   ├── __init__.py         # Public API
│   ├── cli.py              # CLI: pyqchem-viz {charge,fukui,esp,mo,nci,bond-order,auto}
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

## Roadmap

- [ ] Interactive 3D viewer (PyVista / Plotly)
- [ ] Density of states (DOS) plots
- [ ] Vibrational mode animation
- [ ] Natural Transition Orbital (NTO) visualization
- [ ] Multi-molecule comparison panels
- [ ] PDF/SVG export
- [ ] GUI application (PyQt6)

## License

MIT
