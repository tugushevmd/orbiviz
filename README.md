# Orbiviz

**Lightweight quantum chemistry toolkit with publication-quality visualization.**

Orbiviz combines a minimal HF/STO-3G engine with a powerful visualization
module (`pyqchem.viz`) designed for computational chemists who need beautiful,
informative, and publication-ready figures from standard quantum chemistry
output files.

Inspired by [IBOview](https://www.iboview.org/) (visual quality) and
[Multiwfn](http://sobereva.com/multiwfn/) (analytical breadth).

## Features

### Visualizations

| Visualization | Description | Input |
|---|---|---|
| **Charge Map** | 2D molecular diagram with atoms colored by charge + ranked bar chart | CSV or Gaussian/ORCA/NWChem/Q-Chem output |
| **Spin Density Map** | 2D map of α-β Mulliken/Loewdin spin populations (open-shell systems) | Open-shell Gaussian/ORCA output |
| **Delta-q Map** | Charge difference (q_B - q_A) between two outputs (e.g. neutral vs. anion) | Two output files |
| **Fukui Map** | Condensed Fukui indices (f+, f-, dual descriptor) | CSV + XYZ, or neutral/anion/cation outputs |
| **ESP Surface** | 3D electrostatic potential mapped onto electron density isosurface | Gaussian cube files |
| **MO Surface** | 3D molecular orbital isosurface, IBOview-style positive/negative lobes | Gaussian cube file |
| **MO Slice 2D** | 2D contour slice of an MO cube on xy/xz/yz plane (no SciPy needed) | Gaussian cube file |
| **NCI Plot** | Non-covalent interaction analysis (scatter + 3D isosurface) | Electron density cube |
| **Bond Order Map** | 2D diagram with bonds colored/sized by Mayer bond order | ORCA output |
| **Trajectory Animation** | Animated 2D ball-and-stick view of an optimization or MD trajectory | Multi-frame XYZ |

All renderers write whatever format Matplotlib infers from the output suffix:
`.png`, `.svg`, `.pdf`, `.eps`. Trajectory animations are written as `.gif`
(no extra dependency) or `.mp4` (requires ffmpeg).

### File Format Support

| Format | Extension | What's parsed |
|---|---|---|
| XYZ | `.xyz` | Geometry (single & multi-frame trajectories) |
| Gaussian cube | `.cube`, `.cub` | Geometry + volumetric data (incl. MO cubes with negative natoms header) |
| Gaussian log | `.log`, `.out` | Geometry, charges (Mulliken, NBO/NPA, CHELPG, APT), spin populations, energy, dipole |
| Gaussian fchk | `.fchk` | Geometry, MO coefficients, orbital energies, density matrix |
| ORCA output | `.out` | Geometry, charges (Mulliken, Loewdin, CHELPG), spin populations, Mayer bond orders, energy |
| NWChem output | `.out`, `.log` | Geometry, charges (Mulliken, ESP, Lowdin), energy |
| Q-Chem output | `.out`, `.log` | Geometry, charges (Mulliken, ChElPG, Hirshfeld), energy |
| NBO output | `.log`, `.out` | NPA charges (geometry must be supplied separately via `--xyz`) |
| Molden | `.molden` | Geometry, MO energies, occupations |

### Design Principles

- **Drop a file, get a figure** — `auto_parse()` detects format automatically.
- **Publication quality** — 200 DPI default, perceptually-uniform colormaps,
  clean typography.
- **Full periodic table** — 86 elements (H-Rn) with correct covalent radii and
  CPK colors.
- **No `jet`** — `coolwarm` for charges, `RdBu_r` for ESP and orbital
  amplitudes, `RdYlGn_r` for NCI.
- **Lazy optional dependencies** — importing `pyqchem.viz` never fails because
  of missing SciPy / scikit-image. The error is raised only if you actually
  call a 3D-isosurface renderer that needs them.
- **Modular** — use from CLI, Python scripts, or Jupyter notebooks.

## Installation

```bash
# Basic (charges, Fukui maps, bond orders, 2D MO slices, trajectory animations)
pip install -e .

# Full (also enables 3D ESP / MO / NCI isosurfaces — requires SciPy + scikit-image)
pip install -e ".[full]"

# Development
pip install -e ".[dev]"
```

### Dependencies

- **Core:** Python >= 3.10, NumPy, Matplotlib (Pillow is pulled in by Matplotlib for GIF export)
- **3D isosurface visualizations:** SciPy, scikit-image
- **MP4 trajectory export:** ffmpeg on PATH

## Quick Start

### Command Line

The CLI is exposed both as `python -m pyqchem.viz` and via the top-level
`main.py` shim. Use `--help` on any subcommand for the full option list.

```bash
# Inspect a file: detect format and show what data is available
python main.py auto results/anion.out

# Atomic charge map (any supported program)
python main.py charge \
    --input-file results/molecule.out \
    --charge-scheme chelpg \
    --output figures/charges.png

# Same thing as a vector SVG (just change the suffix)
python main.py charge --input-file results/molecule.out --charge-scheme mulliken --output figures/charges.svg

# Spin density (alpha - beta) for an open-shell calculation
python main.py spin \
    --input-file results/radical.out \
    --scheme mulliken \
    --output figures/spin.png

# Charge difference: q(anion) - q(neutral)
python main.py diff \
    --file-a results/neutral.out \
    --file-b results/anion.out \
    --charge-scheme mulliken \
    --output figures/delta_q.png

# Condensed Fukui (dual descriptor) from a triplet of outputs
python main.py fukui \
    --neutral-file results/neutral.out \
    --anion-file   results/anion.out \
    --cation-file  results/cation.out \
    --metric dual_descriptor \
    --output figures/fukui_dual.png

# Bond order map from an ORCA Mayer block
python main.py bond-order --orca-out results/molecule.out --output figures/bo.png

# 2D MO contour slice from a cube file (no SciPy needed)
python main.py mo-slice --cube cubes/HOMO.cube --plane xy --output figures/homo_xy.png

# 3D MO isosurface (requires the [full] install profile)
python main.py mo --cube cubes/HOMO.cube --output figures/homo_3d.png

# 3D ESP-mapped density surface
python main.py esp --density-cube cubes/density.cube --esp-cube cubes/esp.cube --output figures/esp.png

# NCI scatter plot
python main.py nci --density-cube cubes/density.cube --mode scatter --output figures/nci_scatter.png

# Animate an optimization trajectory
python main.py trajectory --trajectory results/opt_trj.xyz --output figures/opt.gif --fps 8 --stride 2
```

Every subcommand accepts `--dpi N` (default 200) and writes the format
implied by the output suffix.

### Python API

```python
from pathlib import Path
from pyqchem.viz import (
    auto_parse,
    render_ranked_charge_map,
    render_spin_density_map,
    render_charge_difference_map,
    render_condensed_fukui,
    render_mo_slice_2d,
    render_trajectory_animation,
)

# 1. Parse anything
result = auto_parse("calc/molecule.out")
print(result["format"], result["charges"].keys())

# 2. Charge map
render_ranked_charge_map(
    result["atoms"],
    result["charges"]["chelpg"],
    Path("figures/charges.svg"),
    title="CHELPG charges",
)

# 3. Spin density (open-shell calculation)
anion = auto_parse("calc/anion.out")
render_spin_density_map(
    anion["atoms"],
    anion["spin_populations"]["mulliken"],
    Path("figures/spin.png"),
    title="Anion spin density",
)

# 4. Charge difference between two states
neutral = auto_parse("calc/neutral.out")
render_charge_difference_map(
    neutral["atoms"],
    neutral["charges"]["mulliken"],
    anion["charges"]["mulliken"],
    Path("figures/delta_q.png"),
    title="q(anion) - q(neutral)",
)

# 5. 2D contour slice through an MO cube — no SciPy required
render_mo_slice_2d(
    Path("cubes/HOMO.cube"),
    Path("figures/homo_xy.png"),
    plane="xy",
    offset=0.0,
)

# 6. Animate an optimization trajectory
render_trajectory_animation(
    Path("calc/opt_trj.xyz"),
    Path("figures/opt.gif"),
    fps=8,
    stride=2,
    title="r2SCAN-3c optimization",
)
```

## Project Structure

```
orbiviz/
├── main.py                 # Top-level CLI entry point (forwards to pyqchem.viz.cli)
├── pyqchem/
│   ├── viz/                # Visualization module
│   │   ├── __init__.py     # Lazy public API
│   │   ├── cli.py          # CLI: orbiviz {charge,spin,diff,fukui,esp,mo,mo-slice,nci,bond-order,trajectory,auto}
│   │   ├── _elements.py    # Periodic table (86 elements, radii, CPK colors)
│   │   ├── _geometry.py    # XYZ/cube readers, bond detection, 2D projection
│   │   ├── _io.py          # CSV data readers
│   │   ├── _parsers.py     # Gaussian/ORCA/NWChem/Q-Chem/Molden/NBO parsers
│   │   ├── _style.py       # Visual theme constants
│   │   ├── _plotting.py    # Shared matplotlib helpers
│   │   ├── charge_map.py   # Charge / spin density / delta-q maps
│   │   ├── fukui_map.py    # Condensed Fukui index visualization
│   │   ├── bond_order_map.py # Bond order diagram
│   │   ├── mo_slice.py     # 2D MO contour slice (no SciPy)
│   │   ├── mo_surface.py   # 3D MO isosurface (IBOview-style)
│   │   ├── esp_surface.py  # 3D ESP isosurface
│   │   ├── nci_plot.py     # NCI scatter + 3D isosurface
│   │   └── trajectory.py   # Multi-frame XYZ animations
│   ├── mol/                # Molecular data structures
│   ├── scf/                # Hartree-Fock SCF solver
│   ├── integrals/          # One- and two-electron integrals
│   ├── basis/              # Basis set data (STO-3G)
│   ├── grad/               # Analytical gradients
│   ├── optimizer/          # Geometry optimization
│   └── io/                 # Input file parsing
└── tests/                  # Pytest suite (uses real ORCA fixtures)
```

## Testing

```bash
pytest tests/ -v
```

The test suite uses a real ORCA r²SCAN-3c calculation on 4-nitroindane
(C₁₀H₉NO₂) shipped under `tests/data/c10h9no2_orca/`, including a multi-frame
optimization trajectory and HOMO/LUMO cube files, so the rendering pipeline
is exercised end-to-end on representative quantum chemistry output.

## License

Apache-2.0
