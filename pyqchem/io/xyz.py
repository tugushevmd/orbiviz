"""XYZ file reader/writer."""

import numpy as np
from ..mol.molecule import Atom, Molecule
from ..mol.constants import BOHR_TO_ANGSTROM


def write_xyz(molecule, filename=None):
    """Write molecule to XYZ format string (and optionally file)."""
    lines = [f"{molecule.n_atoms}", ""]
    for atom in molecule.atoms:
        c = atom.coords * BOHR_TO_ANGSTROM
        lines.append(f"{atom.symbol:2s}  {c[0]:14.8f}  {c[1]:14.8f}  {c[2]:14.8f}")
    text = "\n".join(lines) + "\n"
    if filename:
        with open(filename, 'w') as f:
            f.write(text)
    return text


def read_xyz(filename):
    """Read an XYZ file and return a Molecule."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(Atom(sym, [x, y, z]))
    return Molecule(atoms)
