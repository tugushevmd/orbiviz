import numpy as np
from .constants import ATOMIC_NUMBERS, ANGSTROM_TO_BOHR


class Atom:
    def __init__(self, symbol, coords_angstrom):
        self.symbol = symbol.capitalize()
        if len(self.symbol) > 1:
            self.symbol = self.symbol[0] + self.symbol[1:].lower()
        self.Z = ATOMIC_NUMBERS[self.symbol]
        self.coords = np.array(coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR

    def __repr__(self):
        return f"Atom({self.symbol}, Z={self.Z})"


class Molecule:
    def __init__(self, atoms, charge=0, multiplicity=1):
        self.atoms = list(atoms)
        self.charge = charge
        self.multiplicity = multiplicity

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_electrons(self):
        return sum(a.Z for a in self.atoms) - self.charge

    @property
    def n_occupied(self):
        return self.n_electrons // 2

    def nuclear_repulsion(self):
        e_nuc = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                r = np.linalg.norm(self.atoms[i].coords - self.atoms[j].coords)
                e_nuc += self.atoms[i].Z * self.atoms[j].Z / r
        return e_nuc

    def get_coords(self):
        return np.array([a.coords for a in self.atoms])

    def set_coords_bohr(self, coords):
        for i, a in enumerate(self.atoms):
            a.coords = coords[i].copy()

    def __repr__(self):
        symbols = [a.symbol for a in self.atoms]
        return f"Molecule({' '.join(symbols)}, charge={self.charge}, mult={self.multiplicity})"
