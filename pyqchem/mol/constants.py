import numpy as np

ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18,
}

ELEMENT_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}

ATOMIC_MASSES = {
    'H': 1.00794, 'He': 4.00260, 'Li': 6.941, 'Be': 9.01218, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
}
