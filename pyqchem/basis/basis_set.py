import numpy as np
from math import factorial
from .sto3g import STO3G_DATA


def double_factorial(n):
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


def normalization(alpha, l, m, n):
    """Normalization constant for a Cartesian Gaussian primitive."""
    L = l + m + n
    num = (2.0 * alpha / np.pi) ** 1.5 * (4.0 * alpha) ** L
    den = double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
    return np.sqrt(num / den)


class BasisFunction:
    """A contracted Gaussian basis function.

    phi(r) = sum_k  N_k * c_k * x_A^l * y_A^m * z_A^n * exp(-alpha_k * |r - A|^2)
    """

    def __init__(self, center, alphas, coeffs, l, m, n):
        self.center = np.array(center, dtype=float)
        self.alphas = np.array(alphas, dtype=float)
        self.l = l
        self.m = m
        self.n = n
        # Store normalized coefficients: coeff * norm
        self.norms = np.array([normalization(a, l, m, n) for a in self.alphas])
        self.coeffs = np.array(coeffs, dtype=float)

    @property
    def ang_mom(self):
        return self.l + self.m + self.n

    def __repr__(self):
        labels = {0: 's', 1: 'p', 2: 'd'}
        return f"BF({labels.get(self.ang_mom, '?')}, center={self.center})"


class BasisSet:
    """Build a list of BasisFunction objects for a molecule."""

    def __init__(self, name='sto-3g'):
        self.name = name.lower().replace('-', '')
        if self.name != 'sto3g':
            raise ValueError(f"Unknown basis set: {name}. Only STO-3G is supported.")

    def build(self, molecule):
        """Return list of BasisFunction for the given molecule."""
        basis = []
        for atom in molecule.atoms:
            symbol = atom.symbol
            if symbol not in STO3G_DATA:
                raise ValueError(f"No STO-3G parameters for element {symbol}")
            shells = STO3G_DATA[symbol]
            center = atom.coords
            for ang_mom, exponents, coeffs in shells:
                if ang_mom == 0:
                    basis.append(BasisFunction(center, exponents, coeffs, 0, 0, 0))
                elif ang_mom == 1:
                    basis.append(BasisFunction(center, exponents, coeffs, 1, 0, 0))
                    basis.append(BasisFunction(center, exponents, coeffs, 0, 1, 0))
                    basis.append(BasisFunction(center, exponents, coeffs, 0, 0, 1))
        return basis
