"""Analytical RHF energy gradient via numerical differentiation of integrals.

For the initial implementation, we use numerical finite differences of the
total energy. This is simpler to implement correctly and sufficient for
geometry optimization of small molecules.

dE/dR_A ≈ [E(R_A + h) - E(R_A - h)] / (2h)
"""

import numpy as np


class RHFGradient:
    """Compute RHF energy gradient using central finite differences."""

    def __init__(self, step=1e-4):
        self.step = step

    def compute(self, molecule, basis_set_builder, rhf_class, verbose=True):
        """Compute the gradient dE/dR for all atoms.

        Parameters
        ----------
        molecule : Molecule
        basis_set_builder : BasisSet object with .build(mol) method
        rhf_class : RHF class to construct calculations
        verbose : bool

        Returns
        -------
        gradient : np.ndarray of shape (n_atoms, 3)
        """
        n_atoms = molecule.n_atoms
        gradient = np.zeros((n_atoms, 3))
        h = self.step

        coords_orig = molecule.get_coords().copy()

        for a in range(n_atoms):
            for x in range(3):
                # Forward step
                coords_plus = coords_orig.copy()
                coords_plus[a, x] += h
                molecule.set_coords_bohr(coords_plus)
                basis_plus = basis_set_builder.build(molecule)
                rhf_plus = rhf_class(molecule, basis_plus, verbose=False)
                res_plus = rhf_plus.compute()

                # Backward step
                coords_minus = coords_orig.copy()
                coords_minus[a, x] -= h
                molecule.set_coords_bohr(coords_minus)
                basis_minus = basis_set_builder.build(molecule)
                rhf_minus = rhf_class(molecule, basis_minus, verbose=False)
                res_minus = rhf_minus.compute()

                gradient[a, x] = (res_plus.energy - res_minus.energy) / (2.0 * h)

        # Restore original coordinates
        molecule.set_coords_bohr(coords_orig)

        if verbose:
            print(f"\n  Energy Gradient (Eh/Bohr):")
            print(f"  {'Atom':>6s}  {'dE/dx':>14s}  {'dE/dy':>14s}  {'dE/dz':>14s}")
            print(f"  {'-'*52}")
            for a in range(n_atoms):
                sym = molecule.atoms[a].symbol
                print(f"  {sym:>6s}  {gradient[a,0]:14.8f}  {gradient[a,1]:14.8f}  {gradient[a,2]:14.8f}")
            max_grad = np.max(np.abs(gradient))
            rms_grad = np.sqrt(np.mean(gradient ** 2))
            print(f"\n  Max gradient: {max_grad:.8f} Eh/Bohr")
            print(f"  RMS gradient: {rms_grad:.8f} Eh/Bohr")

        return gradient
