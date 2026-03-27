"""Geometry optimizer using L-BFGS-B from scipy."""

import numpy as np
from scipy.optimize import minimize


class GeometryOptimizer:
    """Optimize molecular geometry by minimizing the RHF energy."""

    def __init__(self, max_steps=100, grad_tol=3e-4, verbose=True):
        self.max_steps = max_steps
        self.grad_tol = grad_tol
        self.verbose = verbose
        self._step = 0

    def optimize(self, molecule, basis_set, rhf_class, gradient_computer):
        """Run geometry optimization.

        Parameters
        ----------
        molecule : Molecule
        basis_set : BasisSet (builder, has .build() method)
        rhf_class : RHF class
        gradient_computer : RHFGradient instance

        Returns
        -------
        result : dict with optimized molecule, final energy, convergence info
        """
        from ..mol.constants import BOHR_TO_ANGSTROM

        n_atoms = molecule.n_atoms
        self._step = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Geometry Optimization")
            print(f"{'='*60}")
            print(f"  Optimizer: L-BFGS-B")
            print(f"  Max steps: {self.max_steps}")
            print(f"  Gradient tolerance: {self.grad_tol:.1e} Eh/Bohr")
            print()

        def energy_and_gradient(coords_flat):
            self._step += 1
            coords = coords_flat.reshape(n_atoms, 3)
            molecule.set_coords_bohr(coords)

            basis = basis_set.build(molecule)
            rhf = rhf_class(molecule, basis, verbose=False)
            scf_result = rhf.compute()

            grad = gradient_computer.compute(molecule, basis_set, rhf_class, verbose=False)

            if self.verbose:
                max_g = np.max(np.abs(grad))
                rms_g = np.sqrt(np.mean(grad ** 2))
                print(f"  Step {self._step:3d}:  E = {scf_result.energy:16.10f} Eh  "
                      f"|grad|_max = {max_g:.6f}  |grad|_rms = {rms_g:.6f}")

            return scf_result.energy, grad.flatten()

        x0 = molecule.get_coords().flatten()

        result = minimize(
            energy_and_gradient,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': self.max_steps,
                'gtol': self.grad_tol,
                'ftol': 1e-12,
            },
        )

        # Update molecule with optimized coords
        opt_coords = result.x.reshape(n_atoms, 3)
        molecule.set_coords_bohr(opt_coords)

        # Final SCF at optimized geometry
        basis = basis_set.build(molecule)
        rhf = rhf_class(molecule, basis, verbose=self.verbose)
        final_result = rhf.compute()

        if self.verbose:
            print(f"\n{'='*60}")
            if result.success:
                print(f"  Optimization CONVERGED in {self._step} steps")
            else:
                print(f"  Optimization did NOT converge: {result.message}")
            print(f"  Final energy: {final_result.energy:18.10f} Eh")
            print(f"\n  Optimized geometry (Angstrom):")
            print(f"  {'Atom':>4s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}")
            print(f"  {'-'*46}")
            for atom in molecule.atoms:
                c = atom.coords * BOHR_TO_ANGSTROM
                print(f"  {atom.symbol:>4s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}")
            print(f"{'='*60}\n")

        return {
            'energy': final_result.energy,
            'scf_result': final_result,
            'converged': result.success,
            'n_steps': self._step,
            'molecule': molecule,
        }
