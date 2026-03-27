"""Restricted Hartree-Fock SCF procedure."""

import numpy as np
from ..integrals import overlap_matrix, kinetic_matrix, nuclear_matrix, electron_repulsion_tensor
from .diis import DIIS


class SCFResult:
    """Container for SCF results."""
    def __init__(self, energy, orbital_energies, C, D, F, H_core, S, ERI, converged, n_iter):
        self.energy = energy
        self.orbital_energies = orbital_energies
        self.C = C          # MO coefficients
        self.D = D          # Density matrix
        self.F = F          # Fock matrix
        self.H_core = H_core
        self.S = S
        self.ERI = ERI
        self.converged = converged
        self.n_iter = n_iter


class RHF:
    """Restricted Hartree-Fock calculation."""

    def __init__(self, molecule, basis, verbose=True):
        self.mol = molecule
        self.basis = basis
        self.verbose = verbose

    def compute(self, max_iter=128, e_conv=1e-10, d_conv=1e-8):
        mol = self.mol
        basis = self.basis
        n_occ = mol.n_occupied
        n_bf = len(basis)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  RHF/STO-3G Calculation")
            print(f"{'='*60}")
            print(f"  Atoms:          {mol.n_atoms}")
            print(f"  Electrons:      {mol.n_electrons}")
            print(f"  Basis functions: {n_bf}")
            print(f"  Occupied MOs:   {n_occ}")
            print()

        # Build one-electron integrals
        if self.verbose:
            print("  Computing one-electron integrals...", end=" ", flush=True)
        S = overlap_matrix(basis)
        T = kinetic_matrix(basis)
        V = nuclear_matrix(basis, mol)
        H_core = T + V
        if self.verbose:
            print("done.")

        # Build two-electron integrals
        if self.verbose:
            print("  Computing two-electron integrals...", end=" ", flush=True)
        ERI = electron_repulsion_tensor(basis)
        if self.verbose:
            print("done.")

        # Orthogonalization matrix S^(-1/2)
        eigvals, eigvecs = np.linalg.eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Initial guess from core Hamiltonian
        Fp = X.T @ H_core @ X
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        E_nuc = mol.nuclear_repulsion()
        E_old = 0.0

        diis = DIIS()

        if self.verbose:
            print(f"\n  {'Iter':>4s}  {'Energy':>18s}  {'Delta E':>14s}  {'RMS(D)':>12s}")
            print(f"  {'-'*52}")

        converged = False
        for iteration in range(1, max_iter + 1):
            # Build Fock matrix
            J = np.einsum('kl,ijkl->ij', D, ERI)
            K = np.einsum('kl,ikjl->ij', D, ERI)
            F = H_core + J - 0.5 * K

            # Electronic energy
            E_elec = 0.5 * np.sum(D * (H_core + F))
            E_total = E_elec + E_nuc

            # DIIS
            diis.push(F, D, S)
            F = diis.extrapolate()

            # Diagonalize Fock matrix
            Fp = X.T @ F @ X
            eps, Cp = np.linalg.eigh(Fp)
            C = X @ Cp

            # New density
            D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

            dE = E_total - E_old
            dD = np.sqrt(np.mean((D_new - D) ** 2))

            if self.verbose:
                print(f"  {iteration:4d}  {E_total:18.10f}  {dE:14.10f}  {dD:12.2e}")

            if abs(dE) < e_conv and dD < d_conv and iteration > 1:
                converged = True
                D = D_new
                break

            E_old = E_total
            D = D_new

        if self.verbose:
            if converged:
                print(f"\n  SCF converged in {iteration} iterations.")
            else:
                print(f"\n  WARNING: SCF did NOT converge in {max_iter} iterations!")
            print(f"\n  Nuclear repulsion energy: {E_nuc:18.10f} Eh")
            print(f"  Electronic energy:        {E_elec:18.10f} Eh")
            print(f"  Total energy:             {E_total:18.10f} Eh")
            print()
            print(f"  Orbital energies (Eh):")
            for i, e in enumerate(eps):
                occ = "occ" if i < n_occ else "vir"
                print(f"    {i:3d}  {e:12.6f}  ({occ})")
            print()

        return SCFResult(
            energy=E_total,
            orbital_energies=eps,
            C=C, D=D, F=F, H_core=H_core, S=S, ERI=ERI,
            converged=converged, n_iter=iteration,
        )
