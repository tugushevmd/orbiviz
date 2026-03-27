"""DIIS (Direct Inversion in the Iterative Subspace) convergence accelerator."""

import numpy as np


class DIIS:
    def __init__(self, max_vectors=8):
        self.max_vectors = max_vectors
        self.fock_list = []
        self.error_list = []

    def push(self, F, D, S):
        """Store a new Fock matrix and its DIIS error vector.

        Error = FDS - SDF (in AO basis).
        """
        err = F @ D @ S - S @ D @ F
        self.fock_list.append(F.copy())
        self.error_list.append(err.copy())

        if len(self.fock_list) > self.max_vectors:
            self.fock_list.pop(0)
            self.error_list.pop(0)

    def extrapolate(self):
        """Return the DIIS-extrapolated Fock matrix."""
        n = len(self.fock_list)
        if n < 2:
            return self.fock_list[-1]

        B = np.zeros((n + 1, n + 1))
        B[-1, :] = -1.0
        B[:, -1] = -1.0
        B[-1, -1] = 0.0

        for i in range(n):
            for j in range(i, n):
                val = np.sum(self.error_list[i] * self.error_list[j])
                B[i, j] = val
                B[j, i] = val

        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0

        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return self.fock_list[-1]

        F_diis = np.zeros_like(self.fock_list[0])
        for i in range(n):
            F_diis += coeffs[i] * self.fock_list[i]

        return F_diis
