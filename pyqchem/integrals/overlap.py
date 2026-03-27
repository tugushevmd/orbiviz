"""Overlap integrals using Obara-Saika recursion."""

import numpy as np


def _overlap_1d(l1, l2, PA, PB, gamma):
    """1D overlap integral using Obara-Saika recursion.

    Computes S(l1, l2) for one Cartesian direction.
    PA = P_i - A_i, PB = P_i - B_i, gamma = alpha + beta.
    """
    # Build table S[i][j] for i=0..l1, j=0..l2
    S = np.zeros((l1 + 2, l2 + 2))
    S[0][0] = 1.0  # normalized by prefactor outside

    # Upward recursion in first index
    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue
            if j > 0:
                # S(i, j) = PB * S(i, j-1) + (i/(2*gamma)) * S(i-1, j-1) + ((j-1)/(2*gamma)) * S(i, j-2)
                S[i][j] = PB * S[i][j - 1]
                if i > 0:
                    S[i][j] += i / (2.0 * gamma) * S[i - 1][j - 1]
                if j > 1:
                    S[i][j] += (j - 1) / (2.0 * gamma) * S[i][j - 2]
            else:
                # S(i, 0) = PA * S(i-1, 0) + ((i-1)/(2*gamma)) * S(i-2, 0)
                S[i][j] = PA * S[i - 1][0]
                if i > 1:
                    S[i][j] += (i - 1) / (2.0 * gamma) * S[i - 2][0]

    return S[l1][l2]


def overlap_primitive(alpha_a, l_a, m_a, n_a, A,
                      alpha_b, l_b, m_b, n_b, B):
    """Overlap integral between two primitive Gaussian functions."""
    gamma = alpha_a + alpha_b
    P = (alpha_a * A + alpha_b * B) / gamma
    PA = P - A
    PB = P - B
    AB = A - B

    prefactor = np.exp(-alpha_a * alpha_b / gamma * np.dot(AB, AB)) * (np.pi / gamma) ** 1.5

    sx = _overlap_1d(l_a, l_b, PA[0], PB[0], gamma)
    sy = _overlap_1d(m_a, m_b, PA[1], PB[1], gamma)
    sz = _overlap_1d(n_a, n_b, PA[2], PB[2], gamma)

    return prefactor * sx * sy * sz


def overlap_contracted(bf_a, bf_b):
    """Overlap integral between two contracted basis functions."""
    s = 0.0
    for i, (alpha_a, ca) in enumerate(zip(bf_a.alphas, bf_a.coeffs)):
        for j, (alpha_b, cb) in enumerate(zip(bf_b.alphas, bf_b.coeffs)):
            s += bf_a.norms[i] * bf_b.norms[j] * ca * cb * \
                 overlap_primitive(alpha_a, bf_a.l, bf_a.m, bf_a.n, bf_a.center,
                                   alpha_b, bf_b.l, bf_b.m, bf_b.n, bf_b.center)
    return s


def overlap_matrix(basis):
    """Build the overlap matrix S."""
    n = len(basis)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = overlap_contracted(basis[i], basis[j])
            S[j, i] = S[i, j]
    return S
