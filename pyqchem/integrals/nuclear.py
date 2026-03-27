"""Nuclear attraction integrals using Obara-Saika recursion."""

import numpy as np
from .boys import boys


def _R(t, u, v, n, p, RPC):
    """Hermite Coulomb auxiliary integral R^n_{t,u,v}.

    Recursive computation for nuclear attraction integrals.
    """
    if t == u == v == 0:
        return (-2.0 * p) ** n * boys(n, p * np.dot(RPC, RPC))
    elif t > 0:
        val = (t - 1) * _R(t - 2, u, v, n + 1, p, RPC)
        val += RPC[0] * _R(t - 1, u, v, n + 1, p, RPC)
        return val
    elif u > 0:
        val = (u - 1) * _R(t, u - 2, v, n + 1, p, RPC)
        val += RPC[1] * _R(t, u - 1, v, n + 1, p, RPC)
        return val
    elif v > 0:
        val = (v - 1) * _R(t, u, v - 2, n + 1, p, RPC)
        val += RPC[2] * _R(t, u, v - 1, n + 1, p, RPC)
        return val
    return 0.0


def _E(i, j, t, Qx, a, b):
    """Hermite expansion coefficient E^{ij}_t.

    Used to expand product of two Gaussians in Hermite Gaussians.
    """
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == j == t == 0:
        return np.exp(-q * Qx ** 2)
    if j == 0:
        # decrement i
        val = (1.0 / (2.0 * p)) * _E(i - 1, j, t - 1, Qx, a, b)
        val -= (q * Qx / a) * _E(i - 1, j, t, Qx, a, b)
        val += (t + 1) * _E(i - 1, j, t + 1, Qx, a, b)
        return val
    # decrement j
    val = (1.0 / (2.0 * p)) * _E(i, j - 1, t - 1, Qx, a, b)
    val += (q * Qx / b) * _E(i, j - 1, t, Qx, a, b)
    val += (t + 1) * _E(i, j - 1, t + 1, Qx, a, b)
    return val


def nuclear_primitive(alpha_a, l_a, m_a, n_a, A,
                      alpha_b, l_b, m_b, n_b, B,
                      C, Z_C):
    """Nuclear attraction integral for one nucleus at C with charge Z_C.

    <a|(-Z_C/|r-C|)|b>
    """
    p = alpha_a + alpha_b
    P = (alpha_a * A + alpha_b * B) / p
    RPC = P - C

    val = 0.0
    for t in range(l_a + l_b + 1):
        for u in range(m_a + m_b + 1):
            for v in range(n_a + n_b + 1):
                e_x = _E(l_a, l_b, t, A[0] - B[0], alpha_a, alpha_b)
                e_y = _E(m_a, m_b, u, A[1] - B[1], alpha_a, alpha_b)
                e_z = _E(n_a, n_b, v, A[2] - B[2], alpha_a, alpha_b)
                val += e_x * e_y * e_z * _R(t, u, v, 0, p, RPC)

    val *= -Z_C * 2.0 * np.pi / p
    return val


def nuclear_contracted(bf_a, bf_b, molecule):
    """Nuclear attraction integral summed over all nuclei."""
    v = 0.0
    for i, (alpha_a, ca) in enumerate(zip(bf_a.alphas, bf_a.coeffs)):
        for j, (alpha_b, cb) in enumerate(zip(bf_b.alphas, bf_b.coeffs)):
            for atom in molecule.atoms:
                v += bf_a.norms[i] * bf_b.norms[j] * ca * cb * \
                     nuclear_primitive(alpha_a, bf_a.l, bf_a.m, bf_a.n, bf_a.center,
                                       alpha_b, bf_b.l, bf_b.m, bf_b.n, bf_b.center,
                                       atom.coords, atom.Z)
    return v


def nuclear_matrix(basis, molecule):
    """Build the nuclear attraction matrix V."""
    n = len(basis)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            V[i, j] = nuclear_contracted(basis[i], basis[j], molecule)
            V[j, i] = V[i, j]
    return V
