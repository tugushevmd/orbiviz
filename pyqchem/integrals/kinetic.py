"""Kinetic energy integrals."""

import numpy as np
from .overlap import _overlap_1d


def kinetic_primitive(alpha_a, l_a, m_a, n_a, A,
                      alpha_b, l_b, m_b, n_b, B):
    """Kinetic energy integral between two primitive Gaussians.

    T = -0.5 * <a|nabla^2|b>
    Using the relation for each direction:
    T_x(a,b) = -2*beta*(2*b_x+1)*S(a_x,b_x) + beta^2 * 4 * S(a_x, b_x+2)
               + b_x*(b_x-1)*S(a_x, b_x-2)
    Wait — use the standard formulation:
    T_x = beta*(2*l_b+1)*S_x(l_a,l_b) - 2*beta^2*S_x(l_a,l_b+2) - 0.5*l_b*(l_b-1)*S_x(l_a,l_b-2)
    Then T = T_x*S_y*S_z + S_x*T_y*S_z + S_x*S_y*T_z
    """
    gamma = alpha_a + alpha_b
    P = (alpha_a * A + alpha_b * B) / gamma
    PA = P - A
    PB = P - B
    AB = A - B

    prefactor = np.exp(-alpha_a * alpha_b / gamma * np.dot(AB, AB)) * (np.pi / gamma) ** 1.5

    def S(la, lb, pa, pb):
        return _overlap_1d(la, lb, pa, pb, gamma)

    sx = S(l_a, l_b, PA[0], PB[0])
    sy = S(m_a, m_b, PA[1], PB[1])
    sz = S(n_a, n_b, PA[2], PB[2])

    def T_component(la, lb, pa, pb):
        t = alpha_b * (2 * lb + 1) * S(la, lb, pa, pb)
        t -= 2.0 * alpha_b ** 2 * S(la, lb + 2, pa, pb)
        if lb >= 2:
            t -= 0.5 * lb * (lb - 1) * S(la, lb - 2, pa, pb)
        return t

    tx = T_component(l_a, l_b, PA[0], PB[0])
    ty = T_component(m_a, m_b, PA[1], PB[1])
    tz = T_component(n_a, n_b, PA[2], PB[2])

    return prefactor * (tx * sy * sz + sx * ty * sz + sx * sy * tz)


def kinetic_contracted(bf_a, bf_b):
    """Kinetic energy integral between two contracted basis functions."""
    t = 0.0
    for i, (alpha_a, ca) in enumerate(zip(bf_a.alphas, bf_a.coeffs)):
        for j, (alpha_b, cb) in enumerate(zip(bf_b.alphas, bf_b.coeffs)):
            t += bf_a.norms[i] * bf_b.norms[j] * ca * cb * \
                 kinetic_primitive(alpha_a, bf_a.l, bf_a.m, bf_a.n, bf_a.center,
                                   alpha_b, bf_b.l, bf_b.m, bf_b.n, bf_b.center)
    return t


def kinetic_matrix(basis):
    """Build the kinetic energy matrix T."""
    n = len(basis)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            T[i, j] = kinetic_contracted(basis[i], basis[j])
            T[j, i] = T[i, j]
    return T
