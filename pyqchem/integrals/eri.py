"""Two-electron repulsion integrals (ERI) using McMurchie-Davidson scheme."""

import numpy as np
from .boys import boys
from .nuclear import _E, _R


def _R4(t, u, v, n, p, RPQ):
    """Hermite Coulomb integral for ERIs — same recursion as nuclear."""
    return _R(t, u, v, n, p, RPQ)


def eri_primitive(alpha_a, l_a, m_a, n_a, A,
                  alpha_b, l_b, m_b, n_b, B,
                  alpha_c, l_c, m_c, n_c, C,
                  alpha_d, l_d, m_d, n_d, D):
    """Compute (ab|cd) ERI for four primitive Gaussians.

    Uses McMurchie-Davidson: expand each pair in Hermite Gaussians, then
    contract with the R auxiliary integrals.
    """
    p = alpha_a + alpha_b
    q = alpha_c + alpha_d
    alpha = p * q / (p + q)
    P = (alpha_a * A + alpha_b * B) / p
    Q = (alpha_c * C + alpha_d * D) / q
    RPQ = P - Q

    val = 0.0
    for t in range(l_a + l_b + 1):
        for u in range(m_a + m_b + 1):
            for v in range(n_a + n_b + 1):
                e1x = _E(l_a, l_b, t, A[0] - B[0], alpha_a, alpha_b)
                e1y = _E(m_a, m_b, u, A[1] - B[1], alpha_a, alpha_b)
                e1z = _E(n_a, n_b, v, A[2] - B[2], alpha_a, alpha_b)
                for tau in range(l_c + l_d + 1):
                    for mu in range(m_c + m_d + 1):
                        for nu in range(n_c + n_d + 1):
                            e2x = _E(l_c, l_d, tau, C[0] - D[0], alpha_c, alpha_d)
                            e2y = _E(m_c, m_d, mu, C[1] - D[1], alpha_c, alpha_d)
                            e2z = _E(n_c, n_d, nu, C[2] - D[2], alpha_c, alpha_d)
                            r_val = _R4(t + tau, u + mu, v + nu, 0, alpha, RPQ)
                            val += e1x * e1y * e1z * e2x * e2y * e2z * r_val * \
                                   (-1) ** (tau + mu + nu)

    val *= 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return val


def eri_contracted(bf_a, bf_b, bf_c, bf_d):
    """ERI between four contracted basis functions."""
    val = 0.0
    for i, (aa, ca) in enumerate(zip(bf_a.alphas, bf_a.coeffs)):
        for j, (ab, cb) in enumerate(zip(bf_b.alphas, bf_b.coeffs)):
            for k, (ac, cc) in enumerate(zip(bf_c.alphas, bf_c.coeffs)):
                for l, (ad, cd) in enumerate(zip(bf_d.alphas, bf_d.coeffs)):
                    val += bf_a.norms[i] * bf_b.norms[j] * bf_c.norms[k] * bf_d.norms[l] * \
                           ca * cb * cc * cd * \
                           eri_primitive(aa, bf_a.l, bf_a.m, bf_a.n, bf_a.center,
                                         ab, bf_b.l, bf_b.m, bf_b.n, bf_b.center,
                                         ac, bf_c.l, bf_c.m, bf_c.n, bf_c.center,
                                         ad, bf_d.l, bf_d.m, bf_d.n, bf_d.center)
    return val


def electron_repulsion_tensor(basis):
    """Build the full ERI tensor (ij|kl).

    Uses 8-fold permutational symmetry to reduce computation.
    """
    n = len(basis)
    ERI = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(i + 1):
            for k in range(n):
                for l in range(k + 1):
                    if (i * (i + 1)) // 2 + j >= (k * (k + 1)) // 2 + l:
                        val = eri_contracted(basis[i], basis[j], basis[k], basis[l])
                        # Apply 8-fold symmetry
                        ERI[i, j, k, l] = val
                        ERI[j, i, k, l] = val
                        ERI[i, j, l, k] = val
                        ERI[j, i, l, k] = val
                        ERI[k, l, i, j] = val
                        ERI[l, k, i, j] = val
                        ERI[k, l, j, i] = val
                        ERI[l, k, j, i] = val

    return ERI
