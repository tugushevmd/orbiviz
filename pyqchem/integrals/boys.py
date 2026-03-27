"""Boys function F_n(x) for molecular integrals."""

import numpy as np
from scipy.special import hyp1f1


def boys(n, x):
    """Compute Boys function F_n(x).

    F_n(x) = integral_0^1 t^(2n) exp(-x t^2) dt
           = (1/(2n+1)) * 1F1(n+0.5, n+1.5, -x)
    """
    if x < 1e-14:
        return 1.0 / (2.0 * n + 1.0)
    return hyp1f1(n + 0.5, n + 1.5, -x) / (2.0 * n + 1.0)


def boys_array(n_max, x):
    """Compute Boys function F_m(x) for m = 0, 1, ..., n_max."""
    result = np.zeros(n_max + 1)
    for m in range(n_max + 1):
        result[m] = boys(m, x)
    return result
