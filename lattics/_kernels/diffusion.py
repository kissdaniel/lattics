"""Implicit diffusion on a two-dimensional grid with zero-flux (Neumann) boundaries.

A time step is split into two implicit (backward Euler) steps, one along each axis (locally one-dimensional, LOD).
Every column sum of the one-dimensional matrix is 1, so the total mass is conserved exactly, and the step is
unconditionally stable. The matrix depends only on the number of cells and on ``r = D dt / dx^2``, so its
factorization (Thomas algorithm) is computed once and reused for every line of the grid.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def thomas_factors(n, r):
    """Factorizes the matrix of an implicit diffusion step along a line of ``n`` cells.

    Args:
        n (int): the number of cells
        r (float): ``D dt / dx^2``

    Returns:
        tuple of arrays: the modified super-diagonal and the inverse pivots
    """
    diag = np.full(n, 1.0 + 2.0 * r)
    if n == 1:
        diag[0] = 1.0
    else:
        diag[0] = 1.0 + r
        diag[n - 1] = 1.0 + r
    c_prime = np.zeros(n)
    inv_pivot = np.zeros(n)
    inv_pivot[0] = 1.0 / diag[0]
    c_prime[0] = -r * inv_pivot[0]
    for i in range(1, n):
        inv_pivot[i] = 1.0 / (diag[i] + r * c_prime[i - 1])
        c_prime[i] = -r * inv_pivot[i]
    return c_prime, inv_pivot


@numba.njit(cache=True)
def implicit_diffusion_2d(C, r, c_prime_0, inv_pivot_0, c_prime_1, inv_pivot_1):
    """Performs one LOD diffusion step: an implicit step along axis 0, then along axis 1.

    Args:
        C (2D array of floats): the concentrations
        r (float): ``D dt / dx^2``
        c_prime_0, inv_pivot_0 (arrays): the factors of :func:`thomas_factors` for axis 0
        c_prime_1, inv_pivot_1 (arrays): the factors of :func:`thomas_factors` for axis 1

    Returns:
        2D array of floats: the new concentrations
    """
    n0, n1 = C.shape
    tmp = np.empty_like(C)
    d_prime = np.empty(n0)
    for j in range(n1):
        d_prime[0] = C[0, j] * inv_pivot_0[0]
        for i in range(1, n0):
            d_prime[i] = (C[i, j] + r * d_prime[i - 1]) * inv_pivot_0[i]
        tmp[n0 - 1, j] = d_prime[n0 - 1]
        for i in range(n0 - 2, -1, -1):
            tmp[i, j] = d_prime[i] - c_prime_0[i] * tmp[i + 1, j]
    out = np.empty_like(C)
    d_prime = np.empty(n1)
    for i in range(n0):
        d_prime[0] = tmp[i, 0] * inv_pivot_1[0]
        for j in range(1, n1):
            d_prime[j] = (tmp[i, j] + r * d_prime[j - 1]) * inv_pivot_1[j]
        out[i, n1 - 1] = d_prime[n1 - 1]
        for j in range(n1 - 2, -1, -1):
            out[i, j] = d_prime[j] - c_prime_1[j] * out[i, j + 1]
    return out
