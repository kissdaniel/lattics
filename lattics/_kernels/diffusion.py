import numba
import numpy as np
from numba import float32


@numba.jit(float32[:](float32[:], float32[:], float32[:], float32[:]), nopython=True, cache=True)
def tridiag_solver(sub, diag, sup, const):
    N = len(diag)
    sol = np.zeros(N, dtype='float32')
    for i in range(1, N):
        w = sub[i] / diag[i - 1]
        diag[i] = diag[i] - w * sup[i - 1]
        const[i] = const[i] - w * const[i - 1]
    sol[N - 1] = const[N - 1] / diag[N - 1]
    for i in range(N - 2, -1, -1):
        sol[i] = (const[i] - sup[i] * sol[i + 1]) / diag[i]
    return sol


@numba.jit(float32[:, :](float32[:, :]), nopython=True, cache=True)
def apply_neumann_bc_2d(array):
    N, M = array.shape
    padded = np.zeros((N + 2, M + 2), dtype=array.dtype)

    for i in range(N):
        for j in range(M):
            padded[i + 1, j + 1] = array[i, j]
    for j in range(M):
        padded[0, j + 1] = array[0, j]
        padded[-1, j + 1] = array[-1, j]
    for i in range(N):
        padded[i + 1, 0] = array[i, 0]
        padded[i + 1, -1] = array[i, -1]
    return padded


@numba.njit(float32[:, :](float32[:, :], float32, float32, float32, float32), cache=True)
def diffusion_solver_ftcs_2d(C, D, d, dt, dx):
    shape_y = C.shape[0]
    shape_x = C.shape[1]
    C_tmp = apply_neumann_bc_2d(C)
    C_old = np.copy(C_tmp)
    r = D * dt / (dx**2)
    for idx, c in np.ndenumerate(C_old):
        if idx[0] == 0 or idx[1] == 0 or idx[0] == shape_y + 1 or idx[1] == shape_x + 1:
            continue
        y, x = idx
        dCdy = C_old[y - 1, x] - 2 * C_old[y, x] + C_old[y + 1, x]
        dCdx = C_old[y, x - 1] - 2 * C_old[y, x] + C_old[y, x + 1]
        C_tmp[y, x] = C_old[y, x] + r * (dCdy + dCdx) - C_old[y, x] * d * dt
    return C_tmp[1:-1, 1:-1]


@numba.jit(float32[:, :](float32[:, :], float32, float32, float32, float32), nopython=True, cache=True)
def diffusion_solver_lod_2d(C, D, d, dt, h):
    r1 = D * dt / (h**2)
    r2 = 0.5 * d * dt
    C_tmp = apply_neumann_bc_2d(C)
    for j in range(C.shape[1]):
        N = C_tmp.shape[0]
        rhs = C_tmp[:, j]
        subdiag = np.float32(-r1) * np.ones(N, dtype='float32')
        diag = np.float32(1 + 2 * r1 + r2) * np.ones(N, dtype='float32')
        diag[0] = np.float32(1 + r1 + r2)
        diag[N - 1] = np.float32(1 + r1 + r2)
        superdiag = np.float32(-r1) * np.ones(N, dtype='float32')
        C_tmp[:, j] = tridiag_solver(subdiag, diag, superdiag, rhs)
    for i in range(C_tmp.shape[0]):
        N = C_tmp.shape[1]
        rhs = C_tmp[i, :]
        subdiag = np.float32(-r1) * np.ones(N, dtype='float32')
        diag = np.float32(1 + 2 * r1 + r2) * np.ones(N, dtype='float32')
        diag[0] = np.float32(1 + r1 + r2)
        diag[N - 1] = np.float32(1 + r1 + r2)
        superdiag = np.float32(-r1) * np.ones(N, dtype='float32')
        C_tmp[i, :] = tridiag_solver(subdiag, diag, superdiag, rhs)
    return C_tmp[1:-1, 1:-1]
