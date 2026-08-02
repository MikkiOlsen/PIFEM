from __future__ import annotations
from collections.abc import Generator
import numpy as np
from numba import njit, types
from numba.typed import Dict

_GAUSS_POINTS = Dict.empty(
    key_type=types.int64,
    value_type=types.UniTuple(types.float64[:], 2)
)
_GAUSS_POINTS[2] = (
    np.array([-1/np.sqrt(3), 1/np.sqrt(3)], dtype=np.float64),
    np.array([1.0, 1.0], dtype=np.float64)
)
_GAUSS_POINTS[3] = (
    np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)], dtype=np.float64),
    np.array([5/9, 8/9, 5/9], dtype=np.float64)
)

@njit(cache=True)
def nquad8(xi: float, eta: float) -> np.ndarray:
    N = np.zeros(8)
    N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
    N[1] = 0.25 * (1 + xi) * (1 - eta) * ( xi - eta - 1)
    N[2] = 0.25 * (1 + xi) * (1 + eta) * ( xi + eta - 1)
    N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
    N[4] = 0.5 * (1 - xi**2) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta**2)
    N[6] = 0.5 * (1 - xi**2) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta**2)
    return N

@njit(cache=True)
def dnquad8(xi: float, eta: float) -> np.ndarray:
    dN = np.zeros((2, 8))
    dN[0, 0] = 0.25 * (1 - eta) * (2*xi + eta)
    dN[0, 1] = 0.25 * (1 - eta) * (2*xi - eta)
    dN[0, 2] = 0.25 * (1 + eta) * (2*xi + eta)
    dN[0, 3] = 0.25 * (1 + eta) * (2*xi - eta)
    dN[0, 4] = -xi * (1 - eta)
    dN[0, 5] = 0.5 * (1 - eta**2)
    dN[0, 6] = -xi * (1 + eta)
    dN[0, 7] = -0.5 * (1 - eta**2)
    dN[1, 0] = 0.25 * (1 - xi) * ( xi + 2*eta)
    dN[1, 1] = 0.25 * (1 + xi) * (-xi + 2*eta)
    dN[1, 2] = 0.25 * (1 + xi) * ( xi + 2*eta)
    dN[1, 3] = 0.25 * (1 - xi) * (-xi + 2*eta)
    dN[1, 4] = -0.5 * (1 - xi**2)
    dN[1, 5] = -eta * (1 + xi)
    dN[1, 6] = 0.5 * (1 - xi**2)
    dN[1, 7] = -eta * (1 - xi)
    return dN

_REF_GP_N  = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
_REF_GP_DN = Dict.empty(key_type=types.int64, value_type=types.float64[:, :, :])
_REF_GP_W  = Dict.empty(key_type=types.int64, value_type=types.float64[:])

for _n in (2, 3):
    _gp, _w = _GAUSS_POINTS[_n]
    _entries = [
        (nquad8(_xi, _eta), dnquad8(_xi, _eta), _w[_ix] * _w[_iy])
        for _ix, _xi in enumerate(_gp)
        for _iy, _eta in enumerate(_gp)
    ]
    _REF_GP_N[_n]  = np.stack([e[0] for e in _entries])                    # (n_pts, 8)
    _REF_GP_DN[_n] = np.stack([e[1] for e in _entries])                    # (n_pts, 2, 8)
    _REF_GP_W[_n]  = np.array([e[2] for e in _entries], dtype=np.float64)  # (n_pts)



@njit(cache=True)
def quad8_gauss_points(Xe: np.ndarray, N_all: np.ndarray, dN_all: np.ndarray, Weight: np.ndarray):
    for i in range(Weight.shape[0]):
        N  = N_all[i]
        dN = dN_all[i]
        J         = dN @ Xe
        detJ      = np.linalg.det(J)
        dN_global = np.ascontiguousarray(np.linalg.solve(J, dN))
        yield N, dN_global, detJ, Weight[i]

@njit(cache = True)
def quad8_gauss_points_force(Xe: np.ndarray, N_all: np.ndarray, dN_all: np.ndarray, Weight: np.ndarray):
    for i in range(Weight.shape[0]):
        N  = N_all[i]
        dN = dN_all[i]
        J    = dN @ Xe
        detJ = np.linalg.det(J)
        yield N, detJ, Weight[i]

_NODE_COORDS: list[tuple[float, float]] = [
    (-1, -1), (1, -1), (1, 1), (-1, 1),
    (0, -1),  (1, 0),  (0, 1), (-1, 0),
]
_DN_NODES: list[np.ndarray] = [dnquad8(xi, eta) for xi, eta in _NODE_COORDS]
_N_NODES:  list[np.ndarray] = [nquad8(xi, eta)  for xi, eta in _NODE_COORDS]