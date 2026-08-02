from __future__ import annotations
import numpy as np
from numba import njit
from .quad8 import quad8_gauss_points, _REF_GP_N, _REF_GP_DN, _REF_GP_W

@njit(cache=True)
def keplate8(Xe: np.ndarray, He: np.ndarray, N_all_3: np.ndarray, dN_all_3: np.ndarray, W_all_3: np.ndarray, N_all_2: np.ndarray, dN_all_2: np.ndarray, W_all_2: np.ndarray) -> np.ndarray:
    E, nu, t = He[0], He[1], He[2]

    D_b = (E * t**3 / (12 * (1 - nu**2))) * np.array([
        [1,  nu,         0],
        [nu,  1,         0],
        [0,   0, (1-nu)/2],
    ])
    k_shear = 5/6
    G   = E / (2 * (1 + nu))
    D_s = k_shear * G * t * np.eye(2)

    Ke    = np.zeros((24, 24))
    _idx  = np.arange(8)

    # Bending (n_pts=3)
    for N, dN_global, detJ, weight in quad8_gauss_points(Xe, N_all_3, dN_all_3, W_all_3):
        Bb = np.zeros((3, 24))
        Bb[0, 3*_idx + 1] = dN_global[0]
        Bb[1, 3*_idx + 2] = dN_global[1]
        Bb[2, 3*_idx + 1] = dN_global[1]
        Bb[2, 3*_idx + 2] = dN_global[0]
        Ke += (Bb.T @ D_b @ Bb) * detJ * weight

    # Shear (n_pts=2)
    for N, dN_global, detJ, weight in quad8_gauss_points(Xe, N_all_2, dN_all_2, W_all_2):
        Bs = np.zeros((2, 24))
        Bs[0, 3*_idx]     = dN_global[0]
        Bs[0, 3*_idx + 1] = N
        Bs[1, 3*_idx]     = dN_global[1]
        Bs[1, 3*_idx + 2] = N
        Ke += (Bs.T @ D_s @ Bs) * detJ * weight

    return Ke