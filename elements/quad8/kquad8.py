from __future__ import annotations
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from .kequad8 import keplate8
from .quad8 import _REF_GP_N, _REF_GP_DN, _REF_GP_W
from core.assem import sparse_assem

@njit(cache=True)
def _kplate8(T: np.ndarray, X: np.ndarray, H: np.ndarray, N_all_3: np.ndarray, dN_all_3: np.ndarray, W_all_3: np.ndarray, N_all_2: np.ndarray, dN_all_2: np.ndarray, W_all_2: np.ndarray, dof: int) -> np.ndarray:
    n_elem = T.shape[0]
    Ke_all = np.empty((n_elem, 24, 24))
    for i in range(n_elem):
        Xe = X[T[i, 0:8] - 1, 0:2]
        He = H[T[i, 8] - 1, :]
        Ke_all[i] = keplate8(
            Xe, He,
            N_all_3, dN_all_3, W_all_3,
            N_all_2, dN_all_2, W_all_2,
        )
    return Ke_all

def kquad8(T: np.ndarray, X: np.ndarray, H: np.ndarray, dof: int = 3) -> csr_matrix:
    n_nodes = X.shape[0]

    if H.ndim != 2 or H.shape[1] != 3:
        raise ValueError(f"H must have shape (n_materials, 3) holding [E, nu, thickness] per row, got {H.shape}")

    propno = T[:, 8]
    bad = (propno < 1) | (propno > H.shape[0])
    if bad.any():
        raise ValueError(f"{bad.sum()} of {len(T)} elements have a property number outside 1..{H.shape[0]}: {np.unique(propno[bad]).tolist()}. T[:, 8] is a 1-based index into the rows of H")

    node_refs = T[:, :8]
    if node_refs.min() < 1 or node_refs.max() > n_nodes:
        raise ValueError(f"Topology references nodes {node_refs.min()}..{node_refs.max()} but X has {n_nodes} nodes; T[:, :8] is 1-based")

    Ke_all = _kplate8(
        T, X, H,
        _REF_GP_N[3], _REF_GP_DN[3], _REF_GP_W[3],
        _REF_GP_N[2], _REF_GP_DN[2], _REF_GP_W[2],
        dof,
    )
    return sparse_assem(T, Ke_all, n_nodes, dof)