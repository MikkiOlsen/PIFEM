from __future__ import annotations
import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from .quad8 import quad8_gauss_points, _REF_GP_N, _REF_GP_DN, _REF_GP_W
from numba import njit, prange


@njit(cache=True, parallel=True)
def _build_foundation_triplets(T: np.ndarray, X: np.ndarray, ks: float, Gs: float, dofs_per_node: int, N_all: np.ndarray, dN_all: np.ndarray, W_all: np.ndarray):
    n_elem = T.shape[0]
    rows = np.empty(n_elem * 64, dtype=np.int32)
    cols = np.empty(n_elem * 64, dtype=np.int32)
    data = np.empty(n_elem * 64, dtype=np.float64)

    for i in prange(n_elem):
        node_indices = T[i, 0:8] - 1
        Xe = np.ascontiguousarray(X[node_indices, 0:2])
        Ke_f = np.zeros((8, 8))

        for N, dN_global, detJ, weight in quad8_gauss_points(Xe, N_all, dN_all, W_all):
            Ke_f += (
                ks * np.outer(N, N)
                + Gs * (
                    np.outer(dN_global[0], dN_global[0])
                    + np.outer(dN_global[1], dN_global[1])
                )
            ) * detJ * weight

        ig = (T[i, :8] - 1) * dofs_per_node
        for r in range(8):
            for c in range(8):
                idx = i * 64 + r * 8 + c
                rows[idx] = ig[r]
                cols[idx] = ig[c]
                data[idx] = Ke_f[r, c]

    return rows, cols, data


def apply_foundation(K: np.ndarray | spmatrix, T: np.ndarray, X: np.ndarray, ks: float, Gs: float = 0.0, dofs_per_node: int = 3, n_pts: int = 3) -> np.ndarray | spmatrix:
    if ks < 0:
        raise ValueError(f"Winkler modulus ks must be non-negative, got {ks}")
    if Gs < 0:
        raise ValueError(f"Pasternak shear modulus Gs must be non-negative, got {Gs}")

    n_nodes = X.shape[0]
    max_node_ref = T[:, :8].max()
    if max_node_ref > n_nodes:
        raise ValueError(
            f"Topology references node {max_node_ref} but X only has {n_nodes} nodes"
        )
    if n_pts not in _REF_GP_N:
        raise ValueError(f"n_pts must be 2 or 3, got {n_pts}")

    ndof = n_nodes * dofs_per_node
    rows, cols, data = _build_foundation_triplets(
        T, X, ks, Gs, dofs_per_node,
        _REF_GP_N[n_pts], _REF_GP_DN[n_pts], _REF_GP_W[n_pts],
    )

    K_f = csr_matrix((data, (rows, cols)), shape=(ndof, ndof))

    if isinstance(K, np.ndarray):
        return K + K_f.toarray()
    return K + K_f
