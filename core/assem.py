from __future__ import annotations

from numba import njit

import numpy as np
from scipy.sparse import csr_matrix

@njit(cache=True)
def _sparse_assem(T: np.ndarray, Ke_all: np.ndarray, dof: int, enodes: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_elem = T.shape[0]
    edof   = enodes * dof
    node_ids = T[:, :enodes] - 1                                        # (n_elem, enodes)
    global_index = (node_ids[:, :, None] * dof + np.arange(dof)).reshape(n_elem, edof)

    total = n_elem * edof * edof
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    data = Ke_all.reshape(n_elem, edof * edof).ravel()

    #Basically just inserting according to row major format. The same can be accomplished using a combination of Numpy tile and repeat commands, but these are not compatible with Numba, at least not presently. 
    #And using these functions instead result in lower performance as the runtime has to return to the Python interpreter repeatedly instead of staying in the Numba compiled code.
    for i in range(n_elem):
        for r in range(edof):
            for c in range(edof):
                idx = i * edof * edof + r * edof + c
                rows[idx] = global_index[i, r]
                cols[idx] = global_index[i, c]

    return data, rows, cols

def sparse_assem(T: np.ndarray, Ke_all: np.ndarray, n_nodes: int, dof: int) -> csr_matrix:
    data, rows, cols = _sparse_assem(T, Ke_all, dof)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes * dof, n_nodes * dof))