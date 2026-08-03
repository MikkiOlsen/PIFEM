from __future__ import annotations

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from numba import njit


@njit(cache=True)
def _apply_joint_springs(node_pairs_arr: np.ndarray, f_arr: np.ndarray, dof_index: int, dofs_per_node: int) -> tuple:
    n = node_pairs_arr.shape[0]

    rows = np.empty(n * 4, dtype=np.int32)
    cols = np.empty(n * 4, dtype=np.int32)
    data = np.empty(n * 4, dtype=np.float64)

    for p in range(n):
        nA = node_pairs_arr[p, 0]
        nB = node_pairs_arr[p, 1]
        iA = nA * dofs_per_node + dof_index
        iB = nB * dofs_per_node + dof_index
        f = f_arr[p]
        base = p * 4
        rows[base]     = iA;  cols[base]     = iA;  data[base]     =  f
        rows[base + 1] = iB;  cols[base + 1] = iB;  data[base + 1] =  f
        rows[base + 2] = iA;  cols[base + 2] = iB;  data[base + 2] = -f
        rows[base + 3] = iB;  cols[base + 3] = iA;  data[base + 3] = -f

    return rows, cols, data


def _trapezoidal_tributary(pairs_arr: np.ndarray, X: np.ndarray, along_axis: int, decimals: int = 6) -> np.ndarray:
    group_axis = 1 - along_axis
    pos   = X[pairs_arr[:, 0], along_axis]
    group = np.round(X[pairs_arr[:, 0], group_axis], decimals)

    tributary = np.empty(len(pairs_arr), dtype=np.float64)
    for key in np.unique(group):
        idx = np.where(group == key)[0]
        order = idx[np.argsort(pos[idx])]
        s = pos[order]
        n = len(s)
        if n == 1:
            tributary[order[0]] = 0.0   # isolated node on this joint - no neighbour to span, no spring
            continue
        t = np.empty(n, dtype=np.float64)
        t[0]  = (s[1] - s[0]) / 2.0
        t[-1] = (s[-1] - s[-2]) / 2.0
        if n > 2:
            t[1:-1] = (s[2:] - s[:-2]) / 2.0
        tributary[order] = t
    return tributary


def apply_joint_springs(K: spmatrix, node_pairs: list, q0: float, dof_index: int = 0, dofs_per_node: int = 3, *, X: np.ndarray, along_axis: int) -> spmatrix:
    if q0 <= 0 or len(node_pairs) == 0:
        return K

    pairs_arr = np.array(node_pairs, dtype=np.int64)
    f_arr = q0 * _trapezoidal_tributary(pairs_arr, X, along_axis)

    rows, cols, data = _apply_joint_springs(pairs_arr, f_arr, dof_index, dofs_per_node)

    K_spring = csr_matrix((data, (rows, cols)), shape=K.shape)

    return K + K_spring
