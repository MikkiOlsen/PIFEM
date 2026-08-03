import numpy as np
from numba import njit
from numba.typed import List

@njit(cache = True)
def find_transverse_joint_pairs(X: np.ndarray, L_s: float, jx: float, num_x: int, tol: float = 1e-5):
    pairs = List()
    if num_x < 2:
        return pairs
    col0 = np.ascontiguousarray(X[:, 0])
    for sx in range(1, num_x):
        x_left = sx * L_s + (sx - 1) * jx
        x_right = sx * (L_s + jx)
        nodes_left = np.where(np.isclose(col0, x_left, atol=tol))[0]
        nodes_right = np.where(np.isclose(col0, x_right, atol=tol))[0]
        for i in nodes_left:
            for j in nodes_right:
                if np.allclose(X[i, 1:], X[j, 1:], atol=tol):
                    pairs.append((i, j))
                    break
    return pairs

@njit(cache = True)
def find_longitudinal_joint_pairs(X: np.ndarray, W_s: float, jy: float, num_y: int, tol: float = 1e-5):
    pairs = List()
    if num_y < 2:
        return pairs
    col0 = np.ascontiguousarray(X[:, 0])
    col1 = np.ascontiguousarray(X[:, 1])
    col2 = np.ascontiguousarray(X[:, 2])
    for sy in range(1, num_y):
        y_bottom = sy * W_s + (sy - 1) * jy
        y_top = sy * (W_s + jy)
        nodes_bottom = np.where(np.isclose(col1, y_bottom, atol=tol))[0]
        nodes_top = np.where(np.isclose(col1, y_top, atol=tol))[0]
        for i in nodes_bottom:
            for j in nodes_top:
                if np.isclose(col0[i], col0[j], atol=tol) and np.isclose(col2[i], col2[j], atol=tol):
                    pairs.append((i, j))
                    break
    return pairs