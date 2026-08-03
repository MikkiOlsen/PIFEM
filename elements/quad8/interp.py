from __future__ import annotations
import numpy as np

from .quad8 import nquad8, dnquad8
from core.utils import find_nearest_node, get_node_dof, _DOF_MAP


def _element_bboxes(T_nodes: np.ndarray, Xc: np.ndarray):
    ex = Xc[T_nodes, 0]
    ey = Xc[T_nodes, 1]
    return ex.min(axis=1), ex.max(axis=1), ey.min(axis=1), ey.max(axis=1)


def _invert_isoparametric(Xe: np.ndarray, point: np.ndarray, max_iter: int = 50, tol: float = 1e-10) -> tuple[float, float, bool]:
    xi, eta = 0.0, 0.0
    for _ in range(max_iter):
        N  = nquad8(xi, eta)
        dN = dnquad8(xi, eta)
        x_iso = N @ Xe
        r = x_iso - point
        if np.dot(r, r) < tol * tol:
            return xi, eta, True
        J = dN @ Xe                      # J[i, j] = d(x_j) / d(xi_i)
        try:
            d_xi = -np.linalg.solve(J.T, r)
        except np.linalg.LinAlgError:
            return xi, eta, False
        xi  += d_xi[0]
        eta += d_xi[1]
    N = nquad8(xi, eta)
    return xi, eta, bool(np.dot(N @ Xe - point, N @ Xe - point) < tol * tol)


def quad8_point_weights(T: np.ndarray, X: np.ndarray, point, dof: str | int = 'w', dofs_per_node: int = 3, bbox_tol: float = 1e-6, edge_tol: float = 1e-6):
    pt = np.asarray(point, dtype=np.float64)[:2]
    T_nodes = (T[:, 0:8] - 1).astype(np.int64)
    Xc = np.ascontiguousarray(X[:, :2], dtype=np.float64)
    xmin, xmax, ymin, ymax = _element_bboxes(T_nodes, Xc)

    span = max(xmax.max() - xmin.min(), ymax.max() - ymin.min(), 1.0)
    tol = bbox_tol * span
    candidates = np.where(
        (xmin - tol <= pt[0]) & (pt[0] <= xmax + tol) &
        (ymin - tol <= pt[1]) & (pt[1] <= ymax + tol)
    )[0]

    for e in candidates:
        Xe = Xc[T_nodes[e]]
        xi, eta, converged = _invert_isoparametric(Xe, pt)
        if converged and abs(xi) <= 1.0 + edge_tol and abs(eta) <= 1.0 + edge_tol:
            weights = nquad8(xi, eta)
            offset = _DOF_MAP[dof] if isinstance(dof, str) else int(dof)
            global_dofs = T_nodes[e] * dofs_per_node + offset
            return global_dofs, weights

    node = find_nearest_node(X, pt)
    global_dofs = np.zeros(8, dtype=np.int64)
    global_dofs[0] = get_node_dof(node, dof, dofs_per_node)
    weights = np.zeros(8, dtype=np.float64)
    weights[0] = 1.0
    return global_dofs, weights


def interpolate_field(T: np.ndarray, X: np.ndarray, U: np.ndarray, point, dof: str | int = 'w', dofs_per_node: int = 3) -> float:
    global_dofs, weights = quad8_point_weights(T, X, point, dof, dofs_per_node)
    return float(weights @ U[global_dofs])
