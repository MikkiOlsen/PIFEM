import numpy as np
from .quad8 import _DN_NODES, _N_NODES

_IDX8 = np.arange(8)


def compute_moments_quad8(T, X, H, U):
    num_nodes = len(X)
    Mx = np.zeros(num_nodes)
    My = np.zeros(num_nodes)
    Mxy = np.zeros(num_nodes)
    count = np.zeros(num_nodes)

    _db_cache = {}

    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        Xe = X[node_indices, 0:2]
        prop_idx = T[i, 8] - 1
        He = H[prop_idx, :]

        if prop_idx not in _db_cache:
            E, nu, t = He[0], He[1], He[2]
            _db_cache[prop_idx] = (E * t**3 / (12 * (1 - nu**2))) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu)/2]
            ])
        Db = _db_cache[prop_idx]

        dof_indices = (node_indices[:, None] * 3 + np.arange(3)).ravel()
        Ue = U[dof_indices]

        for n, dN in enumerate(_DN_NODES):
            J = dN @ Xe
            dN_global = np.linalg.solve(J, dN)

            Bb = np.zeros((3, 24))
            Bb[0, 3*_IDX8 + 1] = dN_global[0]
            Bb[1, 3*_IDX8 + 2] = dN_global[1]
            Bb[2, 3*_IDX8 + 1] = dN_global[1]
            Bb[2, 3*_IDX8 + 2] = dN_global[0]

            M = Db @ (Bb @ Ue)

            node_idx = node_indices[n]
            Mx[node_idx] += M[0]
            My[node_idx] += M[1]
            Mxy[node_idx] += M[2]
            count[node_idx] += 1

    Mx /= count
    My /= count
    Mxy /= count

    return Mx, My, Mxy


def compute_shear_quad8(T, X, H, U):
    num_nodes = len(X)
    Qx = np.zeros(num_nodes)
    Qy = np.zeros(num_nodes)
    count = np.zeros(num_nodes)

    _ds_cache = {}

    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        Xe = X[node_indices, 0:2]
        prop_idx = T[i, 8] - 1
        He = H[prop_idx, :]

        if prop_idx not in _ds_cache:
            E, nu, t = He[0], He[1], He[2]
            k_shear = 5 / 6
            G = E / (2 * (1 + nu))
            _ds_cache[prop_idx] = k_shear * G * t
        Ds = _ds_cache[prop_idx]

        dof_indices = (node_indices[:, None] * 3 + np.arange(3)).ravel()
        Ue = U[dof_indices]

        for n, (N, dN) in enumerate(zip(_N_NODES, _DN_NODES)):
            J = dN @ Xe
            dN_global = np.linalg.solve(J, dN)

            Bs = np.zeros((2, 24))
            Bs[0, 3*_IDX8]     = dN_global[0]
            Bs[0, 3*_IDX8 + 1] = N
            Bs[1, 3*_IDX8]     = dN_global[1]
            Bs[1, 3*_IDX8 + 2] = N

            Q = Ds * (Bs @ Ue)

            node_idx = node_indices[n]
            Qx[node_idx] += Q[0]
            Qy[node_idx] += Q[1]
            count[node_idx] += 1

    Qx /= count
    Qy /= count

    return Qx, Qy


def compute_surface_strains_quad8(T, X, U, z_surface, dof=3):
    num_nodes = len(X)
    kappa_x = np.zeros(num_nodes)
    kappa_y = np.zeros(num_nodes)
    kappa_xy = np.zeros(num_nodes)
    count = np.zeros(num_nodes)

    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        Xe = X[node_indices, 0:2]

        dof_indices = (node_indices[:, None] * dof + np.arange(dof)).ravel()
        Ue = U[dof_indices]

        for n, dN in enumerate(_DN_NODES):
            J = dN @ Xe
            dN_global = np.linalg.solve(J, dN)

            Bb = np.zeros((3, 24))
            Bb[0, dof*_IDX8 + 1] = dN_global[0]
            Bb[1, dof*_IDX8 + 2] = dN_global[1]
            Bb[2, dof*_IDX8 + 1] = dN_global[1]
            Bb[2, dof*_IDX8 + 2] = dN_global[0]

            kappa = Bb @ Ue

            node_idx = node_indices[n]
            kappa_x[node_idx] += kappa[0]
            kappa_y[node_idx] += kappa[1]
            kappa_xy[node_idx] += kappa[2]
            count[node_idx] += 1

    count = np.maximum(count, 1)
    kappa_x /= count
    kappa_y /= count
    kappa_xy /= count

    return z_surface * kappa_x, z_surface * kappa_y, z_surface * kappa_xy


def calculate_lte(U, node_pairs, load_node, dofs_per_node: int = 3):
    partner_node = None
    for nA, nB in node_pairs:
        if nA == load_node:
            partner_node = nB
            break
        elif nB == load_node:
            partner_node = nA
            break

    if partner_node is None:
        return None

    w_loaded = abs(U[load_node * dofs_per_node])
    w_unloaded = abs(U[partner_node * dofs_per_node])

    if w_loaded == 0:
        return 0.0

    return (w_unloaded / w_loaded) * 100.0
