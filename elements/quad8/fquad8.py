import numpy as np
from .quad8 import _REF_GP_N, _REF_GP_DN, _REF_GP_W
from numba import njit, prange


@njit(cache=True)
def _consistent_plate_load(T_nodes, X, q, use_patch, center, radius, dof, N_all, dN_all, W_all, elem_xmin, elem_xmax, elem_ymin, elem_ymax):
    n_nodes = X.shape[0]
    n_gp    = W_all.shape[0]
    F  = np.zeros(n_nodes * dof)
    r2 = radius * radius
    tol = radius * 0.05

    for e in range(T_nodes.shape[0]):
        if use_patch:
            if (elem_xmax[e] < center[0] - radius - tol or
                    elem_xmin[e] > center[0] + radius + tol or
                    elem_ymax[e] < center[1] - radius - tol or
                    elem_ymin[e] > center[1] + radius + tol):
                continue

        Xe = np.empty((8, 2))
        for k in range(8):
            nk = T_nodes[e, k]
            Xe[k, 0] = X[nk, 0]
            Xe[k, 1] = X[nk, 1]

        fe = np.zeros(8)
        for ig in range(n_gp):
            N  = N_all[ig]
            dN = dN_all[ig]
            J  = dN @ Xe
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if use_patch:
                x_gp = N @ Xe
                dx = x_gp[0] - center[0]
                dy = x_gp[1] - center[1]
                if dx * dx + dy * dy > r2:
                    continue
            fe += q * N * (detJ * W_all[ig])

        for k in range(8):
            F[T_nodes[e, k] * dof] += fe[k]

    return np.ascontiguousarray(F)


def consistent_plate_load(T: np.ndarray, X: np.ndarray, q: float, center: np.ndarray = None, radius: float = None, dof: int = 3) -> np.ndarray:
    use_patch = (center is not None) and (radius is not None)
    c = np.asarray(center, dtype=np.float64) if use_patch else np.zeros(2)
    r = float(radius) if use_patch else 0.0
    T_nodes = (T[:, 0:8] - 1).astype(np.int64)
    Xc      = np.ascontiguousarray(X[:, :2], dtype=np.float64)
    elem_xmin = Xc[T_nodes, 0].min(axis=1)
    elem_xmax = Xc[T_nodes, 0].max(axis=1)
    elem_ymin = Xc[T_nodes, 1].min(axis=1)
    elem_ymax = Xc[T_nodes, 1].max(axis=1)
    return _consistent_plate_load(T_nodes, Xc, q, use_patch, c, r, dof, _REF_GP_N[3], _REF_GP_DN[3], _REF_GP_W[3], elem_xmin, elem_xmax, elem_ymin, elem_ymax)


def apply_tire_load(T: np.ndarray, X: np.ndarray, load: float, tire_pressure: float, center: np.ndarray, dof: int = 3, out: np.ndarray = None) -> np.ndarray:
    area   = load / tire_pressure
    radius = np.sqrt(area / np.pi)
    result = consistent_plate_load(T, X, -tire_pressure, center=center, radius=radius, dof=dof)
    if out is not None:
        out += result
        return out
    return result


@njit(cache=True)
def _disk_anti(x, r):
    if x <= -r:
        x = -r
    elif x >= r:
        x = r
    return 0.5 * (x * np.sqrt(r * r - x * x) + r * r * np.arcsin(x / r))


@njit(cache=True)
def _circle_rect_overlap(cx, cy, r, x0, x1, y0, y1):
    ax0 = x0 - cx; ax1 = x1 - cx
    ay0 = y0 - cy; ay1 = y1 - cy

    xa = max(ax0, -r); xb = min(ax1, r)
    if xb <= xa:
        return 0.0

    bp = np.empty(6, dtype=np.float64)
    nb = 0
    bp[nb] = xa; nb += 1
    bp[nb] = xb; nb += 1


    if -r <= ay1 <= r:
        t = np.sqrt(r * r - ay1 * ay1)
        bp[nb] = -t; nb += 1
        bp[nb] =  t; nb += 1
    if -r <= ay0 <= r:
        t = np.sqrt(r * r - ay0 * ay0)
        bp[nb] = -t; nb += 1
        bp[nb] =  t; nb += 1
    bp = np.sort(bp[:nb])

    area = 0.0
    for i in range(nb - 1):
        p = bp[i]; q = bp[i + 1]
        if q <= xa or p >= xb or q - p <= 0.0:
            continue
        m  = 0.5 * (p + q)
        sm = np.sqrt(max(0.0, r * r - m * m))         # chord half-height at midpoint
        upper = min(ay1, sm)
        lower = max(ay0, -sm)
        if upper <= lower:
            continue
        if sm >= ay1:                                  # upper clamp = ay1 (constant)
            int_up = ay1 * (q - p)
        else:                                          # upper clamp = +sqrt(...)
            int_up = _disk_anti(q, r) - _disk_anti(p, r)
        if -sm <= ay0:                                 # lower clamp = ay0 (constant)
            int_lo = ay0 * (q - p)
        else:                                          # lower clamp = -sqrt(...)
            int_lo = -(_disk_anti(q, r) - _disk_anti(p, r))
        area += int_up - int_lo
    return area


@njit(cache=True)
def _build_single_force_vector(centroid, elem_intN, elem_area, ref_x, wheel_x_off, wheel_y, wheel_q, wheel_r, T_nodes, dof, n_nodes, elem_xmin, elem_xmax, elem_ymin, elem_ymax):
    n_wheels = wheel_x_off.shape[0]
    n_elem   = T_nodes.shape[0]
    F        = np.zeros(n_nodes * dof)

    for w in range(n_wheels):
        cx = ref_x + wheel_x_off[w]
        cy = wheel_y[w]
        r  = wheel_r[w]
        # wheel_q is the (signed) contact pressure; total force = pressure * pi r^2.
        # Distribute it by continuous circle-element overlap, force-conserving.
        force = wheel_q[w] * np.pi * r * r
        tol = r * 0.05

        total_cov  = 0.0
        nearest_e  = 0
        nearest_d2 = 1e30

        for e in range(n_elem):
            if (elem_xmax[e] < cx - r - tol or elem_xmin[e] > cx + r + tol or
                    elem_ymax[e] < cy - r - tol or elem_ymin[e] > cy + r + tol):
                continue
            dx = centroid[e, 0] - cx
            dy = centroid[e, 1] - cy
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_e  = e
            ov = _circle_rect_overlap(cx, cy, r, elem_xmin[e], elem_xmax[e], elem_ymin[e], elem_ymax[e])
            if ov > 0.0:
                bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                total_cov += (ov / bbox) * elem_area[e]

        if total_cov == 0.0:
            qn = force / elem_area[nearest_e]
            for k in range(8):
                F[T_nodes[nearest_e, k] * dof] += qn * elem_intN[nearest_e, k]
        else:
            qn = force / total_cov
            for e in range(n_elem):
                if (elem_xmax[e] < cx - r - tol or elem_xmin[e] > cx + r + tol or
                        elem_ymax[e] < cy - r - tol or elem_ymin[e] > cy + r + tol):
                    continue
                ov = _circle_rect_overlap(cx, cy, r, elem_xmin[e], elem_xmax[e], elem_ymin[e], elem_ymax[e])
                if ov > 0.0:
                    bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                    frac = ov / bbox
                    for k in range(8):
                        F[T_nodes[e, k] * dof] += qn * frac * elem_intN[e, k]

    return F


def build_single_tire_force_vector(T: np.ndarray, X: np.ndarray, wheel_x_off: np.ndarray, wheel_y: np.ndarray, wheel_q: np.ndarray, wheel_r: np.ndarray, ref_x: float, dof: int = 3) -> np.ndarray:
    T_nodes, Xc, xmin, xmax, ymin, ymax, centroid, elem_intN, elem_area = _block_precompute(T, X)
    return _build_single_force_vector(
        centroid, elem_intN, elem_area, ref_x,
        np.ascontiguousarray(wheel_x_off, dtype=np.float64),
        np.ascontiguousarray(wheel_y,     dtype=np.float64),
        np.ascontiguousarray(wheel_q,     dtype=np.float64),
        np.ascontiguousarray(wheel_r,     dtype=np.float64),
        T_nodes, dof, Xc.shape[0], xmin, xmax, ymin, ymax,
    )


@njit(cache=True, parallel=True)
def _build_all_force_vectors(centroid, elem_intN, elem_area, ref_x_arr, wheel_x_off, wheel_y, wheel_q, wheel_r, T_nodes, dof, n_nodes, elem_xmin, elem_xmax, elem_ymin, elem_ymax):
    n_steps  = ref_x_arr.shape[0]
    n_wheels = wheel_x_off.shape[0]
    n_elem   = T_nodes.shape[0]
    F_all    = np.zeros((n_steps, n_nodes * dof))

    for si in prange(n_steps):
        ref_x = ref_x_arr[si]
        for w in range(n_wheels):
            cx = ref_x + wheel_x_off[w]
            cy = wheel_y[w]
            r  = wheel_r[w]
            force = wheel_q[w] * np.pi * r * r   # pressure * area = total wheel force
            tol = r * 0.05

            total_cov  = 0.0
            nearest_e  = 0
            nearest_d2 = 1e30

            for e in range(n_elem):
                if (elem_xmax[e] < cx - r - tol or elem_xmin[e] > cx + r + tol or
                        elem_ymax[e] < cy - r - tol or elem_ymin[e] > cy + r + tol):
                    continue
                dx = centroid[e, 0] - cx
                dy = centroid[e, 1] - cy
                d2 = dx * dx + dy * dy
                if d2 < nearest_d2:
                    nearest_d2 = d2
                    nearest_e  = e
                ov = _circle_rect_overlap(cx, cy, r, elem_xmin[e], elem_xmax[e], elem_ymin[e], elem_ymax[e])
                if ov > 0.0:
                    bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                    total_cov += (ov / bbox) * elem_area[e]

            if total_cov == 0.0:
                qn = force / elem_area[nearest_e]
                for k in range(8):
                    F_all[si, T_nodes[nearest_e, k] * dof] += qn * elem_intN[nearest_e, k]
            else:
                qn = force / total_cov
                for e in range(n_elem):
                    if (elem_xmax[e] < cx - r - tol or elem_xmin[e] > cx + r + tol or
                            elem_ymax[e] < cy - r - tol or elem_ymin[e] > cy + r + tol):
                        continue
                    ov = _circle_rect_overlap(cx, cy, r, elem_xmin[e], elem_xmax[e], elem_ymin[e], elem_ymax[e])
                    if ov > 0.0:
                        bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                        frac = ov / bbox
                        for k in range(8):
                            F_all[si, T_nodes[e, k] * dof] += qn * frac * elem_intN[e, k]

    return F_all


def build_all_tire_force_vectors(T: np.ndarray, X: np.ndarray, wheel_x_off: np.ndarray, wheel_y: np.ndarray, wheel_q: np.ndarray, wheel_r: np.ndarray, ref_x_arr: np.ndarray, dof: int = 3) -> np.ndarray:  # noqa: E501
    T_nodes, Xc, xmin, xmax, ymin, ymax, centroid, elem_intN, elem_area = _block_precompute(T, X)
    return _build_all_force_vectors(
        centroid, elem_intN, elem_area,
        np.ascontiguousarray(ref_x_arr,   dtype=np.float64),
        np.ascontiguousarray(wheel_x_off, dtype=np.float64),
        np.ascontiguousarray(wheel_y,     dtype=np.float64),
        np.ascontiguousarray(wheel_q,     dtype=np.float64),
        np.ascontiguousarray(wheel_r,     dtype=np.float64),
        T_nodes, dof, Xc.shape[0], xmin, xmax, ymin, ymax,
    )

@njit(cache=True)
def _precompute_block_geometry(T_nodes, X, N_all, dN_all, W_all):
    n_elem = T_nodes.shape[0]
    n_gp   = W_all.shape[0]

    elem_intN = np.zeros((n_elem, 8))
    centroid  = np.zeros((n_elem, 2))

    for e in range(n_elem):
        for k in range(4):
            nk = T_nodes[e, k]
            centroid[e, 0] += X[nk, 0]
            centroid[e, 1] += X[nk, 1]
        centroid[e, 0] *= 0.25
        centroid[e, 1] *= 0.25

        Xe = np.empty((8, 2))
        for k in range(8):
            nk = T_nodes[e, k]
            Xe[k, 0] = X[nk, 0]
            Xe[k, 1] = X[nk, 1]

        for ig in range(n_gp):
            N  = N_all[ig]
            dN = dN_all[ig]
            J  = dN @ Xe
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            w    = W_all[ig]
            for k in range(8):
                elem_intN[e, k] += N[k] * detJ * w

    elem_area = np.zeros(n_elem)
    for e in range(n_elem):
        for k in range(8):
            elem_area[e] += elem_intN[e, k]

    return centroid, elem_intN, elem_area


@njit(cache=True, parallel=True)
def _build_all_block_force_vectors(centroid, elem_intN, elem_area, ref_x_arr, wheel_x_off, wheel_y, wheel_force, wheel_half, T_nodes, dof, n_nodes, elem_xmin, elem_xmax, elem_ymin, elem_ymax):
    n_steps  = ref_x_arr.shape[0]
    n_wheels = wheel_x_off.shape[0]
    n_elem   = T_nodes.shape[0]
    F_all    = np.zeros((n_steps, n_nodes * dof))

    for si in prange(n_steps):
        ref_x = ref_x_arr[si]
        for w in range(n_wheels):
            cx   = ref_x + wheel_x_off[w]
            cy   = wheel_y[w]
            half = wheel_half[w]
            tol  = half * 0.5
            lo_x = cx - half; hi_x = cx + half
            lo_y = cy - half; hi_y = cy + half

            total_cov  = 0.0
            nearest_e  = 0
            nearest_d2 = 1e30

            for e in range(n_elem):
                if (elem_xmax[e] < cx - half - tol or elem_xmin[e] > cx + half + tol or
                        elem_ymax[e] < cy - half - tol or elem_ymin[e] > cy + half + tol):
                    continue
                dx = centroid[e, 0] - cx
                dy = centroid[e, 1] - cy
                d2 = dx * dx + dy * dy
                if d2 < nearest_d2:
                    nearest_d2 = d2
                    nearest_e  = e
                ox = min(elem_xmax[e], hi_x) - max(elem_xmin[e], lo_x)
                oy = min(elem_ymax[e], hi_y) - max(elem_ymin[e], lo_y)
                if ox > 0.0 and oy > 0.0:
                    bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                    total_cov += (ox * oy / bbox) * elem_area[e]

            if total_cov == 0.0:
                # Footprint smaller than / outside every nearby element bbox:
                # fall back to loading the single nearest element.
                q = -wheel_force[w] / elem_area[nearest_e]
                for k in range(8):
                    F_all[si, T_nodes[nearest_e, k] * dof] += q * elem_intN[nearest_e, k]
            else:
                q = -wheel_force[w] / total_cov
                for e in range(n_elem):
                    if (elem_xmax[e] < cx - half - tol or elem_xmin[e] > cx + half + tol or
                            elem_ymax[e] < cy - half - tol or elem_ymin[e] > cy + half + tol):
                        continue
                    ox = min(elem_xmax[e], hi_x) - max(elem_xmin[e], lo_x)
                    oy = min(elem_ymax[e], hi_y) - max(elem_ymin[e], lo_y)
                    if ox > 0.0 and oy > 0.0:
                        bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                        frac = ox * oy / bbox
                        for k in range(8):
                            F_all[si, T_nodes[e, k] * dof] += q * frac * elem_intN[e, k]

    return F_all


@njit(cache=True)
def _build_single_block_force_vector(centroid, elem_intN, elem_area, ref_x, wheel_x_off, wheel_y, wheel_force, wheel_half, T_nodes, dof, n_nodes, elem_xmin, elem_xmax, elem_ymin, elem_ymax):
    n_wheels = wheel_x_off.shape[0]
    n_elem   = T_nodes.shape[0]
    F        = np.zeros(n_nodes * dof)

    for w in range(n_wheels):
        cx   = ref_x + wheel_x_off[w]
        cy   = wheel_y[w]
        half = wheel_half[w]
        tol  = half * 0.5
        lo_x = cx - half; hi_x = cx + half
        lo_y = cy - half; hi_y = cy + half

        total_cov  = 0.0
        nearest_e  = 0
        nearest_d2 = 1e30

        for e in range(n_elem):
            if (elem_xmax[e] < cx - half - tol or elem_xmin[e] > cx + half + tol or
                    elem_ymax[e] < cy - half - tol or elem_ymin[e] > cy + half + tol):
                continue
            dx = centroid[e, 0] - cx
            dy = centroid[e, 1] - cy
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_e  = e
            ox = min(elem_xmax[e], hi_x) - max(elem_xmin[e], lo_x)
            oy = min(elem_ymax[e], hi_y) - max(elem_ymin[e], lo_y)
            if ox > 0.0 and oy > 0.0:
                bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                total_cov += (ox * oy / bbox) * elem_area[e]

        if total_cov == 0.0:
            q = -wheel_force[w] / elem_area[nearest_e]
            for k in range(8):
                F[T_nodes[nearest_e, k] * dof] += q * elem_intN[nearest_e, k]
        else:
            q = -wheel_force[w] / total_cov
            for e in range(n_elem):
                if (elem_xmax[e] < cx - half - tol or elem_xmin[e] > cx + half + tol or
                        elem_ymax[e] < cy - half - tol or elem_ymin[e] > cy + half + tol):
                    continue
                ox = min(elem_xmax[e], hi_x) - max(elem_xmin[e], lo_x)
                oy = min(elem_ymax[e], hi_y) - max(elem_ymin[e], lo_y)
                if ox > 0.0 and oy > 0.0:
                    bbox = (elem_xmax[e] - elem_xmin[e]) * (elem_ymax[e] - elem_ymin[e])
                    frac = ox * oy / bbox
                    for k in range(8):
                        F[T_nodes[e, k] * dof] += q * frac * elem_intN[e, k]

    return F


def _block_precompute(T: np.ndarray, X: np.ndarray):
    T_nodes   = (T[:, 0:8] - 1).astype(np.int64)
    Xc        = np.ascontiguousarray(X[:, :2], dtype=np.float64)
    elem_xmin = Xc[T_nodes, 0].min(axis=1)
    elem_xmax = Xc[T_nodes, 0].max(axis=1)
    elem_ymin = Xc[T_nodes, 1].min(axis=1)
    elem_ymax = Xc[T_nodes, 1].max(axis=1)
    centroid, elem_intN, elem_area = _precompute_block_geometry(
        T_nodes, Xc, _REF_GP_N[3], _REF_GP_DN[3], _REF_GP_W[3]
    )
    return T_nodes, Xc, elem_xmin, elem_xmax, elem_ymin, elem_ymax, centroid, elem_intN, elem_area


def build_all_block_force_vectors(T: np.ndarray, X: np.ndarray, wheel_x_off: np.ndarray, wheel_y: np.ndarray, wheel_force: np.ndarray, wheel_radius: np.ndarray, ref_x_arr: np.ndarray, dof: int = 3) -> np.ndarray:
    T_nodes, Xc, xmin, xmax, ymin, ymax, centroid, elem_intN, elem_area = _block_precompute(T, X)
    wheel_half = (0.5 * np.sqrt(np.pi) * np.asarray(wheel_radius, dtype=np.float64))
    return _build_all_block_force_vectors(
        centroid, elem_intN, elem_area,
        np.ascontiguousarray(ref_x_arr,    dtype=np.float64),
        np.ascontiguousarray(wheel_x_off,  dtype=np.float64),
        np.ascontiguousarray(wheel_y,      dtype=np.float64),
        np.ascontiguousarray(wheel_force,  dtype=np.float64),
        np.ascontiguousarray(wheel_half,   dtype=np.float64),
        T_nodes, dof, X.shape[0],
        xmin, xmax, ymin, ymax,
    )


def build_single_block_force_vector(T: np.ndarray, X: np.ndarray, wheel_x_off: np.ndarray, wheel_y: np.ndarray, wheel_force: np.ndarray, wheel_radius: np.ndarray, ref_x: float, dof: int = 3) -> np.ndarray:
    T_nodes, Xc, xmin, xmax, ymin, ymax, centroid, elem_intN, elem_area = _block_precompute(T, X)
    wheel_half = (0.5 * np.sqrt(np.pi) * np.asarray(wheel_radius, dtype=np.float64))
    return _build_single_block_force_vector(
        centroid, elem_intN, elem_area,
        float(ref_x),
        np.ascontiguousarray(wheel_x_off,  dtype=np.float64),
        np.ascontiguousarray(wheel_y,      dtype=np.float64),
        np.ascontiguousarray(wheel_force,  dtype=np.float64),
        np.ascontiguousarray(wheel_half,   dtype=np.float64),
        T_nodes, dof, X.shape[0],
        xmin, xmax, ymin, ymax,
    )
