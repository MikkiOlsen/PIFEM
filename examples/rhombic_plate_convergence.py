import os
import numpy as np
import matplotlib.pyplot as plt

import core
from elements.quad8 import kquad8, consistent_plate_load, interpolate_field
from mesh.gmsh import import_msh

MESHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')

MESH_NAMES = ['rhombic_plate_3_x_3', 'rhombic_plate_8_x_8', 'rhombic_plate_15_x_15', 'rhombic_plate_25_x_25']

t = 0.05
q = -1000.0

material = core.load_material("pavement_concrete")
H = core.make_H(material, t)


def corners(X):
    # The rhombus has a vertex at each coordinate extreme, ordered around
    # the boundary: top, right, bottom, left.
    return [X[np.argmax(X[:, 1])], X[np.argmax(X[:, 0])],
            X[np.argmin(X[:, 1])], X[np.argmin(X[:, 0])]]


def acute_angle(pts):
    top, right, _, left = pts
    v1 = (right - top)[:2]
    v2 = (left - top)[:2]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def solve(name):
    X, T = import_msh(os.path.join(MESHES, name + '.msh'))
    K = kquad8(T, X, H, dof=3)

    FE = core.FESystem(X, dofs_per_node=3)
    pts = corners(X)
    for i in range(4):
        FE.simply_supported_edge(pts[i], pts[(i + 1) % 4], tol=1e-6)

    F = consistent_plate_load(T, X, q)
    U = FE.solve_direct(K, F)

    return T.shape[0], interpolate_field(T, X, U, (0.0, 0.0), dof='w')


if __name__ == "__main__":
    X0, _ = import_msh(os.path.join(MESHES, MESH_NAMES[0] + '.msh'))
    pts = corners(X0)
    side = np.linalg.norm((pts[1] - pts[0])[:2])
    E, nu = material['E'], material['nu']
    D = E * t**3 / (12.0 * (1.0 - nu**2))

    print(f"Rhombic plate, side = {side:.3f} m, acute angle = {acute_angle(pts):.0f}°")
    print(f"{'mesh':<24}{'elements':>10}{'w_centre [um]':>16}{'change':>10} {'w D / q L^4':>14}")

    n_elements, deflections = [], []
    for name in MESH_NAMES:
        n_elem, w = solve(name)
        n_elements.append(n_elem)
        deflections.append(w * 1e6)
        change = ('' if len(deflections) < 2 else f"{abs(deflections[-1] - deflections[-2]) / abs(deflections[-1]) * 100:.2f}%")
        print(f"{name:<24}{n_elem:>10}{w * 1e6:>16.4f}{change:>10} {w * D / (q * side**4):>14.6f}")

    fig, ax = plt.subplots()
    ax.plot(n_elements, deflections, 'o-', color='black', linewidth=2, markerfacecolor='white', markeredgecolor='black')
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Centre deflection [µm]')
    ax.set_title('Skew (rhombic) plate - simply supported on all four edges')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()