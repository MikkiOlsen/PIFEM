import os
import numpy as np
import matplotlib.pyplot as plt

import core
from elements.quad8 import kquad8, consistent_plate_load, interpolate_field
from mesh.gmsh import import_msh

MESHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')

MESH_NAMES = ['quartercircle_2_x_2', 'quartercircle_4_x_4',
              'quartercircle_8_x_8', 'quartercircle_16_x_16']

R = 3.0
t = 0.05
q = -1000.0

material = core.load_material("pavement_concrete")
H = core.make_H(material, t)
E, nu = material['E'], material['nu']
D = E * t**3 / (12.0 * (1.0 - nu**2))

# Simply supported circular plate under uniform load, thin-plate theory:
#   w_max = q R^4 (5 + nu) / (64 D (1 + nu))
W_ANALYTICAL = q * R**4 * (5.0 + nu) / (64.0 * D * (1.0 + nu))


def solve(name):
    X, T = import_msh(os.path.join(MESHES, name + '.msh'))
    K = kquad8(T, X, H, dof=3)

    radius = np.hypot(X[:, 0], X[:, 1])
    arc_nodes = np.where(np.isclose(radius, R, atol=1e-5))[0]

    FE = core.FESystem(X, dofs_per_node=3)
    FE.symmetry(axis='x', value=0.0)
    FE.symmetry(axis='y', value=0.0)
    FE.simply_supported_nodes(arc_nodes)

    F = consistent_plate_load(T, X, q)
    U = FE.solve_direct(K, F)

    return T.shape[0], interpolate_field(T, X, U, (0.0, 0.0), dof='w')


if __name__ == "__main__":
    print(f"D = {D:.4e} N·m   R/t = {R / t:.0f}")
    print(f"{'mesh':<24}{'elements':>10}{'w_centre [mm]':>16}{'error':>10}")

    n_elements, deflections = [], []
    for name in MESH_NAMES:
        n_elem, w = solve(name)
        n_elements.append(n_elem)
        deflections.append(w * 1000)
        error = (w - W_ANALYTICAL) / W_ANALYTICAL * 100
        print(f"{name:<24}{n_elem:>10}{w * 1000:>16.4f}{error:>9.2f}%")

    print(f"{'analytical (thin plate)':<24}{'':>10}{W_ANALYTICAL * 1000:>16.4f}")

    fig, ax = plt.subplots()
    ax.plot(n_elements, deflections, 'o-', color='black', linewidth=2, markerfacecolor='white', markeredgecolor='black', label='PIFEM')
    ax.invert_yaxis()
    ax.axhline(W_ANALYTICAL * 1000, color='red', linestyle='--', linewidth=1.5, label='Analytical')
    ax.set_xscale('log')
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Centre deflection [mm]')
    ax.set_title(f'Quarter circular plate - simply supported, R = {R:.0f} m')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
