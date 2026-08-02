import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from core import *
from mesh.quad8 import mesh_slab_grid
from elements.quad8 import *


L_SLAB    = 5.0
W_SLAB    = 5.0
THICKNESS = 0.36
JOINT_GAP = 0.001
NUM_X     = 3
NUM_Y     = 3
NX_SLAB   = 1
NY_SLAB   = 1
GRAVITY   = 9.81

K_WINKLER    = 0.3e9
K_JOINT      = 10e9
GS_PASTERNAK = 0.0

SPEED   = 10
DT      = 1         # s
X_START = -7.0


truck       = Vehicle.from_dict(read_vehicle("volvo_fh16"))
TRUCK_REF_Y = 0

X, T, elem_slab, node_slab = mesh_slab_grid(
    L_SLAB, W_SLAB, NX_SLAB, NY_SLAB, NUM_X, NUM_Y, JOINT_GAP, JOINT_GAP, nodes_indices=True)

H = make_H(load_material("cph_rafa"), THICKNESS)
K = kquad8(T, X, H, dof=3)
K = apply_foundation(K, T, X, ks=K_WINKLER, Gs=GS_PASTERNAK, dofs_per_node=3)


longi_pairs = find_longitudinal_joint_pairs(X, W_SLAB, JOINT_GAP, NUM_Y)
trans_pairs = find_transverse_joint_pairs(X, L_SLAB, JOINT_GAP, NUM_X)
K = apply_joint_springs(K, trans_pairs, q0=K_JOINT, dof_index=0, X=X, along_axis=1)
K = apply_joint_springs(K, longi_pairs, q0=K_JOINT, dof_index=0, X=X, along_axis=0)

FE = FESystem(X, dofs_per_node=3)
y_north = NUM_Y * W_SLAB + (NUM_Y - 1) * JOINT_GAP
FE.simply_supported(axis='y', value=0.0)
FE.simply_supported(axis='y', value=y_north)

t_hist = np.arange(0, 10, 1)

ref_x_arr = X_START + SPEED * t_hist

F_batch = truck.build_all_F(T, X, ref_x_arr, TRUCK_REF_Y)
F_batch_block = truck.build_all_F_block(T, X, ref_x_arr, TRUCK_REF_Y)

print("Numba warmup complete - all @njit functions compiled and cached.")