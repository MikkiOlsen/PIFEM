import numpy as np
from .slab import mesh_slab

def mesh_slab_grid(L_slab, W_slab, nx_slab, ny_slab, num_slabs_x, num_slabs_y, joint_x, joint_y, nodes_indices: bool = False):
    X_all = np.empty((0, 3))
    T_all = np.empty((0, 9), dtype=int)

    n_nodes_per_slab = (2 * nx_slab + 1) * (2 * ny_slab + 1) - nx_slab * ny_slab
    n_elems_per_slab = nx_slab * ny_slab
    n_slabs = num_slabs_x * num_slabs_y

    if nodes_indices:
        node_slab = np.empty(n_slabs * n_nodes_per_slab, dtype=int) #the slab ID (1-based) of nodes of elements
        elem_slab = np.empty(n_slabs * n_elems_per_slab, dtype=int) #the slab ID (1-based) of elements
        #Allows for easy identification of which nodes/elements belong to which slab, which is useful for postprocessing.
        #For instance: if I wanted to get all elements in slab 5 I could do: T[elem_slab == 5]
        #Or if I wanted to get the max deflection for every single slab I could do:
        #uz = extract_dof(U, 'w')
        #max_deflection_per_slab = [np.max(uz[node_slab == slab_id]) for slab_id in range(1, n_slabs + 1)] or something along those lines. Haven't tested out this code snippet but it shold hopefully work.
        #This might not be useful at all, but I thought it could be a nice addition for future postprocessing purposes.
        #And they take up very little memory, and takes it basically no arithmetics to calculate as these things will need to be calculated anyways. So I don't see any reason not to include them

    for sy in range(num_slabs_y):
        for sx in range(num_slabs_x):
            slab_id = sy * num_slabs_x + sx + 1  # 1-based

            X_slab, T_slab = mesh_slab(L_slab, W_slab, nx_slab, ny_slab)

            x_off = sx * (L_slab + joint_x)
            y_off = sy * (W_slab + joint_y)

            X_slab[:, 0] += x_off
            X_slab[:, 1] += y_off

            node_offset = len(X_all)
            T_slab[:, 0:8] += node_offset

            if nodes_indices:
                n0 = (slab_id - 1) * n_nodes_per_slab
                e0 = (slab_id - 1) * n_elems_per_slab

                node_slab[n0:n0 + n_nodes_per_slab] = slab_id
                elem_slab[e0:e0 + n_elems_per_slab] = slab_id

            X_all = np.vstack((X_all, X_slab))
            T_all = np.vstack((T_all, T_slab))

    match nodes_indices:
        case True:
            return X_all, T_all, elem_slab, node_slab
        case False:
            return X_all, T_all
