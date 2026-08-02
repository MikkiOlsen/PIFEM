import numpy as np

def mesh_slab(L, W, nx, ny):
    nnx = 2 * nx + 1
    nny = 2 * ny + 1

    x = np.linspace(0, L, nnx)
    y = np.linspace(0, W, nny)
    xv, yv = np.meshgrid(x, y)
    X_full = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(nnx * nny)])

    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    base = (2 * jj.ravel()) * nnx + (2 * ii.ravel())           # (nx*ny)

    offsets = np.array([0, 2, 2 + 2*nnx, 2*nnx, 1, 2 + nnx, 2*nnx + 1, nnx])
    elem_nodes = base[:, None] + offsets                         # (nx*ny, 8)

    sequence = elem_nodes.ravel()
    _, first_pos = np.unique(sequence, return_index=True)
    del _
    new_node_order = sequence[np.sort(first_pos)]

    old_to_new = np.empty(nnx * nny, dtype=int)
    old_to_new[new_node_order] = np.arange(len(new_node_order))

    X = X_full[new_node_order]

    T = np.empty((nx * ny, 9), dtype=int)
    T[:, :8] = old_to_new[elem_nodes] + 1
    T[:, 8]  = 1

    return X, T
