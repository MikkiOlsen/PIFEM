import numpy as np
from core.options import OPTIONS

def mesh_slab_grid_gmsh(L_slab, W_slab, num_x, num_y, joint_gap, lc_fine, lc_med, lc_coarse, fine_radius, med_inner_radius, med_outer_radius, coarse_radius, x_refine, y_refine, row_labels=None):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("slab_grid_gmsh")
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("General.NumThreads", OPTIONS.NumThreads)

    surf_tags = {}
    for sy in range(num_y):
        for sx in range(num_x):
            x0 = sx * (L_slab + joint_gap)
            y0 = sy * (W_slab + joint_gap)
            pts = [
                gmsh.model.geo.addPoint(x0,          y0,          0, lc_coarse),
                gmsh.model.geo.addPoint(x0 + L_slab, y0,          0, lc_coarse),
                gmsh.model.geo.addPoint(x0 + L_slab, y0 + W_slab, 0, lc_coarse),
                gmsh.model.geo.addPoint(x0,          y0 + W_slab, 0, lc_coarse),
            ]
            lines = [
                gmsh.model.geo.addLine(pts[0], pts[1]),
                gmsh.model.geo.addLine(pts[1], pts[2]),
                gmsh.model.geo.addLine(pts[2], pts[3]),
                gmsh.model.geo.addLine(pts[3], pts[0]),
            ]
            cl = gmsh.model.geo.addCurveLoop(lines)
            surf_tags[(sx, sy)] = gmsh.model.geo.addPlaneSurface([cl])

    refine_pt = gmsh.model.geo.addPoint(x_refine, y_refine, 0, lc_fine)
    gmsh.model.geo.synchronize()

    if row_labels is not None:
        for sy in range(num_y):
            for sx in range(num_x):
                label = row_labels[sy] if sy < len(row_labels) else None
                name = f"Tile_{sx}_{sy}_{label}" if label else f"Tile_{sx}_{sy}"
                gmsh.model.addPhysicalGroup(2, [surf_tags[(sx, sy)]], name=name)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [refine_pt])
    gmsh.model.mesh.field.setNumber(f_dist,  "Sampling",   100)

    f_inner = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_inner, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_inner, "SizeMin", lc_fine)
    gmsh.model.mesh.field.setNumber(f_inner, "SizeMax", lc_coarse)
    gmsh.model.mesh.field.setNumber(f_inner, "DistMin", fine_radius)
    gmsh.model.mesh.field.setNumber(f_inner, "DistMax", med_inner_radius)

    f_outer = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_outer, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_outer, "SizeMin", lc_med)
    gmsh.model.mesh.field.setNumber(f_outer, "SizeMax", lc_coarse)
    gmsh.model.mesh.field.setNumber(f_outer, "DistMin", med_outer_radius)
    gmsh.model.mesh.field.setNumber(f_outer, "DistMax", coarse_radius)

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_inner, f_outer])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.option.setNumber("Mesh.RecombineAll",          1)
    gmsh.option.setNumber("Mesh.Algorithm",             8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.ElementOrder",          2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = np.empty(int(node_tags.max()) + 1, dtype=np.intp)
    tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.intp)
    X = coords.reshape(-1, 3)

    T_blocks = []

    for sy in range(num_y):
        propno = sy + 1
        for sx in range(num_x):
            _, _, element_node_tags  = gmsh.model.mesh.getElements(dim=2, tag=surf_tags[(sx, sy)])
            nodes = element_node_tags [0].reshape(-1, 8)
            block = np.empty((nodes.shape[0], 9), dtype=np.intp)
            block[:, :8] = tag_to_idx[nodes] + 1
            block[:, 8]  = propno
            T_blocks.append(block)

    gmsh.finalize()

    T = np.vstack(T_blocks)


    used = np.unique(T[:, :8] - 1)
    X_compact = X[used]
    remap = np.empty(len(X), dtype=np.intp)
    remap[used] = np.arange(len(used), dtype=np.intp)
    T[:, :8] = remap[T[:, :8] - 1] + 1

    return X_compact, T
