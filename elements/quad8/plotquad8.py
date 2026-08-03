import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

def plotelem8(T, X, elnum=False, nodenum=False, title="Plate Topology", color='b', linewidth=1.0, figsize=[10,8], ax=None, set_aspect=True):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    # Quad8 edge connectivity: 1-5-2, 2-6-3, 3-7-4, 4-8-1
    edges = [[0,4,1], [1,5,2], [2,6,3], [3,7,0]]

    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        elem_coords = X[node_indices, :]
        for edge in edges:
            ax.plot(elem_coords[edge, 0], elem_coords[edge, 1], elem_coords[edge, 2], color=color, linewidth=linewidth)
        if elnum:
            centroid = np.mean(elem_coords[0:4], axis=0) # Use corner nodes for centroid
            ax.text(centroid[0], centroid[1], centroid[2], str(i+1), color='red', fontsize=10, fontweight='bold')

    if nodenum:
        for i in range(len(X)):
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(i+1), color='blue', fontsize=8)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(title)

    if set_aspect:
        L = X[:,0].max() - X[:,0].min()
        W = X[:,1].max() - X[:,1].min()
        L = L if L > 0 else 1.0
        W = W if W > 0 else 1.0
        ax.set_box_aspect((L, W, (L+W)/4)) 

    return ax

def plotmesh2d8(T, X, elnum=False, nodenum=False, title=None, linewidth=0.8, figsize=[8, 6], ax=None):
    from matplotlib.ticker import MaxNLocator

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    edges = [[0,4,1], [1,5,2], [2,6,3], [3,7,0]]

    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        elem_coords = X[node_indices, :]
        for edge in edges:
            ax.plot(elem_coords[edge, 0], elem_coords[edge, 1], color='k', linewidth=linewidth, rasterized=True)
        if elnum:
            centroid = np.mean(elem_coords[0:4], axis=0)
            ax.text(centroid[0], centroid[1], str(i+1), color='k', fontsize=8, ha='center', va='center')

    if nodenum:
        for i in range(len(X)):
            ax.text(X[i, 0], X[i, 1], str(i+1), color='k', fontsize=7, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_axis_off()

    xmin, xmax = X[:,0].min(), X[:,0].max()
    ymin, ymax = X[:,1].min(), X[:,1].max()
    span = max(xmax - xmin, ymax - ymin)
    tick_len = 0.025 * span

    x_ticks = np.array(MaxNLocator(nbins=6).tick_values(xmin, xmax))
    x_ticks = x_ticks[(x_ticks >= xmin - 1e-10) & (x_ticks <= xmax + 1e-10)]
    y_ticks = np.array(MaxNLocator(nbins=6).tick_values(ymin, ymax))
    y_ticks = y_ticks[(y_ticks >= ymin - 1e-10) & (y_ticks <= ymax + 1e-10)]

    for xt in x_ticks:
        ax.plot([xt, xt], [ymin, ymin - tick_len], color='k', linewidth=0.8, clip_on=False)
        ax.text(xt, ymin - tick_len * 1.8, f'{xt:g}', ha='center', va='top', fontsize=8, clip_on=False)

    for yt in y_ticks:
        ax.plot([xmin, xmin - tick_len], [yt, yt], color='k', linewidth=0.8, clip_on=False)
        ax.text(xmin - tick_len * 1.8, yt, f'{yt:g}', ha='right', va='center', fontsize=8, clip_on=False)

    arrow_len = 0.07 * span
    ox = xmin - 0.16 * span
    oy = ymin - 0.16 * span
    ap = dict(arrowstyle='->', color='k', lw=0.6, shrinkA=0, shrinkB=0)

    ax.annotate('', xy=(ox + arrow_len, oy), xytext=(ox, oy), arrowprops=ap, clip_on=False)
    ax.text(ox + arrow_len * 1.35, oy, 'X', ha='center', va='center', fontsize=8, clip_on=False)

    ax.annotate('', xy=(ox, oy + arrow_len), xytext=(ox, oy), arrowprops=ap, clip_on=False)
    ax.text(ox, oy + arrow_len * 1.35, 'Y', ha='center', va='bottom', fontsize=8, clip_on=False)

    # Z-axis indicator: circle with centre dot = out-of-plane (toward viewer)
    from matplotlib.patches import Circle
    r = arrow_len * 0.18
    z_circle = Circle((ox, oy), r, fill=False, color='k', linewidth=0.6,
                      clip_on=False, zorder=5)
    ax.add_patch(z_circle)
    ax.plot(ox, oy, 'k.', markersize=2.5, clip_on=False, zorder=6)
    ax.text(ox - r * 2.0, oy + r * 2.0, 'Z', ha='center', va='center',
            fontsize=8, clip_on=False)

    ax.set_xlim(xmin - 0.22 * span, xmax + 0.05 * span)
    ax.set_ylim(ymin - 0.22 * span, ymax + 0.05 * span)

    if title is not None:
        ax.set_title(title)

    return ax

def plotelemdisp8(T, X, U, scale=100.0, title="Deformed Plate Wireframe", set_aspect=True):
    X_def = X.copy()
    # For plates, DOFs are [w, theta_x, theta_y]
    # w is vertical displacement (Z)
    uz = U[0::3]
    X_def[:, 2] += uz * scale 
    
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original (grey) and deformed (blue)
    plotelem8(T, X, color='grey', linewidth=0.5, ax=ax, title=title)
    plotelem8(T, X_def, color='blue', linewidth=1.5, ax=ax, title=title)
    
    if set_aspect:
        # Set aspect ratio to keep it looking like a slab
        L = X[:,0].max() - X[:,0].min()
        W = X[:,1].max() - X[:,1].min()
        L = L if L > 0 else 1.0
        W = W if W > 0 else 1.0
        # We want Z to have some height in the box even if small, but proportional
        ax.set_box_aspect((L, W, (L+W)/4)) 

def plotcontour8(T, X, U, scale=100.0, title="Plate Deflection Contours", set_aspect=True):
    X_def = X.copy()
    uz = U[0::3]
    X_def[:, 2] += uz * scale
    
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(uz.min(), uz.max())
    cmap = plt.get_cmap('jet_r') 

    faces = []
    face_colors = []
    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        # For visualization, we can just use the corner nodes [1,2,3,4] for the face
        face_nodes = node_indices[[0, 1, 2, 3]]
        verts = X_def[face_nodes]
        faces.append(verts)
        avg_disp = np.mean(uz[node_indices]) # Average of all 8 nodes
        face_colors.append(cmap(norm(avg_disp)))

    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolors='black', alpha=0.8, linewidths=0.2)
    ax.add_collection3d(poly)
    
    # Add Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(uz)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cb.set_label('Vertical Displacement [m]')

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(title)
    
    if set_aspect:
        # Set limits for visibility
        L = X[:,0].max() - X[:,0].min()
        W = X[:,1].max() - X[:,1].min()
        L = L if L > 0 else 1.0
        W = W if W > 0 else 1.0
        ax.set_xlim(X[:,0].min(), X[:,0].max())
        ax.set_ylim(X[:,1].min(), X[:,1].max())
        
        # Fixed Z-limits relative to slab size makes scale changes visible
        # Or use box_aspect to ensure the scale factor has a visual impact
        z_center = (X_def[:,2].min() + X_def[:,2].max()) / 2
        z_range = max(abs(X_def[:,2]).max() * 2, (L+W)/20)
        ax.set_zlim(z_center - z_range/2, z_center + z_range/2)
        
        ax.set_box_aspect((L, W, (L+W)/4))

def plotmoment8(T, X, M, title="Bending Moment Contour", label="Moment [Nm/m]", set_aspect=True):
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(M.min(), M.max())
    cmap = plt.get_cmap('viridis')

    faces = []
    face_colors = []
    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        # For visualization, we can just use the corner nodes [1,2,3,4] for the face
        face_nodes = node_indices[[0, 1, 2, 3]]
        verts = X[face_nodes]
        faces.append(verts)
        avg_val = np.mean(M[node_indices])
        face_colors.append(cmap(norm(avg_val)))

    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolors='black', alpha=0.9, linewidths=0.1)
    ax.add_collection3d(poly)
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(M)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cb.set_label(label)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(title)
    
    if set_aspect:
        L = X[:,0].max() - X[:,0].min()
        W = X[:,1].max() - X[:,1].min()
        L = L if L > 0 else 1.0
        W = W if W > 0 else 1.0
        ax.set_xlim(X[:,0].min(), X[:,0].max())
        ax.set_ylim(X[:,1].min(), X[:,1].max())
        ax.set_zlim(-0.1, 0.1) # Keep it flat for contour
        ax.set_box_aspect((L, W, (L+W)/10))

    return ax

def plotbc8(T, X, FE, F=None, title="Boundary Conditions & Loads", figsize=(10, 8), ax=None, set_aspect=True):
    # Note: visualization for loads is limited to vertical loads. Visualization for other DOFS loads (theta_x, theta_y) is not implemented.
    from matplotlib.lines import Line2D

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    edges = [[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]
    for i in range(len(T)):
        ni = T[i, 0:8] - 1
        coords = X[ni]
        for edge in edges:
            ax.plot(coords[edge, 0], coords[edge, 1], coords[edge, 2],
                    color='0.75', lw=0.8)

    dof = FE.dofs_per_node  # 3 for plate8

    diag = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
    G = max(diag * 0.025, 1e-9)

    node_offsets = {}
    for d in FE.fixed_dofs:
        n, o = d // dof, d % dof
        node_offsets.setdefault(n, set()).add(o)

    Fw = F[0::dof] if F is not None else np.zeros(len(X))
    peak = np.max(np.abs(Fw))
    loaded_mask = np.abs(Fw) > peak * 1e-6 if peak > 0 else np.zeros(len(X), dtype=bool)
    n_load_nodes = int(np.sum(loaded_mask))

    dofs_present = {o for offs in node_offsets.values() for o in offs}

    use_fallback = len(node_offsets) > 5000 or n_load_nodes > 5000
    if use_fallback:
        print(f"plotbc8: {len(node_offsets)} BC nodes / {n_load_nodes} load nodes - using scatter fallback.")
        _style = {0: ('v','tab:orange',r'$w=0$'),
                  1: ('s','tab:green', r'$\theta_x=0$'),
                  2: ('D','tab:blue',  r'$\theta_y=0$')}
        for offset, (marker, color, label) in _style.items():
            nodes = [n for n, offs in node_offsets.items() if offset in offs]
            if nodes:
                ax.scatter(X[nodes,0], X[nodes,1], X[nodes,2], marker=marker, s=60, c=color, edgecolors='k', linewidths=0.5, label=label, depthshade=False)
    else:
        nan = float('nan')
        H  = G * np.sqrt(3) / 2   # triangle height (equilateral)
        hb = 0.5 * G               # half base

        xs = {0: [], 1: [], 2: []}
        ys = {0: [], 1: [], 2: []}
        zs = {0: [], 1: [], 2: []}

        for node, offsets in node_offsets.items():
            cx, cy, cz = X[node]
            for o in offsets:
                if o == 0:
                    p1 = (cx, cy - hb, cz - H)
                    p2 = (cx, cy + hb, cz - H)
                elif o == 1:
                    p1 = (cx - hb, cy - H, cz)
                    p2 = (cx + hb, cy - H, cz)
                else:
                    p1 = (cx - H, cy, cz - hb)
                    p2 = (cx - H, cy, cz + hb)
                xs[o] += [cx, p1[0], p2[0], cx, nan]
                ys[o] += [cy, p1[1], p2[1], cy, nan]
                zs[o] += [cz, p1[2], p2[2], cz, nan]

        _colors = {0: 'tab:orange', 1: 'tab:green', 2: 'tab:blue'}
        for o in (0, 1, 2):
            if xs[o]:
                ax.plot(xs[o], ys[o], zs[o], color=_colors[o], lw=1.4)

        Lx_bc = max(np.ptp(X[:, 0]), 1.0)
        Wy_bc = max(np.ptp(X[:, 1]), 1.0)
        z_mid = float(np.mean(X[:, 2]))
        z_half = (Lx_bc + Wy_bc) / 8   # half of (Lx+Wy)/4
        ax.set_zlim(z_mid - z_half, z_mid + z_half)
        
    proxies = []
    if peak > 0:
        loaded = np.where(loaded_mask)[0]
        forces = Fw[loaded]
        arrow_len = G * 4 * np.abs(forces) / peak

        down = forces < 0
        if np.any(down):
            idx = loaded[down]
            al  = arrow_len[down]
            ax.quiver(X[idx,0], X[idx,1], X[idx,2],
                      np.zeros(len(idx)), np.zeros(len(idx)), -al,
                      color='tab:red', arrow_length_ratio=0.3, linewidth=1.0)
            proxies.append(Line2D([0],[0], color='tab:red', lw=1.5,
                                  label=f'Load ↓ ({np.sum(forces[down]):.0f} N)'))
        up = ~down
        if np.any(up):
            idx = loaded[up]
            al  = arrow_len[up]
            ax.quiver(X[idx,0], X[idx,1], X[idx,2],
                      np.zeros(len(idx)), np.zeros(len(idx)), al,
                      color='tab:blue', arrow_length_ratio=0.3, linewidth=1.0)
            proxies.append(Line2D([0],[0], color='tab:blue', lw=1.5,
                                  label=f'Load ↑ ({np.sum(forces[up]):.0f} N)'))

    legend_handles = []
    if not use_fallback:
        _dof_legend = {
            0: ('tab:orange', r'Fixed $w = 0$'),
            1: ('tab:green',  r'Fixed $\theta_x = 0$'),
            2: ('tab:blue',   r'Fixed $\theta_y = 0$'),
        }
        for o in (0, 1, 2):
            if o in dofs_present:
                col, lbl = _dof_legend[o]
                legend_handles.append(Line2D([0],[0], color=col, lw=1.4, label=lbl))
    legend_handles += proxies

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

    if set_aspect:
        Lx = max(X[:, 0].ptp(), 1.0)
        Wy = max(X[:, 1].ptp(), 1.0)
        ax.set_box_aspect((Lx, Wy, (Lx + Wy) / 4))

    return ax


def plotrotation8(T, X, Rot, title="Rotation Contour", label="Rotation [rad]", set_aspect=True):
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(Rot.min(), Rot.max())
    cmap = plt.get_cmap('RdBu') # Red-Blue is good for rotations (pos/neg)

    faces = []
    face_colors = []
    for i in range(np.size(T, 0)):
        node_indices = T[i, 0:8] - 1
        face_nodes = node_indices[[0, 1, 2, 3]]
        verts = X[face_nodes]
        faces.append(verts)
        avg_val = np.mean(Rot[node_indices])
        face_colors.append(cmap(norm(avg_val)))

    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolors='black', alpha=0.9, linewidths=0.1)
    ax.add_collection3d(poly)
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Rot)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cb.set_label(label)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(title)
    
    if set_aspect:
        L = X[:,0].max() - X[:,0].min()
        W = X[:,1].max() - X[:,1].min()
        L = L if L > 0 else 1.0
        W = W if W > 0 else 1.0
        ax.set_xlim(X[:,0].min(), X[:,0].max())
        ax.set_ylim(X[:,1].min(), X[:,1].max())
        ax.set_zlim(-0.1, 0.1)
        ax.set_box_aspect((L, W, (L+W)/10))

    return ax
