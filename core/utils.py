from __future__ import annotations
import numpy as np
import json
import os

_DOF_MAP = {'w': 0, 'theta_x': 1, 'theta_y': 2}


def find_nearest_node(X: np.ndarray, position: np.ndarray | list | tuple) -> int:
    pos = np.asarray(position, dtype=float)
    if pos.ndim == 0 or pos.shape[0] == 2:
        pos = np.append(pos.ravel()[:2], 0.0)
    return int(np.argmin(np.linalg.norm(X - pos, axis=1)))


def get_node_dof(node: int, dof: str | int = 'w', dofs_per_node: int = 3) -> int:
    offset = _DOF_MAP[dof] if isinstance(dof, str) else int(dof)
    return node * dofs_per_node + offset


def make_H(mat: dict, thickness: float) -> np.ndarray:
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    return np.array([[mat['E'], mat['nu'], thickness]])

def extract_dof(U: np.ndarray, dof: str | int = 'w', dofs_per_node: int = 3) -> np.ndarray:
    offset = _DOF_MAP[dof] if isinstance(dof, str) else int(dof)
    return U[offset::dofs_per_node]


def extract_dof_at_position(U: np.ndarray, X: np.ndarray, position: np.ndarray | list | tuple, dof: str | int = 'w', dofs_per_node: int = 3, tol = 1e-5) -> float:
    node = find_nearest_node(X, position)
    return float(U[get_node_dof(node, dof, dofs_per_node)])


def load_material(material_name: str, config_path: str | None = None) -> dict:
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', 'materials.json')

    with open(config_path, 'r') as f:
        data = json.load(f)

    if material_name not in data:
        raise ValueError(f"Material '{material_name}' not found in {config_path}")

    return data[material_name]

def build_material_table(T: np.ndarray, num_x: int, num_y: int, nx_slab: int, ny_slab: int, thickness: float, default_material: str, slab_materials: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    slab_materials = slab_materials or {}

    all_labels = [
        slab_materials.get(sy * num_x + sx + 1, default_material)
        for sy in range(num_y)
        for sx in range(num_x)
    ]
    max_id_w = len(str(num_x * num_y))
    max_lbl_w = max(len(lbl) for lbl in all_labels)
    cell_w = max_id_w + 2 + max_lbl_w

    h_bar   = "─" * (cell_w + 2)
    top     = "┌" + "┬".join([h_bar] * num_x) + "┐"
    mid_sep = "├" + "┼".join([h_bar] * num_x) + "┤"
    bottom  = "└" + "┴".join([h_bar] * num_x) + "┘"

    print("Slab layout  (X →,  Y ↑)")
    print(top)
    for sy in range(num_y - 1, -1, -1):
        cells = []
        for sx in range(num_x):
            slab_id = sy * num_x + sx + 1
            label = slab_materials.get(slab_id, default_material)
            content = f"{str(slab_id).rjust(max_id_w)}  {label}".ljust(cell_w)
            cells.append(f" {content} ")
        print("│" + "│".join(cells) + "│")
        if sy > 0:
            print(mid_sep)
    print(bottom)
    
    n_slabs = num_x * num_y
    elems_per_slab = nx_slab * ny_slab

    for sid in slab_materials:
        if not (1 <= sid <= n_slabs):
            print(f"Warning: slab_id {sid} is out of range [1, {n_slabs}] and will be ignored.")

    per_slab_name = [
        slab_materials.get(slab_id, default_material)
        for slab_id in range(1, n_slabs + 1)
    ]

    seen: dict = {}
    unique_names: list = []
    for name in per_slab_name:
        if name not in seen:
            seen[name] = len(unique_names) + 1  # 1-based propno
            unique_names.append(name)

    H = np.vstack([make_H(load_material(name), thickness) for name in unique_names])

    T_out = T.copy()
    for idx, name in enumerate(per_slab_name):
        start = idx * elems_per_slab
        end = start + elems_per_slab
        T_out[start:end, 8] = seen[name]

    return H, T_out
