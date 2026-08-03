from __future__ import annotations

import meshio
import numpy as np

ELEMENT_TYPE = "quad8"


def import_msh(file_path: str, propno: int | None = None, materials: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(file_path)

    blocks = [(i, block.data) for i, block in enumerate(mesh.cells)
              if block.type == ELEMENT_TYPE]

    if not blocks:
        found = sorted({block.type for block in mesh.cells})
        raise ValueError(f"No '{ELEMENT_TYPE}' elements in {file_path}. Found: {', '.join(found) if found else 'no cells'}")

    connectivity = np.vstack([data for _, data in blocks]) + 1

    if propno is not None:
        tags = np.full(len(connectivity), propno, dtype=np.int64)
    elif materials is not None:
        tags = _named_tags(mesh, blocks, file_path, materials)
    else:
        tags = _physical_tags(mesh, blocks, len(connectivity))

    T = np.column_stack([connectivity, tags]).astype(np.int64)
    X = np.ascontiguousarray(mesh.points, dtype=np.float64)

    return X, T


def _named_tags(mesh, blocks: list, file_path: str, materials: list[str]) -> np.ndarray:
    rank_of_tag = {}
    for rank, name in enumerate(materials, start=1):
        entry = mesh.field_data.get(name)
        if entry is None or int(entry[1]) != 2:
            available = ', '.join(sorted(n for n, e in mesh.field_data.items()
                                         if int(e[1]) == 2))
            raise ValueError(f"Physical surface '{name}' not found in {file_path}. Available: {available if available else 'none'}")
        rank_of_tag[int(entry[0])] = rank

    raw = _raw_tags(mesh, blocks, file_path)

    tags = np.zeros(len(raw), dtype=np.int64)
    for tag, rank in rank_of_tag.items():
        tags[raw == tag] = rank

    if not tags.all():
        missing = sorted(set(raw[tags == 0].tolist()))
        raise ValueError(f"{file_path} has elements in physical groups {missing} that are not listed in materials={materials}")
    return tags


def _raw_tags(mesh, blocks: list, file_path: str) -> np.ndarray:
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is None or len(physical) != len(mesh.cells):
        raise ValueError(f"{file_path} has no usable 'gmsh:physical' data; assign physical surfaces or pass propno instead")

    per_block = []
    for i, data in blocks:
        if len(physical[i]) != len(data):
            raise ValueError(f"{file_path} has incomplete 'gmsh:physical' data; assign physical surfaces or pass propno instead")
        per_block.append(np.asarray(physical[i], dtype=np.int64))

    return np.concatenate(per_block)


def _physical_tags(mesh, blocks: list, n_elements: int) -> np.ndarray:
    physical = mesh.cell_data.get("gmsh:physical")

    if physical is None or len(physical) != len(mesh.cells):
        return np.ones(n_elements, dtype=np.int64)

    per_block = []
    for i, data in blocks:
        if len(physical[i]) != len(data):
            return np.ones(n_elements, dtype=np.int64)
        per_block.append(np.asarray(physical[i], dtype=np.int64))

    # Gmsh physical tags are arbitrary integers; T[:, 8] indexes rows of the
    # material table H, so remap them onto a consecutive 1-based range.
    _, inverse = np.unique(np.concatenate(per_block), return_inverse=True)
    return inverse.astype(np.int64) + 1
