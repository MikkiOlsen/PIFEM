from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

from .utils import find_nearest_node, get_node_dof

if TYPE_CHECKING:
    from .FESystem import FESystem

_DOF_MAP = {'w': 0, 'theta_x': 1, 'theta_y': 2}


def apply_point_load(X: np.ndarray, position: np.ndarray | list | tuple, magnitude: float, dof: str | int = 'w', dofs_per_node: int = 3, F: np.ndarray = None) -> np.ndarray:
    if F is None:
        F = np.zeros(X.shape[0] * dofs_per_node)
    F[get_node_dof(find_nearest_node(X, position), dof, dofs_per_node)] += magnitude
    return F


def read_vehicle(vehicle_name: str, config_path: str | None = None) -> dict:
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', 'vehicles', f'{vehicle_name}.json')

    with open(config_path, 'r') as f:
        data = json.load(f)

    return data
