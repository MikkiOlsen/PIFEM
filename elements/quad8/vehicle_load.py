from __future__ import annotations
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from .fquad8 import (
    build_single_tire_force_vector, build_all_tire_force_vectors,
    build_single_block_force_vector, build_all_block_force_vectors,
)
from core.utils import find_nearest_node, get_node_dof
from core.loading import apply_point_load
from core.options import OPTIONS

if TYPE_CHECKING:
    from core.FESystem import FESystem


@dataclass(frozen=True)
class Vehicle:
    wheel_x_off:    np.ndarray  # (n_wheels) x offset from vehicle reference point
    wheel_y_off:    np.ndarray  # (n_wheels) y offset from vehicle reference point
    force_per_tire: np.ndarray  # (n_wheels) positive magnitude [N]
    tire_pressure:  np.ndarray  # (n_wheels) [Pa]
    tire_radius:    np.ndarray  # (n_wheels) [m]

    @classmethod
    def from_dict(cls, vehicle: dict, gravity: float = 9.81) -> "Vehicle":
        x_off, y_off, force, pressure, radius = [], [], [], [], []
        for axle in vehicle['axles']:
            n_tires = len(axle['y_offsets'])
            f = (axle['mass_kg'] / n_tires) * gravity
            r = np.sqrt(f / axle['tire_pressure_pa'] / np.pi)
            for y in axle['y_offsets']:
                x_off.append(axle['x_offset'])
                y_off.append(y)
                force.append(f)
                pressure.append(axle['tire_pressure_pa'])
                radius.append(r)
        return cls(
            wheel_x_off=np.array(x_off,    dtype=np.float64),
            wheel_y_off=np.array(y_off,    dtype=np.float64),
            force_per_tire=np.array(force,  dtype=np.float64),
            tire_pressure=np.array(pressure, dtype=np.float64),
            tire_radius=np.array(radius,   dtype=np.float64),
        )

    def build_F(self, T: np.ndarray, X: np.ndarray, ref_x: float, ref_y: float, dofs_per_node: int = 3, use_tire_load: bool = True) -> np.ndarray:
        if use_tire_load:
            return build_single_tire_force_vector(
                T, X,
                self.wheel_x_off,
                self.wheel_y_off + ref_y,
                -self.tire_pressure,
                self.tire_radius,
                ref_x,
                dofs_per_node,
            )
        F = np.zeros(X.shape[0] * dofs_per_node)
        for x_off, y_off, f in zip(self.wheel_x_off, self.wheel_y_off, self.force_per_tire):
            apply_point_load(X, position=(ref_x + x_off, ref_y + y_off),
                             magnitude=-f, dof='w', dofs_per_node=dofs_per_node, out=F)
        return F

    def build_F_block(self, T: np.ndarray, X: np.ndarray, ref_x: float, ref_y: float, dofs_per_node: int = 3) -> np.ndarray:
        return build_single_block_force_vector(
            T, X,
            self.wheel_x_off,
            self.wheel_y_off + ref_y,
            self.force_per_tire,
            self.tire_radius,
            ref_x,
            dofs_per_node,
        )

    def build_all_F_block(self, T: np.ndarray, X: np.ndarray, ref_x_arr: np.ndarray, ref_y: float, dofs_per_node: int = 3) -> np.ndarray:
        return build_all_block_force_vectors(
            T, X,
            self.wheel_x_off,
            self.wheel_y_off + ref_y,
            self.force_per_tire,
            self.tire_radius,
            np.asarray(ref_x_arr, dtype=np.float64),
            dofs_per_node,
        )

    def build_all_F(self, T: np.ndarray, X: np.ndarray, ref_x_arr: np.ndarray, ref_y: float, dofs_per_node: int = 3) -> np.ndarray:
        wheel_y = self.wheel_y_off + ref_y
        return build_all_tire_force_vectors(
            T, X,
            self.wheel_x_off,
            wheel_y,
            -self.tire_pressure,
            self.tire_radius,
            np.asarray(ref_x_arr, dtype=np.float64),
            dofs_per_node,
        )


@dataclass
class ForceTable:
    _table: sp.csr_matrix  # (n_positions, n_dofs)
    x_grid: np.ndarray     # (n_positions) uniform, sorted

    def lookup(self, x_positions) -> np.ndarray:
        scalar = np.ndim(x_positions) == 0
        x = np.atleast_1d(np.asarray(x_positions, dtype=np.float64))
        dx = self.x_grid[1] - self.x_grid[0]
        idx = np.round((x - self.x_grid[0]) / dx).astype(np.int64).clip(0, len(self.x_grid) - 1)
        result = self._table[idx].toarray()
        return result[0] if scalar else result


def build_force_table(vehicle: Vehicle, T: np.ndarray, X: np.ndarray, x_grid: np.ndarray, ref_y: float, dofs_per_node: int = 3, batch_size: int = 500, use_block: bool = False) -> ForceTable:
    x_grid = np.asarray(x_grid, dtype=np.float64)
    batches = []
    for lo in range(0, len(x_grid), batch_size):
        hi = min(lo + batch_size, len(x_grid))
        if use_block:
            F_dense = vehicle.build_all_F_block(T, X, x_grid[lo:hi], ref_y, dofs_per_node)
        else:
            F_dense = vehicle.build_all_F(T, X, x_grid[lo:hi], ref_y, dofs_per_node)
        batches.append(sp.csr_matrix(F_dense))
        del F_dense
        if OPTIONS.Verbose:
            print(f'  ForceTable: {hi}/{len(x_grid)} positions computed')
    return ForceTable(
        _table=sp.vstack(batches, format='csr'),
        x_grid=x_grid,
    )


def moving_load_history(K: np.ndarray, X: np.ndarray, T: np.ndarray, vehicle: Vehicle, FE: FESystem, speed: float, ref_y: float, x_start: float, x_end: float, dt: float, measure_pos: tuple | list | np.ndarray, measure_dof: str | int = 'w', dofs_per_node: int = 3, use_tire_load: bool = True, print_every: int = 0, use_direct_solver: bool = True, PARDISO: bool = True, preconditioner_type: str = 'Jacobi') -> tuple[np.ndarray, np.ndarray]:
    veh = vehicle
    n_steps = int(np.ceil((x_end - x_start) / (speed * dt))) + 1

    if use_direct_solver:
        FE.setup_factorized(K, PARDISO)

    meas_node = find_nearest_node(X, measure_pos)
    meas_global = get_node_dof(meas_node, measure_dof, dofs_per_node)
    free, _ = FE.get_indices()
    meas_in_free = np.searchsorted(free, meas_global)
    meas_is_free = meas_in_free < len(free) and free[meas_in_free] == meas_global
    if not meas_is_free:
        print(f"Warning: measurement DOF {meas_global} is fixed - result will be constant.")

    t_hist = np.zeros(n_steps)
    d_hist = np.zeros(n_steps)
    U = None

    for i in range(n_steps):
        t = i * dt
        ref_x = x_start + speed * t

        F = veh.build_F(T, X, ref_x, ref_y, dofs_per_node, use_tire_load)

        if use_direct_solver:
            U = FE.solve_factorized(F)
        else:
            U = FE.solve_iterative(K, F, x0=U, preconditioner_type=preconditioner_type, tol=1e-6)

        d_hist[i] = U[meas_global]
        t_hist[i] = t

        if print_every > 0 and (i % print_every == 0 or i == n_steps - 1):
            print(f"  Step {i+1}/{n_steps}, t={t:.4f}s, ref_x={ref_x:.3f}m, dof={d_hist[i]:.6e}")

    if PARDISO:
        FE.cleanup()
    return t_hist, d_hist


def moving_load_history_parallel(K: np.ndarray, X: np.ndarray, T: np.ndarray, vehicle: Vehicle, FE: FESystem, speed: float, ref_y: float, x_start: float, x_end: float, dt: float, measure_pos: tuple | list | np.ndarray, measure_dof: str | int = 'w', dofs_per_node: int = 3, use_tire_load: bool = True, print_every: int = 0, n_workers: int = 4, tol: float = 1e-6, maxiter: int | None = None, use_direct_solver: bool = False, PARDISO: bool = True) -> tuple[np.ndarray, np.ndarray]:
    veh = vehicle
    n_steps = int(np.ceil((x_end - x_start) / (speed * dt))) + 1

    meas_node = find_nearest_node(X, measure_pos)
    meas_global = get_node_dof(meas_node, measure_dof, dofs_per_node)
    free, _ = FE.get_indices()
    meas_in_free = np.searchsorted(free, meas_global)
    meas_is_free = meas_in_free < len(free) and free[meas_in_free] == meas_global
    if not meas_is_free:
        print(f"Warning: measurement DOF {meas_global} is fixed - result will be constant.")

    if use_direct_solver:
        FE.setup_factorized(K, PARDISO)

    t_hist = np.arange(n_steps, dtype=np.float64) * dt
    ref_x_arr = x_start + speed * t_hist

    force_start_time = time.time()
    if use_tire_load:
        F_all = veh.build_all_F(T, X, ref_x_arr, ref_y, dofs_per_node)
        F_list = [F_all[i] for i in range(n_steps)]
    else:
        F_list = [veh.build_F(T, X, ref_x_arr[i], ref_y, dofs_per_node, use_tire_load=False)
                  for i in range(n_steps)]
    force_elapsed = time.time() - force_start_time
    if OPTIONS.Verbose:
        print(f'Force vector assembly: {n_steps} vectors in {force_elapsed:.2f}s')

    if use_direct_solver:
        d_hist = np.array(FE.solve_batch_factorized(
            F_list, n_workers=n_workers, extract_dofs=meas_global
        ))
    else:
        U_list = FE.solve_batch_parallel(K, F_list, tol=tol, maxiter=maxiter, n_workers=n_workers)
        d_hist = np.array([U[meas_global] for U in U_list])

    if print_every > 0:
        for i in range(n_steps):
            if i % print_every == 0 or i == n_steps - 1:
                ref_x = x_start + speed * t_hist[i]
                print(f"  Step {i+1}/{n_steps}, t={t_hist[i]:.4f}s, ref_x={ref_x:.3f}m, dof={d_hist[i]:.6e}")

    if use_direct_solver and PARDISO:
        FE.cleanup()

    return t_hist, d_hist
