from __future__ import annotations

import os
import numpy as np
from scipy.sparse import spmatrix, diags
from scipy.sparse.linalg import cg, splu, spsolve
import time
import threading
import pyamg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from core.options import OPTIONS

_pyamgx = None
_pyamgx_initialized = False

def _ensure_pyamgx():
    global _pyamgx, _pyamgx_initialized
    if _pyamgx is None:
        import pyamgx as _pyx
        _pyamgx = _pyx
    if not _pyamgx_initialized:
        _pyamgx.initialize()
        _pyamgx_initialized = True

class FESystem:
    _AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, X: np.ndarray, dofs_per_node: int = 3) -> None:
        self.n_nodes = len(X)
        self.X_matrix = X
        self.dofs_per_node = dofs_per_node
        self.fixed_dofs: set[int] = set()
        self.prescribed_values: dict[int, float] = {}
        self._has_nonzero_prescribed: bool = False
        self._K_free = None
        self._K_free_fixed = None
        self._preconditioner = None
        self._cached_free = None
        self._cached_fixed = None
        self._cached_fixed_vals = None
        self._cached_correction = None

        # CPU AMG cache
        self._amg_ml_cached = None

        # AmgX cache
        self._amgx_cfg = None
        self._amgx_rsc = None
        self._amgx_mtx = None
        self._amgx_x = None
        self._amgx_b = None
        self._amgx_sol = None
        self._amgx_prescribed_correction = None
        self._amgx_x0 = None

        # MUMPS cache
        self._mumps_ctx = None
        self._mpi_comm = None
        self._mpi_rank = 0
        self._mumps_n_dof = None

        self.option = OPTIONS

    def _resolve_axis(self, axis: str) -> int:
        if not isinstance(axis, str) or axis.lower() not in self._AXIS_MAP:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        return self._AXIS_MAP[axis.lower()]

    def fix_nodes(self, node_indices: np.ndarray | list[int], dof_offsets: tuple[int, ...] | int = (0, 1, 2), value: float = 0.0) -> None:
        if isinstance(dof_offsets, int):
            dof_offsets = (dof_offsets,)
        for node in node_indices:
            for offset in dof_offsets:
                idx = node * self.dofs_per_node + offset
                self.fixed_dofs.add(idx)
                self.prescribed_values[idx] = value
        self._has_nonzero_prescribed = any(v != 0.0 for v in self.prescribed_values.values())

    def _fix_on_plane(self, axis: str, value: float, dof_offsets: tuple[int, ...] | int, tol: float = 1e-5) -> None:
        axis_idx = self._resolve_axis(axis)
        nodes = np.where(np.isclose(self.X_matrix[:, axis_idx], value, atol=tol))[0]
        self.fix_nodes(nodes, dof_offsets)

    def _fix_on_edge(self, p_start: np.ndarray | list | tuple, p_end: np.ndarray | list | tuple, dof_offsets: tuple[int, ...] | int, tol: float = 1e-3) -> None:
        
        p_start = np.array(p_start)
        p_end = np.array(p_end)
        edge_vec = p_end - p_start
        edge_length = np.linalg.norm(edge_vec)
        unit_edge = edge_vec / edge_length
        normal = np.array([-unit_edge[1], unit_edge[0], 0])
        distances = np.abs(np.dot(self.X_matrix - p_start, normal))
        projections = np.dot(self.X_matrix - p_start, unit_edge)
        mask = (distances < tol) & (projections > -tol) & (projections < edge_length + tol)
        self.fix_nodes(np.where(mask)[0], dof_offsets)

    def simply_supported(self, axis: str, value: float, tol: float = 1e-5) -> None:
        self._fix_on_plane(axis, value, dof_offsets=(0,), tol=tol)

    def simply_supported_edge(self, p_start, p_end, tol: float = 1e-3) -> None:
        self._fix_on_edge(p_start, p_end, dof_offsets=(0,), tol=tol)

    def simply_supported_nodes(self, nodes: np.ndarray | list[int]) -> None:
        self.fix_nodes(nodes, dof_offsets=(0,))

    def clamped(self, axis: str, value: float, tol: float = 1e-5) -> None:
        self._fix_on_plane(axis, value, dof_offsets=(0, 1, 2), tol=tol)

    def clamped_edge(self, p_start, p_end, tol: float = 1e-3) -> None:
        self._fix_on_edge(p_start, p_end, dof_offsets=(0, 1, 2), tol=tol)

    def clamped_nodes(self, nodes: np.ndarray | list[int]) -> None:
        self.fix_nodes(nodes, dof_offsets=(0, 1, 2))

    def symmetry(self, axis: str, value: float, tol: float = 1e-5) -> None:
        axis_idx = self._resolve_axis(axis)
        nodes = np.where(np.isclose(self.X_matrix[:, axis_idx], value, atol=tol))[0]
        if axis_idx == 0:
            self.fix_nodes(nodes, [1])
        elif axis_idx == 1:
            self.fix_nodes(nodes, [2])

    def get_indices(self) -> tuple[np.ndarray, np.ndarray]:
        all_dofs = np.arange(self.n_nodes * self.dofs_per_node)
        fixed = np.array(sorted(self.fixed_dofs))
        free = np.setdiff1d(all_dofs, fixed)
        return free, fixed

    def get_fixed_nodes(self) -> np.ndarray:
        return np.unique(np.array(sorted(self.fixed_dofs)) // self.dofs_per_node)

    def _check_K(self, K: np.ndarray | spmatrix) -> None:
        n_dof = self.n_nodes * self.dofs_per_node
        if K.shape != (n_dof, n_dof):
            raise ValueError(f"Stiffness matrix shape {K.shape} does not match expected ({n_dof}, {n_dof})")
        if len(self.fixed_dofs) == 0:
            raise ValueError("No boundary conditions have been applied - system is singular")

    def _check_system(self, K: np.ndarray | spmatrix, F: np.ndarray) -> None:
        n_dof = self.n_nodes * self.dofs_per_node
        if any(v != 0.0 for v in self.prescribed_values.values()):
            self._has_nonzero_prescribed = True
        if F.shape[0] != n_dof:
            raise ValueError(f"Force vector length {F.shape[0]} does not match n_nodes * dofs_per_node = {n_dof}")
        self._check_K(K)

    def _cache_submatrices(self, K: np.ndarray | spmatrix, use_preconditioner: bool = True, preconditioner_type: str = 'AMG') -> None:
        free, fixed = self.get_indices()
        self._cached_free = free
        self._cached_fixed = fixed
        self._cached_fixed_vals = np.array([self.prescribed_values.get(int(i), 0.0) for i in fixed])
        self._K_free = K[np.ix_(free, free)].tocsr()

        if self._has_nonzero_prescribed:
            self._K_free_fixed = K[np.ix_(free, fixed)]
            self._cached_correction = self._K_free_fixed @ self._cached_fixed_vals
        else:
            self._K_free_fixed = None
            self._cached_correction = np.zeros(len(free))

        if use_preconditioner:
            match preconditioner_type:
                case 'Jacobi':
                    diag_K = self._K_free.diagonal().copy()
                    if np.any(diag_K < -1e-10):
                        print("Warning: Zero or negative diagonal detected. Check BCs.")
                        diag_K[diag_K <= 0] = 1.0
                    diag_K = np.maximum(diag_K, 1e-10)
                    self._preconditioner = diags(1.0 / diag_K)
                case 'AMG':
                    pass
        else:
            self._preconditioner = None
        

    
    def solve_direct(self, K: np.ndarray | spmatrix, F: np.ndarray, PARDISO = True) -> np.ndarray:
        self._check_system(K, F)
        self._cache_submatrices(K, use_preconditioner=False)
        free = self._cached_free
        start_time = time.time()
        U = np.zeros(len(F))
        U[self._cached_fixed] = self._cached_fixed_vals
        rhs = F[free]
        if self._K_free_fixed is not None:
            rhs = rhs - self._K_free_fixed @ U[self._cached_fixed]
        if PARDISO:
            os.environ['MKL_NUM_THREADS'] = str(self.option.NumThreads)
            from pypardiso import PyPardisoSolver
            solver = PyPardisoSolver()
            U[free] = solver.solve(self._K_free, rhs)
            solver.free_memory(everything=True)
        else:
            U[free] = spsolve(self._K_free, rhs)
        if self.option.Verbose:
            print(f'Solving took: {time.time() - start_time:.2f} seconds')
        return U

    def solve_iterative(self, K: np.ndarray | spmatrix, F: np.ndarray, tol: float = 1e-5, maxiter: int | None = None, use_preconditioner: bool = True, x0: np.ndarray | None = None, preconditioner_type: str = 'AMG') -> np.ndarray:
        start_time = time.time()
        self._check_system(K, F)
        self._cache_submatrices(K, use_preconditioner=use_preconditioner, preconditioner_type=preconditioner_type)
        free = self._cached_free
        U = np.zeros(len(F))
        U[self._cached_fixed] = self._cached_fixed_vals
        rhs = F[free]
        if self._K_free_fixed is not None:
            rhs = rhs - self._K_free_fixed @ U[self._cached_fixed]
        x0_free = x0[free] if x0 is not None else None

        match preconditioner_type:
            case 'AMG':
                if self._amg_ml_cached is None:
                    self._amg_ml_cached = pyamg.smoothed_aggregation_solver(self._K_free)
                u_free = self._amg_ml_cached.solve(rhs, x0=x0_free, tol=tol, maxiter=maxiter, cycle='V', accel='cg')
            case 'Jacobi':
                u_free, info = cg(self._K_free, rhs, x0=x0_free, rtol=tol, atol=1e-6, maxiter=maxiter, M=self._preconditioner)
                if info > 0:
                    print(f"Warning: CG failed to converge after {info} iterations.")

        U[free] = u_free
        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(f'Solving took: {elapsed:.2f} seconds')
        return U
    '''
    def solve_batch_parallel(self, K: np.ndarray | spmatrix, F_list: list[np.ndarray],
                             tol: float = 1e-5, maxiter: int | None = None,
                             n_workers: int = 4) -> list[np.ndarray]:
        from scipy.sparse import issparse, csc_matrix
        total_start = time.time()
        self._check_system(K, F_list[0])
        self._cache_submatrices(K, use_preconditioner=True, preconditioner_type='AMG')
        free = self._cached_free
        fixed = self._cached_fixed
        fixed_vals = self._cached_fixed_vals
        K_free = self._K_free

        correction = self._cached_correction

        F_list_sparse = [
            F if issparse(F) else csc_matrix(F.reshape(-1, 1))
            for F in F_list
        ]

        n_total = len(F_list_sparse)
        chunk_splits = np.array_split(range(n_total), n_workers)
        F_chunks = [[F_list_sparse[i] for i in chunk] for chunk in chunk_splits]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_solve_chunk_worker, K_free, F_chunk,
                                free, fixed, fixed_vals, correction, tol, maxiter)
                for F_chunk in F_chunks
            ]
            chunk_results = [future.result() for future in futures]

        results = []
        for chunk in chunk_results:
            results.extend(chunk)

        total_elapsed = time.time() - total_start
        avg_per_solve = total_elapsed / n_total
        print(f'Batch parallel ({n_workers} workers): {n_total} solves in {total_elapsed:.2f}s '
              f'(avg {avg_per_solve:.4f}s per solve)')
        return results
    '''

    def solve_batch_factorized(self, F_list: list[np.ndarray], n_workers: int | None = None, extract_dofs=None) -> list:
        if n_workers is None:
            n_workers = self.option.NumThreads
        total_start = time.time()
        n_total = len(F_list)
        chunk_splits = np.array_split(np.arange(n_total), n_workers)

        def _solve_chunk(indices):
            results = []
            for idx in indices:
                U = self.solve_factorized(F_list[idx])
                results.append((idx, U[extract_dofs] if extract_dofs is not None else U))
            return results

        all_solutions = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_solve_chunk, chunk) for chunk in chunk_splits]
            for future in futures:
                for idx, val in future.result():
                    all_solutions[idx] = val

        results = [all_solutions[i] for i in range(n_total)]

        if self.option.Verbose:
            total_elapsed = time.time() - total_start
            avg_per_solve = total_elapsed / n_total
            print(f'Batch factorized ({n_workers} workers): {n_total} solves in {total_elapsed:.2f}s '
                f'(avg {avg_per_solve:.4f}s per solve)')
        return results

    def setup_amgx(self, K: np.ndarray | spmatrix, config: str | None = None) -> None:
        start_time = time.time()
        _ensure_pyamgx()
        self._check_K(K)
        self._cache_submatrices(K, use_preconditioner=False)

        self._amgx_prescribed_correction = self._cached_correction

        if config is None:
            import json
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'amgx_config.json')
            with open(config_path, 'r') as f:
                config = json.dumps(json.load(f))

        self._amgx_cfg = _pyamgx.Config().create(config)
        self._amgx_rsc = _pyamgx.Resources().create_simple(self._amgx_cfg)
        self._amgx_mtx = _pyamgx.Matrix().create(self._amgx_rsc)
        self._amgx_x = _pyamgx.Vector().create(self._amgx_rsc)
        self._amgx_b = _pyamgx.Vector().create(self._amgx_rsc)
        self._amgx_sol = _pyamgx.Solver().create(self._amgx_rsc, self._amgx_cfg)

        K_csr = self._K_free.tocsr().astype(np.float64)
        self._amgx_mtx.upload_CSR(K_csr)
        self._amgx_sol.setup(self._amgx_mtx)
        self._amgx_x0 = None

        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(f'AmgX setup took {elapsed:.4f} seconds')

    def solve_amgx(self, F: np.ndarray) -> np.ndarray:
        start_time = time.time()
        rhs = F[self._cached_free] - self._amgx_prescribed_correction

        self._amgx_b.upload(rhs.astype(np.float64))
        if self._amgx_x0 is not None:
            self._amgx_x.upload(self._amgx_x0)
        else:
            self._amgx_x.upload(np.zeros(len(rhs), dtype=np.float64))

        self._amgx_sol.solve(self._amgx_b, self._amgx_x)

        u_free = np.zeros(len(rhs), dtype=np.float64)
        self._amgx_x.download(u_free)
        self._amgx_x0 = u_free.copy()

        U = np.zeros(len(F))
        U[self._cached_fixed] = self._cached_fixed_vals
        U[self._cached_free] = u_free

        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(f'AmgX solve in: {elapsed:.4f} seconds')
        return U

    def solve_batch_amgx(self, F_list: list[np.ndarray]) -> list[np.ndarray]:
        total_start = time.time()
        n_total = len(F_list)

        results = []
        for F in F_list:
            results.append(self.solve_amgx(F))

        total_elapsed = time.time() - total_start
        avg_per_solve = total_elapsed / n_total
        if self.option.Verbose:
            print(f'Batch AmgX: {n_total} solves in {total_elapsed:.2f}s '
                  f'(avg {avg_per_solve:.4f}s per solve)')
        return results

    def setup_mumps(self, K: "np.ndarray | spmatrix", sym: bool = False, ordering: str = 'metis') -> None:
        from mumps import DMumpsContext
        from mpi4py import MPI

        start_time = time.time()
        self._check_K(K)
        self._cache_submatrices(K, use_preconditioner=False)

        _ord = {'amd': 0, 'amf': 2, 'scotch': 3, 'pord': 4, 'metis': 5, 'qamd': 6, 'auto': 7}

        ctx = DMumpsContext(sym=1 if sym else 0, par=1, comm=MPI.COMM_SELF)
        if not self.option.Verbose:
            ctx.set_silent()
        ctx.set_icntl(7, _ord.get(ordering.lower(), 5))

        if sym:
            from scipy.sparse import triu
            ctx.set_centralized_sparse(triu(self._K_free, format='csr'))
        else:
            ctx.set_centralized_sparse(self._K_free)

        ctx.run(job=4)  # analysis + factorization

        self._mumps_ctx = ctx

        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(f'MUMPS setup (analysis + factorisation) took {elapsed:.4f} seconds')

    def solve_mumps(self, F: np.ndarray) -> np.ndarray:
        if self._mumps_ctx is None:
            raise RuntimeError(
                "MUMPS context not initialised - call setup_mumps() first."
            )

        start_time = time.time()
        ctx = self._mumps_ctx
        n_free = len(self._cached_free)

        rhs = np.asfortranarray(
            (F[self._cached_free] - self._cached_correction).astype(np.float64).reshape(n_free, 1)
        )
        ctx._refs['rhs'] = rhs
        ctx.id.nrhs = 1
        ctx.id.lrhs = n_free
        ctx.id.rhs = ctx.cast_array(rhs)
        ctx.run(job=3)

        U = np.zeros(len(F), dtype=np.float64)
        U[self._cached_fixed] = self._cached_fixed_vals
        U[self._cached_free] = rhs[:, 0]

        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(f'MUMPS solve in: {elapsed:.4f} seconds')
        return U

    def solve_batch_mumps(self, F_list: list) -> list:
        if self._mumps_ctx is None:
            raise RuntimeError(
                "MUMPS context not initialised - call setup_mumps() first."
            )

        total_start = time.time()
        n_total = len(F_list)
        n_free = len(self._cached_free)
        ctx = self._mumps_ctx

        rhs_matrix = np.asfortranarray(np.empty((n_free, n_total), dtype=np.float64))
        for i, F in enumerate(F_list):
            rhs_matrix[:, i] = F[self._cached_free] - self._cached_correction

        ctx._refs['rhs_matrix'] = rhs_matrix
        ctx.id.nrhs = n_total
        ctx.id.lrhs = n_free
        ctx.id.rhs = ctx.cast_array(rhs_matrix)
        ctx.run(job=3)  # solution written in place

        n_dof = len(F_list[0])
        solutions = []
        for i in range(n_total):
            U = np.zeros(n_dof, dtype=np.float64)
            U[self._cached_fixed] = self._cached_fixed_vals
            U[self._cached_free] = rhs_matrix[:, i]
            solutions.append(U)

        if self.option.Verbose:
            total_elapsed = time.time() - total_start
            avg_per_solve = total_elapsed / n_total
            print(
                f'Batch MUMPS (NRHS={n_total}): {total_elapsed:.2f}s '
                f'(avg {avg_per_solve:.4f}s per solve)'
            )
        return solutions

    def cleanup_mumps(self) -> None:
        if self._mumps_ctx is not None:
            self._mumps_ctx.destroy()
            self._mumps_ctx = None


    def setup_mumps_mpi(self, K, sym: bool = False, ordering: str = 'metis') -> None:
        from mumps import DMumpsContext
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start_time = time.time()

        if rank == 0:
            self._check_K(K)
            self._cache_submatrices(K, use_preconditioner=False)

        self._cached_free = comm.bcast(self._cached_free, root=0)
        self._cached_fixed = comm.bcast(self._cached_fixed, root=0)
        self._cached_fixed_vals = comm.bcast(self._cached_fixed_vals, root=0)
        self._cached_correction = comm.bcast(self._cached_correction, root=0)
        self._mumps_n_dof = comm.bcast(self.n_nodes * self.dofs_per_node if rank == 0 else None, root=0)

        _ord = {'amd': 0, 'amf': 2, 'scotch': 3, 'pord': 4, 'metis': 5, 'qamd': 6, 'auto': 7}

        ctx = DMumpsContext(sym=1 if sym else 0, par=1, comm=comm)
        if not self.option.Verbose:
            ctx.set_silent()
        ctx.set_icntl(7, _ord.get(ordering.lower(), 5))  #ordering method (1-based)

        if rank == 0:
            if sym:
                # MUMPS sym=1 expects only the upper triangle; filter out lower entries.
                from scipy.sparse import triu
                ctx.set_centralized_sparse(triu(self._K_free, format='csr'))
            else:
                ctx.set_centralized_sparse(self._K_free)

        ctx.run(job=4)  #analysis + factorization

        self._mumps_ctx = ctx
        self._mpi_comm = comm
        self._mpi_rank = rank

        if self.option.Verbose and rank == 0:
            elapsed = time.time() - start_time
            print(f'MUMPS MPI setup (analysis + factorisation) took {elapsed:.4f} seconds')

    def solve_batch_mumps_mpi(self, F_list, extract_dofs=None):
        if self._mumps_ctx is None:
            raise RuntimeError("MUMPS MPI context not initialised - call setup_mumps_mpi() first.")

        ctx = self._mumps_ctx
        comm = self._mpi_comm
        rank = self._mpi_rank
        start_time = time.time()

        n_total = comm.bcast(len(F_list) if rank == 0 else None, root=0)
        n_free = len(self._cached_free)

        if rank == 0:
            # Build Fortran-order RHS matrix (n_free × n_total)
            rhs_matrix = np.asfortranarray(np.empty((n_free, n_total), dtype=np.float64))

            #F_all = np.array(F_list)
            rhs_matrix[:] = (F_list[:, self._cached_free] - self._cached_correction).T
            '''
            for i, F in enumerate(F_list):
                rhs_matrix[:, i] = F[self._cached_free] - self._cached_correction
            '''

            ctx._refs['rhs_matrix'] = rhs_matrix  # prevent GC during solve
            ctx.id.nrhs = n_total
            ctx.id.lrhs = n_free
            ctx.id.rhs = ctx.cast_array(rhs_matrix)

        ctx.run(job=3)  #solve - solution written back into rhs_matrix on rank 0

        if rank != 0:
            return None

        sol_matrix = rhs_matrix  #shape (n_free, n_total), F-order solution is in place

        if self.option.Verbose:
            elapsed = time.time() - start_time
            print(
                f'Batch MUMPS MPI (NRHS={n_total}): {elapsed:.2f}s '
                f'(avg {elapsed / n_total * 1000:.2f} ms/solve)'
            )

        if extract_dofs is not None:
            free_idx = np.searchsorted(self._cached_free, extract_dofs)
            return sol_matrix[free_idx, :]

        solutions = []
        for i in range(n_total):
            U = np.zeros(self._mumps_n_dof, dtype=np.float64)
            U[self._cached_fixed] = self._cached_fixed_vals
            U[self._cached_free] = sol_matrix[:, i]
            solutions.append(U)
        return solutions

    def cleanup_mumps_mpi(self) -> None:
        if self._mumps_ctx is not None:
            self._mumps_ctx.destroy()
            self._mumps_ctx = None
        self._mpi_comm = None
        self._mpi_rank = 0
        self._mumps_n_dof = None

    def setup_factorized(self, K, PARDISO: bool = True):
        start_time = time.time()
        self._cache_submatrices(K, use_preconditioner=False)

        if PARDISO:
            os.environ['MKL_NUM_THREADS'] = str(self.option.NumThreads)
            from pypardiso import PyPardisoSolver
            self._pardiso = PyPardisoSolver()
            self._pardiso_lock = threading.Lock()  # Prevents corruption of PARDISO's internal state
            self._K_free_csr = self._K_free.tocsr()
            self._pardiso.factorize(self._K_free_csr)
        else:
            self._factor = splu(self._K_free.tocsc())

        self._prescribed_correction = self._cached_correction
        self._factorized_pardiso = PARDISO
        if self.option.Verbose:
            print(f'Factorization took {time.time() - start_time:.4f} seconds')

    def solve_factorized(self, F):
        start_time = time.time()
        rhs = F[self._cached_free] - self._prescribed_correction
        U = np.zeros(len(F))
        if self._factorized_pardiso:
            with self._pardiso_lock:
                U[self._cached_free] = self._pardiso.solve(self._K_free_csr, rhs)
        else:
            U[self._cached_free] = self._factor.solve(rhs)
        U[self._cached_fixed] = self._cached_fixed_vals
        if self.option.Verbose:
            print(f'Solving took {time.time() - start_time:.4f} seconds')
        return U

    def cleanup(self):
        """Free PARDISO and AmgX memory."""
        global _pyamgx_initialized
        if hasattr(self, '_pardiso') and self._pardiso is not None:
            self._pardiso.free_memory(everything=True)
        if _pyamgx_initialized:
            if self._amgx_sol is not None:
                self._amgx_sol.destroy()
            if self._amgx_x is not None:
                self._amgx_x.destroy()
            if self._amgx_b is not None:
                self._amgx_b.destroy()
            if self._amgx_mtx is not None:
                self._amgx_mtx.destroy()
            if self._amgx_rsc is not None:
                self._amgx_rsc.destroy()
            if self._amgx_cfg is not None:
                self._amgx_cfg.destroy()
            self._amgx_cfg = None
            self._amgx_rsc = None
            self._amgx_mtx = None
            self._amgx_x = None
            self._amgx_b = None
            self._amgx_sol = None
            self._amgx_prescribed_correction = None
            self._amgx_x0 = None
            _pyamgx.finalize()
            _pyamgx_initialized = False