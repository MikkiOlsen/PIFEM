# PIFEM

A Python finite element code for Mindlin plates on an elastic foundation.

PIFEM solves plate-bending problems with 8-node serendipity (quad8) elements on a Winkler or Pasternak foundation. It was written for jointed concrete pavements, so it also covers slab grids with load-transfer springs across joints and moving vehicle loads, but the plate solver itself is general.

## Features

- Mindlin (shear-deformable) kinematics, with Winkler and Pasternak foundations
- Jointed slab grids with load transfer springs, and moving multi-axle vehicle loads
- Numba-compiled assembly, and a choice of direct, iterative, GPU and MUMPS solvers
- Structured slab meshing, plus mesh import through meshio

## Install

```
pip install -e .
```

Optional direct solvers:

```
pip install -e ".[solvers]"
```

## First run - JIT warm-up

PIFEM JIT-compiles its computationally intensive routines with numba, so the first call into each compiled function pays a compilation cost of a few seconds. Compile the whole codebase once up front instead:

```
python examples/Numba_warmup_script.py
```

This takes about a minute and executes every `@njit` function - element stiffness, foundation, joint springs, and the vehicle force builders. Results are cached to `__pycache__` as `.nbc`/`.nbi` files, so later runs start immediately. The cache is gitignored and machine-specific.

If you ever see a numba compilation error after moving or editing files, the cache has gone stale: delete the `__pycache__` directories and run the warm-up again.

## Usage

```python
import numpy as np
import core
import mesh.quad8
from elements.quad8 import kquad8, apply_foundation, consistent_plate_load

X, T = mesh.quad8.mesh_slab(L=10.0, W=10.0, nx=64, ny=64)

material = core.load_material("pavement_concrete")
H = core.make_H(material, thickness=0.05) # Example of making a material matrix with one material entry
H = np.vstack([core.make_H(material, thickness=0.05),
               core.make_H(core.load_material("steel"), thickness=0.1)]) # Example of multi-entry material matrix

K = kquad8(T, X, H, dof=3)
K = apply_foundation(K, T, X, ks=0.3e9, Gs=0.0, dofs_per_node=3)

FE = core.FESystem(X, dofs_per_node=3)
for axis, value in (('x', 0.0), ('x', 10.0), ('y', 0.0), ('y', 10.0)):
    FE.simply_supported(axis=axis, value=value)

F = core.apply_point_load(X, position=(5.0, 5.0), magnitude=-50000) # Point load
F = consistent_plate_load(T, X, q=-50000) # Uniform load across entire surface if radius and center are None.
U = FE.solve_direct(K, F)
```

## Conventions

- Unit agnostic - no conversion happens anywhere, so any dimensionally consistent
  system works. The bundled materials, vehicles, and examples are all SI (metres,
  pascals, newtons, N/m³ for the subgrade modulus), and results come back in
  whatever system you supplied.
- Three DOF per node, ordered `w`, `theta_x`, `theta_y`. DOF `i` of node `n` sits at
  `U[n * 3 + i]`; `core.get_node_dof` and `core.extract_dof_at_position` handle this
  for you, and `elements.quad8.interpolate_field` samples at an exact coordinate
  rather than snapping to the nearest node.
- `X` is `(n_nodes, 3)` - node coordinates, with `z` normally zero.
- `T` is `(n_elem, 9)` - eight node indices followed by a property number.
- Node indices in `T[:, :8]` are **1-based**, so `X[T[:, :8] - 1]` gives coordinates.
  Ordering follows the serendipity convention: corners 1-4 counter-clockwise, then
  midsides 5-8.
- `T[:, 8]` is the property number, a 1-based index into the rows of `H`.
- `H` is `(n_materials, 3)` - one `[E, nu, thickness]` row per property number.
  Build it with `core.make_H`.

## Loads

Every loading helper returns a global force vector of length
`n_nodes * dofs_per_node`, so results can simply be added together.

| Call | Load |
| --- | --- |
| `core.apply_point_load(X, position, magnitude, dof='w')` | single value at the nearest node to `position` |
| `consistent_plate_load(T, X, q)` | uniform pressure over the whole plate |
| `consistent_plate_load(T, X, q, center, radius)` | uniform pressure over a circular patch |
| `apply_tire_load(T, X, load, tire_pressure, center)` | circular patch, radius derived from load and pressure |
| `build_single_tire_force_vector(...)` | several circular patches in one pass |
| `build_all_tire_force_vectors(...)` | one force vector per position along a sweep |

`apply_point_load` writes `magnitude` into whichever DOF you name - `'w'`,
`'theta_x'` or `'theta_y'` - and takes an existing `F` to accumulate into.

Patch loads are integrated against the element shape functions rather than lumped
onto nodes, and the circle-element overlap is evaluated analytically, so a contact
patch smaller than a single element is still represented correctly.

### Vehicles

A vehicle is a set of wheel offsets with per-wheel loads and tire pressures, read
from JSON in `core/config/vehicles/`:

```python
truck = core.Vehicle.from_dict(core.read_vehicle("volvo_fh16"))
F = truck.build_F(T, X, ref_x=12.0, ref_y=0.0)
```

`ref_x` and `ref_y` place the vehicle reference point; the wheel offsets come from the JSON. `build_F_block` substitutes rectangular footprints for the circular patches. For a sweep, `build_all_F(T, X, ref_x_arr, ref_y)` builds every position in a single compiled pass, which is much faster than calling `build_F` in a loop.

`moving_load_history` runs the whole sweep - building each force vector, solving, and returning the time history of one DOF:

```python
t, w = core.moving_load_history(K, X, T, truck, FE, speed=10.0, ref_y=0.0, x_start=-7.0, x_end=20.0, dt=0.05, measure_pos=(7.5, 0.0), measure_dof='w')
```

This is a sequence of independent static solves, not a transient analysis. `moving_load_history_parallel` spreads them across processes. If the same force vectors are needed repeatedly, `build_force_table` precomputes them on a grid into a sparse table with a `.lookup(x)` method.

### Joints

Adjacent slabs transfer load through spring pairs. Locate the node pairs, then add the springs into `K`:

```python
pairs = core.find_transverse_joint_pairs(X, L_slab, joint_gap, num_x)
K = core.apply_joint_springs(K, pairs, q0=10e9, X=X, along_axis=1)
```

`q0` is a stiffness per unit length of joint, multiplied by each pair's tributary length, so refining the mesh does not change the total joint stiffness. `X` and `along_axis` are keyword-only. `find_longitudinal_joint_pairs` is the counterpart for joints running the other way.

## Solvers

`FESystem` exposes several solvers. `solve_direct` is the default and is a reasonable choice well into the hundreds of thousands of DOF.

| Method | Backend | Use for |
| --- | --- | --- |
| `solve_direct` | PyPardiso, else SciPy `spsolve` | default; `PARDISO=False` forces SciPy |
| `solve_iterative` | PyAMG + CG | large systems; `preconditioner_type='AMG'` or `'Jacobi'` |
| `solve_factorized` | cached LU | many right-hand sides, one stiffness matrix |
| `solve_amgx` | AmgX on GPU | very large systems; needs `pyamgx` |
| `solve_mumps` | MUMPS | large sparse direct solves, especially many right-hand sides |
| `solve_batch_*` | process pool or backend batching | sweeps such as a moving load |

`solve_direct` and `solve_iterative` work straight away. The AmgX, MUMPS, and factorized paths keep a factorisation alive between calls, so each has a `setup_*` to call once and a `cleanup_*` to release it. All are imported lazily, so they cost nothing if unused. AmgX reads its settings from `core/config/amgx_config.json`.

### MUMPS

MUMPS is the recommended solver for large models and for batch evaluations that reuse one stiffness matrix. Factorise once, then solve as often as needed:

```python
FE.setup_mumps(K, sym=True, ordering='metis')   # analysis + factorisation
U = FE.solve_mumps(F)
solutions = FE.solve_batch_mumps(F_list)        # all right-hand sides at once
FE.cleanup_mumps()
```

`sym=True` passes only the upper triangle, which reduces memory use. It holds for most models built here, and MUMPS raises an error if the matrix turns out not to be symmetric - fall back to `sym=False` if that happens. `ordering` accepts `'metis'` (default), `'scotch'`, `'pord'`, `'amd'`, `'amf'`, `'qamd'` or `'auto'`.

**Watch the memory on batch solves.** MUMPS itself supports a sparse right-hand side, but the Python binding this code was developed against did not expose that interface - another binding may. `solve_batch_mumps` therefore assumes a dense right-hand side and converts the whole batch to a Fortran-ordered `(n_free, n_rhs)` float64 array, regardless of how sparse the individual load vectors are. For a system with 500,000 free DOF and 400 positions, the load matrix alone requires roughly 1.6 GB of memory, independent of the factorization overhead.

The function does not chunk internally, so split the list yourself and keep the factorisation across chunks:

```python
solutions = []
for lo in range(0, len(F_list), 50):
    solutions += FE.solve_batch_mumps(F_list[lo:lo + 50])
```

Chunking costs almost nothing - the expensive analysis and factorisation happen once in `setup_mumps`, and each `solve_batch_mumps` call is only the triangular solves.

`setup_mumps_mpi` and `solve_batch_mumps_mpi` distribute the factorisation across MPI ranks, with the right-hand side assembled and the solution collected on rank 0. They need `mpi4py` and a MUMPS build with MPI support.

## Plotting

All plotting helpers live in `elements.quad8` and draw with matplotlib, so call `plt.show()` or `plt.savefig()` yourself afterwards.

| Function | Draws |
| --- | --- |
| `plotelem8(T, X)` | element topology, with optional element and node numbering |
| `plotmesh2d8(T, X)` | flat 2-D mesh, lighter weight for figures |
| `plotbc8(T, X, FE, F)` | supports and loads - check boundary conditions before solving |
| `plotelemdisp8(T, X, U, scale)` | deformed wireframe |
| `plotcontour8(T, X, U, scale)` | deflection contours |
| `plotmoment8(T, X, M)` | bending moment field |
| `plotrotation8(T, X, Rot)` | rotation field |

`plotbc8` is the one worth reaching for first - it renders the model in 3-D with a distinct marker per constrained DOF (`w`, `theta_x`, `theta_y`), which catches a mis-specified support far faster than reading a deflection plot. Two caveats: load symbols are drawn only for the vertical component, as `theta_x` and `theta_y` loads are not implemented in the visualisation, and above 5000 constrained or loaded nodes it falls back to a plain scatter for speed.

The moment and rotation plotters take a single nodal field, so pick one component <from the postprocessing helpers:

```python
from elements.quad8 import compute_moments_quad8, plotmoment8

Mx, My, Mxy = compute_moments_quad8(T, X, H, U)
plotmoment8(T, X, Mx, label='Mx [Nm/m]')
plt.show()
```

`compute_shear_quad8` returns `(Qx, Qy)`, and `compute_surface_strains_quad8` returns the three strain components at a given surface offset.

### A note on performance

Matplotlib is a poor fit for large meshes here. The wireframe plotters draw every element edge separately, so the work grows quickly with element count - a mesh of a few thousand elements already takes noticeable time to appear, and panning or zooming turns sluggish. It is fine at the sizes used in `examples/`; beyond that, plot a subset or export the fields and view them elsewhere.

Moving to a VTK-based viewer such as PyVista would be the natural fix, and would bring proper 3-D interaction with it. Nothing in the postprocessing depends on matplotlib - `compute_moments_quad8` and the other helpers return plain nodal arrays, so the plotting layer could be swapped out on its own without touching the solver.

## Examples

Run these from the repo root after `pip install -e .`.

- `examples/Numba_warmup_script.py` - compile and cache every `@njit` path.
- `examples/westergaard_interior.py` - interior load on a plate on grade, checked against the Westergaard analytical solution for both a point load and a circular contact patch, with a mesh convergence sweep.
- `examples/quartercircle_convergence.py` - a quarter of a simply supported circular plate under uniform load, using symmetry boundary conditions on the two straight edges. Converges to within 0.01% of the thin-plate closed form.
- `examples/rhombic_plate_convergence.py` - a 30° skew plate, simply supported on all four edges. Distorted elements and the obtuse-corner singularity make this converge far more slowly than the other two.
- `examples/import_external_mesh.py` - import a mesh from gmsh, or any other format meshio reads, and dump the resulting node and element arrays.
- `examples/meshes/` - small pre-generated `.msh` meshes to import.

## Configuration

`core/config/` holds `materials.json`, `options.json` (thread count, verbosity),
`amgx_config.json`, and vehicle definitions under `vehicles/`.

## Scope

PIFEM does linear static analysis of Mindlin plates on an elastic foundation, with 8-node quadrilateral elements and small displacements. It does not do dynamics, buckling, geometric or material nonlinearity, contact, or solid/shell elements. Moving vehicle loads are handled as a sequence of independent static solves, not as a transient analysis.

## License

MIT - see `LICENSE`.
