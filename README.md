# PIFEM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21759993.svg)](https://doi.org/10.5281/zenodo.21759993)

A Python finite element code for Mindlin plates on an elastic foundation.

PIFEM solves plate-bending problems with 8-node serendipity (quad8) elements on a Winkler or Pasternak foundation, with Numba-compiled assembly and a choice of direct, iterative, GPU and MUMPS solvers. It was written for jointed concrete pavements, so it also covers slab grids with load-transfer springs across joints, moving vehicle loads, and structured slab meshing with meshio import, but the plate solver itself is general.

## Scope

PIFEM does linear static analysis of Mindlin plates on an elastic foundation, with 8-node quadrilateral elements and small displacements. It does not do dynamics, buckling, geometric or material nonlinearity, contact, or solid/shell elements. Moving vehicle loads are handled as a sequence of independent static solves, not as a transient analysis.

## Install

### Requirements
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- Git (for cloning the repository)

### Clone the repository

```bash
git clone https://github.com/MikkiOlsen/PIFEM.git
cd PIFEM
```

### Recommended: virtual environment with uv

```bash
# Create and activate a virtual environment
uv venv PIFEM-env
source PIFEM-env/bin/activate  # Linux/macOS
# or
PIFEM-env\Scripts\activate  # Windows

# Install PIFEM
uv pip install -e .
```

### Alternative: plain pip

```bash
pip install -e .
```

### Optional direct solvers

```bash
uv pip install -e ".[solvers]"   # or: pip install -e ".[solvers]"
```

## First run - JIT warm-up

PIFEM JIT-compiles its computationally intensive routines with numba, so the first call into each compiled function pays a compilation cost of a few seconds. Running the warmup script is optional but avoids this cost on the first real run:

```
python examples/Numba_warmup_script.py
```

Results are cached to `__pycache__` as `.nbc`/`.nbi` files, so later runs start immediately.

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

## Meshing

`mesh.quad8` builds structured quad8 meshes directly. Anything more involved can be built elsewhere and imported.

| Call | Produces |
| --- | --- |
| `mesh_slab(L, W, nx, ny)` | a single rectangular slab of `nx` by `ny` elements |
| `mesh_slab_grid(L_slab, W_slab, nx_slab, ny_slab, num_slabs_x, num_slabs_y, joint_x, joint_y)` | a grid of separate slabs with joint gaps between them |
| `mesh_slab_grid_gmsh(...)` | the same grid meshed by gmsh, graded towards a point |

All return `(X, T)`. `mesh_slab_grid` also takes `nodes_indices=True`, which additionally returns `elem_slab` and `node_slab` holding the 1-based slab id of every element and node - useful for pulling results out one slab at a time, as in `T[elem_slab == 5]`.

`mesh_slab_grid_gmsh` uses the gmsh API, so it needs gmsh installed. It grades element size from `lc_fine` near `(x_refine, y_refine)` out to `lc_coarse`, and takes an optional `row_labels` to name the physical surface of each slab row.

### Importing a mesh

`import_msh` reads any format meshio can parse that contains 8-node quadrilateral elements, and returns PIFEM's `(X, T)` with 1-based connectivity:

```python
from mesh.gmsh import import_msh

X, T = import_msh("slabs.msh")
```

Property numbers come from gmsh only. `import_msh` takes the physical surfaces recorded in the file, ranks them by tag number, and remaps them onto a consecutive 1-based range. Other formats do not store anything like that, so every element comes back as property number 1.

Group *names* are ignored by default, so the lower tag becomes property number 1 regardless of its name. If the mesh has more than one material, name them explicitly instead:

```python
X, T = import_msh("slabs.msh", materials=["mat_a", "mat_b"])   # -> propno 1, 2
```

A missing name raises an error, rather than a mesh whose two materials have swapped places. Passing `propno=<int>` overrides both and assigns a single property number to every element.

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
| `build_all_tire_force_vectors(...)` | one force vector per position along a run |

`apply_point_load` writes `magnitude` into whichever DOF you name, `'w'`, `'theta_x'` or `'theta_y'`, and takes an existing `F` to accumulate into.

Patch loads are integrated against the element shape functions rather than lumped onto nodes, and the circle-element overlap is evaluated analytically, so a contact patch smaller than a single element is still represented correctly.

### Vehicles

A vehicle is a set of wheel offsets with per-wheel loads and tire pressures, read
from JSON in `core/config/vehicles/`:

```python
truck = core.Vehicle.from_dict(core.read_vehicle("volvo_fh16"))
F = truck.build_F(T, X, ref_x=12.0, ref_y=0.0)
```

`ref_x` and `ref_y` place the vehicle reference point; the wheel offsets come from the JSON. `build_F_block` substitutes rectangular footprints for the circular patches. For a run, `build_all_F(T, X, ref_x_arr, ref_y)` builds every position in a single compiled pass, which is much faster than calling `build_F` in a loop.

`moving_load_history` runs the whole run, building each force vector, solving, and returning the time history of one DOF:

```python
t, w = core.moving_load_history(K, X, T, truck, FE, speed=10.0, ref_y=0.0, x_start=-7.0, x_end=20.0, dt=0.05, measure_pos=(7.5, 0.0), measure_dof='w')
```

`moving_load_history_parallel` spreads them across processes. If the same loading run needs to be evaluated multiple times, for example, when varying stiffness or boundary conditions while keeping the vehicle path fixed, `build_force_table` precomputes the force vectors on a uniform x-grid into a sparse table. `.lookup(x)` snaps to the nearest grid point and returns a dense force vector for a scalar position, or a batch array for multiple positions.

### Joints

Adjacent slabs transfer load through spring pairs. Locate the node pairs, then add the springs into `K`:

```python
pairs = core.find_transverse_joint_pairs(X, L_slab, joint_gap, num_x)
K = core.apply_joint_springs(K, pairs, q0=10e9, X=X, along_axis=1)
```

`q0` is a stiffness per unit length of joint, multiplied by each pair's tributary length, so refining the mesh does not change the total joint stiffness. `X` and `along_axis` are keyword-only. `find_longitudinal_joint_pairs` is the counterpart for joints running the other way.

## Boundary conditions

`FESystem` records the constraints; they are applied when you solve, not when you set them.

| Method | Constrains |
| --- | --- |
| `simply_supported(axis, value)` | `w` on the plane `axis = value` |
| `simply_supported_edge(p_start, p_end)` | `w` along the line between two points |
| `simply_supported_nodes(nodes)` | `w` at an explicit list of nodes |
| `clamped(axis, value)` | `w`, `theta_x` and `theta_y` on a plane |
| `clamped_edge(p_start, p_end)` | the same along a line |
| `clamped_nodes(nodes)` | the same at given nodes |
| `symmetry(axis, value)` | `theta_x` on an x-plane, `theta_y` on a y-plane |
| `fix_nodes(nodes, dof_offsets, value)` | any set of DOFs, optionally to a non-zero value |

The plane methods match coordinates within `tol=1e-5` by default; the edge methods use `tol=1e-3` and take every node within that distance of the line. The edge form is the one to use on a mesh whose boundaries are not axis-aligned, as in `rhombic_plate_convergence.py`.

`symmetry` deliberately leaves `w` free, which is what a symmetry plane requires. `quartercircle_convergence.py` applies it to the two straight edges of the quarter model.

`fix_nodes` is the general case the others are built on: pass `dof_offsets=(0, 1, 2)` for all three DOFs, or a single offset, together with a `value` to prescribe a non-zero displacement. `get_indices()` returns the free and fixed DOF arrays and `get_fixed_nodes()` the constrained nodes.

## Solvers

`FESystem` exposes several solvers. `solve_direct` is the default and is a reasonable choice well into the hundreds of thousands of DOF.

| Method | Backend | Use for |
| --- | --- | --- |
| `solve_direct` | PyPardiso, else SciPy `spsolve` | default; `PARDISO=False` forces SciPy |
| `solve_iterative` | PyAMG + CG | large systems; `preconditioner_type='AMG'` or `'Jacobi'` |
| `solve_factorized` | cached LU | many right-hand sides, one stiffness matrix |
| `solve_amgx` | AmgX on GPU | very large systems; needs `pyamgx` as well as an AMGX build |
| `solve_mumps` | MUMPS | large sparse direct solves, especially many right-hand sides |
| `solve_batch_*` | process pool or backend batching | a whole list of load vectors in one call, e.g. a moving load |

`solve_direct` and `solve_iterative` work straight away. The AmgX, MUMPS, and factorized paths keep a factorisation alive between calls, so each has a `setup_*` to call once and a `cleanup_*` to release it. All are imported lazily, so they cost nothing if unused. AmgX reads its settings from `core/config/amgx_config.json`.

`solve_factorized` is a lighter alternative to MUMPS for sequential single right-hand side solves: `setup_factorized` factorises once with PyPardiso or SciPy's splu, and each `solve_factorized` call is then just the triangular solve. It has no batch interface, making it the natural fit for `moving_load_history`, where each step is solved one at a time.

### MUMPS

MUMPS is the recommended solver for large models and for batch evaluations that reuse one stiffness matrix. Factorise once, then solve as often as needed:

```python
FE.setup_mumps(K, sym=True, ordering='metis')   # analysis + factorisation
U = FE.solve_mumps(F)
solutions = FE.solve_batch_mumps(F_list)        # all right-hand sides at once
FE.cleanup_mumps()
```

`sym=True` passes only the upper triangle, which reduces memory use. It holds for most models built here, and MUMPS raises an error if the matrix turns out not to be symmetric. Fall back to `sym=False` if that happens. `ordering` accepts `'metis'` (default), `'scotch'`, `'pord'`, `'amd'`, `'amf'`, `'qamd'` or `'auto'`.

**Watch the memory on batch solves.** MUMPS itself supports a sparse right-hand side, but the Python binding this code was developed for did not expose that interface; another binding may. `solve_batch_mumps` therefore assumes a dense right-hand side and converts the whole batch to a Fortran-ordered `(n_free, n_rhs)` float64 array, regardless of how sparse the individual load vectors are. For a system with 500,000 free DOF and 400 positions, the load matrix alone requires roughly 1.6 GB of memory, independent of the factorization overhead.

The function does not chunk internally, so split the list yourself and reuse the factorisation across chunks:

```python
solutions = []
for lo in range(0, len(F_list), 50):
    solutions += FE.solve_batch_mumps(F_list[lo:lo + 50])
```

`setup_mumps_mpi` and `solve_batch_mumps_mpi` distribute the factorisation across MPI ranks, with the right-hand side assembled and the solution collected on rank 0. They need `mpi4py` and a MUMPS build with MPI support.

## Plotting

All plotting helpers live in `elements.quad8` and draw with matplotlib, so call `plt.show()` or `plt.savefig()` yourself afterwards.

| Function | Draws |
| --- | --- |
| `plotelem8(T, X)` | element topology, with optional element and node numbering |
| `plotmesh2d8(T, X)` | flat 2D mesh, lighter weight for figures |
| `plotbc8(T, X, FE, F)` | supports and loads - check boundary conditions before solving |
| `plotelemdisp8(T, X, U, scale)` | deformed wireframe |
| `plotcontour8(T, X, U, scale)` | deflection contours |
| `plotmoment8(T, X, M)` | bending moment field |
| `plotrotation8(T, X, Rot)` | rotation field |

`plotbc8` is the one worth reaching for first: it renders the model in 3D with a distinct marker per constrained DOF (`w`, `theta_x`, `theta_y`), which catches a mis-specified support far faster than reading a deflection plot. Two caveats: load symbols are drawn only for the vertical component, as `theta_x` and `theta_y` loads are not implemented in the visualisation, and above 5000 constrained or loaded nodes it falls back to a plain scatter for speed.

The moment and rotation plotters take a single nodal field, so pick one component from the postprocessing helpers:

```python
from elements.quad8 import compute_moments_quad8, plotmoment8

Mx, My, Mxy = compute_moments_quad8(T, X, H, U)
plotmoment8(T, X, Mx, label='Mx [Nm/m]')
plt.show()
```

`compute_shear_quad8` returns `(Qx, Qy)`, and `compute_surface_strains_quad8` returns the three strain components at a given surface offset.

### A note on performance

Matplotlib is a poor fit for large meshes here. The wireframe plotters draw every element edge separately, so the work grows quickly with element count: a mesh of a few thousand elements already takes noticeable time to appear, and panning or zooming turns sluggish. It is fine at the sizes used in `examples/`; beyond that, plot a subset or export the fields and view them elsewhere.

Moving to a VTK-based viewer such as PyVista would be the natural fix, and would bring proper 3-D interaction with it. Nothing in the postprocessing depends on matplotlib - `compute_moments_quad8` and the other helpers return plain nodal arrays, so the plotting layer could be swapped out on its own without touching the solver.

## Examples

The following examples are included in `examples/`:

- `Numba_warmup_script.py` - compile and cache every `@njit` path.
- `westergaard_interior.py` - interior load on a plate on grade, checked against the Westergaard analytical solution for both a point load and a circular contact patch, with a mesh convergence study.
- `quartercircle_convergence.py` - a quarter of a simply supported circular plate under uniform load, using symmetry boundary conditions on the two straight edges. Converges to within 0.01% of the thin-plate closed form.
- `rhombic_plate_convergence.py` - a 30° skew plate, simply supported on all four edges. Distorted elements and the obtuse-corner singularity make this converge far more slowly than the other two.
- `import_external_mesh.py` - import a mesh from gmsh, or any other format meshio reads, and dump the resulting node and element arrays.
- `meshes/` - small pre-generated `.msh` meshes to import.

## Configuration

`core/config/` holds `materials.json`, `options.json` (thread count, verbosity),
`amgx_config.json`, and vehicle definitions under `vehicles/`.


## License

MIT - see `LICENSE`.