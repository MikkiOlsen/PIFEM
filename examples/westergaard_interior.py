import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy.integrate import quad
from scipy.special import j0, j1
import core
import mesh.quad8
from elements.quad8 import kquad8, apply_foundation

FIG_W = 4.5
FIG_H = 3.5
C_PIFEM = 'black'

t         = 0.05
K_WINKLER = 0.3e9
P         = -50000

material  = core.load_material("pavement_concrete")
H         = core.make_H(material, t)
E         = material['E']
nu        = material['nu']

D   = E * t**3 / (12.0 * (1.0 - nu**2))
ell = (D / K_WINKLER) ** 0.25

print(f"D  = {D:.4e} N·m")
print(f"l  = {ell:.4f} m")


def westergaard_point_load(dist_array, y_dist=0.0):
    def _integrand(m, r):
        sq      = np.sqrt(m**4 + 1)
        beta_w  = np.sqrt(max((sq - m**2) / 2.0, 0.0))
        gamma_w = np.sqrt((sq + m**2) / 2.0)
        num = (np.cos(m * r / ell)
               * (beta_w  * np.cos(beta_w  * y_dist / ell)
                  + gamma_w * np.sin(beta_w  * y_dist / ell))
               * np.exp(-gamma_w * y_dist / ell))
        return num / (beta_w**2 + gamma_w**2)

    scale = np.sign(P) * abs(P) / (2.0 * np.pi * K_WINKLER * ell**2) * 1e3  # mm

    w = np.empty(len(dist_array))
    for i, r in enumerate(dist_array):
        val, _ = quad(_integrand, 0, 50, args=(r,), limit=1000)
        w[i] = val * scale
    return w


def westergaard_circular_load(dist_array, tire_radius):
    a = tire_radius
    p = P / (np.pi * a**2)                              # pressure (same sign as P)

    prefactor_circ  = p * a / (K_WINKLER * ell) * 1e3  # mm
    prefactor_point = np.sign(P) * abs(P) / (2.0 * np.pi * K_WINKLER * ell**2) * 1e3

    def _integrand_circ(m, r):
        return j0(m * r / ell) * j1(m * a / ell) / (1.0 + m**4)

    def _integrand_point(m, r):
        sq      = np.sqrt(m**4 + 1)
        beta_w  = np.sqrt(max((sq - m**2) / 2.0, 0.0))
        gamma_w = np.sqrt((sq + m**2) / 2.0)
        return np.cos(m * r / ell) * beta_w / (beta_w**2 + gamma_w**2)

    w = np.empty(len(dist_array))
    for i, r in enumerate(dist_array):
        r_abs = abs(r)
        if r_abs >= 3.0 * a:
            val, _ = quad(_integrand_point, 0, 50, args=(r_abs,), limit=1000)
            w[i]   = val * prefactor_point
        else:
            val, _ = quad(_integrand_circ, 0, np.inf, args=(r_abs,), limit=1000)
            w[i]   = val * prefactor_circ
    return w


def _solve(L, W, NX, NY, use_circular=False, tire_radius=None):
    from elements.quad8.fquad8 import build_single_tire_force_vector

    X, T = mesh.quad8.mesh_slab(L, W, NX, NY)
    K    = kquad8(T, X, H, dof=3)
    K    = apply_foundation(K, T, X, ks=K_WINKLER, Gs=0.0, dofs_per_node=3)

    FE = core.FESystem(X, dofs_per_node=3)
    FE.simply_supported(axis='x', value=0.0)
    FE.simply_supported(axis='x', value=L)
    FE.simply_supported(axis='y', value=0.0)
    FE.simply_supported(axis='y', value=W)

    if use_circular:
        tire_pressure = abs(P) / (np.pi * tire_radius**2)
        F = build_single_tire_force_vector(
            T, X,
            wheel_x_off = np.array([0.0]),
            wheel_y     = np.array([W / 2.0]),
            wheel_q     = np.array([-tire_pressure]),
            wheel_r     = np.array([tire_radius]),
            ref_x       = L / 2.0,
            dof         = 3,
        )
    else:
        F = core.apply_point_load(X, position=(L / 2.0, W / 2.0), magnitude=P)

    U = FE.solve_direct(K, F)
    return X, T, U

def profile_convergence(L=10.0, W=10.0):
    resolutions = [(16, 16), (32, 32), (64, 64), (80, 80), (100, 100), (150,150), (200, 200)]

    print(f"\nProfile convergence  ({L:.0f}×{W:.0f} m,  l = {ell:.4f} m)")

    NX_ref, NY_ref = resolutions[-1]
    X_ref, _, _ = _solve(L, W, NX_ref, NY_ref)
    y_mid   = W / 2.0
    mask    = np.isclose(X_ref[:, 1], y_mid, atol=1e-8)
    x_ref   = np.sort(X_ref[mask, 0])

    data = {'x [m]': x_ref, 'dist from centre [m]': x_ref - L / 2.0}

    for NX, NY in resolutions:
        X, T, U = _solve(L, W, NX, NY)
        n_elem  = T.shape[0]

        mask  = np.isclose(X[:, 1], y_mid, atol=1e-8)
        idx   = np.where(mask)[0]
        order = np.argsort(X[idx, 0])
        idx   = idx[order]

        x_line = X[idx, 0]
        w_line = core.extract_dof(U, 'w')[idx] * 1000   # mm

        w_interp = np.interp(x_ref, x_line, w_line)
        label    = f'w_FE {NX}x{NY} [mm]  (n={n_elem})'
        data[label] = w_interp
        print(f"{NX:4d}×{NY:<4d}  n_elem={n_elem:>8d}  w_centre={w_line[len(w_line)//2]:.4f} mm")

    return data


def boundary_effects(NX=64, NY=64, L=2.0, W=2.0):
    X, T, U = _solve(L, W, NX, NY)

    y_mid = W / 2.0
    mask  = np.isclose(X[:, 1], y_mid, atol=1e-8)
    idx   = np.where(mask)[0]
    order = np.argsort(X[idx, 0])
    idx   = idx[order]

    dist = X[idx, 0] - L / 2.0
    w_fe = core.extract_dof(U, 'w')[idx] * 1000

    print(f"\nComputing Westergaard analytical solution for {L:.0f}x{W:.0f} m plate...")
    w_analytical = westergaard_point_load(dist, y_dist=0.0)

    h_mm = L / (2 * NX) * 1000
    print(f"Boundary effects {L:.0f}×{W:.0f} m ({NX}x{NY} mesh, h = {h_mm:.1f} mm = {h_mm/1e3/ell:.2f}ℓ)")

    return {
        'dist_from_centre [m]': dist,
        'w_FE [mm]':            w_fe,
        'w_analytical [mm]':    w_analytical,
        'L': L, 'W': W,
    }


def plot_profile_convergence(data, L=10.0, Point_line=True, analytical=None):
    dist = np.asarray(data['dist from centre [m]'])
    keys = [k for k in data if k.startswith('w_FE')]

    n = len(keys)
    colours = [str(v) for v in np.linspace(0.65, 0.0, n)]

    def _n_elem(key):
        m = re.search(r'n=(\d+)', key)
        return int(m.group(1))

    n_elems = [_n_elem(k) for k in keys]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, key in enumerate(keys):
        ax.plot(dist, data[key], color=colours[i], linewidth=1.5)

    if analytical is not None:
        ax.plot(dist, analytical, color='red', linestyle='--', linewidth=1.5, label='Analytical')
    if Point_line:
        ax.axvline(0, color='black', linestyle=':', linewidth=1, label='Load point')
    ax.set_xlabel('Distance from centre [m]')
    ax.set_ylabel('Deflection [mm]')
    ax.set_title(f'Profile convergence - interior load, {L:.0f}×{L:.0f} m plate')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    if analytical is not None or Point_line:
        ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Discrete colorbar: one band per mesh, ticked by its total element count.
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(n))
    cbar.ax.set_yticklabels([f'{ne:,}' for ne in n_elems])
    cbar.set_label('Number of elements')

    fig.tight_layout()
    return fig


def plot_boundary_effects(data):
    fig, ax = plt.subplots()

    dist = data['dist_from_centre [m]']
    L, W = data['L'], data['W']

    ax.plot(dist, data['w_FE [mm]'],         color='black',  linewidth=2,   label=f'FE ({L:.0f}×{W:.0f} m)')
    ax.plot(dist, data['w_analytical [mm]'], color='red',    linewidth=1.5, linestyle='--', label='Analytical')

    ax.axvline(0, color='grey', linestyle=':', linewidth=1, label='Load point')
    ax.set_xlabel('Distance from centre [m]')
    ax.set_ylabel('Deflection [mm]')
    ax.set_title(f'Boundary effects - {L:.0f}×{W:.0f} m plate vs analytical')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()


def circular_load_comparison(NX=200, NY=200, L=10.0, W=10.0, tire_pressure_pa=0.83e6):
    a = np.sqrt(abs(P) / (tire_pressure_pa * np.pi))

    X, T, U = _solve(L, W, NX, NY, use_circular=True, tire_radius=a)

    y_mid = W / 2.0
    mask  = np.isclose(X[:, 1], y_mid, atol=1e-8)
    idx   = np.where(mask)[0]
    order = np.argsort(X[idx, 0])
    idx   = idx[order]

    dist = X[idx, 0] - L / 2.0
    w_fe = core.extract_dof(U, 'w')[idx] * 1000

    print(f"\nCircular load: a = {a:.4f} m  "
          f"(P = {abs(P)/1e3:.1f} kN,  p = {tire_pressure_pa/1e6:.2f} MPa)")
    print(f"Computing Westergaard circular analytical solution for {L:.0f}x{W:.0f} m plate...")
    w_analytical = westergaard_circular_load(dist, a)

    return {
        'dist_from_centre [m]': dist,
        'w_FE [mm]':            w_fe,
        'w_analytical [mm]':    w_analytical,
        'L': L, 'W': W,
        'tire_radius': a,
    }


def circular_profile_convergence(resolutions=None, L=10.0, W=10.0, tire_pressure_pa=0.83e6):
    if resolutions is None:
        resolutions = [(16, 16), (32, 32), (64, 64), (80, 80), (100, 100), (150, 150), (200, 200)]

    a = np.sqrt(abs(P) / (tire_pressure_pa * np.pi))

    print(f"\nCircular-load convergence  ({L:.0f}×{W:.0f} m, a = {a:.4f} m,  l = {ell:.4f} m)")

    # Reference mid-line node positions from the finest mesh
    NX_ref, NY_ref = resolutions[-1]
    X_ref, _, _ = _solve(L, W, NX_ref, NY_ref, use_circular=True, tire_radius=a)
    y_mid = W / 2.0
    mask  = np.isclose(X_ref[:, 1], y_mid, atol=1e-8)
    x_ref = np.sort(X_ref[mask, 0])

    data = {'x [m]': x_ref, 'dist from centre [m]': x_ref - L / 2.0}
    meta = []

    for NX, NY in resolutions:
        X, T, U = _solve(L, W, NX, NY, use_circular=True, tire_radius=a)
        n_elem  = T.shape[0]

        mask  = np.isclose(X[:, 1], y_mid, atol=1e-8)
        idx   = np.where(mask)[0]
        order = np.argsort(X[idx, 0])
        idx   = idx[order]

        x_line = X[idx, 0]
        w_line = core.extract_dof(U, 'w')[idx] * 1000   # mm

        w_interp = np.interp(x_ref, x_line, w_line)
        label    = f'w_FE {NX}x{NY} [mm]  (n={n_elem})'
        data[label] = w_interp
        meta.append({'NX': NX, 'NY': NY, 'h': L / NX, 'n_elem': n_elem})
        print(f"{NX:4d}×{NY:<4d}  n_elem={n_elem:>8d}  h={L/NX:.4f} m  w_centre={w_line[len(w_line)//2]:.4f} mm")

    return data, meta, a


def plot_circular_convergence_error(data, meta, a, L):
    dist = np.asarray(data['dist from centre [m]'])
    w_ana = westergaard_circular_load(dist, a)

    keys   = [k for k in data if k.startswith('w_FE')]
    h      = np.array([m['h'] for m in meta])
    n_elem = np.array([m['n_elem'] for m in meta])

    rmse = np.array([np.sqrt(np.mean((np.asarray(data[k]) - w_ana) ** 2)) for k in keys])

    w_peak = np.max(np.abs(w_ana))
    rmse_pct = rmse / w_peak * 100.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax.plot(n_elem, rmse_pct, color=C_PIFEM, marker='o', linestyle='-', label=rf'RMSE over mid-line')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Deflection RMSE vs analytical [% of peak]')
    ax.set_title(f'Circular-load convergence rate - {L:.0f}×{L:.0f} m plate')
    ax.tick_params(which='both', direction='in')
    ax.legend(loc='best', frameon=True, edgecolor='#cccccc', fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    def _minor_fmt(y, _):
        if y <= 0:
            return ''
        mantissa = round(y / 10.0 ** np.floor(np.log10(y)))
        return f'{y:g}' if mantissa in (2, 5) else ''

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(_minor_fmt))
    ax.tick_params(axis='y', which='minor', labelsize=7)
    return fig


def plot_circular_comparison(data):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    dist = data['dist_from_centre [m]']
    L, W = data['L'], data['W']
    a    = data['tire_radius']

    ax.plot(dist, data['w_FE [mm]'],         color='black',  linewidth=2, label="PIFEM")
    ax.plot(dist, data['w_analytical [mm]'], color='red',    linewidth=1.5, linestyle='--', label='Analytical')

    ax.axvline( a, color='grey', linestyle=':', linewidth=1)
    ax.axvline(-a, color='grey', linestyle=':', linewidth=1, label=f'Contact radius a = {a:.3f} m')
    ax.set_xlabel('Distance from centre [m]')
    ax.set_ylabel('Deflection [mm]')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_title(f'Circular load comparison - {L:.0f}x{W:.0f} m plate  '
                 f'(P = {abs(P)/1e3:.0f} kN)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    L = W = 10.0
    resolutions = [(16, 16), (32, 32), (64, 64), (100, 100)]

    data, meta, a = circular_profile_convergence(resolutions=resolutions, L=L, W=W)
    dist  = np.asarray(data['dist from centre [m]'])
    w_ana = westergaard_circular_load(dist, a)

    data_cmp = circular_load_comparison(NX=meta[-1]['NX'], NY=meta[-1]['NY'], L=L, W=W)

    plot_profile_convergence(data, L=L, Point_line=False, analytical=w_ana)
    plot_circular_comparison(data_cmp)
    plot_circular_convergence_error(data, meta, a, L)
    plt.show()