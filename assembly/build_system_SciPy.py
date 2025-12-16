"""
SciPy/Numpy assembly of linear system A x = b for Tg + interface (Ts, mpp) + radius (Rd).

Differences vs PETSc build_system:
- Returns dense numpy arrays (A, b) for SciPy-based solvers.
- No PETSc dependency; intended for Windows / SciPy workflow.

Scope (Step 11+12, SciPy backend):
- Gas temperature Tg:
  - fully implicit time term (theta = 1.0),
  - implicit diffusion in gas cells,
  - explicit Stefan convection (using physics.stefan_velocity + flux_convective_gas),
  - outer boundary: strong Dirichlet T = T_inf.
- Interface conditions (optional, controlled by cfg.physics.include_Ts / include_mpp):
  - Ts energy jump: gas/liquid conduction + latent heat (single-condensable MVP, no diffusion enthalpy yet),
  - mpp Stefan mass balance: diffusive flux of condensable species vs eq_result["Yg_eq"].
- Radius evolution (optional, cfg.physics.include_Rd):
  - backward-Euler dR/dt = -mpp / rho_l_if,
  - coupled to mpp via radius_eq.build_radius_row.

Inputs:
- cfg, grid, layout, state_old, props, dt,
- optional eq_result (interface equilibrium Yg_eq) and state_guess (for interface / radius diagnostics).

Contract:
- layout must contain Tg block; Ts / mpp / Rd blocks only used if cfg.physics.include_* is True.
- if include_mpp is True, eq_result must contain 'Yg_eq' with full gas-species mass fractions.
"""


from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.stefan_velocity import compute_stefan_velocity
from physics.flux_convective_gas import compute_gas_convective_flux_T
from physics.interface_bc import build_interface_coeffs, EqResultLike
from physics.radius_eq import build_radius_row
from physics.interface_bc import InterfaceCoeffs  # type hint only
from physics.radius_eq import RadiusCoeffs  # type hint only


def _apply_center_bc_Tg(A: np.ndarray, row: int, coeff: float, col_neighbor: int) -> None:
    """Inner gas boundary Neumann (zero-gradient) using a reflected ghost cell."""
    A[row, col_neighbor] += -coeff


def _apply_outer_dirichlet_Tg(A: np.ndarray, b: np.ndarray, row: int, T_far: float) -> None:
    """Strong Dirichlet at outer boundary: T = T_far."""
    A[row, :] = 0.0
    A[row, row] = 1.0
    b[row] = T_far


def _scatter_interface_rows(
    A: np.ndarray,
    b: np.ndarray,
    iface_coeffs: "InterfaceCoeffs",
) -> None:
    """
    Scatter interface rows (Ts, mpp) into global matrix/vector.

    Each InterfaceRow.row is a global row index; cols/vals are global unknown indices/coefficients.
    """
    for row_def in iface_coeffs.rows:
        r = row_def.row
        if not row_def.cols:
            # Allow placeholder rows with no coefficients.
            continue
        for c, v in zip(row_def.cols, row_def.vals):
            A[r, c] += v
        b[r] += row_def.rhs


def _scatter_radius_row(
    A: np.ndarray,
    b: np.ndarray,
    rad_coeffs: "RadiusCoeffs",
) -> None:
    """Scatter radius equation row into global matrix/vector."""
    r = rad_coeffs.row
    for c, v in zip(rad_coeffs.cols, rad_coeffs.vals):
        A[r, c] += v
    b[r] += rad_coeffs.rhs


def build_transport_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    eq_result: EqResultLike | None = None,
    state_guess: State | None = None,
    return_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assemble the coupled linear system A x = b for one time step.

    Unknowns (depending on cfg/layout):
    - Tg: gas temperature in Ng cells;
    - Ts: interface temperature (single scalar);
    - mpp: interface mass flux (single scalar, positive for evaporation);
    - Rd: droplet radius (single scalar).

    Contents:
    - Tg block:
      time term + implicit diffusion + explicit Stefan convection;
      outer gas cell has strong Dirichlet T = cfg.initial.T_inf.
    - Interface block (via physics.interface_bc.build_interface_coeffs):
      Ts energy jump (q_g + q_l - q_lat = 0),
      mpp Stefan mass condition (single condensable species).
    - Radius block (via physics.radius_eq.build_radius_row):
      backward-Euler dR/dt = -mpp / rho_l_if.

    Parameters
    ----------
    cfg, grid, layout, state_old, props, dt :
        Core problem definition and previous time level state.
    eq_result : mapping, optional
        Must contain 'Yg_eq' (full gas mass-fraction vector) if include_mpp is True.
    state_guess : State, optional
        Current nonlinear guess used in interface / radius diagnostics (MVP: coefficients do not depend on it).
    return_diag : bool, default False
        If True, returns (A, b, diag_sys) where diag_sys aggregates diagnostics
        from interface_bc and radius_eq; otherwise returns (A, b) only.
    """
    if not layout.has_block("Tg"):
        raise ValueError("layout missing Tg block required for Step 6 assembly.")
    if state_old.Tg.shape != (grid.Ng,):
        raise ValueError(f"Tg shape {state_old.Tg.shape} != ({grid.Ng},)")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Step 6 MVP only supports theta=1.0 (fully implicit Tg diffusion).")

    if state_guess is None:
        state_guess = state_old

    Ng = grid.Ng
    Nc = grid.Nc
    N = layout.n_dof()
    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    diag_sys: Dict[str, Any] = {}

    gas_start = grid.gas_slice.start if grid.gas_slice is not None else grid.Nl

    for ig in range(Ng):
        row = layout.idx_Tg(ig)
        rho = float(props.rho_g[ig])
        cp = float(props.cp_g[ig])
        k_i = float(props.k_g[ig])

        cell_idx = gas_start + ig
        V = float(grid.V_c[cell_idx])
        aP_time = rho * cp * V / dt

        aP = aP_time
        b_i = aP_time * state_old.Tg[ig]

        # Left face (ig-1/2)
        if ig > 0:
            rL = grid.r_c[cell_idx - 1]
            rC = grid.r_c[cell_idx]
            A_f = float(grid.A_f[cell_idx])
            dr = rC - rL
            k_face = 0.5 * (k_i + float(props.k_g[ig - 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, layout.idx_Tg(ig - 1)] += -coeff
        else:
            if Ng > 1:
                rC = grid.r_c[cell_idx]
                rR = grid.r_c[cell_idx + 1]
                A_f = float(grid.A_f[cell_idx])
                dr = rR - rC
                k_face = 0.5 * (k_i + float(props.k_g[ig + 1]))
                coeff = k_face * A_f / dr
                _apply_center_bc_Tg(A, row, coeff, layout.idx_Tg(ig + 1))
                aP += coeff

        # Right face (ig+1/2)
        if ig < Ng - 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f = float(grid.A_f[cell_idx + 1])
            dr = rR - rC
            k_face = 0.5 * (k_i + float(props.k_g[ig + 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A[row, layout.idx_Tg(ig + 1)] += -coeff

        A[row, row] += aP
        b[row] += b_i

    # --- Stefan velocity and convective flux (explicit) ---
    stefan = compute_stefan_velocity(cfg, grid, props, state_old)
    u_face = stefan.u_face

    q_conv = compute_gas_convective_flux_T(
        cfg=cfg,
        grid=grid,
        props=props,
        Tg=state_old.Tg,  # explicit in time
        u_face=u_face,
    )

    # Add explicit convective source to RHS: b[row] -= (A_R*q_R - A_L*q_L)
    for ig in range(Ng):
        row = layout.idx_Tg(ig)
        cell_idx = gas_start + ig
        f_L = cell_idx
        f_R = cell_idx + 1
        A_L = float(grid.A_f[f_L])
        A_R = float(grid.A_f[f_R])
        q_L = float(q_conv[f_L])
        q_R = float(q_conv[f_R])
        S_conv = A_R * q_R - A_L * q_L  # net outward convective power (W)
        b[row] -= S_conv

    # Outer boundary Dirichlet on last gas cell
    Tg_far = float(cfg.initial.T_inf)
    row_bc = layout.idx_Tg(Ng - 1)
    _apply_outer_dirichlet_Tg(A, b, row_bc, Tg_far)

    # --- Interface equations: Ts energy jump + mpp Stefan (single-condensable) ---
    phys = cfg.physics
    if (phys.include_Ts or phys.include_mpp) and (layout.has_block("Ts") or layout.has_block("mpp")):
        if phys.include_mpp and layout.has_block("mpp") and eq_result is None:
            raise ValueError("Step 11: mpp equation requires eq_result with 'Yg_eq'.")
        iface_coeffs = build_interface_coeffs(
            grid=grid,
            state=state_guess,
            props=props,
            layout=layout,
            cfg=cfg,
            eq_result=eq_result,
        )
        _scatter_interface_rows(A, b, iface_coeffs)
        diag_sys.update(iface_coeffs.diag)

    # --- Radius evolution equation (Rd–mpp coupling) ---
    if phys.include_Rd and layout.has_block("Rd"):
        rad_coeffs = build_radius_row(
            grid=grid,
            state_old=state_old,
            state_guess=state_guess,
            props=props,
            layout=layout,
            dt=dt,
            cfg=cfg,
        )
        _scatter_radius_row(A, b, rad_coeffs)
        diag_sys.update(rad_coeffs.diag)

    if return_diag:
        return A, b, diag_sys
    return A, b


def build_transport_system_petsc(*args, **kwargs):
    raise RuntimeError(
        "PETSc backend not available in SciPy workflow. Use build_transport_system() with SciPy solvers."
    )
