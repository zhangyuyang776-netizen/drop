"""
SciPy/Numpy assembly of a single-species transport equation (gas phase) for Y_k.

Scope (Step 9.4 MVP):
- Gas-phase only, single species k_spec.
- Implicit diffusion using rho_g * D_g (frozen at state_old).
- Explicit convection using Stefan velocity + upwind convective flux (state_old).
- Inner boundary: zero diffusive flux (no interface coupling yet).
- Outer boundary: Dirichlet Y_inf_k = 0.0 (MVP placeholder).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State
from physics.stefan_velocity import compute_stefan_velocity
from physics.flux_convective_gas import compute_gas_convective_flux_Y


def build_species_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    k_spec: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble one gas species Y_k equation into dense numpy system A x = b.

    Assumptions:
    - Gas-phase only, single species index k_spec.
    - Theta=1.0 (fully implicit diffusion).
    - Explicit convection (Stefan velocity), using state_old.
    - No interface source; inner boundary Neumann zero-flux.
    - Outer boundary Dirichlet: Y_k = 0.0 (MVP).
    """
    # layout must provide Yg block and idx_Yg
    if not hasattr(layout, "idx_Yg"):
        raise ValueError("layout must provide idx_Yg(k_spec, ig) for species block.")
    if not layout.has_block("Yg"):
        raise ValueError("layout missing 'Yg' block required for species assembly.")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("species system MVP only supports theta=1.0 (fully implicit diffusion).")

    Ns_g, Ng_y = state_old.Yg.shape
    Ng = grid.Ng
    if Ng_y != Ng:
        raise ValueError(f"state_old.Yg shape {state_old.Yg.shape} inconsistent with Ng={Ng}")
    if not (0 <= k_spec < Ns_g):
        raise ValueError(f"k_spec={k_spec} out of range for Ns_g={Ns_g}")

    if props.D_g is None:
        raise ValueError("Props.D_g is None; gas diffusion coeffs required for species assembly.")
    if props.D_g.shape != (Ns_g, Ng):
        raise ValueError(f"D_g shape {props.D_g.shape} != ({Ns_g}, {Ng})")
    if props.rho_g.shape != (Ng,):
        raise ValueError(f"rho_g shape {props.rho_g.shape} != ({Ng},)")

    Nc = grid.Nc
    Nl = grid.Nl
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.V_c.shape != (Nc,):
        raise ValueError(f"V_c shape {grid.V_c.shape} != ({Nc},)")
    if grid.A_f.shape != (Nc + 1,):
        raise ValueError(f"A_f shape {grid.A_f.shape} != ({Nc+1},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")

    N = layout.n_dof()
    if N != Ng:
        raise ValueError(f"Step 9.4 assumes scalar layout: n_dof={N} must equal Ng={Ng}")

    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    gas_start = grid.gas_slice.start if grid.gas_slice is not None else Nl

    rho_g = props.rho_g
    D_g = props.D_g
    Yg_old = state_old.Yg

    # Time + diffusion (implicit)
    for ig in range(Ng):
        row = layout.idx_Yg(k_spec, ig)
        cell_idx = gas_start + ig

        rho_i = float(rho_g[ig])
        Dk_i = float(D_g[k_spec, ig])
        V = float(grid.V_c[cell_idx])

        aP_time = rho_i * V / dt
        aP = aP_time
        b_i = aP_time * float(Yg_old[k_spec, ig])

        # Left face diffusion (ig>0; ig=0 uses zero-flux at interface)
        if ig > 0:
            iL = cell_idx - 1
            rL = grid.r_c[iL]
            rC = grid.r_c[cell_idx]
            A_f_L = float(grid.A_f[cell_idx])  # left face index = cell_idx
            dr_L = rC - rL
            if dr_L <= 0.0:
                raise ValueError("Non-positive dr on left face in species diffusion assembly")

            rho_L = float(rho_g[ig - 1])
            Dk_L = float(D_g[k_spec, ig - 1])

            rho_f_L = 0.5 * (rho_L + rho_i)
            Dk_f_L = 0.5 * (Dk_L + Dk_i)
            coeff_L = rho_f_L * Dk_f_L * A_f_L / dr_L

            aP += coeff_L
            A[row, layout.idx_Yg(k_spec, ig - 1)] += -coeff_L

        # Right face diffusion
        if ig < Ng - 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f_R = float(grid.A_f[cell_idx + 1])
            dr = rR - rC
            if dr <= 0.0:
                raise ValueError("Non-positive dr in species diffusion assembly")
            rho_R = float(rho_g[ig + 1])
            Dk_R = float(D_g[k_spec, ig + 1])
            rho_f = 0.5 * (rho_i + rho_R)
            Dk_f = 0.5 * (Dk_i + Dk_R)
            coeff_R = rho_f * Dk_f * A_f_R / dr  # kg/s
            aP += coeff_R
            A[row, layout.idx_Yg(k_spec, ig + 1)] += -coeff_R

        A[row, row] += aP
        b[row] += b_i

    # Explicit convection using Stefan velocity + convective flux (state_old)
    stefan = compute_stefan_velocity(cfg, grid, props, state_old)
    u_face = stefan.u_face
    J_conv_all = compute_gas_convective_flux_Y(cfg, grid, props, state_old.Yg, u_face)
    J_conv_k = J_conv_all[k_spec, :]

    for ig in range(Ng):
        row = layout.idx_Yg(k_spec, ig)
        cell_idx = gas_start + ig
        f_L = cell_idx
        f_R = cell_idx + 1
        A_L = float(grid.A_f[f_L])
        A_R = float(grid.A_f[f_R])
        J_L = float(J_conv_k[f_L])
        J_R = float(J_conv_k[f_R])
        S_conv = A_R * J_R - A_L * J_L  # outward positive mass flow (kg/s)
        b[row] -= S_conv

    # Outer boundary Dirichlet: Y_k = 0.0 (MVP)
    ig_bc = Ng - 1
    row_bc = layout.idx_Yg(k_spec, ig_bc)
    A[row_bc, :] = 0.0
    A[row_bc, row_bc] = 1.0
    b[row_bc] = 0.0

    return A, b
