"""
SciPy/Numpy assembly of liquid-phase temperature equation (Step 10/12 split stage).

Scope:
- Liquid cells only (0..Nl-1), assembled as a standalone Nl x Nl system.
- Implicit conduction: rho_l, cp_l, k_l frozen at state_old.
- Center r=0: symmetry (zero-gradient) via ghost-cell-like stencil.
- Interface r=Rd: strong Dirichlet T_l(Nl-1) = Ts_given (state_old.Ts).
- No interface coupling with gas/mpp in this split stage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State


def build_liquid_T_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    couple_interface: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble liquid-phase temperature equation into dense numpy system A x = b (Tl-only).

    Scope:
    - Only Tl in liquid cells (0..Nl-1); returned system size is (Nl, Nl).
    - Implicit conduction with rho_l, cp_l, k_l frozen at state_old.
    - Center r=0: symmetry (zero-gradient) via ghost-cell-like stencil.
    - Interface r=R_d:
      * If couple_interface=False (Stage 2 mode): Dirichlet T_l(Nl-1) = Ts_given.
      * If couple_interface=True (Stage 1 coupled mode): keep conduction equation,
        coupling provided by Ts energy equation in build_transport_system.

    Parameters
    ----------
    couple_interface : bool, default False
        If False: apply Dirichlet BC at interface (for Stage 2 split solve).
        If True: keep conduction equation at interface (for Stage 1 coupled solve).
    """
    if not layout.has_block("Tl"):
        raise ValueError("layout missing 'Tl' block required for liquid T assembly.")

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Liquid T system MVP only supports theta=1.0 (fully implicit).")

    Nl = grid.Nl
    if state_old.Tl.shape != (Nl,):
        raise ValueError(f"Tl shape {state_old.Tl.shape} != ({Nl},)")
    if props.rho_l.shape != (Nl,):
        raise ValueError(f"rho_l shape {props.rho_l.shape} != ({Nl},)")
    if props.cp_l.shape != (Nl,):
        raise ValueError(f"cp_l shape {props.cp_l.shape} != ({Nl},)")
    if props.k_l.shape != (Nl,):
        raise ValueError(f"k_l shape {props.k_l.shape} != ({Nl},)")

    Nc = grid.Nc
    if grid.r_c.shape != (Nc,):
        raise ValueError(f"r_c shape {grid.r_c.shape} != ({Nc},)")
    if grid.V_c.shape != (Nc,):
        raise ValueError(f"V_c shape {grid.V_c.shape} != ({Nc},)")
    if grid.r_f.shape != (Nc + 1,):
        raise ValueError(f"r_f shape {grid.r_f.shape} != ({Nc+1},)")
    if grid.A_f.shape != (Nc + 1,):
        raise ValueError(f"A_f shape {grid.A_f.shape} != ({Nc+1},)")

    iface_f = grid.iface_f
    if iface_f <= 0 or iface_f > Nc:
        raise ValueError(f"Invalid iface_f={iface_f} for Nc={Nc}")

    A = np.zeros((Nl, Nl), dtype=np.float64)
    b = np.zeros(Nl, dtype=np.float64)

    rho_l = props.rho_l
    cp_l = props.cp_l
    k_l = props.k_l
    Tl_old = state_old.Tl

    for il in range(Nl):
        row = il  # local index 0..Nl-1
        cell_idx = il  # liquid cells occupy 0..Nl-1 globally

        rho_i = float(rho_l[il])
        cp_i = float(cp_l[il])
        k_i = float(k_l[il])
        V = float(grid.V_c[cell_idx])

        aP_time = rho_i * cp_i * V / dt
        aP = aP_time
        b_i = aP_time * float(Tl_old[il])

        # Left face diffusion (il > 0); il == 0 uses symmetry (handled via ghost-like stencil)
        if il > 0:
            iL = cell_idx - 1
            iC = cell_idx
            f_L = iC  # left face index
            rL = grid.r_c[iL]
            rC = grid.r_c[iC]
            dr_L = rC - rL
            if dr_L <= 0.0:
                raise ValueError("Non-positive dr_L in liquid diffusion assembly")

            k_L = float(k_l[il - 1])
            k_face_L = 0.5 * (k_L + k_i)
            A_f_L = float(grid.A_f[f_L])
            coeff_L = k_face_L * A_f_L / dr_L
            aP += coeff_L
            A[row, il - 1] += -coeff_L
        else:
            # center symmetry (dT/dr=0): reflect to neighbor if exists
            if Nl > 1:
                iC = cell_idx
                iR = cell_idx + 1
                f_center = iC  # face index 0
                rC = grid.r_c[iC]
                rR = grid.r_c[iR]
                dr_center = rR - rC
                if dr_center <= 0.0:
                    raise ValueError("Non-positive dr at center in liquid diffusion assembly")
                k_R = float(k_l[il + 1])
                k_face_center = 0.5 * (k_i + k_R)
                A_f_center = float(grid.A_f[f_center])
                coeff_center = k_face_center * A_f_center / dr_center
                aP += coeff_center
                A[row, il + 1] += -coeff_center
            # if Nl == 1, Dirichlet will override this row later

        # Right face diffusion
        if il < Nl - 1:
            iC = cell_idx
            iR = cell_idx + 1
            f_R = iC + 1
            rC = grid.r_c[iC]
            rR = grid.r_c[iR]
            dr_R = rR - rC
            if dr_R <= 0.0:
                raise ValueError("Non-positive dr_R in liquid diffusion assembly")
            k_R = float(k_l[il + 1])
            k_face_R = 0.5 * (k_i + k_R)
            A_f_R = float(grid.A_f[f_R])
            coeff_R = k_face_R * A_f_R / dr_R
            aP += coeff_R
            A[row, il + 1] += -coeff_R
        else:
            # interface cell: right face handled by Dirichlet below
            pass

        A[row, row] += aP
        b[row] += b_i

    # Interface boundary condition
    il_bc = Nl - 1
    row_bc = il_bc

    if not couple_interface:
        # Stage 2 mode: Dirichlet BC (Tl[Nl-1] = Ts_given)
        Ts_given = float(state_old.Ts)
        A[row_bc, :] = 0.0
        A[row_bc, row_bc] = 1.0
        b[row_bc] = Ts_given
    else:
        # Stage 1 coupled mode: keep conduction equation at interface
        # Interface energy coupling is handled by _build_Ts_row
        # The conduction equation remains as assembled in the loop above
        pass

    return A, b


def build_liquid_T_system_petsc(*args, **kwargs):
    raise RuntimeError("PETSc backend not available in SciPy workflow. Use build_liquid_T_system with SciPy solvers.")
