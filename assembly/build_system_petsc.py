"""
Assembly of linear system A x = b for transport equations (Step 6 MVP).

Scope (Step 6):
- Assemble gas temperature Tg diffusion only (v=0), theta-scheme, Dirichlet at outer boundary, symmetric at center.
- Inputs: cfg, grid, layout, state_old, props, dt.
- Outputs: PETSc Mat (AIJ) and Vec.

Not in scope:
- Updating State
- Computing properties
- Pack/unpack layout (only use layout.idx_* for indices)
- PETSc solver configuration
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from petsc4py import PETSc

from core.layout import UnknownLayout
from core.types import CaseConfig, Grid1D, Props, State


def _apply_outer_dirichlet_Tg(A: PETSc.Mat, b: PETSc.Vec, row: int, T_far: float) -> None:
    """Strong Dirichlet at outer boundary: T = T_far."""
    # clear existing row
    cols, _ = A.getRow(row)
    for c in cols:
        A.setValue(row, c, 0.0, addv=False)
    A.setValue(row, row, 1.0, addv=False)
    b.setValue(row, T_far, addv=False)


def build_transport_system_petsc(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    comm: Optional[PETSc.Comm] = None,
) -> Tuple[PETSc.Mat, PETSc.Vec]:
    """
    Step 6: scalar diffusion for Tg only (v=0), theta-scheme.

    Contract:
    - layout must contain Tg block; n_dof equals global vector size.
    - cfg.physics.include_gas_energy must be True; other equations ignored in this MVP.
    - props shapes validated externally; Tg shape must match grid.Ng.
    """
    if comm is None:
        comm = PETSc.COMM_WORLD

    if not layout.has_block("Tg"):
        raise ValueError("layout missing Tg block required for Step 6 assembly.")
    if state_old.Tg.shape != (grid.Ng,):
        raise ValueError(f"Tg shape {state_old.Tg.shape} != ({grid.Ng},)")

    N = layout.n_dof()
    A = PETSc.Mat().createAIJ([N, N], comm=comm)
    A.setPreallocationNNZ(5)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    b = PETSc.Vec().createMPI(N, comm=comm)

    theta = float(cfg.discretization.theta)
    if abs(theta - 1.0) > 1e-12:
        raise ValueError("Step 6 MVP only supports theta=1.0 (fully implicit Tg diffusion).")
    Ng = grid.Ng

    gas_start = grid.gas_slice.start

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
            A.setValue(row, layout.idx_Tg(ig - 1), -coeff, addv=True)
        else:
            # Step 6: inner gas boundary treated as Neumann (zero flux).
            # No contribution from the left face.
            pass

        # Right face (ig+1/2)
        if ig < Ng - 1:
            rC = grid.r_c[cell_idx]
            rR = grid.r_c[cell_idx + 1]
            A_f = float(grid.A_f[cell_idx + 1])
            dr = rR - rC
            k_face = 0.5 * (k_i + float(props.k_g[ig + 1]))
            coeff = k_face * A_f / dr
            aP += coeff
            A.setValue(row, layout.idx_Tg(ig + 1), -coeff, addv=True)
        else:
            # outer Dirichlet handled after loop
            pass

        A.setValue(row, row, aP, addv=True)
        b.setValue(row, b_i, addv=True)

    # Outer boundary Dirichlet on last gas cell
    Tg_far = float(cfg.initial.T_inf)
    row_bc = layout.idx_Tg(Ng - 1)
    _apply_outer_dirichlet_Tg(A, b, row_bc, Tg_far)

    A.assemblyBegin()
    A.assemblyEnd()
    b.assemblyBegin()
    b.assemblyEnd()
    return A, b
