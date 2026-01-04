from __future__ import annotations

import copy

import numpy as np
import pytest

from tests._helpers_step15 import build_min_problem, make_cfg_base


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("PETSc PC smoke tests are serial-only.")
    return PETSc


def _build_case():
    cfg = make_cfg_base(
        Nl=1,
        Ng=5,
        solve_Yg=True,
        include_mpp=False,
        include_Ts=False,
        include_Rd=False,
    )
    grid, layout, state0, props0 = build_min_problem(cfg)
    return cfg, grid, layout, state0, props0


def _build_system(cfg, grid, layout, state0, props0, PETSc):
    from assembly.build_system_petsc import build_transport_system_petsc_native  # noqa: E402

    A, b = build_transport_system_petsc_native(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=float(cfg.time.dt),
        state_guess=state0,
        eq_result=None,
        return_diag=False,
        comm=PETSc.COMM_SELF,
    )
    return A, b


@pytest.mark.parametrize("pc_type", ["asm", "bjacobi"])
def test_petsc_linear_pc_smoke(pc_type):
    PETSc = _import_petsc_or_skip()
    cfg, grid, layout, state0, props0 = _build_case()
    A, b = _build_system(cfg, grid, layout, state0, props0, PETSc)

    cfg_pc = copy.deepcopy(cfg)
    cfg_pc.solver.linear.pc_type = pc_type
    cfg_pc.petsc.ksp_type = "gmres"
    cfg_pc.petsc.rtol = 1.0e-10
    cfg_pc.petsc.atol = 1.0e-12
    cfg_pc.petsc.max_it = 200

    from solvers.petsc_linear import solve_linear_system_petsc  # noqa: E402

    res = solve_linear_system_petsc(A=A, b=b, cfg=cfg_pc, layout=layout, P=A)

    assert np.all(np.isfinite(res.x))
    diag = res.diag or {}
    pc_type_eff = str(diag.get("pc_type", "")).lower()
    diag_pc = diag.get("pc", {}) or {}
    diag_global = diag_pc.get("global", {}) if isinstance(diag_pc, dict) else {}
    assert pc_type_eff == pc_type or diag_global.get("pc_type") == pc_type
