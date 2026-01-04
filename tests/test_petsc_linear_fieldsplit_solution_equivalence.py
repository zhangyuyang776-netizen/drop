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
        pytest.skip("FieldSplit test is serial-only.")
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


def test_petsc_linear_fieldsplit_solution_equivalence():
    PETSc = _import_petsc_or_skip()
    cfg, grid, layout, state0, props0 = _build_case()

    from assembly.build_system_petsc import build_transport_system_petsc_native  # noqa: E402
    from solvers.petsc_linear import solve_linear_system_petsc  # noqa: E402

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

    cfg_ilu = copy.deepcopy(cfg)
    cfg_ilu.solver.linear.pc_type = "ilu"
    cfg_ilu.solver.linear.fieldsplit = None
    cfg_ilu.petsc.ksp_type = "gmres"
    cfg_ilu.petsc.rtol = 1e-10
    cfg_ilu.petsc.atol = 1e-12
    cfg_ilu.petsc.max_it = 200

    cfg_fs = copy.deepcopy(cfg)
    cfg_fs.solver.linear.pc_type = "fieldsplit"
    cfg_fs.solver.linear.fieldsplit = {
        "type": "additive",
        "split_mode": "by_layout",
        "sub_ksp_type": "preonly",
        "sub_pc_type": "ilu",
    }
    cfg_fs.petsc.ksp_type = "gmres"
    cfg_fs.petsc.rtol = 1e-10
    cfg_fs.petsc.atol = 1e-12
    cfg_fs.petsc.max_it = 200

    res_ilu = solve_linear_system_petsc(A=A, b=b, cfg=cfg_ilu, layout=layout, P=A)
    res_fs = solve_linear_system_petsc(A=A, b=b, cfg=cfg_fs, layout=layout, P=A)

    assert res_ilu.converged
    assert res_fs.converged

    diag_fs = res_fs.diag or {}
    assert str(diag_fs.get("pc_type", "")).lower() == "fieldsplit" or diag_fs.get("pc", {}).get("enabled", False)

    err = float(np.max(np.abs(res_ilu.x - res_fs.x)))
    assert err < 1e-10
