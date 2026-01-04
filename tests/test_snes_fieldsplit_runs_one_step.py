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
        pytest.skip("SNES fieldsplit test is serial-only.")
    return PETSc


def _build_case_base():
    cfg = make_cfg_base(
        Nl=1,
        Ng=3,
        solve_Yg=True,
        include_mpp=False,
        include_Ts=False,
        include_Rd=False,
    )
    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 10

    cfg.petsc.jacobian_mode = "fd"
    cfg.petsc.ksp_type = "gmres"
    cfg.petsc.rtol = 1e-10
    cfg.petsc.atol = 1e-12
    cfg.petsc.max_it = 200

    grid, layout, state0, props0 = build_min_problem(cfg)
    return cfg, grid, layout, state0, props0


def _solve_one(cfg, grid, layout, state0, props0):
    from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402
    from solvers.petsc_snes import solve_nonlinear_petsc  # noqa: E402

    ctx, u0 = build_nonlinear_context_for_step(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg.time.t0),
        dt=float(cfg.time.dt),
    )
    return solve_nonlinear_petsc(ctx, u0)


def test_snes_fieldsplit_runs_one_step():
    _import_petsc_or_skip()
    cfg, grid, layout, state0, props0 = _build_case_base()

    cfg_ilu = copy.deepcopy(cfg)
    cfg_ilu.solver.linear.pc_type = "ilu"
    cfg_ilu.solver.linear.fieldsplit = None

    cfg_fs = copy.deepcopy(cfg)
    cfg_fs.solver.linear.pc_type = "fieldsplit"
    cfg_fs.solver.linear.fieldsplit = {
        "type": "additive",
        "split_mode": "by_layout",
        "sub_ksp_type": "preonly",
        "sub_pc_type": "ilu",
    }

    out_ilu = _solve_one(cfg_ilu, grid, layout, state0, props0)
    out_fs = _solve_one(cfg_fs, grid, layout, state0, props0)

    extra_fs = out_fs.diag.extra or {}
    assert extra_fs.get("snes_reason", -1) >= 0
    assert extra_fs.get("ksp_reason", -1) >= 0
    assert np.all(np.isfinite(out_fs.u))

    it_ilu = int(out_ilu.diag.n_iter)
    it_fs = int(out_fs.diag.n_iter)
    assert it_fs <= 2 * it_ilu + 3
