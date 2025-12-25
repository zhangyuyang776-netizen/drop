from __future__ import annotations

import copy

import numpy as np
import pytest

from tests._helpers_step15 import build_min_problem, make_cfg_base


def _import_petsc_or_skip():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("SNES fieldsplit schur test is serial-only.")
    return PETSc


def _build_case_base():
    cfg = make_cfg_base(
        Nl=1,
        Ng=3,
        solve_Yg=True,
        include_mpp=True,
        include_Ts=True,
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


def _patch_props_and_equilibrium(monkeypatch, cfg, props0, layout):
    import assembly.residual_global as residual_global  # noqa: E402
    import properties.compute_props as cp_mod  # noqa: E402
    import properties.equilibrium as eq_mod  # noqa: E402

    def _fake_compute_props(cfg_in, grid_in, state_in):
        return props0, {}

    monkeypatch.setattr(residual_global, "compute_props", _fake_compute_props)
    monkeypatch.setattr(cp_mod, "compute_props", _fake_compute_props)
    monkeypatch.setattr(residual_global, "_get_or_build_eq_model", lambda ctx, state: object())

    cond_name = None
    try:
        eq_cfg = getattr(getattr(cfg, "physics", None), "interface", None)
        eq_cfg = getattr(eq_cfg, "equilibrium", None)
        cond_list = getattr(eq_cfg, "condensables_gas", None)
        if cond_list:
            cond_name = list(cond_list)[0]
    except Exception:
        cond_name = None

    def _fake_equilibrium(eq_model, Ts, Pg, Yl_face, Yg_face):
        Yg_face_arr = np.asarray(Yg_face, dtype=np.float64).reshape(-1)
        if Yg_face_arr.size == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return empty, empty, empty

        Yg_eq = Yg_face_arr.copy()
        k_cond_full = 0
        if cond_name and cond_name in layout.gas_species_full:
            k_cond_full = layout.gas_species_full.index(cond_name)
        Yg_eq[k_cond_full] = max(float(Yg_eq[k_cond_full]), 0.1)

        s = float(np.sum(Yg_eq))
        if not np.isfinite(s) or s <= 0.0:
            Yg_eq[:] = 0.0
            Yg_eq[k_cond_full] = 1.0
        else:
            Yg_eq /= s

        y_cond = np.array([Yg_eq[k_cond_full]], dtype=np.float64)
        psat = np.array([1.0], dtype=np.float64)
        return Yg_eq, y_cond, psat

    monkeypatch.setattr(residual_global, "compute_interface_equilibrium", _fake_equilibrium)
    monkeypatch.setattr(eq_mod, "compute_interface_equilibrium", _fake_equilibrium)


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


def test_snes_fieldsplit_schur_runs(monkeypatch):
    _import_petsc_or_skip()
    cfg, grid, layout, state0, props0 = _build_case_base()
    _patch_props_and_equilibrium(monkeypatch, cfg, props0, layout)

    cfg_add = copy.deepcopy(cfg)
    cfg_add.solver.linear.pc_type = "fieldsplit"
    cfg_add.solver.linear.fieldsplit = {
        "type": "additive",
        "split_mode": "by_layout",
        "sub_ksp_type": "preonly",
        "sub_pc_type": "ilu",
    }

    cfg_sch = copy.deepcopy(cfg)
    cfg_sch.solver.linear.pc_type = "fieldsplit"
    cfg_sch.solver.linear.fieldsplit = {
        "type": "schur",
        "schur_fact_type": "lower",
        "bulk_ksp_type": "gmres",
        "bulk_pc_type": "ilu",
        "iface_ksp_type": "preonly",
        "iface_pc_type": "lu",
    }

    out_add = _solve_one(cfg_add, grid, layout, state0, props0)
    out_sch = _solve_one(cfg_sch, grid, layout, state0, props0)

    assert np.all(np.isfinite(out_add.u))
    assert np.all(np.isfinite(out_sch.u))

    extra_add = out_add.diag.extra or {}
    extra_sch = out_sch.diag.extra or {}
    assert int(extra_add.get("ksp_reason", 0)) >= 0
    assert int(extra_sch.get("ksp_reason", 0)) >= 0

    it_add = int(extra_add.get("ksp_its_total", 10**9))
    it_sch = int(extra_sch.get("ksp_its_total", 10**9))
    assert it_sch <= 10 * it_add + 5

    diag_pc = extra_sch.get("pc_structured", {}) or {}
    assert diag_pc.get("fieldsplit", {}).get("type") == "schur"
    assert diag_pc.get("fieldsplit", {}).get("plan") == "bulk_iface"
    assert set(diag_pc.get("splits", {})) >= {"bulk", "iface"}
