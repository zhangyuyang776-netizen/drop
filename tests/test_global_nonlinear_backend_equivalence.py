from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Bridge phase is serial-only (COMM_WORLD size must be 1).")
    return PETSc


def _import_chemistry_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("Cantera not available")
    try:
        import CoolProp  # noqa: F401
    except Exception:
        pytest.skip("CoolProp not available")


def _rel_inf(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(1.0, float(np.linalg.norm(b, ord=np.inf)))
    return float(np.linalg.norm(a - b, ord=np.inf) / denom)


def test_global_nonlinear_backend_equivalence(tmp_path: Path):
    _import_petsc_or_skip()
    _import_chemistry_or_skip()

    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg_base = _load_case_config(str(yml))
    cfg_base.geometry.N_liq = 1
    cfg_base.geometry.N_gas = 3
    cfg_base.geometry.mesh.enforce_interface_continuity = False

    cfg_base.nonlinear.enabled = True
    cfg_base.nonlinear.max_outer_iter = 25
    cfg_base.nonlinear.verbose = False

    cfg_base.paths.output_root = tmp_path
    cfg_base.paths.case_dir = tmp_path / "case"
    cfg_base.paths.case_dir.mkdir(parents=True, exist_ok=True)
    cfg_base.io.write_every = 10**9
    cfg_base.io.scalars_write_every = 10**9
    cfg_base.io.formats = []
    cfg_base.io.fields.scalars = []
    cfg_base.io.fields.gas = []
    cfg_base.io.fields.liquid = []
    cfg_base.io.fields.interface = []

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402
    from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402
    from solvers.solver_nonlinear import solve_nonlinear  # noqa: E402
    from assembly.residual_global import residual_only  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg_base)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg_base, gas_model)

    grid = build_grid(cfg_base)
    layout = build_layout(cfg_base, grid)
    state0 = build_initial_state_erfc(cfg_base, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg_base, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg_base, grid, state0)

    cfg_s = copy.deepcopy(cfg_base)
    cfg_s.nonlinear.backend = "scipy"
    ctx_s, u0_s = build_nonlinear_context_for_step(
        cfg=cfg_s,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg_s.time.t0),
        dt=float(cfg_s.time.dt),
    )
    t0 = time.perf_counter()
    res_s = solve_nonlinear(ctx_s, u0_s)
    t_s = time.perf_counter() - t0
    print(
        f"[global_nl] scipy it={res_s.diag.n_iter} t={t_s:.3f}s "
        f"inf={res_s.diag.res_norm_inf:.3e}"
    )
    assert res_s.diag.converged, res_s.diag.message

    cfg_p = copy.deepcopy(cfg_base)
    cfg_p.nonlinear.backend = "petsc"
    cfg_p.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg_p.petsc.ksp_type = "preonly"
    cfg_p.petsc.pc_type = "lu"
    cfg_p.petsc.max_it = 1
    cfg_p.petsc.monitor = False
    ctx_p, u0_p = build_nonlinear_context_for_step(
        cfg=cfg_p,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg_p.time.t0),
        dt=float(cfg_p.time.dt),
    )
    t0 = time.perf_counter()
    res_p = solve_nonlinear(ctx_p, u0_p)
    t_p = time.perf_counter() - t0
    print(
        f"[global_nl] petsc it={res_p.diag.n_iter} t={t_p:.3f}s "
        f"inf={res_p.diag.res_norm_inf:.3e}"
    )
    extra_p = res_p.diag.extra
    print("\n[global_nl petsc extra]", extra_p)
    assert "n_func_eval" in extra_p and extra_p["n_func_eval"] > 0
    assert "n_jac_eval" in extra_p and extra_p["n_jac_eval"] >= 0
    assert "ksp_its_total" in extra_p and extra_p["ksp_its_total"] >= 0
    assert "snes_reason" in extra_p and np.isfinite(extra_p["snes_reason"])
    assert "time_func" in extra_p and np.isfinite(extra_p["time_func"])
    assert "time_jac" in extra_p and np.isfinite(extra_p["time_jac"])
    assert "time_linear_total" in extra_p and np.isfinite(extra_p["time_linear_total"])
    assert res_p.diag.converged, res_p.diag.message

    rel = _rel_inf(res_p.u, res_s.u)
    assert rel < 1.0e-6

    res_inf_s = float(np.linalg.norm(residual_only(res_s.u, ctx_s), ord=np.inf))
    res_inf_p = float(np.linalg.norm(residual_only(res_p.u, ctx_p), ord=np.inf))
    assert np.isfinite(res_inf_s) and np.isfinite(res_inf_p)
