from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
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


@pytest.mark.parametrize("jacobian_mode", ["fd", "mfpc_aijA", "mfpc_sparse_fd"])
def test_snes_smoke(tmp_path: Path, jacobian_mode: str):
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

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = 3
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 20

    cfg.petsc.jacobian_mode = jacobian_mode
    cfg.petsc.ksp_type = "preonly"
    cfg.petsc.pc_type = "lu"
    cfg.petsc.max_it = 1
    cfg.petsc.monitor = False

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)
    cfg.io.write_every = 10**9
    cfg.io.scalars_write_every = 10**9
    cfg.io.formats = []
    cfg.io.fields.scalars = []
    cfg.io.fields.gas = []
    cfg.io.fields.liquid = []
    cfg.io.fields.interface = []

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402
    from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402
    from solvers.petsc_snes import solve_nonlinear_petsc  # noqa: E402
    from assembly.residual_global import residual_only  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg, grid, state0)

    ctx, u0 = build_nonlinear_context_for_step(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg.time.t0),
        dt=float(cfg.time.dt),
    )

    res0 = residual_only(u0, ctx)
    res0_inf = float(np.linalg.norm(res0, ord=np.inf))

    nl_result = solve_nonlinear_petsc(ctx, u0)
    extra = nl_result.diag.extra
    print("\n[SNES extra]", extra)
    print("[SNES] n_func_eval =", extra.get("n_func_eval"))
    print("[SNES] n_jac_eval  =", extra.get("n_jac_eval"))
    print("[SNES] ksp_its_total =", extra.get("ksp_its_total"))
    print("[SNES] time_func =", extra.get("time_func"))
    print("[SNES] time_jac  =", extra.get("time_jac"))
    print("[SNES] time_linear_total =", extra.get("time_linear_total"))
    print("[SNES] snes_reason =", extra.get("snes_reason"))

    assert "n_func_eval" in extra and extra["n_func_eval"] > 0
    assert "n_jac_eval" in extra and extra["n_jac_eval"] >= 0
    assert "ksp_its_total" in extra and extra["ksp_its_total"] >= 0
    assert "snes_reason" in extra and np.isfinite(extra["snes_reason"])
    assert "time_func" in extra and np.isfinite(extra["time_func"])
    assert "time_jac" in extra and np.isfinite(extra["time_jac"])
    assert "time_linear_total" in extra and np.isfinite(extra["time_linear_total"])

    res1 = residual_only(nl_result.u, ctx)
    res1_inf = float(np.linalg.norm(res1, ord=np.inf))

    assert np.isfinite(res1_inf)
    if nl_result.diag.converged:
        assert res1_inf <= 1.0e-6 * max(1.0, res0_inf)
    else:
        assert res1_inf < res0_inf
