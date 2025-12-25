from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Native PETSc assembly test is serial-only.")
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


def _petsc_mat_to_dense(PETSc, A):
    n, m = A.getSize()
    assert n == m
    idx = np.arange(n, dtype=PETSc.IntType)
    return np.asarray(A.getValues(idx, idx), dtype=np.float64)


def _petsc_vec_to_dense(b):
    return np.asarray(b.getArray(), dtype=np.float64).copy()


def _apply_case_flags(
    cfg,
    *,
    solve_Yg: bool,
    solve_Yl: bool,
    include_Ts: bool,
    include_mpp: bool,
    include_Rd: bool,
    solve_Tl: bool,
) -> None:
    include_Ts = bool(include_Ts)
    include_Rd = bool(include_Rd)
    include_mpp = bool(include_mpp) or include_Ts or include_Rd

    solve_Yg = bool(solve_Yg) or include_mpp
    solve_Yl = bool(solve_Yl)
    solve_Tl = bool(solve_Tl) or solve_Yl or include_Ts

    cfg.physics.solve_Yg = solve_Yg
    cfg.physics.solve_Yl = solve_Yl
    cfg.physics.solve_Tl = solve_Tl
    cfg.physics.include_Ts = include_Ts
    cfg.physics.include_mpp = include_mpp
    cfg.physics.include_Rd = include_Rd

    solver_cfg = getattr(cfg, "solver", None)
    linear_cfg = getattr(solver_cfg, "linear", None)
    if linear_cfg is not None:
        linear_cfg.backend = "petsc"
        linear_cfg.assembly_mode = "native_aij"


def _build_case(
    tmp_path: Path,
    *,
    Ng: int = 3,
    Nl: int = 1,
    solve_Yg: bool,
    solve_Yl: bool,
    include_Ts: bool,
    include_mpp: bool,
    include_Rd: bool,
    solve_Tl: bool,
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any] | None]:
    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = Nl
    cfg.geometry.N_gas = Ng
    cfg.geometry.mesh.enforce_interface_continuity = False

    _apply_case_flags(
        cfg,
        solve_Yg=solve_Yg,
        solve_Yl=solve_Yl,
        include_Ts=include_Ts,
        include_mpp=include_mpp,
        include_Rd=include_Rd,
        solve_Tl=solve_Tl,
    )

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
    from solvers.timestepper import (  # noqa: E402
        _build_eq_result_for_step,
        _complete_Yg_eq_with_closure,
    )

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

    eq_result = None
    if cfg.physics.include_mpp:
        eq_result = _build_eq_result_for_step(cfg, grid, state0, props0)
        eq_result = _complete_Yg_eq_with_closure(cfg, layout, eq_result)

    return cfg, grid, layout, state0, props0, eq_result


@pytest.mark.parametrize(
    "case_flags",
    [
        dict(
            solve_Yg=False,
            solve_Yl=False,
            include_Ts=False,
            include_mpp=False,
            include_Rd=False,
            solve_Tl=False,
        ),
        dict(
            solve_Yg=False,
            solve_Yl=False,
            include_Ts=False,
            include_mpp=False,
            include_Rd=False,
            solve_Tl=True,
        ),
        dict(
            solve_Yg=True,
            solve_Yl=False,
            include_Ts=False,
            include_mpp=False,
            include_Rd=False,
            solve_Tl=True,
        ),
        dict(
            solve_Yg=True,
            solve_Yl=True,
            include_Ts=False,
            include_mpp=False,
            include_Rd=False,
            solve_Tl=True,
        ),
        dict(
            solve_Yg=True,
            solve_Yl=False,
            include_Ts=True,
            include_mpp=True,
            include_Rd=False,
            solve_Tl=True,
        ),
        dict(
            solve_Yg=True,
            solve_Yl=False,
            include_Ts=False,
            include_mpp=True,
            include_Rd=True,
            solve_Tl=True,
        ),
    ],
)
def test_transport_assembly_petsc_native_vs_numpy(tmp_path: Path, case_flags):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0, eq_result = _build_case(tmp_path, **case_flags)
    dt = float(cfg.time.dt)

    from assembly.build_system_SciPy import build_transport_system as build_numpy  # noqa: E402
    from assembly.build_system_petsc import build_transport_system_petsc_native as build_native  # noqa: E402

    A_np, b_np = build_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=eq_result,
        return_diag=False,
    )

    A_p, b_p = build_native(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=eq_result,
        return_diag=False,
        comm=PETSc.COMM_SELF,
    )

    A_p_dense = _petsc_mat_to_dense(PETSc, A_p)
    b_p_dense = _petsc_vec_to_dense(b_p)

    np.testing.assert_allclose(A_np, A_p_dense, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(b_np, b_p_dense, rtol=1.0e-12, atol=1.0e-12)
