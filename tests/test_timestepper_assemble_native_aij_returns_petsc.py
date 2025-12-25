from __future__ import annotations

import sys
from pathlib import Path

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


def _build_case(tmp_path: Path):
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
    cfg.physics.include_Ts = False
    cfg.physics.include_mpp = False
    cfg.physics.include_Rd = False

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

    return cfg, grid, layout, state0, props0


def test_timestepper_assemble_native_aij_returns_petsc(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0 = _build_case(tmp_path)
    cfg.solver.linear.backend = "petsc"
    cfg.solver.linear.assembly_mode = "native_aij"

    from solvers import timestepper as ts  # noqa: E402

    A, b, diag = ts._assemble_transport_system_step12(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        state_guess=state0,
        props=props0,
        dt=float(cfg.time.dt),
    )

    assert isinstance(A, PETSc.Mat)
    assert isinstance(b, PETSc.Vec)
    assert isinstance(diag, dict)
