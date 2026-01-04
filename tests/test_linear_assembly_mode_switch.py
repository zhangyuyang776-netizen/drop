from __future__ import annotations

import sys
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
    cfg.physics.include_mpp = False
    cfg.physics.include_Ts = False
    cfg.physics.include_Rd = False

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)

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


def test_assembly_mode_default_bridge_dense(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0 = _build_case(tmp_path)
    cfg.solver.linear.backend = "petsc"
    cfg.solver.linear.assembly_mode = "bridge_dense"

    from solvers import timestepper as ts  # noqa: E402

    called = {"bridge": False}

    def _fake_bridge(**kwargs):
        called["bridge"] = True
        n = int(kwargs["layout"].n_dof())
        A = PETSc.Mat().createAIJ([n, n], comm=PETSc.COMM_SELF)
        A.setUp()
        A.assemblyBegin()
        A.assemblyEnd()
        b = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
        b.set(0.0)
        b.assemblyBegin()
        b.assemblyEnd()
        return A, b, {}

    monkeypatch.setattr(ts, "build_transport_system_petsc_bridge", _fake_bridge)

    A, b, diag = ts._assemble_transport_system_step12(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        state_guess=state0,
        props=props0,
        dt=float(cfg.time.dt),
    )

    assert called["bridge"] is True
    assert A.getSize()[0] == layout.n_dof()
    assert b.getSize() == layout.n_dof()
    assert isinstance(diag, dict)


def test_assembly_mode_native_aij_returns_petsc(tmp_path: Path):
    _import_petsc_or_skip()
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

    assert A.getSize()[0] == layout.n_dof()
    assert b.getSize() == layout.n_dof()
    assert isinstance(diag, dict)
