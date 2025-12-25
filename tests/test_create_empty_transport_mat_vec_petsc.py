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
        pytest.skip("create_empty_transport_mat_vec_petsc test is serial-only.")
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


def test_create_empty_transport_mat_vec_shapes_and_zeros(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, _state0, _props0 = _build_case(tmp_path)

    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
    from assembly.build_system_petsc import create_empty_transport_mat_vec_petsc  # noqa: E402

    pattern = build_jacobian_pattern(cfg, grid, layout)
    N = layout.n_dof()

    A, b = create_empty_transport_mat_vec_petsc(pattern, comm=PETSc.COMM_SELF)

    assert isinstance(A, PETSc.Mat)
    m, n = A.getSize()
    assert (m, n) == (N, N)

    assert isinstance(b, PETSc.Vec)
    assert b.getSize() == N

    arr_b = b.getArray(readonly=True)
    assert np.allclose(arr_b, 0.0)


def test_create_empty_transport_mat_vec_set_diagonal_and_mult(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, _state0, _props0 = _build_case(tmp_path)

    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
    from assembly.build_system_petsc import create_empty_transport_mat_vec_petsc  # noqa: E402

    pattern = build_jacobian_pattern(cfg, grid, layout)
    N = layout.n_dof()

    A, _b = create_empty_transport_mat_vec_petsc(pattern, comm=PETSc.COMM_SELF)

    diag = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    diag.set(2.0)
    diag.assemblyBegin()
    diag.assemblyEnd()

    A.setDiagonal(diag)
    A.assemblyBegin()
    A.assemblyEnd()

    x = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    x.set(3.0)
    x.assemblyBegin()
    x.assemblyEnd()

    y = x.duplicate()
    A.mult(x, y)

    arr_y = y.getArray(readonly=True)
    assert np.allclose(arr_y, 6.0)
