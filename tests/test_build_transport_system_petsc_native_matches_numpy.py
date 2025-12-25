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


def test_build_transport_system_petsc_native_matches_numpy(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0 = _build_case(tmp_path)
    dt = float(cfg.time.dt)

    from assembly.build_system_SciPy import build_transport_system as build_transport_system_numpy  # noqa: E402
    from assembly.build_system_petsc import build_transport_system_petsc_native  # noqa: E402

    A_np, b_np = build_transport_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=None,
        return_diag=False,
    )

    A_p, b_p = build_transport_system_petsc_native(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=None,
        return_diag=False,
        comm=PETSc.COMM_SELF,
    )

    N = layout.n_dof()
    A_native = np.zeros_like(A_np)
    for i in range(N):
        cols, vals = A_p.getRow(i)
        if cols.size:
            A_native[i, cols] = vals

    b_native = b_p.getArray(readonly=True).copy()

    assert A_np.shape == A_native.shape
    assert b_np.shape == b_native.shape

    np.testing.assert_allclose(A_native, A_np, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(b_native, b_np, rtol=1.0e-12, atol=1.0e-12)
