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


def test_build_precond_aij_matches_dense(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
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
    from assembly.residual_global import build_transport_system_from_ctx  # noqa: E402
    from assembly.build_precond_aij import build_precond_aij_from_ctx  # noqa: E402

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

    A_dense, _b = build_transport_system_from_ctx(ctx, u0)
    P, diag = build_precond_aij_from_ctx(ctx, u0, drop_tol=0.0)

    n = int(u0.size)
    rng = np.random.default_rng(123)
    v = rng.standard_normal(n).astype(np.float64)

    y_dense = A_dense @ v

    vin = PETSc.Vec().createSeq(n)
    vout = PETSc.Vec().createSeq(n)
    idx = np.arange(n, dtype=PETSc.IntType)
    vin.setValues(idx, v)
    vin.assemblyBegin()
    vin.assemblyEnd()
    P.mult(vin, vout)
    y_aij = vout.getArray().copy()

    np.testing.assert_allclose(y_aij, y_dense, rtol=1.0e-12, atol=1.0e-12)

    assert diag["nnz_sum"] > 0
    assert diag["nnz_max"] > 0
    assert diag["nnz_total"] > 0
    assert diag["nnz_max_row"] >= 1

    finite = np.isfinite(A_dense)
    if np.any(finite):
        max_abs = float(np.max(np.abs(A_dense[finite])))
    else:
        max_abs = 0.0
    drop_high = max_abs * 1.01 if max_abs > 0.0 else 1.0
    _P2, diag2 = build_precond_aij_from_ctx(ctx, u0, drop_tol=drop_high)
    assert diag2["nnz_sum"] <= diag["nnz_sum"]
