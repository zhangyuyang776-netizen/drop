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


def _allclose(a, b, rtol=1e-8, atol=1e-12) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol))


@pytest.mark.parametrize("Ng", [3, 5])
def test_one_step_backend_equivalence(Ng: int, tmp_path: Path):
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

    cfg0 = _load_case_config(str(yml))
    cfg0.geometry.N_liq = 1
    cfg0.geometry.N_gas = Ng
    cfg0.geometry.mesh.enforce_interface_continuity = False
    cfg0.nonlinear.enabled = False

    cfg0.paths.output_root = tmp_path
    cfg0.paths.case_dir = tmp_path / "case"
    cfg0.paths.case_dir.mkdir(parents=True, exist_ok=True)

    cfg0.io.write_every = 10**9
    cfg0.io.scalars_write_every = 10**9
    cfg0.io.formats = []
    cfg0.io.fields.scalars = []
    cfg0.io.fields.gas = []
    cfg0.io.fields.liquid = []
    cfg0.io.fields.interface = []

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402
    try:
        from solvers.timestepper import advance_one_step  # noqa: E402
    except Exception:
        from solvers.timestepper import advance_one_step_scipy as advance_one_step  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg0)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg0, gas_model)

    grid = build_grid(cfg0)
    layout = build_layout(cfg0, grid)
    state0 = build_initial_state_erfc(cfg0, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg0, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg0, grid, state0)

    cfg_s = copy.deepcopy(cfg0)
    cfg_s.nonlinear.backend = "scipy"
    cfg_s.solver.linear.backend = "scipy"

    t0 = time.perf_counter()
    out_s = advance_one_step(cfg_s, grid, layout, state0, props0, t=float(cfg0.time.t0))
    t_scipy = time.perf_counter() - t0
    assert out_s.success

    cfg_p = copy.deepcopy(cfg0)
    cfg_p.nonlinear.backend = "petsc"
    cfg_p.solver.linear.backend = "petsc"
    cfg_p.petsc.ksp_type = "gmres"
    cfg_p.petsc.pc_type = "lu"
    cfg_p.petsc.restart = 30
    cfg_p.petsc.max_it = 5
    cfg_p.petsc.monitor = False

    t0 = time.perf_counter()
    out_p = advance_one_step(cfg_p, grid, layout, state0, props0, t=float(cfg0.time.t0))
    t_petsc = time.perf_counter() - t0
    assert out_p.success

    ds = out_s.diag
    dp = out_p.diag

    assert _allclose(ds.Ts, dp.Ts)
    assert _allclose(ds.mpp, dp.mpp)
    assert _allclose(ds.Rd, dp.Rd)
    assert _allclose(ds.Tg_min, dp.Tg_min)
    assert _allclose(ds.Tg_max, dp.Tg_max)

    assert _allclose(out_s.state_new.Tg, out_p.state_new.Tg)
    if getattr(cfg0.physics, "solve_Yg", False):
        assert _allclose(out_s.state_new.Yg, out_p.state_new.Yg)

    print(
        f"[one_step Ng={Ng}] "
        f"scipy={t_scipy*1e3:.2f}ms petsc={t_petsc*1e3:.2f}ms "
        f"n_iter=({ds.linear_n_iter},{dp.linear_n_iter}) "
        f"rel_res=({ds.linear_rel_residual:.3e},{dp.linear_rel_residual:.3e})"
    )
