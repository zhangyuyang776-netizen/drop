from __future__ import annotations

import logging
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
    return PETSc


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def _build_case(tmp_path: Path):
    try:
        from driver.run_scipy_case import _load_case_config, _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config, _maybe_fill_gas_species  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))

    cfg.geometry.N_liq = max(4, int(getattr(cfg.geometry, "N_liq", 4)))
    cfg.geometry.N_gas = max(8, int(getattr(cfg.geometry, "N_gas", 8)))
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 2

    cfg.petsc.jacobian_mode = "mf"
    cfg.petsc.ksp_type = "preonly"
    cfg.petsc.pc_type = "lu"
    cfg.petsc.max_it = 2

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case_serial"
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

    gas_model, liq_model = get_or_build_models(cfg)
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
    return cfg, layout, ctx, u0


def test_snes_serial_regression(tmp_path: Path, caplog):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() != 1:
        pytest.skip("Serial regression: run without mpiexec (COMM_WORLD size must be 1).")

    caplog.set_level(logging.WARNING, logger="solvers.petsc_snes")

    cfg, layout, ctx, u0 = _build_case(tmp_path)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes import solve_nonlinear_petsc  # noqa: E402

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    result = solve_nonlinear_petsc(ctx, u0)
    diag = result.diag
    extra = diag.extra

    assert diag.n_iter >= 0
    assert np.isfinite(diag.res_norm_inf)

    assert int(extra.get("comm_size", 0)) == 1
    x_type = str(extra.get("X_vec_type", "")).lower()
    f_type = str(extra.get("F_vec_type", "")).lower()
    assert x_type.startswith("seq")
    assert f_type.startswith("seq")

    assert extra.get("snes_mf_enabled", False)
    j_type = str(extra.get("J_mat_type", "")).lower()
    p_type = str(extra.get("P_mat_type", "")).lower()
    if j_type and j_type != "none":
        assert any(tag in j_type for tag in ("snesmf", "mffd", "matfree", "mf"))
    assert "aij" in p_type

    pc_type = str(extra.get("pc_type", "")).lower()
    assert pc_type in ("lu", "ilu")
    assert extra.get("pc_type_overridden", False) is False

    bad = [
        r for r in caplog.records if "Disable use_scaled_unknowns in parallel" in (r.getMessage() or "")
    ]
    assert not bad
