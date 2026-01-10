from __future__ import annotations

import os
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


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI
    return MPI


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def _build_case(tmp_path: Path, nproc: int, rank: int):
    try:
        from driver.run_scipy_case import _load_case_config, _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config, _maybe_fill_gas_species  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = int(os.environ.get("DROPLET_P2_5_NL", "8"))
    cfg.geometry.N_gas = int(os.environ.get("DROPLET_P2_5_NG", "16"))
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc_mpi"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 30

    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.petsc.ksp_type = "gmres"
    cfg.petsc.pc_type = "asm"
    cfg.petsc.max_it = 50
    if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
        cfg.solver.linear.pc_type = "asm"

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / f"case_rank_{rank:03d}"
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


def _norm_inf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.max(np.abs(x))) if x.size else 0.0


def test_petsc_snes_mfpc_sparse_fd_parallel_smoke_mpi(tmp_path: Path):
    MPI = _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI smoke requires COMM_WORLD size >= 2")

    rank = comm.getRank()
    tmp_rank = tmp_path / f"p2_5_rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(tmp_rank, comm.getSize(), rank)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402
    from assembly.residual_global import residual_only  # noqa: E402

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    result = solve_nonlinear_petsc_parallel(ctx, u0)
    diag = result.diag
    extra = diag.extra or {}

    tol = float(os.environ.get("DROPLET_P2_5_TOL", "1.0e-8"))
    tol_ref = float(os.environ.get("DROPLET_P2_5_TOL_REF", "1.0e-6"))

    assert diag.converged is True
    assert extra.get("jacobian_mode") == "mfpc_sparse_fd"
    assert extra.get("snes_mf_enabled", False) is True
    assert extra.get("KSP_A_is_P", False) is True
    ksp_a_type = str(extra.get("KSP_A_type", "")).lower()
    ksp_p_type = str(extra.get("KSP_P_type", "")).lower()
    if ksp_a_type:
        assert "aij" in ksp_a_type
    if ksp_p_type:
        assert "aij" in ksp_p_type

    assert np.all(np.isfinite(result.u))
    assert np.isfinite(diag.res_norm_inf)
    assert float(diag.res_norm_inf) <= tol

    r_inf = None
    if comm.getRank() == 0:
        r_inf = _norm_inf(residual_only(result.u, ctx))
    r_inf = MPI.COMM_WORLD.bcast(r_inf, root=0)
    assert r_inf <= tol_ref

    try:
        comm.barrier()
    except Exception:
        pass
