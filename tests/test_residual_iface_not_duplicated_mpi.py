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
    return PETSc


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


def _build_case(tmp_path: Path, nproc: int):
    try:
        from driver.run_scipy_case import _load_case_config, _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config, _maybe_fill_gas_species  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = max(4, int(nproc))
    cfg.geometry.N_gas = max(8, int(2 * nproc))
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 2

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


def test_residual_iface_not_duplicated_mpi(tmp_path: Path):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI iface residual test requires nproc >= 2.")

    cfg, layout, ctx, u0 = _build_case(tmp_path, comm.getSize())

    from parallel.dm_manager import build_dm, local_to_global_add  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from assembly.residual_local import ResidualLocalCtx, scatter_layout_to_local  # noqa: E402
    from assembly.residual_global import residual_petsc, residual_only  # noqa: E402

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)

    Xl_liq = mgr.dm_liq.createLocalVec()
    Xl_gas = mgr.dm_gas.createLocalVec()
    Xl_if = mgr.dm_if.createLocalVec()
    Xl_liq.set(0.0)
    Xl_gas.set(0.0)
    Xl_if.set(0.0)

    aXl_liq = mgr.dm_liq.getVecArray(Xl_liq)
    aXl_gas = mgr.dm_gas.getVecArray(Xl_gas)
    aXl_if = Xl_if.getArray()

    ctx_local = ResidualLocalCtx(layout=layout, ld=ld)
    scatter_layout_to_local(ctx_local, u0, aXl_liq, aXl_gas, aXl_if, rank=comm.getRank())

    Xg = local_to_global_add(mgr, Xl_liq, Xl_gas, Xl_if)
    Fg = residual_petsc(mgr, ld, ctx, Xg)

    if comm.getRank() != 0:
        return

    res_ref = residual_only(u0, ctx)
    tol = 1e-8 * max(1.0, float(np.linalg.norm(res_ref, ord=np.inf)))

    def _check_block(name: str, idx_layout_fn):
        if not layout.has_block(name):
            return
        idx_dm = ld.owned_global_indices(name)
        if idx_dm.size == 0:
            return
        val = float(Fg.getValues(idx_dm.tolist())[0])
        ref = float(res_ref[idx_layout_fn()])
        assert abs(val - ref) <= tol, f"{name} residual mismatch: {val} vs {ref}"

    _check_block("Ts", layout.idx_Ts)
    _check_block("mpp", layout.idx_mpp)
    _check_block("Rd", layout.idx_Rd)
