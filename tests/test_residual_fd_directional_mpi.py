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


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


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
    cfg.geometry.N_liq = max(4, int(nproc))
    cfg.geometry.N_gas = max(8, int(2 * nproc))
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 2

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


def _gather_vec_to_root(comm, vec):
    mpicomm = comm.tompi4py()
    r0, r1 = vec.getOwnershipRange()
    try:
        local_view = vec.getArray(readonly=True)
    except TypeError:
        local_view = vec.getArray()
    local = np.asarray(local_view, dtype=np.float64)
    ranges = mpicomm.gather((r0, r1), root=0)
    data = mpicomm.gather(local, root=0)
    if mpicomm.rank != 0:
        return None
    out = np.zeros(vec.getSize(), dtype=np.float64)
    for (s, e), part in zip(ranges, data):
        out[int(s) : int(e)] = part
    return out


def test_residual_fd_directional_mpi(tmp_path: Path):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    rank = comm.getRank()
    tmp_rank = tmp_path / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(tmp_rank, comm.getSize(), rank)

    from parallel.dm_manager import build_dm, local_state_to_global  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from assembly.residual_local import ResidualLocalCtx, scatter_layout_to_local  # noqa: E402
    from assembly.residual_global import residual_petsc, residual_only  # noqa: E402

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx_local = ResidualLocalCtx(layout=layout, ld=ld)

    from mpi4py import MPI

    mpicomm = comm.tompi4py()
    if rank == 0:
        rng = np.random.default_rng(12345)
        v = rng.standard_normal(u0.size).astype(np.float64)
    else:
        v = None
    v = mpicomm.bcast(v, root=0)

    eps = 1.0e-4
    if hasattr(layout, "idx_Rd"):
        i_Rd = int(layout.idx_Rd())
        Rd0 = float(u0[i_Rd])
        vRd = float(v[i_Rd])
        if Rd0 > 0.0 and abs(vRd) > 0.0:
            eps_max = 0.49 * Rd0 / abs(vRd)
            if eps_max > 0.0:
                eps = min(eps, eps_max)

    u1 = u0 + eps * v
    u2 = u0 - eps * v
    if hasattr(layout, "idx_Rd"):
        i_Rd = int(layout.idx_Rd())
        assert u1[i_Rd] > 0.0
        assert u2[i_Rd] > 0.0

    def _build_Xg_from_u(u_vec: np.ndarray):
        Xl_liq = mgr.dm_liq.createLocalVec()
        Xl_gas = mgr.dm_gas.createLocalVec()
        Xl_if = mgr.dm_if.createLocalVec()
        Xl_liq.set(0.0)
        Xl_gas.set(0.0)
        Xl_if.set(0.0)

        aXl_liq = mgr.dm_liq.getVecArray(Xl_liq)
        aXl_gas = mgr.dm_gas.getVecArray(Xl_gas)
        aXl_if = Xl_if.getArray()
        scatter_layout_to_local(ctx_local, u_vec, aXl_liq, aXl_gas, aXl_if, rank=rank)
        return local_state_to_global(mgr, Xl_liq, Xl_gas, Xl_if)

    u_tag = np.arange(u0.size, dtype=np.float64)
    Xg_tag = _build_Xg_from_u(u_tag)
    tag_raw = _gather_vec_to_root(comm, Xg_tag)
    perm_dm_to_layout = None
    if rank == 0:
        perm_dm_to_layout = np.rint(tag_raw).astype(np.int64)
        assert np.array_equal(np.sort(perm_dm_to_layout), np.arange(u0.size))

    def _dmraw_to_layout(raw_dm: np.ndarray | None):
        if raw_dm is None:
            return None
        out = np.empty_like(raw_dm)
        out[perm_dm_to_layout] = raw_dm
        return out

    Xg0 = _build_Xg_from_u(u0)
    Xg1 = _build_Xg_from_u(u1)
    Xg2 = _build_Xg_from_u(u2)

    u0_rt = _dmraw_to_layout(_gather_vec_to_root(comm, Xg0))
    ok_rt = True
    if rank == 0:
        du = u0_rt - u0
        norm_u0 = float(np.linalg.norm(u0, ord=np.inf))
        norm_du = float(np.linalg.norm(du, ord=np.inf))
        j = int(np.argmax(np.abs(du)))
        print(
            f"[rt] ||u0_rt-u0||inf={norm_du:.3e}, ||u0||inf={norm_u0:.3e}",
            flush=True,
        )
        print(
            f"[rt] max@{j}: u0_rt={u0_rt[j]:.6e}, u0={u0[j]:.6e}, diff={du[j]:.6e}",
            flush=True,
        )
        ok_rt = norm_du <= 1.0e-10 * max(1.0, norm_u0)
    ok_rt = mpicomm.bcast(ok_rt, root=0)
    assert ok_rt

    F0 = residual_petsc(mgr, ld, ctx, Xg0)
    F1 = residual_petsc(mgr, ld, ctx, Xg1)
    F2 = residual_petsc(mgr, ld, ctx, Xg2)

    F0_global = _dmraw_to_layout(_gather_vec_to_root(comm, F0))
    ok_base = True
    if rank == 0:
        r0 = residual_only(u0, ctx)
        diff0 = F0_global - r0
        norm_r0 = float(np.linalg.norm(r0, ord=np.inf))
        norm_d0 = float(np.linalg.norm(diff0, ord=np.inf))
        j = int(np.argmax(np.abs(diff0)))
        print(
            f"[fd] base check: ||F0-r0||inf={norm_d0:.3e}, ||r0||inf={norm_r0:.3e}",
            flush=True,
        )
        print(
            f"[fd] base max@{j}: F0={F0_global[j]:.6e}, r0={r0[j]:.6e}, diff={diff0[j]:.6e}",
            flush=True,
        )
        ok_base = norm_d0 <= 1.0e-8 * max(1.0, norm_r0)
    ok_base = mpicomm.bcast(ok_base, root=0)
    assert ok_base

    dF = F1.duplicate()
    F1.copy(dF)
    dF.axpy(-1.0, F2)
    dF.scale(1.0 / (2.0 * eps))

    dF_global = _dmraw_to_layout(_gather_vec_to_root(comm, dF))
    ok = True
    if rank == 0:
        r1 = residual_only(u1, ctx)
        r2 = residual_only(u2, ctx)
        dF_ref = (r1 - r2) / (2.0 * eps)
        diff = dF_global - dF_ref
        norm_ref = float(np.linalg.norm(dF_ref, ord=np.inf))
        norm_diff = float(np.linalg.norm(diff, ord=np.inf))
        ok = norm_diff <= 1.0e-3 * max(1.0, norm_ref)
    ok = mpicomm.bcast(ok, root=0)
    assert ok
    try:
        comm.barrier()
    except Exception:
        pass
