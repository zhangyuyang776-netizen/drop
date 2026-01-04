from __future__ import annotations

import sys
from pathlib import Path

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


def _build_cfg_and_layout(nproc: int):
    from tests._helpers_step15 import build_min_problem, make_cfg_base

    Nl = max(8, int(nproc))
    Ng = max(64, int(8 * nproc))

    cfg = make_cfg_base(
        Nl=Nl,
        Ng=Ng,
        solve_Yg=True,
        include_mpp=True,
        include_Ts=True,
        include_Rd=False,
    )
    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc"
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"

    grid, layout, _state0, _props0 = build_min_problem(cfg)
    return cfg, layout


def _petsc_allreduce_sum_scalar(PETSc, comm, local_value: int) -> int:
    """
    MPI sum without mpi4py:
    Put local_value into a length-(nproc) MPI Vec (one entry per rank),
    and use Vec.sum() as the global reduction.

    Note: petsc4py 3.22 Vec.createMPI signature is easy to misuse.
    Use (local, global) tuple explicitly and force blocksize=1.
    """
    nproc = comm.getSize()
    rank = comm.getRank()
    v = PETSc.Vec().createMPI((1, nproc), comm=comm)
    v.setBlockSize(1)
    v.setUp()
    v.setValue(rank, float(local_value))
    v.assemblyBegin()
    v.assemblyEnd()
    s = v.sum()
    v.destroy()
    return int(round(s))


def _as_1d_scalar(x) -> int:
    if isinstance(x, (tuple, list)):
        x = x[0]
    return int(x)


def test_dm_composite_builds_in_serial_or_mpi():
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD

    cfg, layout = _build_cfg_and_layout(comm.getSize())

    from parallel.dm_manager import build_dm, create_global_vec

    mgr = build_dm(cfg, layout, comm=comm)

    Xg = create_global_vec(mgr)
    assert Xg.getSize() > 0

    v_if = mgr.dm_if.createGlobalVec()
    nloc = v_if.getLocalSize()
    assert nloc in (0, mgr.n_if)

    sum_loc = _petsc_allreduce_sum_scalar(PETSc, comm, nloc)
    assert sum_loc == mgr.n_if
    assert v_if.getSize() == mgr.n_if


def test_dm_composite_mpi_ownership_ranges():
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI ownership test: run with mpiexec -n 2/4 ...")

    cfg, layout = _build_cfg_and_layout(comm.getSize())

    from parallel.dm_manager import build_dm

    mgr = build_dm(cfg, layout, comm=comm)

    xs_l, xm_l = mgr.dm_liq.getCorners()
    xs_g, xm_g = mgr.dm_gas.getCorners()

    xs_l = _as_1d_scalar(xs_l)
    xm_l = _as_1d_scalar(xm_l)
    xs_g = _as_1d_scalar(xs_g)
    xm_g = _as_1d_scalar(xm_g)

    r0_l, r1_l = xs_l, xs_l + xm_l
    r0_g, r1_g = xs_g, xs_g + xm_g

    assert 0 <= r0_l <= r1_l <= mgr.Nl
    assert 0 <= r0_g <= r1_g <= mgr.Ng

    sum_l = _petsc_allreduce_sum_scalar(PETSc, comm, int(xm_l))
    sum_g = _petsc_allreduce_sum_scalar(PETSc, comm, int(xm_g))
    assert sum_l == mgr.Nl
    assert sum_g == mgr.Ng
