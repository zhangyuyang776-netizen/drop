from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
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
    sum_loc = comm.allreduce(nloc, op=PETSc.Sum)
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

    r0_l, r1_l = mgr.dm_liq.getOwnershipRange()
    r0_g, r1_g = mgr.dm_gas.getOwnershipRange()

    assert 0 <= r0_l <= r1_l <= mgr.Nl
    assert 0 <= r0_g <= r1_g <= mgr.Ng

    nloc_l = r1_l - r0_l
    nloc_g = r1_g - r0_g
    sum_l = comm.allreduce(nloc_l, op=PETSc.Sum)
    sum_g = comm.allreduce(nloc_g, op=PETSc.Sum)
    assert sum_l == mgr.Nl
    assert sum_g == mgr.Ng
