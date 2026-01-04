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

    _grid, layout, _state0, _props0 = build_min_problem(cfg)
    return cfg, layout


def test_dm_if_shell_no_collectives_mpi():
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    cfg, layout = _build_cfg_and_layout(comm.getSize())

    from parallel.dm_manager import build_dm

    mgr = build_dm(cfg, layout, comm=comm)
    if mgr.n_if <= 0:
        pytest.skip("Interface block is empty in this configuration.")

    X_if = mgr.dm_if.createGlobalVec()
    Xl_if = mgr.dm_if.createLocalVec()

    if comm.getRank() == 0:
        arr = X_if.getArray()
        arr[:] = 1.0
        X_if.assemble()

    mgr.dm_if.globalToLocal(X_if, Xl_if, addv=PETSc.InsertMode.INSERT_VALUES)
    mgr.dm_if.localToGlobal(Xl_if, X_if, addv=PETSc.InsertMode.ADD_VALUES)

    assert Xl_if.getSize() == mgr.n_if
