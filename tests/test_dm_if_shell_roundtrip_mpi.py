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


def test_dm_if_shell_roundtrip_mpi():
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    cfg, layout = _build_cfg_and_layout(comm.getSize())

    from parallel.dm_manager import build_dm, global_to_local, local_to_global_add, _dmcomposite_access

    mgr = build_dm(cfg, layout, comm=comm)
    if mgr.n_if <= 0:
        pytest.skip("Interface block is empty in this configuration.")

    # --- Global to local: root initializes interface values, broadcast to all ranks ---
    Xg = mgr.dm.createGlobalVec()
    with _dmcomposite_access(mgr.dm, Xg) as (_X_liq, _X_gas, X_if):
        if comm.getRank() == 0:
            arr = X_if.getArray()
            arr[:] = np.arange(1, mgr.n_if + 1, dtype=np.float64)
    _Xl_liq, _Xl_gas, Xl_if = global_to_local(mgr, Xg)
    vals = np.asarray(Xl_if.getArray(), dtype=np.float64)
    np.testing.assert_allclose(vals, np.arange(1, mgr.n_if + 1, dtype=np.float64))

    # --- Local to global: each rank contributes rank+1, root sees sum ---
    Fl_liq = mgr.dm_liq.createLocalVec()
    Fl_gas = mgr.dm_gas.createLocalVec()
    Fl_if = mgr.dm_if.createLocalVec()
    Fl_liq.set(0.0)
    Fl_gas.set(0.0)
    Fl_if.set(0.0)
    Fl_if.getArray()[:] = float(comm.getRank() + 1)

    Fg = local_to_global_add(mgr, Fl_liq, Fl_gas, Fl_if)
    if comm.getRank() == 0:
        with _dmcomposite_access(mgr.dm, Fg) as (_F_liq, _F_gas, F_if_g):
            f_if = np.asarray(F_if_g.getArray(), dtype=np.float64)
            expected = float(comm.getSize() * (comm.getSize() + 1) / 2)
            np.testing.assert_allclose(f_if, expected)
