from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    return PETSc


def _import_mpi4py():
    return pytest.importorskip("mpi4py")


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


def _is_to_array(isobj) -> np.ndarray:
    idx = isobj.getIndices()
    if idx is None:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(idx, dtype=np.int64)
    if hasattr(isobj, "restoreIndices"):
        try:
            isobj.restoreIndices()
        except TypeError:
            isobj.restoreIndices(idx)
    return arr


def test_layout_dist_builds_and_is_partitions_global():
    _import_mpi4py()
    PETSc = _import_petsc()
    comm = PETSc.COMM_WORLD
    cfg, layout = _build_cfg_and_layout(comm.getSize())

    from parallel.dm_manager import build_dm
    from core.layout_dist import LayoutDistributed

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)

    groups = {
        "bulk": ["Tl", "Yl", "Tg", "Yg"],
        "iface": ["Ts", "mpp", "Rd"],
    }

    is_bulk = ld.build_is_petsc(groups, "bulk")
    is_iface = ld.build_is_petsc(groups, "iface")

    N_total = ld.N_total
    assert is_bulk.getSize() + is_iface.getSize() == N_total

    Xg = mgr.dm.createGlobalVec()
    r0, r1 = Xg.getOwnershipRange()

    def _check_owned(isobj):
        idx = _is_to_array(isobj)
        if idx.size == 0:
            return
        assert np.all((idx >= r0) & (idx < r1))

    _check_owned(is_bulk)
    _check_owned(is_iface)

    from mpi4py import MPI  # noqa: F401

    mpicomm = comm.tompi4py()
    bulk_loc = _is_to_array(is_bulk)
    iface_loc = _is_to_array(is_iface)

    all_bulk = mpicomm.gather(bulk_loc.tolist(), root=0)
    all_iface = mpicomm.gather(iface_loc.tolist(), root=0)

    if mpicomm.rank == 0:
        bulk_flat = [x for sub in all_bulk for x in sub]
        iface_flat = [x for sub in all_iface for x in sub]
        set_bulk = set(bulk_flat)
        set_iface = set(iface_flat)
        assert len(set_bulk.intersection(set_iface)) == 0
        union = set_bulk.union(set_iface)
        assert len(union) == N_total
