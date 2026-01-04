from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
from assembly.jacobian_pattern_dist import (  # noqa: E402
    LocalJacPattern,
    build_jacobian_pattern_local,
)
from tests.test_jacobian_pattern_dist_serial import (  # noqa: E402
    _make_dummy_cfg_layout,
)


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    return PETSc


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


def test_local_pattern_mpi_matches_global_subblocks():
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()

    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()
    if size < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    cfg, grid, layout = _make_dummy_cfg_layout()
    if rank == 0:
        global_pat = build_jacobian_pattern(cfg, grid, layout)
        n = global_pat.shape[0]
        g_indptr = global_pat.indptr
        g_indices = global_pat.indices
    else:
        global_pat = None
        n = layout.N_total
        g_indptr = None
        g_indices = None

    from mpi4py import MPI

    mpicomm = comm.tompi4py()
    n = mpicomm.bcast(n, root=0)
    g_indptr = mpicomm.bcast(g_indptr, root=0)
    g_indices = mpicomm.bcast(g_indices, root=0)

    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes(n)
    vec.setFromOptions()
    vec.setUp()
    rstart, rend = vec.getOwnershipRange()

    local_pat = build_jacobian_pattern_local(
        cfg,
        grid,
        layout,
        ownership_range=(rstart, rend),
    )

    assert isinstance(local_pat, LocalJacPattern)
    assert np.array_equal(
        local_pat.rows_global,
        np.arange(rstart, rend, dtype=np.int32),
    )

    nloc = rend - rstart
    assert local_pat.indptr.shape == (nloc + 1,)

    for k, gi in enumerate(local_pat.rows_global):
        g_slice = slice(g_indptr[gi], g_indptr[gi + 1])
        l_slice = slice(local_pat.indptr[k], local_pat.indptr[k + 1])

        g_row = np.sort(g_indices[g_slice])
        l_row = np.sort(local_pat.indices[l_slice])

        assert np.array_equal(g_row, l_row)

    local_flags = np.zeros(n, dtype=np.int32)
    for gi in local_pat.rows_global:
        local_flags[int(gi)] = 1

    all_flags = np.zeros_like(local_flags)
    mpicomm.Allreduce(local_flags, all_flags, op=MPI.SUM)

    assert np.all(all_flags == 1)
