from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel.mat_prealloc import (  # noqa: E402
    _get_petsc,
    build_owner_map_from_ownership_ranges,
    count_diag_off_nnz_for_local_rows,
    get_global_ownership_ranges_from_vec,
    get_ownership_range_from_mat_or_vec,
    split_row_cols_by_owner,
)


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    return _get_petsc()


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


def test_owner_map_mpi_ranges_and_owners():
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()
    if size < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    n = 16
    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes(n)
    vec.setFromOptions()
    vec.setUp()

    ranges = get_global_ownership_ranges_from_vec(vec)
    assert ranges.size == size + 1
    assert int(ranges[0]) == 0
    assert int(ranges[-1]) == n
    assert np.all(ranges[1:] >= ranges[:-1])

    owner_map = build_owner_map_from_ownership_ranges(ranges)

    rstart, rend = get_ownership_range_from_mat_or_vec(vec)
    for i in range(rstart, rend):
        assert owner_map.owner_of(i) == rank

    test_indices = list(range(n))
    local_ok = True
    for j in test_indices:
        owner = owner_map.owner_of(j)
        if not (ranges[owner] <= j < ranges[owner + 1]):
            local_ok = False
            break

    from mpi4py import MPI

    mpicomm = comm.tompi4py()
    all_ok = mpicomm.allreduce(1 if local_ok else 0, op=MPI.MIN)
    assert all_ok == 1


def test_split_and_count_nnz_mpi():
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()
    if size < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    n = 32
    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes(n)
    vec.setFromOptions()
    vec.setUp()

    ranges = get_global_ownership_ranges_from_vec(vec)
    owner_map = build_owner_map_from_ownership_ranges(ranges)
    rstart, rend = get_ownership_range_from_mat_or_vec(vec)

    local_rows = []
    local_row_cols = []
    for i in range(rstart, rend):
        cols = [i]
        if i - 1 >= 0:
            cols.append(i - 1)
        if i + 1 < n:
            cols.append(i + 1)
        cols = sorted(set(cols))
        local_rows.append(i)
        local_row_cols.append(cols)

        diag_cols, off_cols = split_row_cols_by_owner(cols, owner_map, rank)
        for c in diag_cols:
            assert owner_map.owner_of(int(c)) == rank
        for c in off_cols:
            assert owner_map.owner_of(int(c)) != rank

    d_nz, o_nz = count_diag_off_nnz_for_local_rows(
        local_rows,
        local_row_cols,
        owner_map,
        rank,
        (rstart, rend),
    )
    assert d_nz.shape == o_nz.shape == (len(local_rows),)

    for k in range(len(local_rows)):
        assert d_nz[k] + o_nz[k] == len(local_row_cols[k])
