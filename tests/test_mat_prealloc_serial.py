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
    PETSc = _get_petsc()
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Serial test: run with COMM_WORLD size == 1.")
    return PETSc


def _make_seq_vec(n: int):
    PETSc = _get_petsc()
    return PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)


def test_owner_map_serial_simple():
    _import_petsc_or_skip()
    n = 10
    vec = _make_seq_vec(n)
    ranges = get_global_ownership_ranges_from_vec(vec)
    assert ranges.shape == (2,)
    assert int(ranges[0]) == 0
    assert int(ranges[1]) == n

    rstart, rend = get_ownership_range_from_mat_or_vec(vec)
    assert rstart == 0
    assert rend == n

    owner_map = build_owner_map_from_ownership_ranges(ranges)
    assert owner_map.size == 1
    for j in range(n):
        assert owner_map.owner_of(j) == 0


def test_split_row_cols_serial_all_diag():
    _import_petsc_or_skip()
    n = 20
    vec = _make_seq_vec(n)
    ranges = get_global_ownership_ranges_from_vec(vec)
    owner_map = build_owner_map_from_ownership_ranges(ranges)
    myrank = 0

    cols = np.array([0, 1, 5, 7, 19], dtype=int)
    diag_cols, off_cols = split_row_cols_by_owner(cols, owner_map, myrank)

    assert np.array_equal(np.sort(diag_cols), np.sort(cols))
    assert off_cols.size == 0


def test_count_nnz_serial_all_diag():
    _import_petsc_or_skip()
    n = 5
    vec = _make_seq_vec(n)
    ranges = get_global_ownership_ranges_from_vec(vec)
    owner_map = build_owner_map_from_ownership_ranges(ranges)
    myrank = 0

    local_rows = [0, 1, 2, 3, 4]
    local_row_cols = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 4],
    ]

    d_nz, o_nz = count_diag_off_nnz_for_local_rows(
        local_rows,
        local_row_cols,
        owner_map,
        myrank,
        (0, n),
    )

    assert np.array_equal(d_nz, np.array([2, 2, 2, 2, 2], dtype=int))
    assert np.all(o_nz == 0)
