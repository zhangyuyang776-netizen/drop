from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_precond_aij import (  # noqa: E402
    build_precond_mat_aij_from_A,
    fill_precond_mat_aij_from_A,
)
from tests.test_build_sparse_fd_jacobian_mpi import (  # noqa: E402
    _build_linear_case,
    _patch_linear_residual,
    _import_mpi4py_or_skip,
    _import_petsc_or_skip,
)


def test_precond_aij_mpi_linear_matches_dense(monkeypatch):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    ctx, x0, A_dense, _pattern, _cfg, _grid, _layout = _build_linear_case()
    _patch_linear_residual(monkeypatch, A_dense)

    petsc_cfg = getattr(ctx.cfg, "petsc", None)
    if petsc_cfg is None:
        ctx.cfg.petsc = SimpleNamespace(fd_eps=1.0e-4)
    else:
        petsc_cfg.fd_eps = 1.0e-4

    u_phys = x0.copy()
    P, diag = build_precond_mat_aij_from_A(ctx, u_phys, drop_tol=0.0)
    assert P.getType().lower() in ("aij", "seqaij", "mpiaij")

    rstart, rend = P.getOwnershipRange()
    for i in range(rstart, rend):
        cols, vals = P.getRow(i)
        cols_arr = np.asarray(cols, dtype=int)
        vals_arr = np.asarray(vals, dtype=np.float64)
        row = np.zeros(A_dense.shape[1], dtype=np.float64)
        if cols_arr.size:
            row[cols_arr] = vals_arr
        np.testing.assert_allclose(row, A_dense[int(i), :], rtol=1.0e-10, atol=1.0e-12)
        if hasattr(P, "restoreRow"):
            P.restoreRow(i, cols, vals)

    diag2 = fill_precond_mat_aij_from_A(P, ctx, u_phys, drop_tol=0.0)
    for i in range(rstart, rend):
        cols, vals = P.getRow(i)
        cols_arr = np.asarray(cols, dtype=int)
        vals_arr = np.asarray(vals, dtype=np.float64)
        row = np.zeros(A_dense.shape[1], dtype=np.float64)
        if cols_arr.size:
            row[cols_arr] = vals_arr
        np.testing.assert_allclose(row, A_dense[int(i), :], rtol=1.0e-10, atol=1.0e-12)
        if hasattr(P, "restoreRow"):
            P.restoreRow(i, cols, vals)

    assert diag.get("builder") == "mfpc_aija_fd"
    assert diag.get("mpi_size") == comm.getSize()
    assert diag.get("n_fd_calls", 0) >= 1
    assert diag.get("n_local_rows") == (rend - rstart)
    assert "mode" in diag and "fd_mpi" in diag["mode"]

    assert diag2.get("builder") == "mfpc_aija_fd"
    assert "mode" in diag2 and "fd_mpi_fill" in diag2["mode"]


def test_precond_aij_mpi_drop_tol(monkeypatch):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    ctx, x0, A_dense, pattern, _cfg, _grid, _layout = _build_linear_case()
    drop_tol = 1.0e-8
    A_small = A_dense.copy()
    changed = 0
    for i in range(A_small.shape[0]):
        row_cols = pattern.indices[pattern.indptr[i] : pattern.indptr[i + 1]]
        off_cols = [int(c) for c in row_cols if int(c) != i]
        if off_cols:
            A_small[i, off_cols[0]] = drop_tol * 0.1
            changed += 1
            if changed >= 3:
                break
    if changed == 0:
        pytest.skip("Pattern has no off-diagonal entries to test drop_tol.")

    _patch_linear_residual(monkeypatch, A_small)

    u_phys = x0.copy()
    P, _diag = build_precond_mat_aij_from_A(ctx, u_phys, drop_tol=drop_tol)
    rstart, rend = P.getOwnershipRange()
    for i in range(rstart, rend):
        cols, vals = P.getRow(i)
        vals_arr = np.asarray(vals, dtype=np.float64)
        assert np.all(np.abs(vals_arr) >= drop_tol)
        if hasattr(P, "restoreRow"):
            P.restoreRow(i, cols, vals)
