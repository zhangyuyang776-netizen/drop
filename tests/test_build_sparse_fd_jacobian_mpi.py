from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian  # noqa: E402
from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
from assembly.jacobian_pattern_dist import build_jacobian_pattern_local  # noqa: E402
from tests.test_jacobian_pattern_dist_serial import _make_dummy_cfg_layout  # noqa: E402
import assembly.residual_global as res_mod  # noqa: E402


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    return PETSc


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


class DummyCtx:
    def __init__(self, cfg, grid, layout, N: int) -> None:
        self.cfg = cfg
        self.grid_ref = grid
        self.layout = layout
        self.scale_u = np.ones(N, dtype=np.float64)
        self.meta = {"residual_scale_F": np.ones(N, dtype=np.float64)}


def _build_linear_case():
    cfg, grid, layout = _make_dummy_cfg_layout()
    pattern = build_jacobian_pattern(cfg, grid, layout)
    N = int(pattern.shape[0])

    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        row_cols = pattern.indices[pattern.indptr[i] : pattern.indptr[i + 1]]
        for k, j in enumerate(row_cols):
            A[i, int(j)] = (i + 1) * 10.0 + (k + 1)

    ctx = DummyCtx(cfg, grid, layout, N)
    x0 = np.linspace(0.1, 1.0, N, dtype=np.float64)
    return ctx, x0, A, pattern, cfg, grid, layout


def _patch_linear_residual(monkeypatch, A: np.ndarray) -> None:
    def residual_only_linear(u_phys, ctx):
        return A @ u_phys

    def residual_only_owned_rows_linear(u_phys, ctx, ownership_range):
        rstart, rend = ownership_range
        return (A @ u_phys)[int(rstart) : int(rend)].copy()

    monkeypatch.setattr(res_mod, "residual_only", residual_only_linear)
    monkeypatch.setattr(res_mod, "residual_only_owned_rows", residual_only_owned_rows_linear)


def test_sparse_fd_jacobian_mpi_linear_matches_pattern(monkeypatch):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    ctx, x0, A, pattern, cfg, grid, layout = _build_linear_case()
    _patch_linear_residual(monkeypatch, A)

    J, info = build_sparse_fd_jacobian(ctx, x0, eps=1.0e-6, drop_tol=0.0, pattern=pattern)
    assert str(J.getType()).lower() in ("mpiaij", "aij")

    rstart, rend = J.getOwnershipRange()
    for i in range(rstart, rend):
        cols, vals = J.getRow(i)
        cols_arr = np.asarray(cols, dtype=int)
        vals_arr = np.asarray(vals, dtype=np.float64)
        row_dense = np.zeros(A.shape[1], dtype=np.float64)
        if cols_arr.size:
            row_dense[cols_arr] = vals_arr
        assert cols_arr.size > 0
        assert np.any(cols_arr == i)
        assert np.allclose(row_dense, A[int(i)], rtol=1.0e-8, atol=1.0e-10)
        if hasattr(J, "restoreRow"):
            J.restoreRow(i, cols, vals)

    local_pat = build_jacobian_pattern_local(cfg, grid, layout, (rstart, rend))
    cols_all = np.asarray(info.get("cols_to_perturb", []), dtype=int)
    used_cols = np.unique(np.asarray(local_pat.indices, dtype=int))
    assert np.all(np.isin(used_cols, cols_all))

    pattern_nnz_local = int(local_pat.indices.size)
    assert info["pattern_nnz_local"] == pattern_nnz_local
    assert info["prealloc_nnz_local"] == pattern_nnz_local
    assert info["nnz_total_local"] == pattern_nnz_local
