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


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def _build_case(tmp_path: Path, nproc: int, rank: int):
    from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402

    cfg, layout, ctx, u0 = _build_case_defaults(tmp_path, nproc, rank)

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc_mpi"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = 10

    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.petsc.ksp_type = "gmres"
    cfg.petsc.pc_type = "asm"
    cfg.petsc.max_it = 10

    return cfg, layout, ctx, u0


def test_petsc_snes_parallel_mfpc_sparse_fd_smoke_mpi(tmp_path: Path):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test: run with mpiexec -n 2/4 ...")

    rank = comm.getRank()
    tmp_rank = tmp_path / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(tmp_rank, comm.getSize(), rank)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    result = solve_nonlinear_petsc_parallel(ctx, u0)
    diag = result.diag
    extra = diag.extra or {}

    assert diag.converged is True
    assert int(extra.get("snes_reason", 0)) > 0

    assert extra.get("jacobian_mode") == "mfpc_sparse_fd"
    assert extra.get("snes_mf_enabled", False) is True

    x_type = str(extra.get("X_vec_type", "")).lower()
    assert "mpi" in x_type

    p_type = str(extra.get("P_mat_type", "")).lower()
    assert "aij" in p_type

    a_type = str(extra.get("KSP_A_type", "")).lower()
    pk_type = str(extra.get("KSP_P_type", "")).lower()
    assert "aij" in a_type
    assert "aij" in pk_type
    assert extra.get("KSP_A_is_P", False) is True

    pc_type = str(extra.get("pc_type", "")).lower()
    assert pc_type in {"asm", "bjacobi"}

    assert np.isfinite(diag.res_norm_inf)
    assert extra.get("ksp_its_total", 0) >= 1

    try:
        comm.barrier()
    except Exception:
        pass
