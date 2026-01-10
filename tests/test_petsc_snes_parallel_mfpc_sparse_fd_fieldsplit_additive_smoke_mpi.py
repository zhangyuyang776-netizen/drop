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


def _prepare_case(tmp_rank: Path, comm, *, pc_type: str, fieldsplit_cfg=None):
    from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402
    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402

    cfg, layout, ctx, u0 = _build_case_defaults(tmp_rank, comm.getSize(), comm.getRank())
    cfg.nonlinear.max_outer_iter = 10
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
        cfg.solver.linear.pc_type = pc_type
        cfg.solver.linear.fieldsplit = fieldsplit_cfg

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld
    return cfg, ctx, u0


def test_mfpc_sparse_fd_fieldsplit_additive_smoke_mpi(tmp_path: Path):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test requires >= 2 ranks")

    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402

    rank = comm.getRank()
    tmp_asm = tmp_path / f"asm_rank_{rank:03d}"
    tmp_fs = tmp_path / f"fs_rank_{rank:03d}"
    tmp_asm.mkdir(parents=True, exist_ok=True)
    tmp_fs.mkdir(parents=True, exist_ok=True)

    _cfg_asm, ctx_asm, u0_asm = _prepare_case(tmp_asm, comm, pc_type="asm")
    res_asm = solve_nonlinear_petsc_parallel(ctx_asm, u0_asm)
    assert res_asm.diag.converged is True

    fs_cfg = {
        "type": "additive",
        "scheme": "bulk_iface",
        "sub_ksp_type": "preonly",
        "sub_pc_type": "asm",
        "sub_pc_asm_sub_pc_type": "ilu",
    }
    _cfg_fs, ctx_fs, u0_fs = _prepare_case(tmp_fs, comm, pc_type="fieldsplit", fieldsplit_cfg=fs_cfg)
    res_fs = solve_nonlinear_petsc_parallel(ctx_fs, u0_fs)
    assert res_fs.diag.converged is True

    diag_pc = (res_fs.diag.extra or {}).get("pc_structured", {}) or {}
    assert diag_pc.get("fieldsplit_type") == "additive"

    assert np.isfinite(res_fs.diag.res_norm_inf)
    assert float(res_fs.diag.res_norm_inf) < 1.0e-6

    du_inf = float(np.max(np.abs(res_fs.u - res_asm.u)))
    assert du_inf < 1.0e-6

    try:
        comm.barrier()
    except Exception:
        pass
