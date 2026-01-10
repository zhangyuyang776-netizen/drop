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


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def test_fieldsplit_schur_defaults_applied_mpi(tmp_path: Path):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("MPI test requires >= 2 ranks")

    from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402
    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402

    rank = comm.getRank()
    tmp_rank = tmp_path / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case_defaults(tmp_rank, comm.getSize(), rank)
    cfg.nonlinear.max_outer_iter = 10
    if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
        cfg.solver.linear.pc_type = "fieldsplit"
        cfg.solver.linear.fieldsplit = {
            "type": "schur",
            "scheme": "bulk_iface",
            "schur_fact_type": "lower",
        }

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    result = solve_nonlinear_petsc_parallel(ctx, u0)
    diag = result.diag
    extra = diag.extra or {}
    diag_pc = extra.get("pc_structured", {}) or {}
    split_cfgs = diag_pc.get("fieldsplit", {}).get("splits", {}) or {}

    assert diag.converged is True
    assert extra.get("jacobian_mode") == "mfpc_sparse_fd"
    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "lower"
    assert "bulk" in split_cfgs
    assert "iface" in split_cfgs

    bulk_cfg = split_cfgs.get("bulk", {}) or {}
    iface_cfg = split_cfgs.get("iface", {}) or {}
    assert bulk_cfg.get("pc_type") == "asm"
    assert bulk_cfg.get("subdomain_ksp_type") == "preonly"
    assert bulk_cfg.get("subdomain_pc_type") == "ilu"
    assert int(bulk_cfg.get("asm_overlap", 0)) == 1

    assert iface_cfg.get("ksp_type") == "preonly"
    assert iface_cfg.get("pc_type") == "ilu"

    try:
        comm.barrier()
    except Exception:
        pass
