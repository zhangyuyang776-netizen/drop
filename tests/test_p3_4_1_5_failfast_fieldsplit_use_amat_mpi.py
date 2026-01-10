from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_mpi4py_or_skip():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI
    return MPI


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    return PETSc


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


@pytest.mark.parametrize("fs_type", ["additive", "schur"])
@pytest.mark.mpi
def test_p3_4_1_5_failfast_allows_valid_config_mpi(tmp_path: Path, fs_type: str, monkeypatch):
    watchdog_prev = os.environ.get("DROPLET_TEST_WATCHDOG")
    os.environ["DROPLET_TEST_WATCHDOG"] = "1"
    try:
        monkeypatch.setenv("DROPLET_PETSC_FAILFAST_FIELDSPLIT", "1")
        _import_mpi4py_or_skip()
        PETSc = _import_petsc_or_skip()
        _import_chemistry_or_skip()

        comm = PETSc.COMM_WORLD
        if comm.getSize() < 2:
            pytest.skip("need MPI size >= 2")

        from tests.test_p3_4_0_fieldsplit_uses_pmat_only_mpi import (
            _build_precond_matrix,
            _build_shell_like_matrix,
            _start_watchdog_abort_after_seconds,
        )
        from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402
        from parallel.dm_manager import build_dm  # noqa: E402
        from core.layout_dist import LayoutDistributed  # noqa: E402
        from solvers.petsc_linear import apply_structured_pc  # noqa: E402

        _start_watchdog_abort_after_seconds(30.0)

        rank = comm.getRank()
        tmp_rank = tmp_path / f"rank_{rank:03d}"
        tmp_rank.mkdir(parents=True, exist_ok=True)

        cfg, layout, ctx, u0 = _build_case_defaults(tmp_rank, comm.getSize(), rank)
        cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
        if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
            cfg.solver.linear.pc_type = "fieldsplit"
            cfg.solver.linear.fieldsplit = {"type": fs_type, "scheme": "bulk_iface"}
            if fs_type == "schur":
                cfg.solver.linear.fieldsplit["schur_fact_type"] = "lower"

        mgr = build_dm(cfg, layout, comm=comm)
        ld = LayoutDistributed.build(comm, mgr, layout)
        ctx.meta["dm"] = mgr.dm
        ctx.meta["dm_manager"] = mgr
        ctx.meta["layout_dist"] = ld

        fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
        drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

        A_shell = _build_shell_like_matrix(PETSc, comm, mgr.dm)
        P_aij = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(A_shell, P_aij)
        ksp.setType("gmres")

        diag_pc = apply_structured_pc(ksp, cfg, layout, A_shell, P_aij)

        pc = ksp.getPC()
        assert pc.getType().lower() == "fieldsplit"

        uses_amat = None
        try:
            uses_amat = bool(pc.getUseAmat())
        except Exception:
            uses_amat = diag_pc.get("uses_amat", None)
        assert uses_amat is False
    finally:
        if watchdog_prev is None:
            os.environ.pop("DROPLET_TEST_WATCHDOG", None)
        else:
            os.environ["DROPLET_TEST_WATCHDOG"] = watchdog_prev


@pytest.mark.parametrize("fs_type", ["additive", "schur"])
@pytest.mark.mpi
def test_p3_4_1_5_failfast_triggers_on_use_amat_mpi(tmp_path: Path, fs_type: str, monkeypatch):
    watchdog_prev = os.environ.get("DROPLET_TEST_WATCHDOG")
    os.environ["DROPLET_TEST_WATCHDOG"] = "1"
    try:
        monkeypatch.setenv("DROPLET_PETSC_FAILFAST_FIELDSPLIT", "1")
        _import_mpi4py_or_skip()
        PETSc = _import_petsc_or_skip()
        _import_chemistry_or_skip()

        comm = PETSc.COMM_WORLD
        if comm.getSize() < 2:
            pytest.skip("need MPI size >= 2")

        from tests.test_p3_4_0_fieldsplit_uses_pmat_only_mpi import (
            _build_precond_matrix,
            _build_shell_like_matrix,
            _start_watchdog_abort_after_seconds,
        )
        from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402
        from parallel.dm_manager import build_dm  # noqa: E402
        from core.layout_dist import LayoutDistributed  # noqa: E402
        import solvers.petsc_linear as pl  # noqa: E402

        _start_watchdog_abort_after_seconds(30.0)

        rank = comm.getRank()
        tmp_rank = tmp_path / f"rank_{rank:03d}"
        tmp_rank.mkdir(parents=True, exist_ok=True)

        cfg, layout, ctx, u0 = _build_case_defaults(tmp_rank, comm.getSize(), rank)
        cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
        if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
            cfg.solver.linear.pc_type = "fieldsplit"
            cfg.solver.linear.fieldsplit = {"type": fs_type, "scheme": "bulk_iface"}
            if fs_type == "schur":
                cfg.solver.linear.fieldsplit["schur_fact_type"] = "lower"

        mgr = build_dm(cfg, layout, comm=comm)
        ld = LayoutDistributed.build(comm, mgr, layout)
        ctx.meta["dm"] = mgr.dm
        ctx.meta["dm_manager"] = mgr
        ctx.meta["layout_dist"] = ld

        fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
        drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

        A_shell = _build_shell_like_matrix(PETSc, comm, mgr.dm)
        P_aij = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(A_shell, P_aij)
        ksp.setType("gmres")

        def _noop_disable(*_args, **_kwargs):
            return {"patched": True, "uses_amat_after": True}

        monkeypatch.setattr(pl, "_disable_pc_use_amat", _noop_disable)
        monkeypatch.setattr(pl, "_get_pc_use_amat", lambda *_args, **_kwargs: True)

        with pytest.raises(RuntimeError, match="failfast|pc_use_amat|use Amat"):
            pl.apply_structured_pc(ksp, cfg, layout, A_shell, P_aij)
    finally:
        if watchdog_prev is None:
            os.environ.pop("DROPLET_TEST_WATCHDOG", None)
        else:
            os.environ["DROPLET_TEST_WATCHDOG"] = watchdog_prev
