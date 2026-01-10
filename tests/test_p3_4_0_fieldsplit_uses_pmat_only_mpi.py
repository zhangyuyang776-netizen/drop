from __future__ import annotations

import os
import sys
import threading
import time
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
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
    except Exception:
        pass
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    return PETSc


def _import_chemistry_or_skip():
    pytest.importorskip("cantera")
    pytest.importorskip("CoolProp")


def _is_aij(mat_type: str) -> bool:
    t = (mat_type or "").lower()
    return "aij" in t


def _is_shell_like(mat_type: str) -> bool:
    t = (mat_type or "").lower()
    return ("shell" in t) or ("mffd" in t) or ("python" in t)


def _get_fieldsplit_subksps(pc):
    sub = pc.getFieldSplitSubKSP()
    if isinstance(sub, tuple) and len(sub) == 2:
        return sub[1]
    return sub


def _start_watchdog_abort_after_seconds(seconds: float):
    if os.environ.get("DROPLET_TEST_WATCHDOG", "0") != "1":
        return None

    MPI = _import_mpi4py_or_skip()

    def _worker():
        time.sleep(seconds)
        try:
            MPI.COMM_WORLD.Abort(99)
        except Exception:
            pass

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th


def _build_precond_matrix(PETSc, comm, dm, ctx, u0, *, fd_eps: float, drop_tol: float):
    from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian

    x_vec = dm.createGlobalVec()
    try:
        nloc = int(x_vec.getLocalSize())
        n_glob = int(x_vec.getSize())
    finally:
        try:
            x_vec.destroy()
        except Exception:
            pass

    P_dm = PETSc.Mat().create(comm=comm)
    P_dm.setSizes(((nloc, n_glob), (nloc, n_glob)))
    try:
        P_dm.setType(PETSc.Mat.Type.AIJ)
    except Exception:
        P_dm.setType("aij")
    try:
        P_dm.setUp()
    except Exception:
        pass

    P, _ = build_sparse_fd_jacobian(
        ctx,
        u0,
        eps=fd_eps,
        drop_tol=drop_tol,
        pattern=None,
        mat=P_dm,
    )
    return P


def _build_shell_like_matrix(PETSc, comm, dm):
    x_vec = dm.createGlobalVec()
    try:
        nloc = int(x_vec.getLocalSize())
        n_glob = int(x_vec.getSize())
    finally:
        try:
            x_vec.destroy()
        except Exception:
            pass

    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((nloc, n_glob), (nloc, n_glob)))
    try:
        A.setType("mffd")
    except Exception:
        try:
            A.setType(PETSc.Mat.Type.MFFD)
        except Exception as exc:
            raise RuntimeError("Unable to create MFFD matrix for shell-like A.") from exc
    try:
        A.setUp()
    except Exception:
        pass
    return A


@pytest.mark.parametrize("fs_type", ["additive", "schur"])
@pytest.mark.mpi
def test_p3_4_0_fieldsplit_uses_pmat_only_mpi(tmp_path: Path, fs_type: str):
    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("need MPI size >= 2")

    _start_watchdog_abort_after_seconds(30.0)

    from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402
    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_structured_pc  # noqa: E402

    rank = comm.getRank()
    tmp_rank = tmp_path / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case_defaults(tmp_rank, comm.getSize(), rank)
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
        cfg.solver.linear.pc_type = "fieldsplit"
        cfg.solver.linear.fieldsplit = {
            "type": fs_type,
            "scheme": "bulk_iface",
        }
        if fs_type == "schur":
            cfg.solver.linear.fieldsplit["schur_fact_type"] = "lower"

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    A = _build_shell_like_matrix(PETSc, comm, mgr.dm)
    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)

    pc = ksp.getPC()
    assert pc.getType().lower() == "fieldsplit"

    Aop, Pop = ksp.getOperators()
    A_type = Aop.getType()
    P_type = Pop.getType()

    assert _is_shell_like(A_type) or (not _is_aij(A_type)), f"outer A must be shell-like, got {A_type}"
    assert _is_aij(P_type), f"outer P must be AIJ/MPIAIJ, got {P_type}"

    uses_amat = None
    try:
        uses_amat = bool(pc.getUseAmat())
    except Exception:
        uses_amat = diag_pc.get("uses_amat", None)
    assert uses_amat is False, "PCFieldSplit must NOT use Amat for split; require pc_use_amat=0"

    ksp.setUp()

    Aop2, Pop2 = ksp.getOperators()
    try:
        assert Aop2.handle == Aop.handle, "outer A operator changed unexpectedly after setup"
        assert Pop2.handle == Pop.handle, "outer P operator changed unexpectedly after setup"
    except Exception:
        assert Aop2 is Aop, "outer A operator changed unexpectedly after setup"
        assert Pop2 is Pop, "outer P operator changed unexpectedly after setup"

    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2, "fieldsplit must create at least 2 sub-KSPs (bulk/iface)"

    for i, sksp in enumerate(subksps):
        As, Ps = sksp.getOperators()
        As_t = As.getType()
        Ps_t = Ps.getType()
        assert _is_aij(Ps_t), f"subKSP[{i}] Psub must be AIJ, got {Ps_t}"
        as_t_l = str(As_t).lower()
        if fs_type == "schur":
            assert (
                _is_aij(As_t) or as_t_l == "schurcomplement"
            ), f"subKSP[{i}] Asub must be AIJ or SchurComplement in schur mode, got {As_t}"
        else:
            assert _is_aij(As_t), f"subKSP[{i}] Asub must be AIJ, got {As_t}"
        assert not _is_shell_like(as_t_l), f"subKSP[{i}] Asub must not be shell-like, got {As_t}"

    try:
        comm.barrier()
    except Exception:
        pass
