"""
P3.5-4 smoke tests: Schur fieldsplit linear solve with stable system construction.

This module implements the P3.5-4-1 construction strategy:
- A: shell-like matrix (mult delegates to P)
- P: AIJ preconditioner (lightly regularized)
- x_true: deterministic (all 1s)
- b = A*x_true

This ensures the solve is stable and convergence-insensitive, covering
the fieldsplit+shell risk path while maintaining numerical reliability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_helpers():
    from tests.test_p3_4_0_fieldsplit_uses_pmat_only_mpi import (  # noqa: E402
        _build_precond_matrix,
        _get_fieldsplit_subksps,
        _import_chemistry_or_skip,
        _import_mpi4py_or_skip,
        _import_petsc_or_skip,
        _start_watchdog_abort_after_seconds,
    )

    return (
        _build_precond_matrix,
        _get_fieldsplit_subksps,
        _import_chemistry_or_skip,
        _import_mpi4py_or_skip,
        _import_petsc_or_skip,
        _start_watchdog_abort_after_seconds,
    )


def _build_context(tmp_path: Path, comm_size: int, rank: int):
    from tests.test_petsc_snes_parallel_defaults_mpi import _build_case as _build_case_defaults  # noqa: E402

    tmp_rank = tmp_path / f"rank_{rank:03d}"
    tmp_rank.mkdir(parents=True, exist_ok=True)
    return _build_case_defaults(tmp_rank, comm_size, rank)


class _ShellMultP:
    """Shell matrix context that delegates mult to a given PETSc matrix P."""

    def __init__(self, P):
        self.P = P

    def mult(self, mat, x, y):
        """Matrix-vector multiplication: y = A*x, delegated to P*x."""
        self.P.mult(x, y)


def build_A_shell_from_P(PETSc, comm, P):
    """
    Build a shell-like matrix A whose MatMult delegates to P.mult().

    This ensures the linear system A*x=b is equivalent to P*x=b,
    while A remains shell-like (covering fieldsplit+shell risk paths).

    Args:
        PETSc: petsc4py.PETSc module
        comm: MPI communicator
        P: PETSc.Mat (AIJ/MPIAIJ preconditioner matrix)

    Returns:
        A: PETSc.Mat (shell-like, with mult delegated to P)
    """
    (M, N) = P.getSize()
    (m, n) = P.getLocalSize()

    try:
        A = PETSc.Mat().createPython([[m, n], [M, N]], comm=comm)
        A.setPythonContext(_ShellMultP(P))
        A.setUp()
        return A
    except Exception:
        try:
            A = PETSc.Mat().createShell([[m, n], [M, N]], comm=comm)
            A.setPythonContext(_ShellMultP(P))
            A.setUp()
            return A
        except Exception as exc:
            raise RuntimeError(
                "Unable to create shell matrix with Python context. "
                "Check petsc4py version and build."
            ) from exc


@pytest.mark.mpi
def test_p3_5_4_schur_solve_smoke_defaults_mpi(tmp_path: Path):
    """
    Smoke test: Schur fieldsplit can solve a well-conditioned system without hanging.

    System construction (P3.5-4-1 strategy):
    - A: shell matrix (mult delegates to P)
    - P: AIJ preconditioner (lightly regularized with shift)
    - x_true: deterministic (all 1s)
    - b = A*x_true
    - Solve A*x=b with Schur fieldsplit, verify x ≈ x_true
    """
    (
        _build_precond_matrix,
        _get_fieldsplit_subksps,
        _import_chemistry_or_skip,
        _import_mpi4py_or_skip,
        _import_petsc_or_skip,
        _start_watchdog_abort_after_seconds,
    ) = _import_helpers()

    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("need MPI size >= 2")

    _start_watchdog_abort_after_seconds(60.0)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc  # noqa: E402
    from solvers.mpi_linear_support import validate_mpi_linear_support  # noqa: E402

    cfg, layout, ctx, u0 = _build_context(tmp_path, comm.getSize(), comm.getRank())
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
    }

    validate_mpi_linear_support(cfg)

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    P_base = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    P_solve = P_base.copy()
    P_solve.shift(1.0)

    A = build_A_shell_from_P(PETSc, comm, P_solve)

    x_true = mgr.dm.createGlobalVec()
    x_true.set(1.0)

    b = mgr.dm.createGlobalVec()
    A.mult(x_true, b)

    tmp1 = mgr.dm.createGlobalVec()
    tmp2 = mgr.dm.createGlobalVec()
    A.mult(x_true, tmp1)
    P_solve.mult(x_true, tmp2)
    tmp1.axpy(-1.0, tmp2)
    consistency_err = tmp1.norm()
    assert consistency_err < 1e-12, f"A and P_solve mult inconsistent: {consistency_err}"

    A_type = str(A.getType()).lower()
    P_type = str(P_solve.getType()).lower()
    assert "python" in A_type or "shell" in A_type, f"A should be shell-like, got {A_type}"
    assert "aij" in P_type, f"P should be AIJ, got {P_type}"

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p354_")
    ksp.setOperators(A, P_solve)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P_solve)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "lower"

    x = mgr.dm.createGlobalVec()
    x.set(0.0)
    ksp.solve(b, x)

    reason = int(ksp.getConvergedReason())
    n_iter = int(ksp.getIterationNumber())

    x_norm = x.norm()
    assert x_norm > 0 and x_norm < 1e10, f"x_norm={x_norm} out of reasonable range"

    x_true_norm = x_true.norm()
    x.axpy(-1.0, x_true)
    err_norm = x.norm()
    rel_err = err_norm / (x_true_norm + 1e-30)

    assert reason > 0, f"KSP did not converge: reason={reason}, n_iter={n_iter}"
    assert rel_err < 1e-6, f"Solution error too large: rel_err={rel_err}, n_iter={n_iter}"


@pytest.mark.mpi
def test_p3_5_4_schur_solve_smoke_yaml_override_mpi(tmp_path: Path):
    """
    Smoke test: Schur fieldsplit with YAML overrides (schur_fact_type, bulk.ksp_type).

    Verifies that user overrides are respected and solve still converges.
    """
    (
        _build_precond_matrix,
        _get_fieldsplit_subksps,
        _import_chemistry_or_skip,
        _import_mpi4py_or_skip,
        _import_petsc_or_skip,
        _start_watchdog_abort_after_seconds,
    ) = _import_helpers()

    _import_mpi4py_or_skip()
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    comm = PETSc.COMM_WORLD
    if comm.getSize() < 2:
        pytest.skip("need MPI size >= 2")

    _start_watchdog_abort_after_seconds(60.0)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc  # noqa: E402
    from solvers.mpi_linear_support import validate_mpi_linear_support  # noqa: E402

    cfg, layout, ctx, u0 = _build_context(tmp_path, comm.getSize(), comm.getRank())
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "schur_fact_type": "upper",
        "subsolvers": {
            "bulk": {"ksp_type": "gmres"},
        },
    }

    validate_mpi_linear_support(cfg)

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    P_base = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    P_solve = P_base.copy()
    P_solve.shift(1.0)

    A = build_A_shell_from_P(PETSc, comm, P_solve)

    x_true = mgr.dm.createGlobalVec()
    x_true.set(1.0)

    b = mgr.dm.createGlobalVec()
    A.mult(x_true, b)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p354_override_")
    ksp.setOperators(A, P_solve)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P_solve)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "upper"

    injected = diag_pc.get("options_injected", {}) or {}
    assert injected.get("fieldsplit_bulk_ksp_type") == "gmres"

    x = mgr.dm.createGlobalVec()
    x.set(0.0)
    ksp.solve(b, x)

    reason = int(ksp.getConvergedReason())
    n_iter = int(ksp.getIterationNumber())

    x_norm = x.norm()
    assert x_norm > 0 and x_norm < 1e10, f"x_norm={x_norm} out of reasonable range"

    x_true_norm = x_true.norm()
    x.axpy(-1.0, x_true)
    err_norm = x.norm()
    rel_err = err_norm / (x_true_norm + 1e-30)

    assert reason > 0, f"KSP did not converge: reason={reason}, n_iter={n_iter}"
    assert rel_err < 1e-6, f"Solution error too large: rel_err={rel_err}, n_iter={n_iter}"
