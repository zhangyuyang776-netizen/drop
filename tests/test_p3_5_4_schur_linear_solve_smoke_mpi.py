"""
P3.5-4 smoke tests: Schur fieldsplit setUp and solve validation.

This module validates that Schur fieldsplit can complete the full production
call chain from configuration through solve, covering the fieldsplit+shell
risk path.

Test phases:
- P3.5-4-2: setUp path structure validation (no solve)
- P3.5-4-3: A(shell) construction with MatMult delegation to P
- P3.5-4-4: Production path configuration and setUp validation
- P3.5-4-5: Complete solve with numerical convergence validation

Key validations:
- uses_amat = False (P3.4 fail-fast)
- fieldsplit_type = schur
- Sub-block operator types correct (P_sub: AIJ, A_sub: AIJ or schurcomplement)
- Solve converges with stable system construction (A.mult := P.mult)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_helpers():
    from tests.test_p3_4_0_fieldsplit_uses_pmat_only_mpi import (  # noqa: E402
        _build_precond_matrix,
        _build_shell_like_matrix,
        _get_fieldsplit_subksps,
        _import_chemistry_or_skip,
        _import_mpi4py_or_skip,
        _import_petsc_or_skip,
        _start_watchdog_abort_after_seconds,
    )

    return (
        _build_precond_matrix,
        _build_shell_like_matrix,
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


def _fill_deterministic(vec):
    """
    Fill vector with deterministic values based on global indices.

    Uses pattern: vec[i] = 1.0 + 1e-3 * i
    This ensures non-trivial, reproducible RHS in parallel.
    """
    r0, r1 = vec.getOwnershipRange()
    idx = list(range(r0, r1))
    vals = [1.0 + 1e-3 * i for i in idx]
    vec.setValues(idx, vals)
    vec.assemblyBegin()
    vec.assemblyEnd()


def build_A_shell_from_P(PETSc, comm, P):
    """
    Build a shell-like (python) matrix A whose MatMult delegates to P.mult.
    A and P must have conforming sizes and parallel layout.

    This construction ensures the linear system A*x=b is equivalent to P*x=b,
    providing a stable, well-conditioned system for Schur fieldsplit smoke tests.

    Args:
        PETSc: petsc4py.PETSc module
        comm: MPI communicator
        P: PETSc.Mat (AIJ/MPIAIJ preconditioner matrix)

    Returns:
        A: PETSc.Mat (python mat with A.mult(x,y) == P.mult(x,y))
    """

    class _AMultFromP:
        """Python context for shell matrix that delegates mult to P."""

        def __init__(self, Pmat):
            self.P = Pmat

        def mult(self, A, x, y):
            """Delegate: y = P * x"""
            self.P.mult(x, y)

    m_loc, n_loc = P.getLocalSize()
    m_glb, n_glb = P.getSize()

    A = PETSc.Mat().createPython([[m_loc, m_glb], [n_loc, n_glb]], comm=comm)
    A.setPythonContext(_AMultFromP(P))
    A.setUp()
    return A


@pytest.mark.mpi
def test_p3_5_4_2_schur_setup_defaults_mpi(tmp_path: Path):
    """
    P3.5-4-2: Schur fieldsplit setUp path (structure only, no solve).

    Validates that Schur fieldsplit can complete setUp without hanging,
    and that structure is correct (uses_amat=False, sub-block types).
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    _start_watchdog_abort_after_seconds(30.0)

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

    A = _build_shell_like_matrix(PETSc, comm, mgr.dm)
    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p342_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # Structure assertions (P3.5-4-2 goal)
    assert diag_pc.get("uses_amat") is False, "Schur must force uses_amat=False"
    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "lower"

    # Sub-block operator type check
    pc = ksp.getPC()
    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2, f"Expected at least 2 sub-KSPs, got {len(subksps)}"

    for idx, sksp in enumerate(subksps):
        try:
            As, Ps = sksp.getOperators()
        except Exception:
            continue
        As_t = str(As.getType()).lower() if As is not None else ""
        Ps_t = str(Ps.getType()).lower() if Ps is not None else ""

        # P_sub must be AIJ
        assert "aij" in Ps_t, f"subKSP[{idx}] Psub={Ps_t} should be AIJ"

        # A_sub: allow AIJ or schurcomplement, but not matshell
        is_aij_or_schur = ("aij" in As_t) or (As_t == "schurcomplement")
        is_shell = ("shell" in As_t) or ("mffd" in As_t) or ("python" in As_t)
        assert is_aij_or_schur, f"subKSP[{idx}] Asub={As_t} should be AIJ or schurcomplement"
        assert not is_shell, f"subKSP[{idx}] Asub={As_t} should not be shell-like"


@pytest.mark.mpi
def test_p3_5_4_2_schur_setup_yaml_override_mpi(tmp_path: Path):
    """
    P3.5-4-2: Schur fieldsplit setUp with YAML overrides (structure only, no solve).

    Validates that user overrides (schur_fact_type='upper', bulk.ksp_type='gmres')
    are respected and setUp completes without hanging.
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    _start_watchdog_abort_after_seconds(30.0)

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

    A = _build_shell_like_matrix(PETSc, comm, mgr.dm)
    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p342_override_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # Structure assertions: verify YAML overrides were applied
    assert diag_pc.get("uses_amat") is False, "Schur must force uses_amat=False"
    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "upper"

    injected = diag_pc.get("options_injected", {}) or {}
    assert injected.get("fieldsplit_bulk_ksp_type") == "gmres"

    # Sub-block operator type check
    pc = ksp.getPC()
    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2, f"Expected at least 2 sub-KSPs, got {len(subksps)}"

    for idx, sksp in enumerate(subksps):
        try:
            As, Ps = sksp.getOperators()
        except Exception:
            continue
        As_t = str(As.getType()).lower() if As is not None else ""
        Ps_t = str(Ps.getType()).lower() if Ps is not None else ""

        # P_sub must be AIJ
        assert "aij" in Ps_t, f"subKSP[{idx}] Psub={Ps_t} should be AIJ"

        # A_sub: allow AIJ or schurcomplement, but not matshell
        is_aij_or_schur = ("aij" in As_t) or (As_t == "schurcomplement")
        is_shell = ("shell" in As_t) or ("mffd" in As_t) or ("python" in As_t)
        assert is_aij_or_schur, f"subKSP[{idx}] Asub={As_t} should be AIJ or schurcomplement"
        assert not is_shell, f"subKSP[{idx}] Asub={As_t} should not be shell-like"


@pytest.mark.mpi
def test_p3_5_4_3_build_A_shell_from_P_and_types_mpi(tmp_path: Path):
    """
    P3.5-4-3: Build A(shell) from P(AIJ) with MatMult delegation.

    Validates the stable system construction where A.mult(x,y) := P.mult(x,y),
    ensuring type conformance and size consistency without solve.

    Key validations:
    - P type: AIJ/MPIAIJ
    - A type: python/shell (shell-like)
    - A.mult(x,y) size conformance (no "Nonconforming object sizes")
    - Schur setUp path still stable
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    _start_watchdog_abort_after_seconds(30.0)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc  # noqa: E402
    from solvers.mpi_linear_support import validate_mpi_linear_support  # noqa: E402

    cfg, layout, ctx, u0 = _build_context(tmp_path, comm.getSize(), comm.getRank())
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {"type": "schur", "scheme": "bulk_iface"}

    validate_mpi_linear_support(cfg)

    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)
    A = build_A_shell_from_P(PETSc, comm, P)

    # --- P3.5-4-3 acceptance assertions (types) ---
    P_type = str(P.getType()).lower()
    assert "aij" in P_type, f"P should be AIJ/MPIAIJ, got {P.getType()}"

    A_type = str(A.getType()).lower()
    assert ("python" in A_type) or ("shell" in A_type), f"A should be python/shell, got {A.getType()}"

    # --- size conformance smoke (prevents the old MatMult nonconforming crash) ---
    # Use P.createVecRight()/Left() to ensure size consistency with matrix layout
    xr = P.createVecRight()
    yl = P.createVecLeft()
    xr.set(1.0)
    A.mult(xr, yl)  # must not raise "Nonconforming object sizes"

    # Verify sizes match
    assert A.getSize() == P.getSize(), f"A.getSize()={A.getSize()} != P.getSize()={P.getSize()}"
    assert A.getLocalSize() == P.getLocalSize(), f"A.getLocalSize()={A.getLocalSize()} != P.getLocalSize()={P.getLocalSize()}"

    # --- still run schur setUp path (structure) ---
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p3543_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("uses_amat") is False


@pytest.mark.mpi
def test_p3_5_4_4_schur_apply_structured_pc_and_setup_defaults_mpi(tmp_path: Path):
    """
    P3.5-4-4: Schur fieldsplit production path with defaults (structure only, no solve).

    Validates the complete production call chain:
    1. apply_structured_pc() injects options (Schur)
    2. ksp.setFromOptions(); ksp.setUp() completes without fail-fast or hang
    3. apply_fieldsplit_subksp_defaults() executes on Schur path
    4. Sub-block operator types satisfy P3.4 constraints

    Uses build_A_shell_from_P() for stable system construction (A.mult := P.mult).
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    _start_watchdog_abort_after_seconds(30.0)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc  # noqa: E402
    from solvers.mpi_linear_support import validate_mpi_linear_support  # noqa: E402

    # 1. Configure (minimal Schur structure)
    cfg, layout, ctx, u0 = _build_context(tmp_path, comm.getSize(), comm.getRank())
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {"type": "schur", "scheme": "bulk_iface"}

    # 2. Validate (ensures defaults/backfill applied)
    validate_mpi_linear_support(cfg)

    # 3. Build dm/layout_dist
    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    # 4. Construct P (AIJ/MPIAIJ)
    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    # 5. Construct A (shell) using proven helper
    A = build_A_shell_from_P(PETSc, comm, P)

    # 6. Configure KSP/PC (production path)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p544_defaults_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # 7. Assertions (structure layer, no solve)
    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("uses_amat") is False, "Schur must force uses_amat=False"
    assert diag_pc.get("schur_fact_type") == "lower", "Default schur_fact_type should be lower"

    # Verify options injection occurred
    injected = diag_pc.get("options_injected", {}) or {}
    assert "fieldsplit_bulk_ksp_type" in injected, "bulk options should be injected"
    assert "fieldsplit_iface_sub_pc_type" in injected, "iface options should be injected"
    assert injected["fieldsplit_bulk_ksp_type"] == "preonly", "Default bulk ksp_type"
    assert injected["fieldsplit_iface_sub_pc_type"] == "lu", "Schur default iface sub_pc_type"

    # Sub-block operator type check (P3.4 constraints)
    pc = ksp.getPC()
    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2, f"Expected at least 2 sub-KSPs, got {len(subksps)}"

    for idx, sksp in enumerate(subksps):
        try:
            As, Ps = sksp.getOperators()
        except Exception:
            continue
        As_t = str(As.getType()).lower() if As is not None else ""
        Ps_t = str(Ps.getType()).lower() if Ps is not None else ""

        # P_sub must be AIJ
        assert "aij" in Ps_t, f"subKSP[{idx}] Psub={Ps_t} should be AIJ"

        # A_sub: allow AIJ or schurcomplement, but not matshell
        is_aij_or_schur = ("aij" in As_t) or (As_t == "schurcomplement")
        is_shell = ("shell" in As_t) or ("mffd" in As_t) or ("python" in As_t)
        assert is_aij_or_schur, f"subKSP[{idx}] Asub={As_t} should be AIJ or schurcomplement"
        assert not is_shell, f"subKSP[{idx}] Asub={As_t} should not be shell-like"


@pytest.mark.mpi
def test_p3_5_4_4_schur_apply_structured_pc_and_setup_yaml_override_mpi(tmp_path: Path):
    """
    P3.5-4-4: Schur fieldsplit production path with YAML overrides (structure only, no solve).

    Validates that user overrides (schur_fact_type='upper', bulk.ksp_type='gmres')
    are correctly applied through the production call chain.

    Uses build_A_shell_from_P() for stable system construction (A.mult := P.mult).
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    _start_watchdog_abort_after_seconds(30.0)

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc  # noqa: E402
    from solvers.mpi_linear_support import validate_mpi_linear_support  # noqa: E402

    # 1. Configure (Schur structure with YAML overrides)
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

    # 2. Validate (ensures defaults/backfill + override validation)
    validate_mpi_linear_support(cfg)

    # 3. Build dm/layout_dist
    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    # 4. Construct P (AIJ/MPIAIJ)
    P = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)

    # 5. Construct A (shell) using proven helper
    A = build_A_shell_from_P(PETSc, comm, P)

    # 6. Configure KSP/PC (production path)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p544_override_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # 7. Assertions (structure layer, verify overrides applied)
    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("uses_amat") is False, "Schur must force uses_amat=False"
    assert diag_pc.get("schur_fact_type") == "upper", "YAML override schur_fact_type should be upper"

    # Verify options injection with YAML overrides
    injected = diag_pc.get("options_injected", {}) or {}
    assert "fieldsplit_bulk_ksp_type" in injected, "bulk options should be injected"
    assert injected["fieldsplit_bulk_ksp_type"] == "gmres", "YAML override bulk.ksp_type=gmres"
    assert injected.get("fieldsplit_iface_sub_pc_type") == "lu", "Schur default iface sub_pc_type=lu"

    # Sub-block operator type check (P3.4 constraints)
    pc = ksp.getPC()
    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2, f"Expected at least 2 sub-KSPs, got {len(subksps)}"

    for idx, sksp in enumerate(subksps):
        try:
            As, Ps = sksp.getOperators()
        except Exception:
            continue
        As_t = str(As.getType()).lower() if As is not None else ""
        Ps_t = str(Ps.getType()).lower() if Ps is not None else ""

        # P_sub must be AIJ
        assert "aij" in Ps_t, f"subKSP[{idx}] Psub={Ps_t} should be AIJ"

        # A_sub: allow AIJ or schurcomplement, but not matshell
        is_aij_or_schur = ("aij" in As_t) or (As_t == "schurcomplement")
        is_shell = ("shell" in As_t) or ("mffd" in As_t) or ("python" in As_t)
        assert is_aij_or_schur, f"subKSP[{idx}] Asub={As_t} should be AIJ or schurcomplement"
        assert not is_shell, f"subKSP[{idx}] Asub={As_t} should not be shell-like"


@pytest.mark.mpi
def test_p3_5_4_5_schur_solve_smoke_defaults_mpi(tmp_path: Path):
    """
    P3.5-4-5: Schur fieldsplit complete solve with defaults.

    Validates the full production call chain including solve:
    - System construction: P.shift(1.0), A.mult := P.mult (stable, well-conditioned)
    - Deterministic x_true construction (1.0 + 1e-3*i)
    - Production path: validate → apply_structured_pc → setUp →
      apply_fieldsplit_subksp_defaults → solve
    - Numerical validation: convergence, finite norm, relative error < 1e-6

    Uses P.createVecRight/Left() to avoid size mismatch (critical for MPI).
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    # 1. Configure (minimal Schur structure)
    cfg, layout, ctx, u0 = _build_context(tmp_path, comm.getSize(), comm.getRank())
    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {"type": "schur", "scheme": "bulk_iface"}

    # 2. Validate (ensures defaults/backfill applied)
    validate_mpi_linear_support(cfg)

    # 3. Build dm/layout_dist
    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    # 4. Construct stable system: P.shift(1.0), A.mult := P.mult
    P_base = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)
    P = P_base.copy()
    P.shift(1.0)  # Light regularization for guaranteed solvability
    A = build_A_shell_from_P(PETSc, comm, P)

    # 5. Construct deterministic x_true and b = A*x_true
    # CRITICAL: Use P.createVecRight/Left() to avoid size mismatch
    x_true = P.createVecRight()
    _fill_deterministic(x_true)

    b = P.createVecLeft()
    A.mult(x_true, b)

    # 6. Configure KSP/PC (production path)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p545_defaults_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # Set tight convergence tolerance for accurate solution
    ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=200)

    # 7. Solve
    x = P.createVecRight()
    x.set(0.0)
    ksp.solve(b, x)

    # 8. Assertions (numerical validation)
    reason = ksp.getConvergedReason()
    n_iter = ksp.getIterationNumber()

    # Must converge (not diverge to NaN/Inf)
    assert reason > 0, f"KSP did not converge: reason={reason}, n_iter={n_iter}"

    # Solution norm must be finite
    x_norm = x.norm()
    assert math.isfinite(x_norm), f"Solution norm is not finite: {x_norm}"
    assert x_norm > 0, f"Solution norm is zero or negative: {x_norm}"

    # Relative error must be small
    err = x.copy()
    err.axpy(-1.0, x_true)
    err_norm = err.norm()
    x_true_norm = x_true.norm()
    rel_err = err_norm / max(x_true_norm, 1e-30)

    assert rel_err < 1e-6, f"Solution error too large: rel_err={rel_err:.2e}, n_iter={n_iter}"


@pytest.mark.mpi
def test_p3_5_4_5_schur_solve_smoke_yaml_override_mpi(tmp_path: Path):
    """
    P3.5-4-5: Schur fieldsplit complete solve with YAML overrides.

    Same as defaults test, but with user overrides:
    - schur_fact_type='upper'
    - bulk.ksp_type='gmres'

    Validates that YAML overrides propagate correctly through the full
    production call chain including solve.
    """
    (
        _build_precond_matrix,
        _build_shell_like_matrix,
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

    # 1. Configure (Schur structure with YAML overrides)
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

    # 2. Validate (ensures defaults/backfill + override validation)
    validate_mpi_linear_support(cfg)

    # 3. Build dm/layout_dist
    mgr = build_dm(cfg, layout, comm=comm)
    ld = LayoutDistributed.build(comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    fd_eps = float(getattr(getattr(cfg, "petsc", None), "fd_eps", 1.0e-8))
    drop_tol = float(getattr(getattr(cfg, "petsc", None), "precond_drop_tol", 0.0))

    # 4. Construct stable system: P.shift(1.0), A.mult := P.mult
    P_base = _build_precond_matrix(PETSc, comm, mgr.dm, ctx, u0, fd_eps=fd_eps, drop_tol=drop_tol)
    P = P_base.copy()
    P.shift(1.0)  # Light regularization for guaranteed solvability
    A = build_A_shell_from_P(PETSc, comm, P)

    # 5. Construct deterministic x_true and b = A*x_true
    # CRITICAL: Use P.createVecRight/Left() to avoid size mismatch
    x_true = P.createVecRight()
    _fill_deterministic(x_true)

    b = P.createVecLeft()
    A.mult(x_true, b)

    # 6. Configure KSP/PC (production path)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix("p545_override_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    # Set tight convergence tolerance for accurate solution
    ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=200)

    # 7. Solve
    x = P.createVecRight()
    x.set(0.0)
    ksp.solve(b, x)

    # 8. Assertions (numerical validation)
    reason = ksp.getConvergedReason()
    n_iter = ksp.getIterationNumber()

    # Must converge (not diverge to NaN/Inf)
    assert reason > 0, f"KSP did not converge: reason={reason}, n_iter={n_iter}"

    # Solution norm must be finite
    x_norm = x.norm()
    assert math.isfinite(x_norm), f"Solution norm is not finite: {x_norm}"
    assert x_norm > 0, f"Solution norm is zero or negative: {x_norm}"

    # Relative error must be small
    err = x.copy()
    err.axpy(-1.0, x_true)
    err_norm = err.norm()
    x_true_norm = x_true.norm()
    rel_err = err_norm / max(x_true_norm, 1e-30)

    assert rel_err < 1e-6, f"Solution error too large: rel_err={rel_err:.2e}, n_iter={n_iter}"

    # Verify YAML overrides were applied
    assert diag_pc.get("schur_fact_type") == "upper", "YAML override schur_fact_type should be upper"
    injected = diag_pc.get("options_injected", {}) or {}
    assert injected.get("fieldsplit_bulk_ksp_type") == "gmres", "YAML override bulk.ksp_type=gmres"
