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


@pytest.mark.mpi
def test_p3_5_3_3_schur_defaults_injected_mpi(tmp_path: Path):
    """Test that Schur fieldsplit uses typed subsolvers → options injection → diag path."""
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
    ksp.setOptionsPrefix("p533_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "lower"
    assert diag_pc.get("uses_amat") is False

    injected = diag_pc.get("options_injected", {}) or {}
    assert injected.get("fieldsplit_bulk_ksp_type") == "preonly"
    assert injected.get("fieldsplit_bulk_pc_type") == "asm"
    assert injected.get("fieldsplit_bulk_pc_asm_overlap") == "1"
    assert injected.get("fieldsplit_bulk_sub_ksp_type") == "preonly"
    assert injected.get("fieldsplit_bulk_sub_pc_type") == "ilu"
    assert injected.get("fieldsplit_iface_ksp_type") == "preonly"
    assert injected.get("fieldsplit_iface_pc_type") == "asm"
    assert injected.get("fieldsplit_iface_pc_asm_overlap") == "1"
    assert injected.get("fieldsplit_iface_sub_ksp_type") == "preonly"
    assert injected.get("fieldsplit_iface_sub_pc_type") == "lu"

    pc = ksp.getPC()
    assert pc.getType().lower() == "fieldsplit"
    subksps = _get_fieldsplit_subksps(pc)
    assert len(subksps) >= 2

    for idx, sksp in enumerate(subksps):
        try:
            As, Ps = sksp.getOperators()
        except Exception:
            continue
        As_t = str(As.getType()).lower() if As is not None else ""
        Ps_t = str(Ps.getType()).lower() if Ps is not None else ""

        assert "aij" in Ps_t, f"subKSP[{idx}] Psub={Ps_t} should be AIJ"

        is_aij_or_schur = ("aij" in As_t) or (As_t == "schurcomplement")
        assert is_aij_or_schur, f"subKSP[{idx}] Asub={As_t} should be AIJ or schurcomplement"

        is_shell = ("shell" in As_t) or ("mffd" in As_t) or ("python" in As_t)
        assert not is_shell, f"subKSP[{idx}] Asub={As_t} should not be shell-like"


@pytest.mark.mpi
def test_p3_5_3_3_schur_yaml_override_injected_mpi(tmp_path: Path):
    """Test that Schur fieldsplit respects YAML subsolvers overrides."""
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
    ksp.setOptionsPrefix("p533_override_")
    ksp.setOperators(A, P)
    ksp.setType("gmres")

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P)
    ksp.setFromOptions()
    ksp.setUp()
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    assert diag_pc.get("fieldsplit_type") == "schur"
    assert diag_pc.get("schur_fact_type") == "upper"

    injected = diag_pc.get("options_injected", {}) or {}
    assert injected.get("fieldsplit_bulk_ksp_type") == "gmres"
    assert injected.get("fieldsplit_iface_ksp_type") == "preonly"

    pc = ksp.getPC()
    assert pc.getType().lower() == "fieldsplit"
