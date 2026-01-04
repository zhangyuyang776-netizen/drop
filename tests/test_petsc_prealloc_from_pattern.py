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
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Preallocation test is serial-only (COMM_WORLD size must be 1).")
    return PETSc


def _import_chemistry_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("Cantera not available")
    try:
        import CoolProp  # noqa: F401
    except Exception:
        pytest.skip("CoolProp not available")


def _build_case(tmp_path: Path):
    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = 3
    cfg.geometry.mesh.enforce_interface_continuity = False
    cfg.physics.include_mpp = False
    cfg.physics.include_Ts = False
    cfg.physics.include_Rd = False

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg, grid, state0)

    return cfg, grid, layout, state0, props0


def test_petsc_prealloc_counts_match_pattern(tmp_path: Path):
    _import_chemistry_or_skip()
    cfg, grid, layout, _state0, _props0 = _build_case(tmp_path)

    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
    from assembly.petsc_prealloc import build_petsc_prealloc_from_pattern  # noqa: E402

    pattern = build_jacobian_pattern(cfg, grid, layout)
    N, d_nz, o_nz = build_petsc_prealloc_from_pattern(pattern)

    assert N == layout.n_dof()
    assert len(d_nz) == N
    assert len(o_nz) == N

    assert all(n >= 0 for n in d_nz)
    assert all(n >= 0 for n in o_nz)
    assert all(n == 0 for n in o_nz)

    nnz_from_dnz = sum(d_nz)
    nnz_from_indptr = int(pattern.indptr[-1])
    nnz_from_indices = int(pattern.indices.size)
    nnz_from_meta = int(pattern.meta["nnz_total"])

    assert nnz_from_dnz == nnz_from_indptr
    assert nnz_from_dnz == nnz_from_indices
    assert nnz_from_dnz == nnz_from_meta


def test_petsc_prealloc_can_create_aij_matrix(tmp_path: Path):
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, _state0, _props0 = _build_case(tmp_path)

    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
    from assembly.petsc_prealloc import build_petsc_prealloc_from_pattern  # noqa: E402

    pattern = build_jacobian_pattern(cfg, grid, layout)
    N, d_nz, o_nz = build_petsc_prealloc_from_pattern(pattern)

    A = PETSc.Mat().createAIJ(size=(N, N), comm=PETSc.COMM_SELF, nnz=(d_nz, o_nz))
    A.setUp()

    diag = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    diag.set(1.0)
    diag.assemblyBegin()
    diag.assemblyEnd()
    A.setDiagonal(diag)
    A.assemblyBegin()
    A.assemblyEnd()

    x = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    x.set(1.0)
    x.assemblyBegin()
    x.assemblyEnd()

    y = x.duplicate()
    A.mult(x, y)

    arr = y.getArray(readonly=True)
    assert np.allclose(arr, 1.0)
