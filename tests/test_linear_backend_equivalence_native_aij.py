from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _petsc_mat_to_dense(PETSc, A_p) -> np.ndarray:
    n, m = A_p.getSize()
    if n != m:
        raise ValueError(f"Expected square matrix, got {(n, m)}")
    rows = list(range(n))
    cols = list(range(n))
    return np.asarray(A_p.getValues(rows, cols), dtype=np.float64)


def _petsc_vec_to_dense(b_p) -> np.ndarray:
    return np.asarray(b_p.getArray(), dtype=np.float64).copy()


def test_linear_backend_equivalence_native_aij(tmp_path: Path):
    from tests.test_build_transport_system_petsc_native_matches_numpy import (  # noqa: E402
        _build_case,
        _import_chemistry_or_skip,
        _import_petsc_or_skip,
    )

    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0 = _build_case(tmp_path)
    dt = float(cfg.time.dt)

    from assembly.build_system_SciPy import (  # noqa: E402
        build_transport_system as build_transport_system_numpy,
    )
    from assembly.build_system_petsc import (  # noqa: E402
        build_transport_system_petsc_native,
    )

    A_np, b_np = build_transport_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=None,
        return_diag=False,
    )

    A_p, b_p = build_transport_system_petsc_native(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=None,
        return_diag=False,
        comm=PETSc.COMM_SELF,
    )

    A_p_dense = _petsc_mat_to_dense(PETSc, A_p)
    b_p_dense = _petsc_vec_to_dense(b_p)

    np.testing.assert_allclose(A_np, A_p_dense, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(b_np, b_p_dense, rtol=1.0e-12, atol=1.0e-12)
