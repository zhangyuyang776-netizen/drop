from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_precond_aij import (  # noqa: E402
    build_precond_mat_aij_from_A,
    fill_precond_mat_aij_from_A,
)
import assembly.residual_global as res_mod  # noqa: E402


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Serial precond test requires COMM_WORLD size == 1.")
    return PETSc


class DummyLinearCfg:
    def __init__(self, assembly_mode: str = "bridge_dense") -> None:
        self.assembly_mode = assembly_mode


class DummySolverCfg:
    def __init__(self, assembly_mode: str = "bridge_dense") -> None:
        self.linear = DummyLinearCfg(assembly_mode)


class DummyCfg:
    def __init__(self, assembly_mode: str = "bridge_dense") -> None:
        self.solver = DummySolverCfg(assembly_mode)


class DummyCtx:
    def __init__(self, cfg, N: int) -> None:
        self.cfg = cfg
        self.grid_ref = None
        self.layout = None
        self.scale_u = np.ones(N, dtype=np.float64)
        self.state_old = None
        self.props_old = None
        self.dt = 1.0


def _make_dense_A(N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            A[i, j] = (i + 1) * 10.0 + (j + 1)
    return A


def _dense_from_csr(P, N: int) -> np.ndarray:
    ia, ja, a_vals = P.getValuesCSR()
    out = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        start = int(ia[i])
        end = int(ia[i + 1])
        cols = ja[start:end]
        vals = a_vals[start:end]
        out[i, cols] = vals
    return out


def test_precond_aij_serial_dense_bridge_matches_denseA(monkeypatch):
    _import_petsc_or_skip()
    N = 5
    A = _make_dense_A(N)

    def fake_build_transport_system_from_ctx(_ctx, _u_phys, *args, **kwargs):
        b = np.zeros(N, dtype=np.float64)
        return A, b

    monkeypatch.setattr(res_mod, "build_transport_system_from_ctx", fake_build_transport_system_from_ctx)

    cfg = DummyCfg(assembly_mode="bridge_dense")
    ctx = DummyCtx(cfg, N)
    u_phys = np.linspace(0.1, 1.0, N, dtype=np.float64)

    P, diag = build_precond_mat_aij_from_A(ctx, u_phys, drop_tol=0.0)
    assert P.getType().lower() in ("aij", "seqaij")

    P_dense = _dense_from_csr(P, N)
    np.testing.assert_allclose(P_dense, A, rtol=1.0e-14, atol=1.0e-14)
    assert diag.get("mode") == "dense_seq"


def test_precond_aij_serial_dense_bridge_fill_works(monkeypatch):
    _import_petsc_or_skip()
    N = 5
    A = _make_dense_A(N)

    def fake_build_transport_system_from_ctx(_ctx, _u_phys, *args, **kwargs):
        b = np.zeros(N, dtype=np.float64)
        return A, b

    monkeypatch.setattr(res_mod, "build_transport_system_from_ctx", fake_build_transport_system_from_ctx)

    cfg = DummyCfg(assembly_mode="bridge_dense")
    ctx = DummyCtx(cfg, N)
    u_phys = np.linspace(0.1, 1.0, N, dtype=np.float64)

    P, _diag = build_precond_mat_aij_from_A(ctx, u_phys, drop_tol=0.0)

    A2 = 2.0 * A

    def fake_build_transport_system_from_ctx2(_ctx, _u_phys, *args, **kwargs):
        b = np.zeros(N, dtype=np.float64)
        return A2, b

    monkeypatch.setattr(res_mod, "build_transport_system_from_ctx", fake_build_transport_system_from_ctx2)

    diag2 = fill_precond_mat_aij_from_A(P, ctx, u_phys, drop_tol=0.0)
    P_dense = _dense_from_csr(P, N)
    np.testing.assert_allclose(P_dense, A2, rtol=1.0e-14, atol=1.0e-14)
    assert diag2.get("mode") == "dense_seq_fill"
