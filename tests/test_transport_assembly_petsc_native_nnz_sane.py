from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.mark.slow
def test_transport_assembly_petsc_native_nnz_sane(tmp_path: Path):
    from tests.test_transport_assembly_petsc_native_vs_numpy import (  # noqa: E402
        _build_case,
        _import_chemistry_or_skip,
        _import_petsc_or_skip,
    )

    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    cfg, grid, layout, state0, props0, eq_result = _build_case(
        tmp_path,
        Ng=200,
        Nl=1,
        solve_Yg=True,
        solve_Yl=False,
        include_Ts=False,
        include_mpp=False,
        include_Rd=False,
        solve_Tl=True,
    )
    dt = float(cfg.time.dt)

    from assembly.build_system_petsc import (  # noqa: E402
        build_transport_system_petsc_bridge as build_bridge,
        build_transport_system_petsc_native as build_native,
    )

    A_native, _b_native = build_native(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=dt,
        state_guess=state0,
        eq_result=eq_result,
        return_diag=False,
        comm=PETSc.COMM_SELF,
    )

    info = A_native.getInfo()
    N = layout.n_dof()
    nz_used = float(info.get("nz_used", 0.0))
    avg_nz = nz_used / max(N, 1)

    assert avg_nz > 1.0, f"avg nnz per row too small: {avg_nz}"
    assert avg_nz < 500.0, f"avg nnz per row too large: {avg_nz}"

    t0 = time.perf_counter()
    for _ in range(3):
        build_native(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state0,
            props=props0,
            dt=dt,
            state_guess=state0,
            eq_result=eq_result,
            return_diag=False,
            comm=PETSc.COMM_SELF,
        )
    t_native = (time.perf_counter() - t0) / 3.0

    t0 = time.perf_counter()
    for _ in range(3):
        build_bridge(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state0,
            props=props0,
            dt=dt,
            state_guess=state0,
            eq_result=eq_result,
            return_diag=False,
            comm=PETSc.COMM_SELF,
        )
    t_bridge = (time.perf_counter() - t0) / 3.0

    assert t_native <= 2.0 * t_bridge + 0.05, (
        f"native too slow: native={t_native:.4g}s, bridge={t_bridge:.4g}s"
    )
