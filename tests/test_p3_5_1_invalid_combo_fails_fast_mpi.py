from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel


def _make_ctx(pc_type: str, fieldsplit):
    linear = SimpleNamespace(pc_type=pc_type, fieldsplit=fieldsplit)
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    cfg = SimpleNamespace(nonlinear=nonlinear, solver=solver)
    return SimpleNamespace(cfg=cfg)


def _start_watchdog(seconds: float = 30.0) -> None:
    def _boom():
        os._exit(2)

    t = threading.Timer(seconds, _boom)
    t.daemon = True
    t.start()


@pytest.mark.mpi
def test_invalid_combo_fails_fast_under_mpi():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("need MPI size >= 2")

    _start_watchdog(30.0)

    ctx = _make_ctx(
        pc_type="fieldsplit",
        fieldsplit={"type": "additive", "scheme": "by_layout"},
    )

    with pytest.raises(ValueError, match=r"fieldsplit\.scheme|allowed"):
        solve_nonlinear_petsc_parallel(ctx, np.zeros(1, dtype=np.float64))
