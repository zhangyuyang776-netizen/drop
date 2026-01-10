from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel


def _make_ctx(pc_type: str = "asm", fieldsplit=None):
    linear = SimpleNamespace(pc_type=pc_type, fieldsplit=fieldsplit)
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    cfg = SimpleNamespace(nonlinear=nonlinear, solver=solver)
    return SimpleNamespace(cfg=cfg)


def test_mpi_entry_calls_validate_via_sentinel(monkeypatch):
    from solvers import petsc_snes_parallel as mod

    def _boom(_cfg):
        raise RuntimeError("SENTINEL_VALIDATE_CALLED")

    monkeypatch.setattr(mod, "validate_mpi_before_petsc", _boom)
    ctx = _make_ctx(pc_type="asm")

    with pytest.raises(RuntimeError, match="SENTINEL_VALIDATE_CALLED"):
        solve_nonlinear_petsc_parallel(ctx, np.zeros(1, dtype=np.float64))


def test_mpi_entry_rejects_invalid_combo_before_petsc():
    fieldsplit = {"type": "additive", "scheme": "by_layout"}
    ctx = _make_ctx(pc_type="fieldsplit", fieldsplit=fieldsplit)

    with pytest.raises(ValueError, match="fieldsplit\\.scheme"):
        solve_nonlinear_petsc_parallel(ctx, np.zeros(1, dtype=np.float64))
