from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from solvers.mpi_linear_support import validate_mpi_linear_support
from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel


class _StopAfterValidate(RuntimeError):
    pass


def _make_ctx(*, pc_type: str, fieldsplit):
    linear = SimpleNamespace(pc_type=pc_type, fieldsplit=fieldsplit)
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    cfg = SimpleNamespace(nonlinear=nonlinear, solver=solver)
    return SimpleNamespace(cfg=cfg)


def test_mpi_entry_fills_additive_subsolvers_defaults(monkeypatch):
    from solvers import petsc_snes_parallel as mod

    fs_cfg = {"type": "additive", "scheme": "bulk_iface"}
    ctx = _make_ctx(pc_type="fieldsplit", fieldsplit=fs_cfg)

    def _validate(cfg):
        validate_mpi_linear_support(cfg)
        raise _StopAfterValidate()

    monkeypatch.setattr(mod, "validate_mpi_before_petsc", _validate)

    with pytest.raises(_StopAfterValidate):
        solve_nonlinear_petsc_parallel(ctx, np.zeros(1, dtype=np.float64))

    fs_raw = ctx.cfg.solver.linear.fieldsplit
    assert isinstance(fs_raw, dict)
    assert fs_raw.get("_subsolvers_filled") is True
    sub = fs_raw.get("subsolvers")
    assert isinstance(sub, dict)
    assert "bulk" in sub
    assert "iface" in sub
