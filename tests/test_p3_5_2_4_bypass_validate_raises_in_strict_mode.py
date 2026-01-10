from __future__ import annotations

from types import SimpleNamespace

import pytest

from solvers.mpi_linear_support import assert_mpi_additive_subsolvers_filled


def _make_cfg():
    linear = SimpleNamespace(
        pc_type="fieldsplit",
        fieldsplit={"type": "additive", "scheme": "bulk_iface"},
    )
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    return SimpleNamespace(nonlinear=nonlinear, solver=solver)


def test_bypass_validate_raises_in_strict_mode(monkeypatch):
    cfg = _make_cfg()
    monkeypatch.setenv("DROPLET_MPI_LINEAR_STRICT", "1")

    with pytest.raises(RuntimeError, match="validate_mpi_linear_support|subsolvers"):
        assert_mpi_additive_subsolvers_filled(cfg)
