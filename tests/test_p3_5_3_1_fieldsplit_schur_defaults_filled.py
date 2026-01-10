from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.linear_types import LinearSolverConfig as LinearSolverConfigTyped  # noqa: E402


def _make_cfg(fieldsplit):
    linear = SimpleNamespace(pc_type="fieldsplit", fieldsplit=fieldsplit)
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    return SimpleNamespace(nonlinear=nonlinear, solver=solver)


def test_schur_defaults_filled_and_backfilled():
    cfg = _make_cfg({"type": "schur", "scheme": "bulk_iface"})

    LinearSolverConfigTyped.from_cfg(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)
    assert fs.get("schur_fact_type") == "lower"
    sub = fs.get("subsolvers", {})
    assert "bulk" in sub
    assert "iface" in sub

    bulk = sub.get("bulk", {})
    iface = sub.get("iface", {})
    assert bulk.get("asm_sub_pc_type") == "ilu"
    assert iface.get("asm_sub_pc_type") == "lu"
    assert bulk.get("pc_type") == "asm"
    assert iface.get("pc_type") == "asm"
    assert int(bulk.get("asm_overlap", 0)) == 1
    assert int(iface.get("asm_overlap", 0)) == 1
