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


def test_schur_override_kept_and_defaults_filled():
    fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "gmres"},
            "iface": {"asm_sub_pc_type": "ilu"},
        },
    }
    cfg = _make_cfg(fieldsplit)

    LinearSolverConfigTyped.from_cfg(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)
    assert fs.get("schur_fact_type") == "lower"

    sub = fs.get("subsolvers", {})
    bulk = sub.get("bulk", {})
    iface = sub.get("iface", {})
    assert bulk.get("ksp_type") == "gmres"
    assert iface.get("asm_sub_pc_type") == "ilu"
