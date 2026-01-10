from __future__ import annotations

import pytest

from solvers.linear_types import LinearSolverConfig


def test_invalid_pc_type_fails():
    with pytest.raises(ValueError, match="linear\\.pc_type: invalid value"):
        LinearSolverConfig.from_dict({"pc_type": "jacobi"})


def test_invalid_fieldsplit_type_fails():
    with pytest.raises(ValueError, match="fieldsplit\\.type: invalid value"):
        LinearSolverConfig.from_dict({"pc_type": "fieldsplit", "fieldsplit": {"type": "add"}})


def test_invalid_scheme_fails():
    with pytest.raises(ValueError, match="fieldsplit\\.scheme: invalid value"):
        LinearSolverConfig.from_dict({"pc_type": "fieldsplit", "fieldsplit": {"scheme": "bulk"}})


def test_asm_forbids_fieldsplit_block():
    with pytest.raises(ValueError, match="forbids"):
        LinearSolverConfig.from_dict({"pc_type": "asm", "fieldsplit": {"type": "additive"}})


def test_valid_fieldsplit_additive_ok():
    cfg = LinearSolverConfig.from_dict(
        {"pc_type": "fieldsplit", "fieldsplit": {"type": "additive", "scheme": "bulk_iface"}}
    )
    assert cfg.pc_type.value == "fieldsplit"
    assert cfg.fieldsplit is not None
    assert cfg.fieldsplit.type.value == "additive"
    assert cfg.fieldsplit.scheme.value == "bulk_iface"


def test_valid_fieldsplit_schur_defaults_fact_type():
    cfg = LinearSolverConfig.from_dict(
        {"pc_type": "fieldsplit", "fieldsplit": {"type": "schur", "scheme": "bulk_iface"}}
    )
    assert cfg.fieldsplit is not None
    assert cfg.fieldsplit.schur_fact_type is not None
