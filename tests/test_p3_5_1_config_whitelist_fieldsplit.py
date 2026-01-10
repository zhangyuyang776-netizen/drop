from __future__ import annotations

from types import SimpleNamespace

import pytest

from solvers.mpi_linear_support import validate_mpi_linear_support


def _base_cfg():
    linear = SimpleNamespace(pc_type="asm", fieldsplit=None)
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    return SimpleNamespace(nonlinear=nonlinear, solver=solver)


@pytest.mark.parametrize(
    "pc_type, fieldsplit, expect_schur_default",
    [
        ("asm", None, False),
        ("fieldsplit", {"type": "additive", "scheme": "bulk_iface"}, False),
        ("fieldsplit", {"type": "schur", "scheme": "bulk_iface", "schur_fact_type": "lower"}, False),
        ("fieldsplit", {"type": "schur", "scheme": "bulk_iface"}, True),
    ],
)
def test_config_whitelist_allows_valid_combos_and_fills_defaults(
    pc_type, fieldsplit, expect_schur_default
):
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = pc_type
    cfg.solver.linear.fieldsplit = fieldsplit

    validate_mpi_linear_support(cfg)

    if expect_schur_default:
        assert isinstance(cfg.solver.linear.fieldsplit, dict)
        assert cfg.solver.linear.fieldsplit.get("schur_fact_type") == "lower"


@pytest.mark.parametrize(
    "pc_type, fieldsplit, msg_re",
    [
        ("ilu", None, r"pc_type|allowed"),
        ("fieldsplit", None, r"fieldsplit|required|must"),
        ("fieldsplit", {"type": "foo", "scheme": "bulk_iface"}, r"fieldsplit\.type|allowed"),
        ("fieldsplit", {"type": "additive", "scheme": "foo"}, r"fieldsplit\.scheme|allowed"),
        ("fieldsplit", {"type": "schur", "scheme": "bulk_iface", "schur_fact_type": "bad"}, r"schur_fact_type|allowed"),
        ("asm", {"type": "additive", "scheme": "bulk_iface"}, r"pc_type.*asm|forbid|forbids"),
    ],
)
def test_config_whitelist_rejects_invalid_combos(pc_type, fieldsplit, msg_re):
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = pc_type
    cfg.solver.linear.fieldsplit = fieldsplit

    with pytest.raises(ValueError, match=msg_re):
        validate_mpi_linear_support(cfg)
