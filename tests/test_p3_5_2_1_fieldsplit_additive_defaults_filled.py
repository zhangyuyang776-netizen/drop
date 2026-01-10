from __future__ import annotations

from types import SimpleNamespace

from solvers.mpi_linear_support import validate_mpi_linear_support


def _base_cfg():
    linear = SimpleNamespace(pc_type="fieldsplit", fieldsplit={"type": "additive", "scheme": "bulk_iface"})
    solver = SimpleNamespace(linear=linear)
    nonlinear = SimpleNamespace(enabled=True)
    return SimpleNamespace(nonlinear=nonlinear, solver=solver)


def test_fieldsplit_additive_defaults_filled():
    cfg = _base_cfg()
    validate_mpi_linear_support(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)

    sub = fs.get("subsolvers", {})
    assert isinstance(sub, dict)

    bulk = sub.get("bulk", {})
    iface = sub.get("iface", {})

    assert bulk.get("ksp_type") == "preonly"
    assert bulk.get("pc_type") == "asm"
    assert int(bulk.get("asm_overlap")) == 1
    assert bulk.get("asm_sub_pc_type") == "ilu"

    assert iface.get("ksp_type") == "preonly"
    assert iface.get("pc_type") == "asm"
    assert int(iface.get("asm_overlap")) == 1
    assert iface.get("asm_sub_pc_type") == "ilu"
