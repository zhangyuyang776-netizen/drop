from __future__ import annotations

import pytest

from solvers.mpi_linear_support import validate_mpi_linear_support
from tests._helpers_step15 import make_cfg_base


def _base_cfg():
    cfg = make_cfg_base(
        Nl=1,
        Ng=3,
        solve_Yg=True,
        include_mpp=False,
        include_Ts=False,
        include_Rd=False,
    )
    cfg.nonlinear.backend = "petsc_mpi"
    return cfg


def test_asm_no_fieldsplit_ok():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "asm"
    cfg.solver.linear.fieldsplit = None
    validate_mpi_linear_support(cfg)


def test_additive_allows_subsolvers_override_and_keeps_user_value():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "additive",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "fgmres"},
        },
    }

    validate_mpi_linear_support(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)
    assert fs["subsolvers"]["bulk"]["ksp_type"] == "fgmres"
    assert fs["subsolvers"]["bulk"]["pc_type"] == "asm"
    assert fs["subsolvers"]["iface"]["pc_type"] == "asm"


def test_additive_rejects_bulk_pc_type_ilu():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "additive",
        "scheme": "bulk_iface",
        "subsolvers": {"bulk": {"pc_type": "ilu"}},
    }
    with pytest.raises(ValueError, match=r"subsolvers\\.bulk\\.pc_type|invalid value"):
        validate_mpi_linear_support(cfg)


def test_additive_rejects_bulk_ksp_type_cg():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "additive",
        "scheme": "bulk_iface",
        "subsolvers": {"bulk": {"ksp_type": "cg"}},
    }
    with pytest.raises(ValueError, match=r"subsolvers\\.bulk\\.ksp_type|invalid value"):
        validate_mpi_linear_support(cfg)


def test_additive_rejects_negative_asm_overlap():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "additive",
        "scheme": "bulk_iface",
        "subsolvers": {"bulk": {"asm_overlap": -1}},
    }
    with pytest.raises(ValueError, match=r"asm_overlap.*>= 0"):
        validate_mpi_linear_support(cfg)


def test_additive_rejects_unknown_subsolver_key_typo():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "additive",
        "scheme": "bulk_iface",
        "subsolvers": {"bulk": {"kspType": "gmres"}},
    }
    with pytest.raises(ValueError, match=r"unknown key|kspType"):
        validate_mpi_linear_support(cfg)
