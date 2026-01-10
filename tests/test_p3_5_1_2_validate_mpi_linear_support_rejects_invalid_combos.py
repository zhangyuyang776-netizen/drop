from __future__ import annotations

import copy

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


def test_invalid_pc_type_fails():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "jacobi"
    with pytest.raises(ValueError, match="linear\\.pc_type: invalid value"):
        validate_mpi_linear_support(cfg)


def test_fieldsplit_missing_config_fails():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = None
    with pytest.raises(ValueError, match="requires cfg\\.solver\\.linear\\.fieldsplit"):
        validate_mpi_linear_support(cfg)


def test_fieldsplit_scheme_not_allowed_for_mpi():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {"type": "additive", "scheme": "by_layout"}
    with pytest.raises(ValueError, match="fieldsplit\\.scheme"):
        validate_mpi_linear_support(cfg)


def test_schur_fact_type_invalid_fails():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "schur_fact_type": "bad",
    }
    with pytest.raises(ValueError, match="schur_fact_type"):
        validate_mpi_linear_support(cfg)


def test_schur_fact_type_default_filled():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {"type": "schur", "scheme": "bulk_iface"}

    validate_mpi_linear_support(cfg)

    fs_cfg = cfg.solver.linear.fieldsplit
    assert isinstance(fs_cfg, dict)
    assert fs_cfg.get("schur_fact_type") == "lower"


def test_asm_forbids_fieldsplit_block():
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "asm"
    cfg.solver.linear.fieldsplit = {"type": "additive", "scheme": "bulk_iface"}
    with pytest.raises(ValueError, match="pc_type=asm"):
        validate_mpi_linear_support(cfg)
