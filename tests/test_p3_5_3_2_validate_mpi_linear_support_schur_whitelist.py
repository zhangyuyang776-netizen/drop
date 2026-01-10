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


def test_schur_default_subsolvers_filled():
    """Test that schur with default subsolvers are filled correctly."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
    }

    validate_mpi_linear_support(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)
    assert fs["schur_fact_type"] == "lower"
    assert "subsolvers" in fs
    assert "bulk" in fs["subsolvers"]
    assert "iface" in fs["subsolvers"]
    assert fs["subsolvers"]["iface"]["ksp_type"] == "preonly"
    assert fs["subsolvers"]["iface"]["asm_sub_pc_type"] == "lu"


def test_schur_bulk_ksp_type_override_preserved():
    """Test that explicit bulk.ksp_type override is preserved."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "gmres"},
        },
    }

    validate_mpi_linear_support(cfg)

    fs = cfg.solver.linear.fieldsplit
    assert isinstance(fs, dict)
    assert fs["subsolvers"]["bulk"]["ksp_type"] == "gmres"
    assert fs["subsolvers"]["bulk"]["pc_type"] == "asm"
    assert fs["subsolvers"]["iface"]["ksp_type"] == "preonly"


def test_schur_rejects_invalid_schur_fact_type():
    """Test that invalid schur_fact_type is rejected."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "schur_fact_type": "bad",
    }

    with pytest.raises(ValueError, match=r"schur_fact_type|allowed"):
        validate_mpi_linear_support(cfg)


def test_schur_rejects_invalid_scheme():
    """Test that invalid scheme is rejected."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "foo",
    }

    with pytest.raises(ValueError, match=r"scheme|allowed"):
        validate_mpi_linear_support(cfg)


def test_schur_rejects_iface_ksp_type_gmres():
    """Test that iface.ksp_type='gmres' is rejected for schur."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "iface": {"ksp_type": "gmres"},
        },
    }

    with pytest.raises(ValueError, match=r"iface\.ksp_type|preonly"):
        validate_mpi_linear_support(cfg)


def test_schur_rejects_iface_ksp_type_fgmres():
    """Test that iface.ksp_type='fgmres' is rejected for schur."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "iface": {"ksp_type": "fgmres"},
        },
    }

    with pytest.raises(ValueError, match=r"iface\.ksp_type|preonly"):
        validate_mpi_linear_support(cfg)


def test_schur_allows_bulk_ksp_type_preonly():
    """Test that bulk.ksp_type='preonly' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "preonly"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["bulk"]["ksp_type"] == "preonly"


def test_schur_allows_bulk_ksp_type_gmres():
    """Test that bulk.ksp_type='gmres' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "gmres"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["bulk"]["ksp_type"] == "gmres"


def test_schur_allows_bulk_ksp_type_fgmres():
    """Test that bulk.ksp_type='fgmres' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "fgmres"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["bulk"]["ksp_type"] == "fgmres"


def test_schur_rejects_invalid_bulk_ksp_type():
    """Test that invalid bulk.ksp_type is rejected."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"ksp_type": "cg"},
        },
    }

    with pytest.raises(ValueError, match=r"bulk\.ksp_type|invalid"):
        validate_mpi_linear_support(cfg)


def test_schur_rejects_invalid_bulk_pc_type():
    """Test that invalid bulk.pc_type is rejected."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"pc_type": "ilu"},
        },
    }

    with pytest.raises(ValueError, match=r"bulk\.pc_type|invalid"):
        validate_mpi_linear_support(cfg)


def test_schur_allows_bulk_asm_sub_pc_type_ilu():
    """Test that bulk.asm_sub_pc_type='ilu' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"asm_sub_pc_type": "ilu"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["bulk"]["asm_sub_pc_type"] == "ilu"


def test_schur_allows_bulk_asm_sub_pc_type_lu():
    """Test that bulk.asm_sub_pc_type='lu' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "bulk": {"asm_sub_pc_type": "lu"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["bulk"]["asm_sub_pc_type"] == "lu"


def test_schur_allows_iface_asm_sub_pc_type_ilu():
    """Test that iface.asm_sub_pc_type='ilu' is allowed."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "iface": {"asm_sub_pc_type": "ilu"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["iface"]["asm_sub_pc_type"] == "ilu"


def test_schur_allows_iface_asm_sub_pc_type_lu():
    """Test that iface.asm_sub_pc_type='lu' is allowed (schur default)."""
    cfg = _base_cfg()
    cfg.solver.linear.pc_type = "fieldsplit"
    cfg.solver.linear.fieldsplit = {
        "type": "schur",
        "scheme": "bulk_iface",
        "subsolvers": {
            "iface": {"asm_sub_pc_type": "lu"},
        },
    }

    validate_mpi_linear_support(cfg)
    assert cfg.solver.linear.fieldsplit["subsolvers"]["iface"]["asm_sub_pc_type"] == "lu"
