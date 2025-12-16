import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grid import build_grid
from core.types import (  # noqa: E402
    CaseChecks,
    CaseConventions,
    CaseDiscretization,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseGeometry,
    CaseMesh,
    CaseTime,
    CaseConfig,
)
from core.types import Grid1D  # noqa: E402


def make_minimal_cfg(*, tol: float, enforce_iface: bool) -> CaseConfig:
    meta = CaseMeta(id="test", title="test", version=1, notes=None)
    paths = CasePaths(
        output_root=Path("."),
        case_dir=Path("."),
        mechanism_dir=Path("."),
        gas_mech="dummy.yaml",
    )
    species = CaseSpecies(
        gas_balance_species="N2",
        liq_species=["NC12H26"],
        liq_balance_species="NC12H26",
        liq2gas_map={"NC12H26": "NC12H26"},
        mw_kg_per_mol={"NC12H26": 0.17034},
        molar_volume_cm3_per_mol={"NC12H26": 227.51},
    )
    conventions = CaseConventions(
        radial_normal="+er",
        flux_sign="outward_positive",
        heat_flux_def="q=-k*dTdr",
        evap_sign="mpp_positive_evaporation",
        gas_closure_species="N2",
        index_source="unknown_layout_only",
        assembly_pure=True,
        grid_state_props_split=True,
    )
    physics = CasePhysics()

    mesh = CaseMesh(
        liq_method="tanh",
        liq_beta=2.0,
        liq_center_bias=0.4,
        gas_method="tanh",
        gas_beta=2.0,
        gas_center_bias=-0.4,
        enforce_interface_continuity=enforce_iface,
        continuity_tol=tol,
    )
    geometry = CaseGeometry(
        a0=1.0e-4,
        R_inf=1.0e-3,
        N_liq=10,
        N_gas=10,
        mesh=mesh,
    )

    time = CaseTime(t0=0.0, dt=1.0e-6, t_end=1.0e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=1100.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 1.0},
        Yl={"NC12H26": 1.0},
        Y_seed=1.0e-12,
    )
    petsc = CasePETSc(
        options_prefix="test_",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1.0e-8,
        atol=1.0e-12,
        max_it=50,
        restart=10,
        monitor=False,
    )
    io_fields = CaseIOFields(scalars=["t"], gas=["Tg"], liquid=["Tl"])
    io = CaseIO(write_every=10, formats=["npz"], save_grid=False, fields=io_fields)
    checks = CaseChecks(
        enforce_sumY=True,
        sumY_tol=1.0e-10,
        clamp_negative_Y=True,
        min_Y=0.0,
        enforce_T_bounds=True,
        T_min=200.0,
        T_max=4000.0,
        enforce_unique_index=True,
        enforce_grid_state_props_split=True,
        enforce_assembly_purity=True,
    )

    return CaseConfig(
        case=meta,
        paths=paths,
        conventions=conventions,
        physics=physics,
        species=species,
        geometry=geometry,
        time=time,
        discretization=disc,
        initial=initial,
        petsc=petsc,
        io=io,
        checks=checks,
    )


def test_grid_no_enforce_continuity():
    cfg = make_minimal_cfg(tol=1e-6, enforce_iface=False)
    grid = build_grid(cfg)
    assert isinstance(grid, Grid1D)
    assert grid.iface_f == cfg.geometry.N_liq


def test_grid_enforce_continuity_pass_with_large_tol():
    cfg = make_minimal_cfg(tol=1.0, enforce_iface=True)
    grid = build_grid(cfg)
    assert isinstance(grid, Grid1D)


def test_grid_enforce_continuity_fail_with_small_tol():
    cfg = make_minimal_cfg(tol=1e-12, enforce_iface=True)
    with pytest.raises(ValueError):
        _ = build_grid(cfg)
