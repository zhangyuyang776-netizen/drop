import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import (  # noqa: E402
    CaseChecks,
    CaseConventions,
    CaseDiscretization,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseMesh,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    CaseConfig,
    Grid1D,
    Props,
)
from physics.flux_convective_gas import compute_gas_convective_flux_T  # noqa: E402


def make_cfg() -> CaseConfig:
    """Minimal CaseConfig satisfying conventions and T_inf requirement."""
    meta = CaseMeta(id="flux_conv", title="flux conv", version=1, notes=None)
    paths = CasePaths(output_root=Path("."), case_dir=Path("."), mechanism_dir=Path("."), gas_mech="dummy.yaml")
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
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["N2"],
        liq_species=["WATER_L"],
        liq_balance_species="WATER_L",
        liq2gas_map={"WATER_L": "H2O"},
        mw_kg_per_mol={"N2": 28.0, "WATER_L": 18.0},
        molar_volume_cm3_per_mol={"WATER_L": 18.0},
    )
    mesh = CaseMesh(
        liq_method="tanh",
        liq_beta=2.0,
        liq_center_bias=0.4,
        gas_method="tanh",
        gas_beta=2.0,
        gas_center_bias=-0.4,
        enforce_interface_continuity=False,
        continuity_tol=1e-12,
    )
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=1, N_gas=3, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-6, t_end=1e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 1.0},
        Yl={"WATER_L": 1.0},
        Y_seed=1e-12,
    )
    petsc = CasePETSc(
        options_prefix="conv_",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-8,
        atol=1e-12,
        max_it=50,
        restart=10,
        monitor=False,
    )
    io_fields = CaseIOFields(scalars=[], gas=[], liquid=[])
    io = CaseIO(write_every=1, formats=["npz"], save_grid=False, fields=io_fields)
    checks = CaseChecks(
        enforce_sumY=True,
        sumY_tol=1e-10,
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


def make_simple_grid() -> Grid1D:
    """Uniform grid with 1 liquid + 3 gas cells."""
    Nl = 1
    Ng = 3
    Nc = Nl + Ng
    r_f = np.array([0.0, 5.0e-5, 1.0e-4, 1.5e-4, 2.0e-4])
    r_c = np.array([2.5e-5, 7.5e-5, 1.25e-4, 1.75e-4])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    return Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nc,
        r_c=r_c,
        r_f=r_f,
        V_c=V_c,
        A_f=A_f,
        iface_f=Nl,
    )


def make_props(grid: Grid1D) -> Props:
    rho_g = np.ones(grid.Ng)
    cp_g = 2.0 * np.ones(grid.Ng)
    k_g = 0.1 * np.ones(grid.Ng)
    rho_l = np.ones(grid.Nl)
    cp_l = np.ones(grid.Nl)
    k_l = np.ones(grid.Nl)
    props = Props(rho_g=rho_g, cp_g=cp_g, k_g=k_g, D_g=None, rho_l=rho_l, cp_l=cp_l, k_l=k_l, D_l=None)
    props.validate_shapes(grid, Ns_g=grid.Ng, Ns_l=grid.Nl)
    return props


def test_convective_flux_shapes_and_zero_velocity():
    cfg = make_cfg()
    grid = make_simple_grid()
    props = make_props(grid)
    Tg = np.array([700.0, 500.0, 300.0])
    u_face = np.zeros(grid.Nc + 1)

    q_conv = compute_gas_convective_flux_T(cfg, grid, props, Tg, u_face)

    assert q_conv.shape == (grid.Nc + 1,)
    assert np.allclose(q_conv, 0.0)


def test_convective_flux_internal_faces_upwind_positive_velocity():
    cfg = make_cfg()
    grid = make_simple_grid()
    props = make_props(grid)
    Tg = np.array([700.0, 500.0, 300.0])
    u_face = np.zeros(grid.Nc + 1)
    # internal gas faces f=2,3; outer f=4
    u_face[2] = 1.0
    u_face[3] = 2.0
    u_face[4] = 3.0

    q_conv = compute_gas_convective_flux_T(cfg, grid, props, Tg, u_face)

    assert np.isclose(q_conv[2], 1400.0)
    assert np.isclose(q_conv[3], 2000.0)
    assert np.isclose(q_conv[4], 1800.0)
    assert q_conv[0] == 0.0
    assert q_conv[grid.iface_f] == 0.0


def test_convective_flux_upwind_negative_velocity_and_outer_boundary():
    cfg = make_cfg()
    grid = make_simple_grid()
    props = make_props(grid)
    cfg.initial.T_inf = 290.0
    Tg = np.array([700.0, 500.0, 300.0])
    u_face = np.zeros(grid.Nc + 1)
    u_face[2] = -1.0
    u_face[3] = -2.0
    u_face[4] = -1.5  # inflow from boundary

    q_conv = compute_gas_convective_flux_T(cfg, grid, props, Tg, u_face)

    assert np.isclose(q_conv[2], -1000.0)
    assert np.isclose(q_conv[3], -1200.0)
    assert np.isclose(q_conv[4], -870.0)


def test_convective_flux_interface_and_liquid_faces_zero():
    cfg = make_cfg()
    grid = make_simple_grid()
    props = make_props(grid)
    Tg = np.array([700.0, 500.0, 300.0])
    u_face = np.zeros(grid.Nc + 1)
    u_face[2] = 1.0
    u_face[3] = 1.0

    q_conv = compute_gas_convective_flux_T(cfg, grid, props, Tg, u_face)

    assert q_conv[0] == 0.0
    assert q_conv[grid.iface_f] == 0.0


def test_convective_flux_conventions_mismatch_raises():
    cfg = make_cfg()
    grid = make_simple_grid()
    props = make_props(grid)
    Tg = np.array([700.0, 500.0, 300.0])
    u_face = np.zeros(grid.Nc + 1)
    # break flux_sign
    cfg_bad = replace(cfg, conventions=replace(cfg.conventions, flux_sign="inward_positive"))
    with pytest.raises(ValueError):
        compute_gas_convective_flux_T(cfg_bad, grid, props, Tg, u_face)
    # break radial_normal
    cfg_bad2 = replace(cfg, conventions=replace(cfg.conventions, radial_normal="-er"))
    with pytest.raises(ValueError):
        compute_gas_convective_flux_T(cfg_bad2, grid, props, Tg, u_face)
    # break heat_flux_def
    cfg_bad3 = replace(cfg, conventions=replace(cfg.conventions, heat_flux_def="q=k*dTdr"))
    with pytest.raises(ValueError):
        compute_gas_convective_flux_T(cfg_bad3, grid, props, Tg, u_face)
