import numpy as np
import pytest
from pathlib import Path

from core.types import (
    CaseChecks,
    CaseConfig,
    CaseConventions,
    CaseDiscretization,
    CaseGeometry,
    CaseInitial,
    CaseIO,
    CaseIOFields,
    CaseMeta,
    CaseMesh,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    Grid1D,
    Props,
    State,
)
from core.layout import build_layout
from physics.radius_eq import build_radius_row


def _make_simple_grid() -> Grid1D:
    """Minimal 1-liquid + 1-gas spherical grid."""
    Nl = 1
    Ng = 1
    Nc = Nl + Ng

    r_f = np.array([0.0, 1.0e-4, 2.0e-4], dtype=float)
    r_c = np.array([0.5e-4, 1.5e-4], dtype=float)
    V_c = (4.0 * np.pi / 3.0) * (r_f[1:] ** 3 - r_f[:-1] ** 3)
    A_f = 4.0 * np.pi * r_f**2

    return Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nc,
        r_c=r_c,
        r_f=r_f,
        V_c=V_c,
        A_f=A_f,
        iface_f=Nl,
        liq_slice=slice(0, Nl),
        gas_slice=slice(Nl, Nc),
    )


def _make_simple_case(grid: Grid1D) -> CaseConfig:
    """Build minimal CaseConfig enabling Rd/mpp layout blocks."""
    physics = CasePhysics(
        solve_Tg=True,
        solve_Yg=True,
        solve_Tl=True,
        solve_Yl=False,
        include_Ts=True,
        include_mpp=True,
        include_Rd=True,
    )

    species = CaseSpecies(
        gas_balance_species="FUEL",
        gas_species=["FUEL"],
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": "FUEL"},
    )

    mesh = CaseMesh(
        liq_method="uniform",
        liq_beta=1.0,
        liq_center_bias=0.0,
        gas_method="uniform",
        gas_beta=1.0,
        gas_center_bias=0.0,
        enforce_interface_continuity=True,
        continuity_tol=1e-12,
    )

    geometry = CaseGeometry(
        a0=float(grid.r_f[grid.iface_f]),
        R_inf=float(grid.r_f[-1]),
        N_liq=grid.Nl,
        N_gas=grid.Ng,
        mesh=mesh,
    )

    time = CaseTime(t0=0.0, dt=1.0e-4, t_end=1.0e-3)

    discretization = CaseDiscretization(
        time_scheme="backward_euler",
        theta=1.0,
        mass_matrix="lumped",
    )

    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"FUEL": 0.0},
        Yl={"FUEL": 1.0},
        Y_seed=0.0,
    )

    petsc = CasePETSc(
        options_prefix="",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-6,
        atol=1e-12,
        max_it=50,
        restart=30,
        monitor=False,
    )

    io = CaseIO(
        write_every=1,
        formats=["none"],
        save_grid=False,
        fields=CaseIOFields(scalars=[], gas=[], liquid=[]),
    )

    checks = CaseChecks(
        enforce_sumY=False,
        sumY_tol=1e-8,
        clamp_negative_Y=False,
        min_Y=0.0,
        enforce_T_bounds=False,
        T_min=0.0,
        T_max=5000.0,
        enforce_unique_index=True,
        enforce_grid_state_props_split=True,
        enforce_assembly_purity=True,
    )

    conventions = CaseConventions(
        radial_normal="+r",
        flux_sign="outward_positive",
        heat_flux_def="q_positive_outward",
        evap_sign="mpp_positive_outward",
        gas_closure_species="FUEL",
        index_source="layout",
        assembly_pure=True,
        grid_state_props_split=True,
    )

    paths = CasePaths(
        output_root=Path("."),
        case_dir=Path("."),
        mechanism_dir=Path("."),
        gas_mech="mech.yaml",
    )

    return CaseConfig(
        case=CaseMeta(id="test_radius", title="test_radius", version=1),
        paths=paths,
        conventions=conventions,
        physics=physics,
        species=species,
        geometry=geometry,
        time=time,
        discretization=discretization,
        initial=initial,
        petsc=petsc,
        io=io,
        checks=checks,
    )


@pytest.fixture
def radius_test_env():
    grid = _make_simple_grid()
    cfg = _make_simple_case(grid)
    layout = build_layout(cfg, grid)

    rho_l_val = 800.0
    rho_l = np.full(grid.Nl, rho_l_val, dtype=float)
    props = Props(
        rho_g=np.zeros(grid.Ng, dtype=float),
        cp_g=np.zeros(grid.Ng, dtype=float),
        k_g=np.zeros(grid.Ng, dtype=float),
        D_g=None,
        rho_l=rho_l,
        cp_l=np.zeros(grid.Nl, dtype=float),
        k_l=np.zeros(grid.Nl, dtype=float),
    )

    dt = cfg.time.dt
    Rd_old = float(grid.r_f[grid.iface_f])

    state_old = State(
        Tg=np.array([300.0], dtype=float),
        Yg=np.zeros((1, grid.Ng), dtype=float),
        Tl=np.array([300.0], dtype=float),
        Yl=np.ones((1, grid.Nl), dtype=float),
        Ts=300.0,
        mpp=0.0,
        Rd=Rd_old,
    )
    state_guess = State(
        Tg=state_old.Tg.copy(),
        Yg=state_old.Yg.copy(),
        Tl=state_old.Tl.copy(),
        Yl=state_old.Yl.copy(),
        Ts=state_old.Ts,
        mpp=0.0,
        Rd=Rd_old,
    )

    return cfg, grid, layout, props, state_old, state_guess, dt, rho_l_val


def test_radius_eq_single_step_matches_analytic(radius_test_env):
    cfg, grid, layout, props, state_old, state_guess, dt, rho_l_val = radius_test_env

    mpp_const = 0.01
    Rd_old = state_old.Rd
    Rd_expected = Rd_old - mpp_const * dt / rho_l_val

    state_guess.Rd = Rd_expected
    state_guess.mpp = mpp_const

    rad = build_radius_row(
        grid=grid,
        state_old=state_old,
        state_guess=state_guess,
        props=props,
        layout=layout,
        dt=dt,
        cfg=cfg,
    )

    coeffs = rad.diag["radius_eq"]["coeffs"]
    assert np.isclose(coeffs["Rd"], 1.0 / dt)
    assert np.isclose(coeffs["mpp"], 1.0 / rho_l_val)
    assert np.isclose(rad.rhs, Rd_old / dt)

    u = np.zeros(layout.n_dof(), dtype=float)
    idx_Rd = rad.diag["radius_eq"]["idx_Rd"]
    idx_mpp = rad.diag["radius_eq"]["idx_mpp"]
    u[idx_Rd] = state_guess.Rd
    u[idx_mpp] = state_guess.mpp
    residual_row = sum(v * u[c] for c, v in zip(rad.cols, rad.vals)) - rad.rhs
    assert abs(residual_row) < 1e-12

    dRdt_expected = (Rd_expected - Rd_old) / dt
    diag = rad.diag["radius_eq"]
    assert np.isclose(diag["dRdt_guess"], dRdt_expected)
    assert np.isclose(dRdt_expected, -mpp_const / rho_l_val, rtol=1e-10, atol=1e-16)

    mass_old = diag["mass_old"]
    mass_bal = diag["mass_balance"]
    assert abs(mass_bal) < 1e-8 * abs(mass_old)


def test_radius_eq_evaporation_decreases_radius(radius_test_env):
    cfg, grid, layout, props, state_old, state_guess, dt, rho_l_val = radius_test_env

    mpp_const = 0.02
    Rd_old = state_old.Rd
    Rd_new = Rd_old - mpp_const * dt / rho_l_val

    state_guess.Rd = Rd_new
    state_guess.mpp = mpp_const

    rad = build_radius_row(
        grid=grid,
        state_old=state_old,
        state_guess=state_guess,
        props=props,
        layout=layout,
        dt=dt,
        cfg=cfg,
    )

    assert Rd_new < Rd_old
    dRdt = rad.diag["radius_eq"]["dRdt_guess"]
    assert dRdt < 0.0


def test_radius_eq_requires_positive_dt(radius_test_env):
    cfg, grid, layout, props, state_old, state_guess, _, _ = radius_test_env
    with pytest.raises(ValueError):
        build_radius_row(
            grid=grid,
            state_old=state_old,
            state_guess=state_guess,
            props=props,
            layout=layout,
            dt=0.0,
            cfg=cfg,
        )
