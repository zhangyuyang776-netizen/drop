import numpy as np
import pytest
from pathlib import Path

import physics.interface_bc as interface_bc
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
from physics.interface_bc import build_interface_coeffs


def _make_simple_grid() -> Grid1D:
    """Minimal 1-cell liquid + 1-cell gas grid."""
    Nl = 1
    Ng = 1
    Nc = Nl + Ng

    r_f = np.array([0.0, 1.0e-4, 2.0e-4], dtype=float)
    r_c = np.array([0.5e-4, 1.5e-4], dtype=float)
    V_c = (4.0 * np.pi / 3.0) * (r_f[1:] ** 3 - r_f[:-1] ** 3)
    A_f = 4.0 * np.pi * r_f**2

    grid = Grid1D(
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
    return grid


def _make_simple_case(grid: Grid1D) -> CaseConfig:
    """Build a minimal CaseConfig sufficient for layout/interface assembly."""
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
        gas_balance_species="N2",
        gas_species=["FUEL", "N2"],
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
        Yg={"FUEL": 0.0, "N2": 1.0},
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
        gas_closure_species="N2",
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

    cfg = CaseConfig(
        case=CaseMeta(id="test_interface", title="test", version=1),
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
    return cfg


@pytest.fixture
def interface_test_env():
    grid = _make_simple_grid()
    cfg = _make_simple_case(grid)
    layout = build_layout(cfg, grid)

    rho_g = np.full(grid.Ng, 1.0, dtype=float)
    cp_g = np.full(grid.Ng, 1.0e3, dtype=float)
    k_g = np.full(grid.Ng, 0.1, dtype=float)
    D_g = np.full((len(cfg.species.gas_species), grid.Ng), 1.0e-5, dtype=float)

    rho_l = np.full(grid.Nl, 800.0, dtype=float)
    cp_l = np.full(grid.Nl, 2.0e3, dtype=float)
    k_l = np.full(grid.Nl, 0.15, dtype=float)

    props = Props(
        rho_g=rho_g,
        cp_g=cp_g,
        k_g=k_g,
        D_g=D_g,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
    )

    Tg = np.full(grid.Ng, 300.0, dtype=float)
    Tl = np.full(grid.Nl, 300.0, dtype=float)
    # Gas species order matches cfg.species.gas_species: ["FUEL", "N2"]
    Yg = np.zeros((len(cfg.species.gas_species), grid.Ng), dtype=float)
    Yl = np.ones((len(cfg.species.liq_species), grid.Nl), dtype=float)

    state = State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=float(grid.r_f[grid.iface_f]),
    )

    return cfg, grid, layout, props, state


def _set_latent_heat(monkeypatch, value: float) -> None:
    """Force latent heat resolution to a fixed value for testing."""
    monkeypatch.setattr(interface_bc, "_get_latent_heat", lambda props, cfg: float(value))


def test_interface_zero_flux(interface_test_env, monkeypatch):
    cfg, grid, layout, props, state = interface_test_env
    _set_latent_heat(monkeypatch, 2.5e6)

    Yg_eq = state.Yg[:, 0].copy()
    eq_result = {"Yg_eq": Yg_eq}

    coeffs = build_interface_coeffs(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
    )

    diag = coeffs.diag
    ts_diag = diag["Ts_energy"]
    mpp_diag = diag["mpp_mass"]

    # Energy balance should be zero
    assert abs(ts_diag["q_g"]) < 1e-12
    assert abs(ts_diag["q_l"]) < 1e-12
    assert abs(ts_diag["q_lat"]) < 1e-12
    assert abs(ts_diag["balance"]) < 1e-12

    # Mass balance should be zero
    assert abs(mpp_diag["J_cond"]) < 1e-12
    assert abs(mpp_diag["mpp"]) < 1e-12
    assert abs(mpp_diag["residual"]) < 1e-12

    # Recompute residuals from matrix rows
    idx_Ts = layout.idx_Ts()
    idx_mpp = layout.idx_mpp()
    row_Ts = next(r for r in coeffs.rows if r.row == idx_Ts)
    row_mpp = next(r for r in coeffs.rows if r.row == idx_mpp)

    u = np.zeros(layout.n_dof(), dtype=float)
    ig_local = 0
    il_local = grid.Nl - 1
    u[layout.idx_Tg(ig_local)] = state.Tg[ig_local]
    u[layout.idx_Tl(il_local)] = state.Tl[il_local]
    u[layout.idx_Ts()] = state.Ts
    u[layout.idx_mpp()] = state.mpp
    # Reduced species index for FUEL (condensable)
    k_red = layout.gas_full_to_reduced["FUEL"]
    u[layout.idx_Yg(k_red, ig_local)] = state.Yg[0, ig_local]

    res_Ts = sum(v * u[c] for c, v in zip(row_Ts.cols, row_Ts.vals)) - row_Ts.rhs
    res_mpp = sum(v * u[c] for c, v in zip(row_mpp.cols, row_mpp.vals)) - row_mpp.rhs

    assert abs(res_Ts) < 1e-12
    assert abs(res_mpp) < 1e-12


def test_interface_evaporation_energy_and_sign(interface_test_env, monkeypatch):
    cfg, grid, layout, props, state = interface_test_env
    L_v = 2.5e6
    _set_latent_heat(monkeypatch, L_v)

    iface_f = grid.iface_f
    il_global = grid.Nl - 1
    ig_global = grid.Nl

    r_if = grid.r_f[iface_f]
    dr_g = grid.r_c[ig_global] - r_if
    dr_l = r_if - grid.r_c[il_global]
    A_if = grid.A_f[iface_f]
    k_g = props.k_g[0]
    k_l = props.k_l[0]

    # Construct an evaporative scenario: liquid hotter, gas cooler -> q_g+q_l > 0 -> mpp > 0
    Tl = 380.0
    Ts = 360.0
    Tg = 350.0

    q_g = -k_g * (Tg - Ts) / dr_g * A_if
    q_l = -k_l * (Ts - Tl) / dr_l * A_if
    mpp_hand = (q_g + q_l) / (L_v * A_if)

    state.Tg[0] = Tg
    state.Tl[0] = Tl
    state.Ts = Ts
    state.mpp = mpp_hand

    Yg_eq = state.Yg[:, 0].copy()
    eq_result = {"Yg_eq": Yg_eq}

    coeffs = build_interface_coeffs(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
    )

    ts_diag = coeffs.diag["Ts_energy"]
    assert abs(ts_diag["balance"]) < 1e-10
    assert state.mpp > 0.0

    idx_Ts = layout.idx_Ts()
    row_Ts = next(r for r in coeffs.rows if r.row == idx_Ts)
    u = np.zeros(layout.n_dof(), dtype=float)
    ig_local = 0
    il_local = grid.Nl - 1
    u[layout.idx_Tg(ig_local)] = state.Tg[ig_local]
    u[layout.idx_Tl(il_local)] = state.Tl[il_local]
    u[layout.idx_Ts()] = state.Ts
    u[layout.idx_mpp()] = state.mpp
    k_red = layout.gas_full_to_reduced["FUEL"]
    u[layout.idx_Yg(k_red, ig_local)] = state.Yg[0, ig_local]

    res_Ts = sum(v * u[c] for c, v in zip(row_Ts.cols, row_Ts.vals)) - row_Ts.rhs
    assert abs(res_Ts) < 1e-10


def test_mpp_sign_convention(interface_test_env, monkeypatch):
    cfg, grid, layout, props, state = interface_test_env
    _set_latent_heat(monkeypatch, 2.5e6)

    state.mpp = 0.0
    state.Yg[0, 0] = 0.0  # condensable in cell
    Yg_eq = np.array([0.1, 0.9], dtype=float)  # larger condensable fraction at interface
    eq_result = {"Yg_eq": Yg_eq}

    coeffs = build_interface_coeffs(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
    )

    mpp_diag = coeffs.diag["mpp_mass"]
    assert mpp_diag["J_cond"] > 0.0
