import numpy as np
import pytest
from pathlib import Path

import solvers.timestepper as timestepper
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
from solvers.timestepper import advance_one_step_scipy


def _make_simple_grid() -> Grid1D:
    """1 liquid cell + 1 gas cell spherical grid."""
    Nl, Ng = 1, 1
    Nc = Nl + Ng
    R_liq = 1.0e-4
    r_f = np.array([0.0, R_liq, 2 * R_liq], dtype=float)
    r_c = np.array([0.5 * R_liq, 1.5 * R_liq], dtype=float)
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
        iface_f=1,
        liq_slice=slice(0, Nl),
        gas_slice=slice(Nl, Nc),
    )


def _make_simple_case(grid: Grid1D) -> CaseConfig:
    case = CaseMeta(id="test_timestepper", title="timestepper_no_flux", version=1)
    paths = CasePaths(
        output_root=Path.cwd(),
        case_dir=Path.cwd() / "tmp_timestepper",
        mechanism_dir=Path("."),
        gas_mech="dummy.yaml",
    )
    physics = CasePhysics(
        solve_Tg=True,
        solve_Yg=False,  # No flux test, Yg not solved
        solve_Tl=True,   # Tl coupled in Stage 1 (fully implicit)
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
        mw_kg_per_mol={"FUEL": 0.1, "N2": 0.028},
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
        radial_normal="+er",
        flux_sign="outward_positive",
        heat_flux_def="q=-k*dTdr",
        evap_sign="mpp_positive_liq_to_gas",
        gas_closure_species="N2",
        index_source="layout",
        assembly_pure=True,
        grid_state_props_split=True,
    )
    return CaseConfig(
        case=case,
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


def _make_simple_props(grid: Grid1D) -> Props:
    rho_g = np.full(grid.Ng, 1.0, dtype=float)
    rho_l = np.full(grid.Nl, 800.0, dtype=float)
    cp_g = np.full(grid.Ng, 1000.0, dtype=float)
    cp_l = np.full(grid.Nl, 2000.0, dtype=float)
    k_g = np.full(grid.Ng, 0.1, dtype=float)
    k_l = np.full(grid.Nl, 0.2, dtype=float)
    D_g = np.full((2, grid.Ng), 1e-5, dtype=float)
    return Props(
        rho_g=rho_g,
        cp_g=cp_g,
        k_g=k_g,
        D_g=D_g,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
    )


def _make_simple_state(cfg: CaseConfig, grid: Grid1D) -> State:
    T0 = float(cfg.initial.T_inf)
    Tg = np.full(grid.Ng, T0, dtype=float)
    Tl = np.full(grid.Nl, T0, dtype=float)
    Ts = T0
    Yg = np.zeros((2, grid.Ng), dtype=float)
    Yg[1, :] = 1.0  # N2 = 1
    Yl = np.ones((1, grid.Nl), dtype=float)
    mpp = 0.0
    Rd = float(grid.r_f[grid.iface_f])
    return State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=Ts,
        mpp=mpp,
        Rd=Rd,
    )


@pytest.fixture
def timestepper_env():
    grid = _make_simple_grid()
    cfg = _make_simple_case(grid)
    layout = build_layout(cfg, grid)
    props = _make_simple_props(grid)
    state = _make_simple_state(cfg, grid)
    return cfg, grid, layout, props, state


def _fake_eq_result_for_no_evap(cfg, grid, state, props):
    ig_if = 0
    Yg_face = np.asarray(state.Yg[:, ig_if], dtype=float)
    return {"Yg_eq": Yg_face.copy()}


def _fake_eq_result_for_evap(cfg, grid, state, props):
    """
    Create a small evaporative driving force: set equilibrium gas FUEL fraction > cell value.
    """
    ig_if = 0
    Yg_cell = state.Yg[:, ig_if].copy()
    gas_names = list(cfg.species.gas_species)
    cond_name = cfg.species.liq_balance_species if isinstance(cfg.species.liq_balance_species, str) else cfg.species.liq_balance_species[0]
    cond_idx = gas_names.index(cond_name)

    Yg_eq = Yg_cell.copy()
    delta = 0.1
    Yg_eq[cond_idx] = Yg_cell[cond_idx] + delta
    others = [i for i in range(Yg_cell.shape[0]) if i != cond_idx]
    sum_others = float(np.sum(Yg_cell[others]))
    if sum_others > 0.0:
        scale = max(1.0 - Yg_eq[cond_idx], 0.0) / sum_others
        for i in others:
            Yg_eq[i] = Yg_cell[i] * scale
    else:
        Yg_eq[cond_idx] = 1.0

    return {"Yg_eq": Yg_eq}


def test_one_step_no_flux_no_evap_keeps_state_constant(timestepper_env, monkeypatch, tmp_path):
    cfg, grid, layout, props, state = timestepper_env
    cfg.paths.case_dir = tmp_path

    # patch equilibrium to enforce Yg_eq == Yg_face -> mpp stays 0
    monkeypatch.setattr(timestepper, "_build_eq_result_for_step", _fake_eq_result_for_no_evap)

    # patch latent heat
    monkeypatch.setattr(interface_bc, "_get_latent_heat", lambda props, cfg: 2.5e6)

    res = advance_one_step_scipy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state=state,
        props=props,
        t=0.0,
    )

    assert res.success
    diag = res.diag
    state_new = res.state_new

    assert diag.linear_converged
    assert diag.linear_residual_norm < 1e-10

    T0 = float(cfg.initial.T_inf)
    assert np.allclose(state_new.Tg, T0, atol=1e-10)
    assert np.allclose(state_new.Tl, T0, atol=1e-10)
    assert abs(state_new.Ts - T0) < 1e-10

    assert abs(state_new.mpp) < 1e-12
    expected_Rd = float(grid.r_f[grid.iface_f])
    assert abs(state_new.Rd - expected_Rd) < 1e-12

    if diag.energy_balance_if is not None:
        assert abs(diag.energy_balance_if) < 1e-8
    if diag.mass_balance_rd is not None:
        assert abs(diag.mass_balance_rd) < 1e-8

    scalars_path = cfg.paths.case_dir / "scalars" / "scalars.csv"
    assert scalars_path.exists()


def test_one_step_simple_evap_radius_response(timestepper_env, monkeypatch, tmp_path):
    cfg, grid, layout, props, state = timestepper_env
    cfg.paths.case_dir = tmp_path

    # induce evaporation: equilibrium has more FUEL than cell, and set latent heat
    monkeypatch.setattr(timestepper, "_build_eq_result_for_step", _fake_eq_result_for_evap)
    monkeypatch.setattr(interface_bc, "_get_latent_heat", lambda props, cfg: 2.5e6)

    Rd_old = float(state.Rd)
    dt = float(cfg.time.dt)
    rho_l_val = float(props.rho_l[0])

    res = advance_one_step_scipy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state=state,
        props=props,
        t=0.0,
    )

    assert res.success
    assert res.diag.linear_converged
    assert np.isfinite(res.diag.linear_residual_norm)

    state_new = res.state_new
    mpp_new = float(state_new.mpp)
    assert mpp_new > 0.0

    Rd_new = float(state_new.Rd)
    Rd_expected = Rd_old - mpp_new * dt / rho_l_val
    assert np.isclose(Rd_new, Rd_expected, rtol=1e-10, atol=1e-16)

    Ts_new = float(state_new.Ts)
    assert 250.0 <= Ts_new <= 400.0

    diag = res.diag
    if diag.mass_balance_rd is not None:
        assert np.isfinite(diag.mass_balance_rd)
        assert abs(diag.mass_balance_rd) < 1e-4
    if diag.energy_balance_if is not None:
        assert np.isfinite(diag.energy_balance_if)
        assert abs(diag.energy_balance_if) < 1e-2

    diag_sys = diag.extra.get("diag_sys", {})
    mpp_diag = diag_sys.get("mpp_mass", {})
    if mpp_diag:
        assert mpp_diag["Yg_eq"] > mpp_diag["Yg_cell"]
        assert mpp_diag["J_cond"] > 0.0

    Tg_new = state_new.Tg
    Tl_new = state_new.Tl
    assert np.all(np.isfinite(Tg_new))
    assert np.all(np.isfinite(Tl_new))
    assert Tg_new.min() >= 250.0 and Tg_new.max() <= 1000.0
    assert Tl_new.min() >= 250.0 and Tl_new.max() <= 1000.0

    scalars_path = cfg.paths.case_dir / "scalars" / "scalars.csv"
    assert scalars_path.exists()
