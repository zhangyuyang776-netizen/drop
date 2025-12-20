from pathlib import Path
from typing import Dict, Sequence

import numpy as np

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
from core.grid import build_grid
from core.layout import build_layout


def make_cfg_base(
    *,
    Nl: int = 1,
    Ng: int = 3,
    gas_species: Sequence[str] = ("FUEL", "N2"),
    gas_balance: str = "N2",
    liq_species: Sequence[str] = ("FUEL_L",),
    liq_balance: str = "FUEL_L",
    liq2gas_map: Dict[str, str] | None = None,
    solve_Yg: bool = True,
    include_mpp: bool = False,
    include_Ts: bool = False,
    include_Rd: bool = False,
    species_convection: bool = False,
    dt: float = 1.0e-4,
) -> CaseConfig:
    """Construct a minimal CaseConfig for step15 tests (no external deps)."""
    liq2gas_map = liq2gas_map or {liq_balance: gas_species[0]}

    meta = CaseMeta(id="step15_test", title="step15", version=1, notes=None)
    paths = CasePaths(output_root=Path("."), case_dir=Path("."), mechanism_dir=Path("."), gas_mech="dummy.yaml")
    conventions = CaseConventions(
        radial_normal="+er",
        flux_sign="outward_positive",
        heat_flux_def="q=-k*dTdr",
        evap_sign="mpp_positive_evaporation",
        gas_closure_species=gas_balance,
        index_source="unknown_layout_only",
        assembly_pure=True,
        grid_state_props_split=True,
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
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=Nl, N_gas=Ng, mesh=mesh)
    time = CaseTime(t0=0.0, dt=dt, t_end=10 * dt, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={gas_balance: 1.0},
        Yl={liq_balance: 1.0},
        Y_seed=0.0,
    )
    petsc = CasePETSc(
        options_prefix="",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-8,
        atol=1e-12,
        max_it=50,
        restart=10,
        monitor=False,
    )
    io = CaseIO(write_every=0, formats=[], save_grid=False, fields=CaseIOFields(scalars=[], gas=[], liquid=[]))
    checks = CaseChecks(
        enforce_sumY=True,
        sumY_tol=1.0e-12,
        clamp_negative_Y=False,
        min_Y=0.0,
        enforce_T_bounds=False,
        T_min=0.0,
        T_max=1.0e6,
        enforce_unique_index=True,
        enforce_grid_state_props_split=True,
        enforce_assembly_purity=True,
    )

    interface = CasePhysics().interface  # default interface, overwrite bc_mode
    interface.bc_mode = "Ts_fixed"
    interface.Ts_fixed = 300.0
    cond_name = gas_species[0]
    interface.equilibrium.condensables_gas = [cond_name]

    physics = CasePhysics(
        model="droplet_1d_spherical_noChem",
        solve_Tg=True,
        solve_Yg=solve_Yg,
        solve_Tl=True,
        solve_Yl=False,
        include_Ts=include_Ts,
        include_mpp=include_mpp,
        include_Rd=include_Rd,
        stefan_velocity=True,
        interface=interface,
    )
    # optional toggle for convection of species (off by default)
    physics.species_convection = species_convection
    # Provide latent heat fallback to avoid Ts-row assembly failures in tests
    physics.latent_heat_default = 2.5e6

    species_cfg = CaseSpecies(
        gas_balance_species=gas_balance,
        gas_species_full=list(gas_species),
        liq_species=list(liq_species),
        liq_balance_species=liq_balance,
        liq2gas_map=liq2gas_map,
    )

    return CaseConfig(
        case=meta,
        paths=paths,
        conventions=conventions,
        physics=physics,
        species=species_cfg,
        geometry=geometry,
        time=time,
        discretization=disc,
        initial=initial,
        petsc=petsc,
        io=io,
        checks=checks,
    )


def make_props_const(
    Ns_g: int,
    Ns_l: int,
    grid: Grid1D,
    *,
    D_g_val: float = 1.0e-5,
    Tg_val: float = 300.0,
    Tl_val: float = 300.0,
) -> Props:
    """Build constant Props without external dependencies."""
    rho_g = np.ones(grid.Ng, dtype=float)
    cp_g = np.ones(grid.Ng, dtype=float)
    k_g = np.ones(grid.Ng, dtype=float)
    D_g = np.full((Ns_g, grid.Ng), D_g_val, dtype=float)
    h_g = cp_g * float(Tg_val)

    rho_l = np.ones(grid.Nl, dtype=float)
    cp_l = np.ones(grid.Nl, dtype=float)
    k_l = np.ones(grid.Nl, dtype=float)
    D_l = np.full((Ns_l, grid.Nl), 1.0e-5, dtype=float)
    h_l = cp_l * float(Tl_val)

    return Props(
        rho_g=rho_g,
        cp_g=cp_g,
        k_g=k_g,
        D_g=D_g,
        h_g=h_g,
        h_l=h_l,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
        D_l=D_l,
    )


def make_state_uniform(cfg: CaseConfig, grid: Grid1D, gas_species: Sequence[str], Yg_full: np.ndarray) -> State:
    """Build a uniform State with provided full Yg array (Ns_g x Ng)."""
    Ns_l = len(cfg.species.liq_species)
    Tg = np.full(grid.Ng, float(cfg.initial.T_inf), dtype=np.float64)
    Tl = np.full(grid.Nl, float(cfg.initial.T_d0), dtype=np.float64)
    Ts = float(cfg.initial.T_d0)
    Yl = np.ones((Ns_l, grid.Nl), dtype=np.float64)
    Rd = float(cfg.geometry.a0)
    return State(Tg=Tg, Yg=Yg_full, Tl=Tl, Yl=Yl, Ts=Ts, mpp=0.0, Rd=Rd)


def build_min_problem(cfg: CaseConfig):
    """Build grid, layout, and a uniform state/props for tests."""
    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    gas_species = cfg.species.gas_species_full
    Ns_g = len(gas_species)
    Ns_l = len(cfg.species.liq_species)

    # default Yg: closure = 1, others 0
    Yg_full = np.zeros((Ns_g, grid.Ng), dtype=np.float64)
    if cfg.species.gas_balance_species in gas_species:
        bal_idx = gas_species.index(cfg.species.gas_balance_species)
        Yg_full[bal_idx, :] = 1.0

    state = make_state_uniform(cfg, grid, gas_species, Yg_full)
    props = make_props_const(Ns_g, Ns_l, grid)
    return grid, layout, state, props


def fake_eq_result(cfg: CaseConfig, layout) -> Dict[str, np.ndarray]:
    """Return a dummy eq_result with zeros (length Ns_g_full)."""
    Ns = len(layout.gas_species_full)
    return {"Yg_eq": np.zeros(Ns, dtype=np.float64)}
