import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import (  # noqa: E402
    CaseChecks,
    CaseConfig,
    CaseConventions,
    CaseCoolProp,
    CaseDiscretization,
    CaseEquilibrium,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseInterface,
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
from core.grid import build_grid  # noqa: E402
from core.layout import build_layout  # noqa: E402
from physics.interface_bc import build_interface_coeffs  # noqa: E402


def _make_cfg() -> CaseConfig:
    meta = CaseMeta(id="step14_diag", title="step14_diag", version=1, notes=None)
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
    physics = CasePhysics(include_Ts=True, include_mpp=True, include_Rd=False)
    # latent heat fallback for tests (overridden in one test)
    setattr(physics, "latent_heat_default", 2.5e6)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["N2", "H2O"],
        liq_species=["H2O_l"],
        liq_balance_species="H2O_l",
        liq2gas_map={"H2O_l": "H2O"},
        mw_kg_per_mol={"N2": 28.0, "H2O": 18.0, "H2O_l": 18.0},
        molar_volume_cm3_per_mol={"H2O_l": 18.0},
    )
    mesh = CaseMesh(
        liq_method="tanh",
        liq_beta=2.0,
        liq_center_bias=0.0,
        gas_method="tanh",
        gas_beta=2.0,
        gas_center_bias=0.0,
        enforce_interface_continuity=False,
        continuity_tol=1e-12,
    )
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=2, N_gas=1, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-4, t_end=1e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 0.8, "H2O": 0.2},
        Yl={"H2O_l": 1.0},
        Y_seed=1e-12,
    )
    petsc = CasePETSc(
        options_prefix="test_",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-6,
        atol=1e-12,
        max_it=20,
        restart=5,
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
    interface_cfg = CaseInterface(type="no_condensation", bc_mode="Ts_fixed", Ts_fixed=300.0, equilibrium=CaseEquilibrium(coolprop=CaseCoolProp(fluids=["Water"])))
    physics.interface = interface_cfg

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


def _make_state(grid: Grid1D, mpp: float = 0.02) -> State:
    Tg = np.array([320.0], dtype=np.float64)
    Tl = np.array([300.0, 305.0], dtype=np.float64)
    Ts = 310.0
    Yg = np.array([[0.8], [0.2]], dtype=np.float64)  # closure + condensable
    Yl = np.ones((1, grid.Nl), dtype=np.float64)
    Rd = float(grid.r_f[grid.iface_f])
    return State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl, Ts=Ts, mpp=mpp, Rd=Rd)


def _make_props(grid: Grid1D) -> Props:
    Ng = grid.Ng
    Nl = grid.Nl
    rho_g = np.full(Ng, 1.0, dtype=np.float64)
    cp_g = np.full(Ng, 1000.0, dtype=np.float64)
    k_g = np.full(Ng, 0.1, dtype=np.float64)
    h_g = cp_g * np.array([320.0], dtype=np.float64)

    rho_l = np.full(Nl, 700.0, dtype=np.float64)
    cp_l = np.full(Nl, 2000.0, dtype=np.float64)
    k_l = np.full(Nl, 0.2, dtype=np.float64)
    h_l = cp_l * np.array([300.0, 305.0], dtype=np.float64)

    D_g = np.full((2, Ng), 1e-5, dtype=np.float64)

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
        D_l=None,
    )


def _build_min_interface(monkeypatch: pytest.MonkeyPatch | None = None, *, lv_from_enthalpy: bool = False):
    cfg = _make_cfg()
    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state = _make_state(grid)
    props = _make_props(grid)

    if lv_from_enthalpy and monkeypatch is not None:
        import physics.interface_bc as ibc

        def fake_lv(props_in, cfg_in):
            return float(props_in.h_g[0] - props_in.h_l[-1])

        monkeypatch.setattr(ibc, "_get_latent_heat", fake_lv)

    eq_result = {"Yg_eq": np.zeros((len(cfg.species.gas_species),), dtype=np.float64)}
    coeffs = build_interface_coeffs(grid, state, props, layout, cfg, eq_result=eq_result)
    return coeffs


def test_Ts_enthalpy_diag_is_self_consistent():
    coeffs = _build_min_interface()
    ent = coeffs.diag["Ts_energy"]["enthalpy_split"]

    scalar_keys = [
        "q_cond_g_pow",
        "q_cond_l_pow",
        "q_diff_g_pow",
        "q_diff_l_pow",
        "q_lat_old_pow",
        "q_lat_eff_pow",
        "latent_mismatch_pow",
        "balance_old_pow",
        "balance_eff_pow",
        "q_total_g_pow",
        "q_total_l_pow",
    ]
    for key in scalar_keys:
        assert key in ent
        assert np.isfinite(ent[key]), f"{key} is not finite"

    # arithmetic consistency
    assert np.isclose(ent["latent_mismatch_pow"], ent["q_lat_old_pow"] - ent["q_lat_eff_pow"], rtol=0.0, atol=1e-12)
    assert np.isclose(ent["balance_old_pow"], ent["q_cond_g_pow"] + ent["q_cond_l_pow"] - ent["q_lat_old_pow"], rtol=0.0, atol=1e-12)
    assert np.isclose(ent["balance_eff_pow"], ent["q_cond_g_pow"] + ent["q_cond_l_pow"] - ent["q_lat_eff_pow"], rtol=0.0, atol=1e-12)
    assert np.isclose(ent["balance_eff_pow"], ent["balance_old_pow"] + ent["latent_mismatch_pow"], rtol=0.0, atol=1e-12)


def test_Ts_enthalpy_diag_latent_matches_enthalpy(monkeypatch: pytest.MonkeyPatch):
    coeffs = _build_min_interface(monkeypatch=monkeypatch, lv_from_enthalpy=True)
    ent = coeffs.diag["Ts_energy"]["enthalpy_split"]

    assert np.isfinite(ent["latent_mismatch_pow"])
    assert np.isfinite(ent["balance_eff_pow"])

    assert abs(ent["latent_mismatch_pow"]) < 1e-12
    assert np.isclose(ent["balance_old_pow"], ent["balance_eff_pow"], rtol=0.0, atol=1e-12)
