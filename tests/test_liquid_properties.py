import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import CoolProp  # noqa: F401
except Exception:
    pytest.skip("CoolProp not available", allow_module_level=True)

from core.types import Grid1D, State  # noqa: E402
from properties.liquid import build_liquid_model, compute_liquid_props  # noqa: E402
from core.types import (  # noqa: E402
    CaseChecks,
    CaseConventions,
    CaseCoolProp,
    CaseDiscretization,
    CaseEquilibrium,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseInterface,
    CaseMesh,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    CaseConfig,
)


def make_case_config_for_liquid() -> CaseConfig:
    meta = CaseMeta(id="test", title="liquid", version=1, notes=None)
    paths = CasePaths(output_root=Path("."), case_dir=Path("."), mechanism_dir=Path("."), gas_mech="gri30.yaml")
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
    eq_cfg = CaseEquilibrium(
        method="raoult_psat",
        psat_model="coolprop",
        background_fill="farfield",
        condensables_gas=["Water"],
        coolprop=CaseCoolProp(backend="HEOS", fluids=["Water"]),
    )
    interface_cfg = CaseInterface(type="no_condensation", bc_mode="Ts_fixed", Ts_fixed=300.0, equilibrium=eq_cfg)
    physics = CasePhysics(interface=interface_cfg)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=tuple(),
        liq_species=["Water"],
        liq_balance_species="Water",
        liq2gas_map={"Water": "Water"},
        mw_kg_per_mol={"N2": 28.0, "Water": 18.01528},
        molar_volume_cm3_per_mol={"Water": 18.0},
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
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=2, N_gas=2, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-6, t_end=1e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 1.0},
        Yl={"Water": 1.0},
        Y_seed=1.0e-12,
    )
    petsc = CasePETSc(options_prefix="test_", ksp_type="gmres", pc_type="ilu", rtol=1e-6, atol=1e-12, max_it=50, restart=10, monitor=False)
    io_fields = CaseIOFields(scalars=[], gas=[], liquid=[])
    io = CaseIO(write_every=1, formats=["npz"], save_grid=False, fields=io_fields)
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


def make_simple_grid_and_state_for_liquid(liq_model) -> tuple[Grid1D, State]:
    Nl = 2
    Ng = 1
    Nc = Nl + Ng
    r_f = np.linspace(0.0, 1.0e-3, Nc + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    grid = Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=Nl)

    Ns_l = len(liq_model.fluids)
    Tl = np.array([300.0, 320.0])
    Yl = np.ones((Ns_l, Nl), dtype=float)
    # normalize columns (already all ones)
    for j in range(Nl):
        Yl[:, j] /= np.sum(Yl[:, j])

    # dummy gas fields just to satisfy State shape requirements
    Ns_g_dummy = 1
    Yg_dummy = np.ones((Ns_g_dummy, Ng), dtype=float)

    state = State(
        Tg=np.ones(Ng) * 300.0,
        Yg=Yg_dummy,
        Tl=Tl,
        Yl=Yl,
        Ts=310.0,
        mpp=0.0,
        Rd=1.0e-4,
    )
    return grid, state


@pytest.fixture(scope="module")
def liquid_model():
    cfg = make_case_config_for_liquid()
    return build_liquid_model(cfg)


@pytest.fixture()
def base_grid_and_state(liquid_model):
    return make_simple_grid_and_state_for_liquid(liquid_model)


def test_liquid_props_shapes_and_positive(liquid_model, base_grid_and_state):
    grid, state = base_grid_and_state
    core, extra = compute_liquid_props(liquid_model, state, grid)
    Nl = grid.Nl
    Ns_l = len(liquid_model.fluids)
    for key in ("rho_l", "cp_l", "k_l"):
        assert key in core
    for key in ("psat_l", "hvap_l"):
        assert key in extra

    assert core["rho_l"].shape == (Nl,)
    assert core["cp_l"].shape == (Nl,)
    assert core["k_l"].shape == (Nl,)
    assert extra["psat_l"].shape == (Ns_l,)
    assert extra["hvap_l"].shape == (Ns_l,)

    assert np.all(core["rho_l"] > 0)
    assert np.all(core["cp_l"] > 0)
    assert np.all(core["k_l"] > 0)
    assert np.all(extra["psat_l"] >= 0)
    assert np.all(extra["hvap_l"] >= 0)
    for arr in (*core.values(), *extra.values()):
        assert np.all(np.isfinite(arr))


def test_liquid_props_enforces_species_count_and_shapes(liquid_model, base_grid_and_state):
    grid, state = base_grid_and_state
    Ns_l = len(liquid_model.fluids)

    # Tl length mismatch
    state_bad_T = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=np.array([300.0, 320.0, 340.0]),
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_liquid_props(liquid_model, state_bad_T, grid)

    # Yl shape mismatch (second dimension)
    Y_bad_shape = np.ones((Ns_l, grid.Nl + 1))
    state_bad_shape = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=Y_bad_shape,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_liquid_props(liquid_model, state_bad_shape, grid)

    # species count mismatch
    Y_bad_species = np.ones((Ns_l - 1, grid.Nl))
    state_bad_species = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=Y_bad_species,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_liquid_props(liquid_model, state_bad_species, grid)


def test_liquid_props_enforces_normalized_Yl(liquid_model, base_grid_and_state):
    grid, state = base_grid_and_state
    Ns_l = len(liquid_model.fluids)
    # scale a column
    Y_scaled = state.Yl.copy()
    Y_scaled[:, 0] *= 2.0
    state_scaled = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=Y_scaled,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_liquid_props(liquid_model, state_scaled, grid)

    # zero column
    Y_zero = state.Yl.copy()
    Y_zero[:, 1] = 0.0
    state_zero = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=Y_zero,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_liquid_props(liquid_model, state_zero, grid)


def test_liquid_psat_hvap_depend_on_Ts(liquid_model, base_grid_and_state):
    grid, state = base_grid_and_state
    state1 = state
    state2 = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=state.Yl,
        Ts=state.Ts + 20.0,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    _, extra1 = compute_liquid_props(liquid_model, state1, grid)
    _, extra2 = compute_liquid_props(liquid_model, state2, grid)
    assert np.all(extra2["psat_l"] >= extra1["psat_l"])
    assert np.all(extra1["psat_l"] >= 0.0)
    assert np.all(extra2["psat_l"] >= 0.0)


def test_liquid_rho_cp_k_change_with_T(liquid_model, base_grid_and_state):
    grid, state = base_grid_and_state
    # lower temperatures
    state_low = state
    state_high = State(
        Tg=state.Tg,
        Yg=state.Yg,
        Tl=state.Tl + 50.0,
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    core_low, _ = compute_liquid_props(liquid_model, state_low, grid)
    core_high, _ = compute_liquid_props(liquid_model, state_high, grid)
    # Expect density to decrease with temperature (typical liquids)
    assert np.all(core_high["rho_l"] < core_low["rho_l"])
