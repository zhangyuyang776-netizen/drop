import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cantera  # noqa: F401
except Exception:
    pytest.skip("Cantera not available", allow_module_level=True)

from core.types import Grid1D, State  # noqa: E402
from properties.gas import build_gas_model, compute_gas_props  # noqa: E402
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
)


def make_case_config_for_gas() -> CaseConfig:
    meta = CaseMeta(id="test", title="gas", version=1, notes=None)
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
    physics = CasePhysics()
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=tuple(),
        liq_species=["FUEL_L"],
        liq_balance_species="FUEL_L",
        liq2gas_map={"FUEL_L": "CH4"},
        mw_kg_per_mol={"N2": 28.0, "O2": 32.0, "CH4": 16.0, "FUEL_L": 16.0},
        molar_volume_cm3_per_mol={"FUEL_L": 200.0},
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
        Yg={"N2": 0.7, "O2": 0.2, "CH4": 0.1},
        Yl={"FUEL_L": 1.0},
        Y_seed=1e-12,
    )
    petsc = CasePETSc(options_prefix="test_", ksp_type="gmres", pc_type="ilu", rtol=1e-6, atol=1e-12, max_it=50, restart=10, monitor=False)
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


def make_simple_grid_and_state_for_gas(gas_model) -> tuple[Grid1D, State]:
    Ng = 3
    Nl = 1
    Nc = Ng + Nl
    r_f = np.linspace(0.0, 1.0e-3, Nc + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    grid = Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=Nl)

    Ns = gas_model.gas.n_species
    Tg = np.array([600.0, 650.0, 700.0])
    Yg = np.zeros((Ns, Ng), dtype=float)
    # give three species non-zero values; rest remain zero
    iN2 = gas_model.gas.species_index("N2")
    iO2 = gas_model.gas.species_index("O2")
    iCH4 = gas_model.gas.species_index("CH4")
    Yg[iN2, :] = 0.7
    Yg[iO2, :] = 0.2
    Yg[iCH4, :] = 0.1
    # normalize each column explicitly (already sums to 1, but keep pattern)
    for j in range(Ng):
        Yg[:, j] /= np.sum(Yg[:, j])

    state = State(
        Tg=Tg,
        Yg=Yg,
        Tl=np.ones(Nl) * 300.0,
        Yl=np.ones((1, Nl)),
        Ts=300.0,
        mpp=0.0,
        Rd=1.0e-4,
    )
    return grid, state


@pytest.fixture(scope="module")
def gas_model():
    cfg = make_case_config_for_gas()
    return build_gas_model(cfg)


@pytest.fixture()
def base_grid_and_state(gas_model):
    return make_simple_grid_and_state_for_gas(gas_model)


def test_gas_props_shapes_and_positive(gas_model, base_grid_and_state):
    grid, state = base_grid_and_state
    core, extra = compute_gas_props(gas_model, state, grid)

    Ns = gas_model.gas.n_species
    Ng = grid.Ng
    for key in ("rho_g", "cp_g", "k_g", "D_g"):
        assert key in core
    for key in ("h_g", "h_gk"):
        assert key in extra

    assert core["rho_g"].shape == (Ng,)
    assert core["cp_g"].shape == (Ng,)
    assert core["k_g"].shape == (Ng,)
    assert core["D_g"].shape == (Ns, Ng)
    assert extra["h_g"].shape == (Ng,)
    assert extra["h_gk"].shape == (Ns, Ng)

    assert np.all(core["rho_g"] > 0)
    assert np.all(core["cp_g"] > 0)
    assert np.all(core["k_g"] > 0)
    assert np.all(core["D_g"] >= 0)
    for arr in (*core.values(), *extra.values()):
        assert np.all(np.isfinite(arr))


def test_gas_props_enforces_species_count(gas_model, base_grid_and_state):
    grid, state = base_grid_and_state
    Y_bad = state.Yg[:-1, :].copy()  # drop one species
    state_bad = State(
        Tg=state.Tg,
        Yg=Y_bad,
        Tl=state.Tl,
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_gas_props(gas_model, state_bad, grid)


def test_gas_props_enforces_normalized_Y(gas_model, base_grid_and_state):
    grid, state = base_grid_and_state

    # Case 1: scale a column to break normalization
    Y_scaled = state.Yg.copy()
    Y_scaled[:, 0] *= 2.0
    state_scaled = State(
        Tg=state.Tg,
        Yg=Y_scaled,
        Tl=state.Tl,
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_gas_props(gas_model, state_scaled, grid)

    # Case 2: zero column
    Y_zero = state.Yg.copy()
    Y_zero[:, 1] = 0.0
    state_zero = State(
        Tg=state.Tg,
        Yg=Y_zero,
        Tl=state.Tl,
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    with pytest.raises(ValueError):
        compute_gas_props(gas_model, state_zero, grid)


def test_gas_props_multiple_cells_consistency(gas_model, base_grid_and_state):
    grid, state = base_grid_and_state
    core, extra = compute_gas_props(gas_model, state, grid)

    assert not np.allclose(core["rho_g"][0], core["rho_g"][1])
    assert not np.allclose(core["cp_g"][0], core["cp_g"][1])

    # Now make Tg identical across cells
    state2 = State(
        Tg=np.ones_like(state.Tg) * state.Tg[1],
        Yg=state.Yg,
        Tl=state.Tl,
        Yl=state.Yl,
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )
    core2, _ = compute_gas_props(gas_model, state2, grid)
    assert np.allclose(core2["rho_g"], core2["rho_g"][0])
    assert np.allclose(core2["cp_g"], core2["cp_g"][0])
    assert np.allclose(core2["k_g"], core2["k_g"][0])
    assert np.allclose(core2["D_g"], core2["D_g"][:, [0]])


def test_gas_props_temperature_trend_sanity(gas_model, base_grid_and_state):
    """At fixed P and composition, density should decrease with increasing T."""
    grid, state_ref = base_grid_and_state
    Ng = grid.Ng
    Yg = state_ref.Yg.copy()

    Tg_low = np.full(Ng, 500.0, dtype=float)
    Tg_high = np.full(Ng, 800.0, dtype=float)
    state_low = State(
        Tg=Tg_low,
        Yg=Yg,
        Tl=state_ref.Tl,
        Yl=state_ref.Yl,
        Ts=state_ref.Ts,
        mpp=state_ref.mpp,
        Rd=state_ref.Rd,
    )
    state_high = State(
        Tg=Tg_high,
        Yg=Yg,
        Tl=state_ref.Tl,
        Yl=state_ref.Yl,
        Ts=state_ref.Ts,
        mpp=state_ref.mpp,
        Rd=state_ref.Rd,
    )
    core_low, _ = compute_gas_props(gas_model, state_low, grid)
    core_high, _ = compute_gas_props(gas_model, state_high, grid)
    assert np.all(core_high["rho_g"] < core_low["rho_g"])
