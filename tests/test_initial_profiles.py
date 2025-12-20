import numpy as np
import pytest

pytest.importorskip("scipy")
from properties.gas import GasPropertiesModel
from core.grid import build_grid
from physics.initial import build_initial_state_erfc
from tests._helpers_step15 import make_cfg_base


class DummyGas:
    """Simple stub for Cantera Solution."""

    def __init__(self, names, rho=1.0, cp=1000.0, k=0.1):
        self.species_names = list(names)
        self.molecular_weights = np.ones(len(names)) * 0.028  # kg/mol placeholder
        self.density = rho
        self.cp_mass = cp
        self.thermal_conductivity = k

    def TPY(self, T, P, Y):
        # keep constants; interface required for compatibility
        self.density = self.density
        self.cp_mass = self.cp_mass
        self.thermal_conductivity = self.thermal_conductivity


@pytest.fixture
def base_setup():
    # Build a minimal config with temperature gradient and condensable fuel species
    cfg = make_cfg_base(Nl=2, Ng=5, gas_species=("FUEL", "N2"), gas_balance="N2")
    cfg.initial.T_inf = 800.0
    cfg.initial.T_d0 = 300.0
    cfg.initial.t_init_T = 1.0e-5
    cfg.initial.t_init_Y = 1.0e-5
    cfg.initial.D_init_Y = 1.0e-5
    cfg.initial.Y_vap_if0 = 1.0e-3
    # farfield gas: pure N2
    cfg.initial.Yg = {"N2": 1.0}
    cfg.initial.Yl = {"FUEL": 1.0}

    grid = build_grid(cfg)

    gas_names = list(cfg.species.gas_species_full)
    gas_stub = DummyGas(gas_names)
    gas_model = GasPropertiesModel(
        gas=gas_stub,
        P_ref=float(cfg.initial.P_inf),
        gas_names=tuple(gas_names),
        name_to_idx={nm: i for i, nm in enumerate(gas_names)},
    )
    liq_model = None

    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    return cfg, grid, gas_model, state0


def test_initial_shapes_and_finite(base_setup):
    cfg, grid, gas_model, state = base_setup
    Ns_g = len(cfg.species.gas_species_full)
    Ns_l = len(cfg.species.liq_species)

    assert state.Tg.shape == (grid.Ng,)
    assert state.Tl.shape == (grid.Nl,)
    assert state.Yg.shape == (Ns_g, grid.Ng)
    assert state.Yl.shape == (Ns_l, grid.Nl)

    assert np.isfinite(state.Tg).all()
    assert np.isfinite(state.Tl).all()
    assert np.isfinite(state.Yg).all()
    assert np.isfinite(state.Yl).all()


def test_initial_Tg_monotone(base_setup):
    cfg, grid, _, state = base_setup
    Tg = state.Tg
    diff = np.diff(Tg)
    atol = 1e-8
    if cfg.initial.T_inf > cfg.initial.T_d0:
        assert (diff >= -atol).all()
    else:
        assert (diff <= atol).all()


def test_initial_Yg_condensable_profile_and_closure(base_setup):
    cfg, grid, gas_model, state = base_setup
    gas_names = list(gas_model.gas.species_names)
    cond_names = list(cfg.physics.interface.equilibrium.condensables_gas)
    assert cond_names, "condensables_gas must not be empty in this test setup."
    cond_idx = gas_names.index(cond_names[0])
    Y = state.Yg[cond_idx, :]
    diff = np.diff(Y)
    atol = 1e-10
    # seeded value should be highest near interface and decay toward farfield (which is zero here)
    assert Y[0] >= Y[-1] - atol
    assert (diff <= atol).all()
    assert Y[-1] <= 1e-6  # farfield fuel should be near zero

    sums = np.sum(state.Yg, axis=0)
    assert np.allclose(sums, 1.0, atol=1.0e-12)


def test_interface_T_match(base_setup):
    cfg, grid, _, state = base_setup
    Tg_if = float(state.Tg[0])
    Ts = float(state.Ts)
    T_inf = float(cfg.initial.T_inf)
    lo = min(Ts, T_inf) - 1e-6
    hi = max(Ts, T_inf) + 1e-6
    assert lo <= Tg_if <= hi
