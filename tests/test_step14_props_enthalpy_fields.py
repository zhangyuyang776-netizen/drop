import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import Grid1D, State  # noqa: E402
from properties import aggregator  # noqa: E402


class DummyGasModel:
    def __init__(self, n_species: int):
        self.gas = type("gas", (), {"n_species": n_species})()


class DummyLiqModel:
    def __init__(self, liq_names):
        self.liq_names = tuple(liq_names)


def _make_grid(Nl: int, Ng: int) -> Grid1D:
    Nc = Nl + Ng
    r_f = np.linspace(0.0, Nc * 1e-4, Nc + 1, dtype=np.float64)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.full(Nc, 1.0, dtype=np.float64)
    A_f = np.full(Nc + 1, 1.0, dtype=np.float64)
    return Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=Nl)


def _make_state(grid: Grid1D, Ns_g: int, Ns_l: int) -> State:
    Tg = np.linspace(300.0, 400.0, grid.Ng, dtype=np.float64)
    Tl = np.linspace(290.0, 320.0, grid.Nl, dtype=np.float64)

    Yg = np.zeros((Ns_g, grid.Ng), dtype=np.float64)
    Yg[0, :] = 1.0

    Yl = np.zeros((Ns_l, grid.Nl), dtype=np.float64)
    Yl[0, :] = 1.0

    return State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl, Ts=300.0, mpp=0.0, Rd=1e-4)


def test_props_enthalpy_fields_shapes_and_finite(monkeypatch: pytest.MonkeyPatch):
    grid = _make_grid(Nl=2, Ng=3)
    Ns_g = 2
    Ns_l = 1
    state = _make_state(grid, Ns_g=Ns_g, Ns_l=Ns_l)

    gas_model = DummyGasModel(Ns_g)
    liq_model = DummyLiqModel(["Water"])

    def fake_compute_gas_props(model, state_in, grid_in):
        Ng = grid_in.Ng
        core = {
            "rho_g": np.full(Ng, 1.0, dtype=np.float64),
            "cp_g": np.full(Ng, 1000.0, dtype=np.float64),
            "k_g": np.full(Ng, 0.1, dtype=np.float64),
            "D_g": np.zeros((model.gas.n_species, Ng), dtype=np.float64),
        }
        return core, {}

    def fake_compute_liquid_props(model, state_in, grid_in):
        Nl = grid_in.Nl
        core = {
            "rho_l": np.full(Nl, 700.0, dtype=np.float64),
            "cp_l": np.full(Nl, 2000.0, dtype=np.float64),
            "k_l": np.full(Nl, 0.2, dtype=np.float64),
        }
        return core, {}

    monkeypatch.setattr(aggregator, "compute_gas_props", fake_compute_gas_props)
    monkeypatch.setattr(aggregator, "compute_liquid_props", fake_compute_liquid_props)

    props, _ = aggregator.build_props_from_state(object(), grid, state, gas_model, liq_model)

    assert props.h_g is not None
    assert props.h_l is not None
    assert props.h_g.shape == state.Tg.shape
    assert props.h_l.shape == state.Tl.shape
    assert np.all(np.isfinite(props.h_g))
    assert np.all(np.isfinite(props.h_l))
    assert np.allclose(props.h_g, props.cp_g * state.Tg)
    assert np.allclose(props.h_l, props.cp_l * state.Tl)


def test_props_enthalpy_fields_liquid_disabled_ok(monkeypatch: pytest.MonkeyPatch):
    grid = _make_grid(Nl=1, Ng=2)
    Ns_g = 2
    state = _make_state(grid, Ns_g=Ns_g, Ns_l=1)
    gas_model = DummyGasModel(Ns_g)

    def fake_compute_gas_props(model, state_in, grid_in):
        Ng = grid_in.Ng
        core = {
            "rho_g": np.full(Ng, 1.0, dtype=np.float64),
            "cp_g": np.full(Ng, 900.0, dtype=np.float64),
            "k_g": np.full(Ng, 0.1, dtype=np.float64),
            "D_g": np.zeros((model.gas.n_species, Ng), dtype=np.float64),
        }
        return core, {}

    monkeypatch.setattr(aggregator, "compute_gas_props", fake_compute_gas_props)

    props, _ = aggregator.build_props_from_state(object(), grid, state, gas_model, liq_model=None)

    assert props.h_g is not None
    assert props.h_g.shape == state.Tg.shape
    assert np.allclose(props.h_g, props.cp_g * state.Tg)

    assert props.h_l is not None
    assert props.h_l.shape == state.Tl.shape
    assert np.all(props.h_l == 0.0)
    assert np.all(props.cp_l == 0.0)
    assert np.all(props.k_l == 0.0)
