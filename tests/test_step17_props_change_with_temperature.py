import numpy as np

import properties.compute_props as cp_mod
from core.types import State
from properties.compute_props import compute_props
from tests._helpers_step15 import build_min_problem, make_cfg_base, make_props_const


def test_step17_props_change_with_temperature(monkeypatch):
    cfg = make_cfg_base(
        include_mpp=False,
        solve_Yg=True,
        species_convection=False,
        include_Ts=False,
        include_Rd=False,
    )
    grid, layout, state, _ = build_min_problem(cfg)

    # Stub models to avoid external dependencies
    monkeypatch.setattr(cp_mod, "get_or_build_models", lambda cfg_in: (None, None))

    def fake_build_props_from_state(cfg_in=None, grid_in=None, state_in=None, gas_model=None, liq_model=None, **kwargs):
        cfg_in = cfg_in or kwargs.get("cfg")
        grid_in = grid_in or kwargs.get("grid")
        state_in = state_in or kwargs.get("state")
        Ns_g = state_in.Yg.shape[0]
        Ns_l = state_in.Yl.shape[0]
        base = make_props_const(Ns_g, Ns_l, grid_in)
        scale = float(np.mean(state_in.Tg))
        base.cp_g = np.full_like(base.cp_g, scale)
        base.k_g = np.full_like(base.k_g, 0.1 * scale)
        extras = {"tag": "fake_props", "scale": scale}
        return base, extras

    monkeypatch.setattr(cp_mod, "build_props_from_state", fake_build_props_from_state)

    props1, _ = compute_props(cfg, grid, state)

    state2 = State(
        Tg=state.Tg + 50.0,
        Yg=state.Yg.copy(),
        Tl=state.Tl.copy(),
        Yl=state.Yl.copy(),
        Ts=state.Ts,
        mpp=state.mpp,
        Rd=state.Rd,
    )

    props2, _ = compute_props(cfg, grid, state2)

    cp_same = np.allclose(props1.cp_g, props2.cp_g, rtol=1e-12, atol=1e-12)
    k_same = np.allclose(props1.k_g, props2.k_g, rtol=1e-12, atol=1e-12)

    assert (not cp_same) or (not k_same), "cp_g or k_g should change when Tg changes"
