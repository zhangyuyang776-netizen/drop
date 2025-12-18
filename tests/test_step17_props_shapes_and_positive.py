import numpy as np

import properties.compute_props as cp_mod
from properties.compute_props import compute_props
from tests._helpers_step15 import build_min_problem, make_cfg_base, make_props_const


def test_step17_props_shapes_and_positive(monkeypatch):
    cfg = make_cfg_base(include_mpp=False, solve_Yg=True, include_Ts=False, include_Rd=False)
    grid, layout, state, _ = build_min_problem(cfg)

    # Stub out model build and property computation to avoid external deps
    monkeypatch.setattr(cp_mod, "get_or_build_models", lambda cfg_in: (None, None))

    def fake_build_props_from_state(cfg_in=None, grid_in=None, state_in=None, gas_model=None, liq_model=None, **kwargs):
        cfg_in = cfg_in or kwargs.get("cfg")
        grid_in = grid_in or kwargs.get("grid")
        state_in = state_in or kwargs.get("state")
        Ns_g = state_in.Yg.shape[0]
        Ns_l = state_in.Yl.shape[0]
        props = make_props_const(Ns_g, Ns_l, grid_in)
        return props, {"tag": "shapes"}

    monkeypatch.setattr(cp_mod, "build_props_from_state", fake_build_props_from_state)

    props, _ = compute_props(cfg, grid, state)

    Ns_g = state.Yg.shape[0]
    Ns_l = state.Yl.shape[0]
    props.validate_shapes(grid, Ns_g=Ns_g, Ns_l=Ns_l)

    assert np.all(props.rho_g > 0)
    assert np.all(props.cp_g > 0)
    assert np.all(props.k_g > 0)
    assert np.all(props.D_g >= 0)

    assert np.all(props.rho_l > 0)
    assert np.all(props.cp_l > 0)
    assert np.all(props.k_l > 0)
    assert np.all(props.D_l >= 0)

    if getattr(props, "h_gk", None) is not None:
        assert props.h_gk.shape == (Ns_g, grid.Ng)
