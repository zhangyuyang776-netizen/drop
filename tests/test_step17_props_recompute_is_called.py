import numpy as np
import pytest

pytest.importorskip("scipy")

from solvers import timestepper
from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import build_min_problem, make_cfg_base, make_props_const


def test_step17_props_recompute_is_called(monkeypatch):
    cfg = make_cfg_base(
        include_mpp=False,
        solve_Yg=False,
        species_convection=False,
        include_Ts=False,
        include_Rd=False,
    )
    grid, layout, state, _ = build_min_problem(cfg)

    Ns_g = len(cfg.species.gas_species_full)
    Ns_l = len(cfg.species.liq_species)
    props0 = make_props_const(Ns_g, Ns_l, grid, D_g_val=1.0e-5)

    calls = {"n": 0}

    # Marked props to ensure passthrough
    props_mark = make_props_const(Ns_g, Ns_l, grid, D_g_val=1.0e-5)
    props_mark.cp_g = props_mark.cp_g + 123.0
    fake_extras = {"gas": {"tag": "step17"}, "liquid": {"tag": "step17"}}

    def fake_compute_props(cfg_in, grid_in, state_in):
        calls["n"] += 1
        return props_mark, fake_extras

    # Patch the function used inside timestepper (imported binding)
    monkeypatch.setattr(timestepper, "compute_props", fake_compute_props)

    res = advance_one_step_scipy(cfg, grid, layout, state, props0, t=0.0)

    assert calls["n"] == 1, "compute_props should be called exactly once per step"
    assert res.success, res.message
    assert res.props_new is props_mark
    assert np.allclose(res.props_new.cp_g, props0.cp_g + 123.0)
