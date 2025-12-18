import numpy as np
import pytest

import solvers.timestepper as timestepper
from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import make_cfg_base, build_min_problem, make_props_const, make_state_uniform


def _fake_eq_result_with_cond(eq_value: float, Ns: int, k_cond_full: int):
    Yg_eq = np.zeros(Ns, dtype=np.float64)
    Yg_eq[k_cond_full] = eq_value
    if k_cond_full < Ns - 1:
        Yg_eq[k_cond_full + 1 :] = 0.0
    return {"Yg_eq": Yg_eq}


def test_species_diffusion_driven_by_interface_bc(monkeypatch):
    cfg = make_cfg_base(include_mpp=True, solve_Yg=True, species_convection=False, include_Ts=False, include_Rd=True,)
    # Species: ["FUEL", "N2"], closure=N2, condensable=FUEL
    grid, layout, state, _ = build_min_problem(cfg)
    Ns_g = len(cfg.species.gas_species)
    Ns_l = len(cfg.species.liq_species)
    props = make_props_const(Ns_g, Ns_l, grid, D_g_val=1.0e-5)

    gas_species = cfg.species.gas_species
    cond_name = gas_species[0]
    bal_name = cfg.species.gas_balance_species
    cond_idx = gas_species.index(cond_name)
    bal_idx = gas_species.index(bal_name)

    # Initial Yg: condensable zero everywhere, closure ~1
    state.Yg[:, :] = 0.0
    state.Yg[bal_idx, :] = 1.0
    cfg.initial.Yg = {cond_name: 0.0, bal_name: 1.0}
    assert cond_name in cfg.physics.interface.equilibrium.condensables_gas

    # Monkeypatch eq_result to enforce interface condensable fraction
    eq_val = 0.05
    monkeypatch.setattr(
        timestepper,
        "_build_eq_result_for_step",
        lambda cfg, grid, state, props: _fake_eq_result_with_cond(eq_val, Ns_g, cond_idx),
    )

    # March a few steps
    res = None
    t = 0.0
    for _ in range(8):
        res = advance_one_step_scipy(cfg, grid, layout, state, props, t=t)
        assert res.success
        state = res.state_new
        t = res.diag.t_new

    assert res is not None
    state_new = res.state_new

    Y_if = float(state_new.Yg[cond_idx, 0])
    assert Y_if > 0.0
    assert Y_if > float(0.0) + 1e-12

    Y_solved = state_new.Yg[cond_idx, :]
    assert np.min(Y_solved) >= -1e-12
    assert np.max(Y_solved) <= 1.0 + 1e-12

    assert state_new.mpp > 0.0
    assert state_new.Rd < cfg.geometry.a0
