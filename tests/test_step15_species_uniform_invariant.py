import numpy as np
import pytest

from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import build_min_problem, make_cfg_base


def test_species_uniform_stays_constant():
    cfg = make_cfg_base(
        include_mpp=False,
        include_Ts=False,
        include_Rd=False,
        solve_Yg=True,
        species_convection=False,
    )
    grid, layout, state, props = build_min_problem(cfg)

    # Make solved species uniform and consistent with farfield
    gas_species = cfg.species.gas_species
    cond_name = gas_species[0]
    bal_name = cfg.species.gas_balance_species
    cond_idx = gas_species.index(cond_name)
    bal_idx = gas_species.index(bal_name)
    state.Yg[:, :] = 0.0
    state.Yg[cond_idx, :] = 0.1
    state.Yg[bal_idx, :] = 0.9
    # farfield matches interior
    cfg.initial.Yg = {cond_name: 0.1, bal_name: 0.9}

    res = advance_one_step_scipy(cfg, grid, layout, state, props, t=0.0)
    assert res.success
    state_new = res.state_new
    Y_old = state.Yg[cond_idx, :].copy()
    Y_new = state_new.Yg[cond_idx, :]

    assert np.all(np.isfinite(Y_new))
    assert np.max(np.abs(Y_new - Y_old)) < 1e-12
    assert np.min(Y_new) >= -1e-12
    assert np.max(Y_new) <= 1.0 + 1e-12
