import numpy as np
import pytest

from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import build_min_problem, make_cfg_base


def test_linear_system_not_singular_with_Yg_enabled():
    cfg = make_cfg_base(include_mpp=False, solve_Yg=True, species_convection=False)
    grid, layout, state, props = build_min_problem(cfg)

    # uniform non-zero condensable to avoid all-zero rows
    gas_species = cfg.species.gas_species
    cond_idx = gas_species.index(gas_species[0])
    bal_idx = gas_species.index(cfg.species.gas_balance_species)
    state.Yg[:, :] = 0.0
    state.Yg[cond_idx, :] = 0.2
    state.Yg[bal_idx, :] = 0.8
    cfg.initial.Yg = {gas_species[0]: 0.2, cfg.species.gas_balance_species: 0.8}

    res = advance_one_step_scipy(cfg, grid, layout, state, props, t=0.0)
    assert res.success
    assert res.diag.linear_converged
    assert res.diag.linear_rel_residual < 1e-8
    assert np.all(np.isfinite(res.state_new.Yg))
