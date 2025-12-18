import numpy as np
import pytest

pytest.importorskip("scipy")

import solvers.timestepper as timestepper
from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import build_min_problem, make_cfg_base, make_props_const


def _fake_eq_result_sum1(eq_value: float, Ns: int, k_cond: int, k_bal: int):
    """Return Yg_eq with sum=1, condensable=eq_value, balance=1-eq_value."""
    Yg_eq = np.zeros(Ns, dtype=np.float64)
    Yg_eq[k_cond] = float(eq_value)
    Yg_eq[k_bal] = float(1.0 - eq_value)
    return {"Yg_eq": Yg_eq}


def test_evap_end_to_end_smoke(monkeypatch):
    # Minimal evap chain: solve_Yg + mpp + Rd shrink (no Ts unknown)
    cfg = make_cfg_base(
        include_mpp=True,
        solve_Yg=True,
        species_convection=False,
        include_Ts=False,
        include_Rd=True,
    )

    grid, layout, state, _ = build_min_problem(cfg)
    Ns_g = len(cfg.species.gas_species)
    Ns_l = len(cfg.species.liq_species)

    # Constant props for determinism (avoid Cantera/CoolProp in this smoke test)
    props_const = make_props_const(Ns_g, Ns_l, grid, D_g_val=1.0e-5)
    monkeypatch.setattr(
        timestepper,
        "compute_props",
        lambda *args, **kwargs: (props_const, {"patched": True}),
    )

    gas_species = cfg.species.gas_species
    bal_name = cfg.species.gas_balance_species
    bal_idx = gas_species.index(bal_name)

    cond_name = cfg.physics.interface.equilibrium.condensables_gas[0]
    cond_idx = gas_species.index(cond_name)

    # Init: condensable=0, balance=1
    state.Yg[:, :] = 0.0
    state.Yg[bal_idx, :] = 1.0
    cfg.initial.Yg = {cond_name: 0.0, bal_name: 1.0}

    eq_val = 0.05
    monkeypatch.setattr(
        timestepper,
        "_build_eq_result_for_step",
        lambda cfg, grid, state, props: _fake_eq_result_sum1(eq_val, Ns_g, cond_idx, bal_idx),
    )

    t = 0.0
    state_now = state
    props_now = props_const

    mpps = []
    rds = []
    y0s = []

    for _ in range(12):
        res = advance_one_step_scipy(cfg, grid, layout, state_now, props_now, t=t)
        assert res.success, res.message

        state_now = res.state_new
        props_now = res.props_new
        t = res.diag.t_new

        mpps.append(float(state_now.mpp))
        rds.append(float(state_now.Rd))
        y0s.append(float(state_now.Yg[cond_idx, 0]))

        s0 = float(np.sum(state_now.Yg[:, 0]))
        assert abs(s0 - 1.0) < 1e-10
        assert np.min(state_now.Yg) >= -1e-12
        assert np.max(state_now.Yg) <= 1.0 + 1e-12

        assert np.isfinite(state_now.mpp)
        assert np.isfinite(state_now.Rd) and state_now.Rd > 0.0

    # Evap should happen
    assert max(mpps) > 0.0

    # Radius should not grow when include_Rd=True (allow tiny numerical noise)
    for i in range(1, len(rds)):
        assert rds[i] <= rds[i - 1] + 1e-14

    # Condensable near interface should increase from 0
    assert y0s[-1] > y0s[0] + 1e-12
