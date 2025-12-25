import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly import residual_global  # noqa: E402
from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402
from solvers.newton_scipy import solve_nonlinear_scipy  # noqa: E402
from tests._helpers_step15 import build_min_problem, make_cfg_base, make_props_const  # noqa: E402


def _patch_props_and_eq(monkeypatch, props):
    monkeypatch.setattr(residual_global, "compute_props", lambda cfg, grid, state: (props, {}))
    monkeypatch.setattr(residual_global, "_get_or_build_eq_model", lambda ctx, state: object())

    def _fake_equilibrium(eq_model, Ts, Pg, Yl_face, Yg_face):
        Yg_eq = np.asarray(Yg_face, dtype=np.float64)
        y_cond = np.zeros_like(Yl_face, dtype=np.float64)
        psat = np.zeros_like(Yl_face, dtype=np.float64)
        return Yg_eq, y_cond, psat

    monkeypatch.setattr(residual_global, "compute_interface_equilibrium", _fake_equilibrium)


def test_interface_no_evaporation_energy_balance(monkeypatch):
    cfg = make_cfg_base(include_mpp=True, include_Ts=True, include_Rd=False, solve_Yg=False)
    cfg.physics.solve_Tl = True
    cfg.physics.solve_Yl = False
    cfg.nonlinear.enabled = True
    cfg.nonlinear.solver = "root_hybr"

    grid, layout, state, _ = build_min_problem(cfg)
    state.Tg[:] = 800.0
    state.Tl[:] = 300.0
    state.Ts = 500.0
    state.mpp = 0.0

    Ns_g = len(cfg.species.gas_species_full)
    Ns_l = len(cfg.species.liq_species)
    props = make_props_const(Ns_g, Ns_l, grid)

    _patch_props_and_eq(monkeypatch, props)

    ctx, u0 = build_nonlinear_context_for_step(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state,
        props_old=props,
        t_old=0.0,
        dt=cfg.time.dt,
    )

    result = solve_nonlinear_scipy(ctx, u0)
    assert result.diag.converged

    state_new = ctx.make_state_from_u(result.u)
    Tmin = min(state_new.Tg[0], state_new.Tl[-1]) - 1.0e-8
    Tmax = max(state_new.Tg[0], state_new.Tl[-1]) + 1.0e-8
    assert Tmin <= state_new.Ts <= Tmax
    assert abs(state_new.mpp) < 1.0e-10

    _, diag = residual_global.build_global_residual(result.u, ctx)
    ts_diag = diag.get("assembly", {}).get("Ts_energy", {})
    balance = ts_diag.get("balance_into_interface", {}).get("balance_eq", np.nan)
    assert np.isfinite(balance)
    assert abs(balance) < 1.0e-6
