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


def test_pure_gas_nonlinear_converges(monkeypatch):
    cfg = make_cfg_base(include_mpp=False, include_Ts=False, include_Rd=False, solve_Yg=False)
    cfg.physics.solve_Tl = False
    cfg.physics.solve_Yl = False
    cfg.physics.interface.bc_mode = "Ts_fixed"
    cfg.physics.interface.Ts_fixed = 300.0
    cfg.initial.T_inf = 1000.0
    cfg.nonlinear.enabled = True
    cfg.nonlinear.solver = "root_hybr"

    grid, layout, state, _ = build_min_problem(cfg)
    state.Tg[:] = 300.0
    state.Ts = 300.0
    state.mpp = 0.0

    Ns_g = len(cfg.species.gas_species_full)
    Ns_l = len(cfg.species.liq_species)
    props = make_props_const(Ns_g, Ns_l, grid)

    monkeypatch.setattr(residual_global, "compute_props", lambda cfg, grid, st: (props, {}))

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

    res = residual_global.residual_only(result.u, ctx)
    assert np.linalg.norm(res, ord=np.inf) < 1.0e-8

    state_new = ctx.make_state_from_u(result.u)
    Tmin = min(cfg.physics.interface.Ts_fixed, cfg.initial.T_inf) - 1.0e-8
    Tmax = max(cfg.physics.interface.Ts_fixed, cfg.initial.T_inf) + 1.0e-8
    assert np.all(state_new.Tg >= Tmin)
    assert np.all(state_new.Tg <= Tmax)
