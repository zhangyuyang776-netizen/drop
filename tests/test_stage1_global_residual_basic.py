import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly import residual_global  # noqa: E402
from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402
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


def test_global_residual_shape_and_diag(monkeypatch):
    cfg = make_cfg_base(include_mpp=True, include_Ts=True, include_Rd=True, solve_Yg=True)
    cfg.physics.solve_Tl = True
    cfg.physics.solve_Yl = False

    grid, layout, state, _ = build_min_problem(cfg)
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

    res, diag = residual_global.build_global_residual(u0, ctx)

    assert res.shape == u0.shape
    assert np.all(np.isfinite(res))
    assert diag["props"]["source"] == "state_guess"
    assert "Ts_energy" in diag.get("assembly", {})
    assert "evaporation" in diag.get("assembly", {})
    assert "Ts_guess" in diag
    assert "Rd_guess" in diag


def test_global_residual_sensitivity_on_interface_unknowns(monkeypatch):
    cfg = make_cfg_base(include_mpp=True, include_Ts=True, include_Rd=True, solve_Yg=True)
    cfg.physics.solve_Tl = True
    cfg.physics.solve_Yl = False

    grid, layout, state, _ = build_min_problem(cfg)
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
    res0, _ = residual_global.build_global_residual(u0, ctx)

    idx_Ts = layout.idx_Ts()
    idx_mpp = layout.idx_mpp()
    idx_Rd = layout.idx_Rd()

    for idx, delta in ((idx_Ts, 1.0e-3), (idx_mpp, 1.0e-6), (idx_Rd, 1.0e-6)):
        du = np.zeros_like(u0)
        du[idx] = delta
        res1, _ = residual_global.build_global_residual(u0 + du, ctx)
        assert abs(res1[idx] - res0[idx]) > 0.0
