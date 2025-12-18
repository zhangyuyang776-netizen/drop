import numpy as np
import pytest

from core.grid import build_grid
from core.layout import build_layout
from solvers.timestepper import advance_one_step_scipy
from tests._helpers_step15 import make_cfg_base, make_props_const, make_state_uniform


def _make_case(include_Ts: bool, Ts_value: float, Tg0: float = 300.0):
    cfg = make_cfg_base(
        Nl=1,
        Ng=3,
        include_mpp=False,
        include_Ts=include_Ts,
        include_Rd=False,
        solve_Yg=False,
        species_convection=False,
    )
    cfg.physics.solve_Tl = False
    cfg.physics.interface.bc_mode = "Ts_fixed"
    cfg.physics.interface.Ts_fixed = Ts_value
    cfg.time.dt = 1.0e-6

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)

    gas_species = cfg.species.gas_species
    Ns_g = len(gas_species)
    Yg = np.zeros((Ns_g, grid.Ng), dtype=np.float64)
    state = make_state_uniform(cfg, grid, gas_species, Yg)
    state.Tg[:] = Tg0
    state.Ts = Ts_value

    props = make_props_const(Ns_g, len(cfg.species.liq_species), grid)
    props.k_g[:] = 1.0
    props.rho_g[:] = 1.0
    props.cp_g[:] = 1.0
    return cfg, grid, layout, state, props


def test_Tg_Ts_interface_conduction_fixed_point():
    cfg, grid, layout, state, props = _make_case(include_Ts=False, Ts_value=300.0, Tg0=300.0)
    res = advance_one_step_scipy(cfg, grid, layout, state, props, t=0.0)
    assert res.success
    assert np.max(np.abs(res.state_new.Tg - state.Tg)) < 1e-12
    assert np.all(np.isfinite(res.state_new.Tg))
    assert abs(res.state_new.Ts - 300.0) < 1e-12


def test_Tg0_moves_toward_hot_Ts_fixed():
    cfg, grid, layout, state, props = _make_case(include_Ts=False, Ts_value=600.0, Tg0=300.0)
    res = advance_one_step_scipy(cfg, grid, layout, state, props, t=0.0)
    assert res.success

    dT = res.state_new.Tg - state.Tg
    assert res.state_new.Tg[0] > state.Tg[0]
    assert np.all(np.isfinite(res.state_new.Tg))
    assert dT[0] >= dT[1] - 1e-14


def test_Tg0_Ts_coupling_matrix_entry():
    # Structural check: Tg0 couples to Ts as an unknown (requires include_mpp for Ts equation)
    cfg = make_cfg_base(
        Nl=1,
        Ng=3,
        include_mpp=True,
        include_Ts=True,
        include_Rd=False,
        solve_Yg=False,
        species_convection=False,
    )
    cfg.physics.solve_Tl = False
    cfg.physics.interface.bc_mode = "Ts_fixed"
    cfg.physics.interface.Ts_fixed = 500.0
    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)

    gas_species = cfg.species.gas_species
    Ns_g = len(gas_species)
    Yg = np.zeros((Ns_g, grid.Ng), dtype=np.float64)
    state = make_state_uniform(cfg, grid, gas_species, Yg)
    state.Tg[:] = 300.0
    state.Ts = 500.0

    props = make_props_const(Ns_g, len(cfg.species.liq_species), grid)
    props.k_g[:] = 1.0
    props.rho_g[:] = 1.0
    props.cp_g[:] = 1.0

    from assembly.build_system_SciPy import build_transport_system

    A, b, diag = build_transport_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state,
        props=props,
        dt=cfg.time.dt,
        eq_result={"Yg_eq": np.zeros(len(cfg.species.gas_species))},
        state_guess=state,
        return_diag=True,
    )

    row_tg0 = layout.idx_Tg(0)
    col_ts = layout.idx_Ts()
    iface_f = grid.iface_f
    gas_start = grid.gas_slice.start if grid.gas_slice is not None else grid.Nl
    cell_idx = gas_start + 0
    A_if = float(grid.A_f[iface_f])
    dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
    k_face = float(props.k_g[0])
    coeff_if = k_face * A_if / dr_if

    assert np.isclose(A[row_tg0, col_ts], -coeff_if, rtol=0.0, atol=1e-12)
    assert A[row_tg0, row_tg0] > coeff_if
    assert diag.get("gas", {}).get("Tg_interface_coupling", {}).get("coeff_if", 0.0) > 0.0
