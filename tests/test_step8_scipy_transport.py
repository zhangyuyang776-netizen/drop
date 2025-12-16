import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import Grid1D, State  # noqa: E402
from properties.aggregator import build_props_from_state  # noqa: E402
from properties.gas import build_gas_model  # noqa: E402
from properties.liquid import build_liquid_model  # noqa: E402
from assembly.build_system_SciPy import build_transport_system  # noqa: E402
from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402
from physics.stefan_velocity import compute_stefan_velocity  # noqa: E402
from physics.flux_convective_gas import compute_gas_convective_flux_T  # noqa: E402
from tests.test_step6_scipy_transport import (  # noqa: E402
    cfg_step6,
    grid_step6,
    gas_model,
    liq_model,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def state_step6(cfg_step6, grid_step6, gas_model) -> State:
    """Baseline state used for Step 6/8 tests."""
    Ng = grid_step6.Ng
    Ns = gas_model.gas.n_species

    Tg = np.array([700.0, 500.0, 300.0], dtype=np.float64)
    assert Tg.shape == (Ng,)

    Yg = np.zeros((Ns, Ng), dtype=np.float64)
    kN2 = gas_model.gas.species_index("N2")
    Yg[kN2, :] = 1.0

    Tl = np.array([300.0], dtype=np.float64)
    Yl = np.ones((1, grid_step6.Nl), dtype=np.float64)

    return State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=1.0e-4,
    )


class DummyLayout:
    """Minimal Tg-only layout."""

    def __init__(self, Ng: int):
        self._Ng = Ng

    def has_block(self, name: str) -> bool:
        return name == "Tg"

    def idx_Tg(self, ig: int) -> int:
        return ig

    def n_dof(self) -> int:
        return self._Ng


def test_stefan_velocity_basic_sanity(cfg_step6, grid_step6, gas_model, state_step6):
    """Check Stefan velocity returns zeros for mpp=0 and positive outward for mpp>0."""
    # mpp = 0
    state_zero = replace(state_step6, mpp=0.0)
    props_zero, _ = build_props_from_state(cfg_step6, grid_step6, state_zero, gas_model, liq_model=None)
    vel_zero = compute_stefan_velocity(cfg_step6, grid_step6, props_zero, state_zero)
    assert vel_zero.u_face.shape == (grid_step6.Nc + 1,)
    assert vel_zero.u_cell.shape == (grid_step6.Nc,)
    assert np.allclose(vel_zero.u_face, 0.0)
    assert np.allclose(vel_zero.u_cell, 0.0)

    # mpp > 0
    mpp_val = 5e-4
    state_evap = replace(state_step6, mpp=mpp_val)
    props_evap, _ = build_props_from_state(cfg_step6, grid_step6, state_evap, gas_model, liq_model=None)
    vel_evap = compute_stefan_velocity(cfg_step6, grid_step6, props_evap, state_evap)

    gas_start = grid_step6.gas_slice.start
    iface_f = grid_step6.iface_f

    assert np.allclose(vel_evap.u_cell[:gas_start], 0.0)
    assert np.all(vel_evap.u_cell[gas_start:] > 0.0)
    assert vel_evap.u_face[iface_f] > 0.0
    assert vel_evap.u_face[-1] > 0.0
    u_gas_faces = vel_evap.u_face[iface_f + 1 :]
    assert np.all(u_gas_faces[1:] <= u_gas_faces[:-1] + 1e-10)

    logger.info("=== Stefan velocity sanity check ===")
    logger.info("mpp = %.3e, Rd = %.3e", state_evap.mpp, state_evap.Rd)
    logger.info("u_face = %s", np.array2string(vel_evap.u_face, precision=4, separator=", "))
    logger.info("u_cell = %s", np.array2string(vel_evap.u_cell, precision=4, separator=", "))


def test_step8_scipy_Tg_diffusion_with_Stefan_convection(cfg_step6, grid_step6, gas_model, state_step6):
    """Closed loop: props -> assembly (with explicit convection) -> SciPy solve."""
    dt = cfg_step6.time.dt
    layout = DummyLayout(grid_step6.Ng)
    # two states: pure diffusion vs diffusion+convection
    state_diff = replace(state_step6, mpp=0.0)
    state_conv = replace(state_step6, mpp=5e-4)

    props_diff, _ = build_props_from_state(cfg_step6, grid_step6, state_diff, gas_model, liq_model=None)
    props_conv, _ = build_props_from_state(cfg_step6, grid_step6, state_conv, gas_model, liq_model=None)

    # assemble + solve: pure diffusion
    A_diff, b_diff = build_transport_system(cfg_step6, grid_step6, layout, state_diff, props_diff, dt)
    res_diff = solve_linear_system_scipy(A_diff, b_diff, cfg_step6, method="direct")
    assert res_diff.converged
    Tg_new_diff = np.array([res_diff.x[layout.idx_Tg(ig)] for ig in range(grid_step6.Ng)])
    state_new_diff = replace(state_diff, Tg=Tg_new_diff)

    # assemble + solve: diffusion + Stefan convection (explicit)
    A_conv, b_conv = build_transport_system(cfg_step6, grid_step6, layout, state_conv, props_conv, dt)
    res_conv = solve_linear_system_scipy(A_conv, b_conv, cfg_step6, method="direct")
    assert res_conv.converged
    Tg_new_conv = np.array([res_conv.x[layout.idx_Tg(ig)] for ig in range(grid_step6.Ng)])
    state_new_conv = replace(state_conv, Tg=Tg_new_conv)

    T_inf = float(cfg_step6.initial.T_inf)
    # boundary cell pinned to T_inf in both cases
    assert abs(state_new_diff.Tg[-1] - T_inf) < 1e-8
    assert abs(state_new_conv.Tg[-1] - T_inf) < 1e-8
    assert abs(state_new_conv.Tg[-1] - state_new_diff.Tg[-1]) < 1e-10

    # at least one cell changed due to convection
    assert np.any(state_new_conv.Tg != state_new_diff.Tg)

    Tg_diff = state_new_diff.Tg
    Tg_conv = state_new_conv.Tg

    # Trend expectations: inner cell cooled more, middle cell warmed slightly (outward convection)
    assert Tg_conv[0] < Tg_diff[0] - 1e-10
    assert Tg_conv[1] > Tg_diff[1] + 1e-10

    # logging table
    vel_conv = compute_stefan_velocity(cfg_step6, grid_step6, props_conv, state_conv)
    u_face = vel_conv.u_face
    q_conv = compute_gas_convective_flux_T(cfg_step6, grid_step6, props_conv, Tg=state_conv.Tg, u_face=u_face)

    gas_start = grid_step6.gas_slice.start
    r_c = grid_step6.r_c
    r_f = grid_step6.r_f

    logger.info("===== Step 8 SciPy: Tg diffusion + Stefan convection =====")
    logger.info(
        "Ng = %d, dt = %.3e s, T_inf = %.2f K, mpp = %.3e kg/m^2/s",
        grid_step6.Ng,
        dt,
        T_inf,
        state_conv.mpp,
    )
    header = " ig |   r_c [m]  |  Tg_old [K] | T_diff [K] | T_conv [K] |  dT_diff |  dT_conv"
    logger.info(header)
    logger.info("-" * len(header))
    for ig in range(grid_step6.Ng):
        cell_idx = gas_start + ig
        logger.info(
            "%3d | %10.4e | %11.4f | %10.4f | %10.4f | % .3e | % .3e",
            ig,
            r_c[cell_idx],
            state_step6.Tg[ig],
            Tg_diff[ig],
            Tg_conv[ig],
            Tg_diff[ig] - state_step6.Tg[ig],
            Tg_conv[ig] - state_step6.Tg[ig],
        )

    logger.info("Tg_old  = %s", np.array2string(state_step6.Tg, precision=4, separator=", "))
    logger.info("Tg_diff = %s", np.array2string(Tg_diff, precision=4, separator=", "))
    logger.info("Tg_conv = %s", np.array2string(Tg_conv, precision=4, separator=", "))
    logger.info("u_face (Stefan) = %s", np.array2string(u_face, precision=4, separator=", "))
    logger.info("q_conv (faces)  = %s", np.array2string(q_conv, precision=4, separator=", "))
    logger.info("r_f (faces)     = %s", np.array2string(r_f, precision=4, separator=", "))
