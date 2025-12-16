import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import Props  # noqa: E402
from physics.flux_convective_gas import compute_gas_convective_flux_Y  # noqa: E402
from tests.test_step6_scipy_transport import cfg_step6, grid_step6  # noqa: E402


def make_props_with_rho(grid, Ng: int, rho_value: float = 1.0) -> Props:
    """Construct minimal Props with rho_g set; other fields dummy."""
    Nl = grid.Nl
    return Props(
        rho_g=np.full(Ng, rho_value, dtype=np.float64),
        cp_g=np.ones(Ng, dtype=np.float64),
        k_g=np.ones(Ng, dtype=np.float64),
        D_g=None,
        rho_l=np.zeros(Nl, dtype=np.float64),
        cp_l=np.zeros(Nl, dtype=np.float64),
        k_l=np.zeros(Nl, dtype=np.float64),
        D_l=None,
    )


def test_species_convective_flux_zero_when_u_zero(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    Ns_g = 1
    Yg = np.array([[0.2, 0.5, 0.9]], dtype=np.float64)  # shape (1, Ng)
    props = make_props_with_rho(grid_step6, Ng, rho_value=1.0)
    u_face = np.zeros(Nc + 1, dtype=np.float64)

    J_conv = compute_gas_convective_flux_Y(cfg_step6, grid_step6, props, Yg, u_face)

    assert J_conv.shape == (Ns_g, Nc + 1)
    assert np.allclose(J_conv, 0.0)


def test_species_convective_flux_positive_u_upwind_left_and_outer(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    iface_f = grid_step6.iface_f

    Ns_g = 1
    Yg = np.array([[1.0, 0.5, 0.0]], dtype=np.float64)
    props = make_props_with_rho(grid_step6, Ng, rho_value=1.0)

    u0 = 2.0
    u_face = np.full(Nc + 1, u0, dtype=np.float64)

    J_conv = compute_gas_convective_flux_Y(cfg_step6, grid_step6, props, Yg, u_face)

    assert J_conv.shape == (Ns_g, Nc + 1)
    assert np.isclose(J_conv[0, iface_f], 0.0)
    assert np.isclose(J_conv[0, 2], 2.0)   # rho=1, u=2, Y_up(left)=1.0
    assert np.isclose(J_conv[0, 3], 1.0)   # rho=1, u=2, Y_up(left)=0.5
    assert np.isclose(J_conv[0, Nc], 0.0)  # uses last cell Y=0.0


def test_species_convective_flux_negative_u_upwind_right(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    iface_f = grid_step6.iface_f

    Ns_g = 1
    Yg = np.array([[1.0, 0.5, 0.0]], dtype=np.float64)
    props = make_props_with_rho(grid_step6, Ng, rho_value=1.0)

    u0 = -2.0
    u_face = np.full(Nc + 1, u0, dtype=np.float64)

    J_conv = compute_gas_convective_flux_Y(cfg_step6, grid_step6, props, Yg, u_face)

    assert J_conv.shape == (Ns_g, Nc + 1)
    assert np.isclose(J_conv[0, iface_f], 0.0)
    # face 2 (between gas0/gas1): upwind = right cell (Y=0.5)
    assert np.isclose(J_conv[0, 2], -1.0)  # 1 * (-2) * 0.5
    # face 3 (between gas1/gas2): upwind = right cell (Y=0.0)
    assert np.isclose(J_conv[0, 3], 0.0)
    # outer boundary: uses last cell Y=0.0
    assert np.isclose(J_conv[0, Nc], 0.0)


def test_species_convective_flux_raises_on_bad_shape(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    Ns_g = 1
    # bad Yg: missing one cell
    Yg_bad = np.array([[1.0, 0.5]], dtype=np.float64)
    props = make_props_with_rho(grid_step6, Ng, rho_value=1.0)
    u_face = np.zeros(Nc + 1, dtype=np.float64)

    with pytest.raises(ValueError):
        compute_gas_convective_flux_Y(cfg_step6, grid_step6, props, Yg_bad, u_face)
