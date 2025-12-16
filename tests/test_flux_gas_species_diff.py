import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import Props  # noqa: E402
from physics.flux_gas import compute_gas_diffusive_flux_Y  # noqa: E402
from tests.test_step6_scipy_transport import cfg_step6, grid_step6  # noqa: E402


def _make_dummy_props_for_species(Ng: int, Ns_g: int, rho_val: float = 1.0, D_val: float = 1e-5) -> Props:
    """Construct minimal Props with specified rho_g and D_g; other fields dummy."""
    rho_g = np.full(Ng, rho_val, dtype=np.float64)
    cp_g = np.ones(Ng, dtype=np.float64)
    k_g = np.ones(Ng, dtype=np.float64)
    D_g = np.full((Ns_g, Ng), D_val, dtype=np.float64)

    rho_l = np.zeros(1, dtype=np.float64)
    cp_l = np.zeros(1, dtype=np.float64)
    k_l = np.zeros(1, dtype=np.float64)
    D_l = None

    return Props(rho_g=rho_g, cp_g=cp_g, k_g=k_g, D_g=D_g, rho_l=rho_l, cp_l=cp_l, k_l=k_l, D_l=D_l)


def test_species_diffusion_zero_for_uniform_Y(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    Ns_g = 3
    props = _make_dummy_props_for_species(Ng=Ng, Ns_g=Ns_g, rho_val=1.0, D_val=1e-5)
    Yg = np.tile(np.array([[0.2], [0.3], [0.5]], dtype=np.float64), (1, Ng))  # shape (Ns_g, Ng)

    J = compute_gas_diffusive_flux_Y(cfg_step6, grid_step6, props, Yg)

    assert J.shape == (Ns_g, Nc + 1)
    assert np.allclose(J, 0.0, atol=1e-14)


def test_species_diffusion_linear_gradient_sign_and_magnitude(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Nc = grid_step6.Nc
    Ns_g = 1
    rho_val = 1.0
    D_val = 1.0
    props = _make_dummy_props_for_species(Ng=Ng, Ns_g=Ns_g, rho_val=rho_val, D_val=D_val)
    # Linear gradient: Y = [0.0, 0.5, 1.0]
    Yg = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)

    J = compute_gas_diffusive_flux_Y(cfg_step6, grid_step6, props, Yg)

    iface_f = grid_step6.iface_f
    f_out = grid_step6.Nc
    gas_start = grid_step6.gas_slice.start
    internal_faces = [f for f in range(gas_start + 1, f_out)]

    assert np.allclose(J[:, iface_f], 0.0)
    assert np.allclose(J[:, f_out], 0.0)

    vals = J[0, internal_faces]
    assert np.all(np.isfinite(vals))
    assert np.all(vals < 0.0)  # dY/dr > 0 => J = -rho D dY/dr < 0

    # check magnitude for first internal face (faces are uniform spacing in this grid)
    gas_start = grid_step6.gas_slice.start
    iL = gas_start + 0
    iR = gas_start + 1
    rL = grid_step6.r_c[iL]
    rR = grid_step6.r_c[iR]
    dr = rR - rL
    dY = 0.5  # 0.5 - 0.0
    dY_dr = dY / dr
    J_expected = -rho_val * D_val * dY_dr
    for f in internal_faces:
        assert np.isclose(J[0, f], J_expected, rtol=1e-12, atol=0.0)


def test_species_diffusion_raises_when_Dg_missing(cfg_step6, grid_step6):
    Ng = grid_step6.Ng
    Ns_g = 2
    rho_g = np.ones(Ng, dtype=np.float64)
    cp_g = np.ones(Ng, dtype=np.float64)
    k_g = np.ones(Ng, dtype=np.float64)

    rho_l = np.zeros(1, dtype=np.float64)
    cp_l = np.zeros(1, dtype=np.float64)
    k_l = np.zeros(1, dtype=np.float64)

    props_bad = Props(
        rho_g=rho_g,
        cp_g=cp_g,
        k_g=k_g,
        D_g=None,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
        D_l=None,
    )

    Yg = np.tile(np.array([[0.3], [0.7]], dtype=np.float64), (1, Ng))
    with pytest.raises(ValueError):
        _ = compute_gas_diffusive_flux_Y(cfg_step6, grid_step6, props_bad, Yg)
