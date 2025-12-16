import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_species_system_SciPy import build_species_system  # noqa: E402
from properties.aggregator import build_props_from_state  # noqa: E402
from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402
from tests.test_step6_scipy_transport import (  # noqa: E402
    cfg_step6,
    grid_step6,
    gas_model,
    state_step6,
)

logger = logging.getLogger(__name__)


class DummySpeciesLayout:
    def __init__(self, Ng: int, k_spec: int):
        self.Ng = Ng
        self.k_spec = k_spec

    def n_dof(self) -> int:
        return self.Ng

    def has_block(self, name: str) -> bool:
        return name == "Yg"

    def idx_Yg(self, k: int, ig: int) -> int:
        if k != self.k_spec:
            raise ValueError(f"Dummy layout only supports species {self.k_spec}, got k={k}")
        if not (0 <= ig < self.Ng):
            raise IndexError(f"ig={ig} out of range for Ng={self.Ng}")
        return ig


def make_state_with_Y_gradient(state_step6):
    Ns_g, Ng = state_step6.Yg.shape
    Yg = np.zeros_like(state_step6.Yg)

    # Only use first two species to keep sum(Y)=1
    # cell:   ig=0   ig=1   ig=2
    # Y0:    1.0    0.5    0.0
    # Y1:    0.0    0.5    1.0
    Y0 = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    Y1 = 1.0 - Y0
    Yg[0, :] = Y0
    Yg[1, :] = Y1
    # others remain zero

    return replace(state_step6, Yg=Yg, mpp=0.0)


def test_step9_scipy_single_species_closed_loop(cfg_step6, grid_step6, state_step6, gas_model, caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    state_grad = make_state_with_Y_gradient(state_step6)
    Ns_g, Ng = state_grad.Yg.shape
    k_spec = 0  # test first species

    # properties (includes D_g)
    props, extras = build_props_from_state(cfg_step6, grid_step6, state_grad, gas_model, liq_model=None)
    assert props.D_g is not None
    assert props.D_g.shape == (Ns_g, Ng)
    assert props.rho_g.shape == (Ng,)

    # assemble species system
    layout = DummySpeciesLayout(Ng=Ng, k_spec=k_spec)
    dt = cfg_step6.time.dt
    A, b = build_species_system(cfg_step6, grid_step6, layout, state_grad, props, dt, k_spec)

    assert A.shape == (Ng, Ng)
    assert b.shape == (Ng,)
    assert np.all(np.diag(A) > 0.0)

    # solve
    result = solve_linear_system_scipy(A, b, cfg_step6, method="direct")
    assert result.converged

    Y_old = state_grad.Yg[k_spec, :].copy()
    Y_new = result.x
    assert Y_new.shape == (Ng,)

    logger.info("=== Step 9 species closed-loop (SciPy) ===")
    logger.info("dt = %.3e", dt)
    logger.info("Y_old = %s", np.array2string(Y_old, precision=4, separator=", "))
    logger.info("Y_new = %s", np.array2string(Y_new, precision=4, separator=", "))
    logger.info("dY    = %s", np.array2string(Y_new - Y_old, precision=4, separator=", "))

    # boundary condition: Y_k at outer boundary pinned to 0
    assert abs(Y_new[-1] - 0.0) < 1e-8

    # middle cell should remain between neighbors (diffusive smoothing, mpp=0)
    assert Y_new[1] <= max(Y_old[0], Y_old[2]) + 1e-6
    assert Y_new[1] >= min(Y_old[0], Y_old[2]) - 1e-6

    # some change should occur
    assert np.any(np.abs(Y_new - Y_old) > 1e-12)
