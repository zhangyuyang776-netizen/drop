import logging
import numpy as np
import pytest

from assembly.build_species_system_SciPy import build_gas_species_system_global
from core.layout import build_layout
from core.types import CaseSpecies, Props, State
from tests.utils_layout import make_case_config, make_simple_grid

logger = logging.getLogger(__name__)


def _make_props(Ns: int, Ng: int) -> Props:
    rho_g = np.ones(Ng, dtype=float)
    cp_g = np.ones(Ng, dtype=float)
    k_g = np.ones(Ng, dtype=float)
    D_g = np.full((Ns, Ng), 1.0e-5, dtype=float)
    rho_l = np.ones(1, dtype=float)
    cp_l = np.ones(1, dtype=float)
    k_l = np.ones(1, dtype=float)
    return Props(rho_g=rho_g, cp_g=cp_g, k_g=k_g, D_g=D_g, rho_l=rho_l, cp_l=cp_l, k_l=k_l)


def test_species_global_diffusion_with_dirichlet_farfield():
    grid = make_simple_grid(Nl=1, Ng=3)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species_full=["FUEL", "N2"],
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": "FUEL"},
    )
    cfg = make_case_config(
        grid,
        species,
        solve_Tg=False,
        solve_Tl=False,
        solve_Yl=False,
        case_id="step9_species",
    )
    # Farfield: only N2 specified -> FUEL farfield = 0

    layout = build_layout(cfg, grid)
    Ns_full = len(species.gas_species_full)
    Ng = grid.Ng

    Yg = np.zeros((Ns_full, Ng), dtype=float)
    Yg[0, :] = np.array([1.0, 0.5, 0.0])  # FUEL
    Yg[1, :] = 1.0 - Yg[0, :]             # closure N2

    state = State(
        Tg=np.full(Ng, 300.0),
        Yg=Yg,
        Tl=np.full(grid.Nl, 300.0),
        Yl=np.ones((len(species.liq_species), grid.Nl), dtype=float),
        Ts=300.0,
        mpp=0.0,
        Rd=float(grid.r_f[grid.iface_f]),
    )
    props = _make_props(Ns_full, Ng)

    dt = 1.0e-3
    A, b, diag = build_gas_species_system_global(cfg, grid, layout, state, props, dt, eq_result=None, return_diag=True)

    assert A.shape == (layout.size, layout.size)
    assert b.shape == (layout.size,)
    assert diag["bc"]["outer"]["Y_far_preview"]["FUEL"] == 0.0

    # Solve dense system (only Yg block present)
    x = np.linalg.solve(A, b)
    Y_new = x.reshape(-1)

    # Outer cell pinned to farfield (0 for FUEL)
    row_last = layout.idx_Yg(0, Ng - 1)
    assert np.isclose(Y_new[row_last], 0.0)

    # Middle cell should smooth between neighbors (diffusion)
    row_mid = layout.idx_Yg(0, 1)
    assert Y_new[row_mid] <= max(Yg[0, 0], Yg[0, 2]) + 1e-6
    assert Y_new[row_mid] >= min(Yg[0, 0], Yg[0, 2]) - 1e-6
