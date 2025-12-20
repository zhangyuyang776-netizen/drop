import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.layout import build_layout, apply_u_to_state  # noqa: E402
from core.types import State  # noqa: E402


def _make_cfg(*, solve_Yl: bool):
    species = SimpleNamespace(
        gas_species_full=["N2"],
        gas_balance_species="N2",
        liq_species=["A_l", "B_l", "C_l"],
        liq_balance_species="C_l",
        liq2gas_map={"A_l": "A", "B_l": "B", "C_l": "C"},
    )
    physics = SimpleNamespace(
        solve_Tg=False,
        solve_Yg=False,
        solve_Tl=False,
        solve_Yl=solve_Yl,
        include_Ts=False,
        include_mpp=False,
        include_Rd=False,
    )
    return SimpleNamespace(species=species, physics=physics)


def test_solve_Yl_layout_size_and_closure_reconstruction():
    Nl = 4
    grid = SimpleNamespace(Ng=0, Nl=Nl)

    cfg_off = _make_cfg(solve_Yl=False)
    cfg_on = _make_cfg(solve_Yl=True)

    layout_off = build_layout(cfg_off, grid)
    layout_on = build_layout(cfg_on, grid)

    Ns_l_full = len(cfg_on.species.liq_species)
    Ns_l_eff = Ns_l_full - 1
    expected_add = Nl * Ns_l_eff

    assert layout_on.size - layout_off.size == expected_add

    Ng = 0
    Tg = np.zeros((Ng,), float)
    Yg = np.zeros((len(cfg_on.species.gas_species_full), Ng), float)
    Tl = np.zeros((Nl,), float)
    Yl0 = np.zeros((Ns_l_full, Nl), float)

    state0 = State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl0, Ts=300.0, mpp=0.0, Rd=1e-4)

    u = np.zeros((layout_on.size,), float)

    AB = np.array([[0.10, 0.05, 0.30, 0.25],
                   [0.20, 0.15, 0.10, 0.25]], dtype=float)

    for e in layout_on.entries:
        if e.kind == "Yl":
            il = e.cell
            k_red = e.spec
            u[e.i] = float(AB[k_red, il])

    state1 = apply_u_to_state(state0, u, layout_on, tol_closure=1e-14)

    sums = state1.Yl.sum(axis=0)
    np.testing.assert_allclose(sums, np.ones((Nl,), float), rtol=0, atol=1e-14)

    C_expected = 1.0 - AB.sum(axis=0)
    np.testing.assert_allclose(state1.Yl[2, :], C_expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(state1.Yl[0, :], AB[0, :], rtol=0, atol=1e-14)
    np.testing.assert_allclose(state1.Yl[1, :], AB[1, :], rtol=0, atol=1e-14)
