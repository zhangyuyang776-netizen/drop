import numpy as np

from core.layout import apply_u_to_state, build_layout, pack_state
from core.types import CaseSpecies, State
from tests.utils_layout import make_case_config, make_simple_grid


def test_closure_reconstruction_counts_inactive_species():
    grid = make_simple_grid(Nl=1, Ng=1)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["O2", "FUEL", "N2"],
        solve_gas_mode="condensables_only",
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": "FUEL"},
    )
    cfg = make_case_config(grid, species, solve_Tg=False)
    layout = build_layout(cfg, grid)

    Yg = np.zeros((len(species.gas_species), grid.Ng))
    Yg[0, :] = 0.21  # inactive species (not solved)
    Yg[1, :] = 0.01  # active condensable
    Yg[2, :] = 0.0   # closure placeholder
    state = State(
        Tg=np.full(grid.Ng, 300.0),
        Yg=Yg,
        Tl=np.full(grid.Nl, 300.0),
        Yl=np.array([[1.0]]),
        Ts=300.0,
        mpp=0.0,
        Rd=float(grid.r_f[grid.iface_f]),
    )

    u, _, _ = pack_state(state, layout)
    k_red = layout.gas_full_to_reduced["FUEL"]
    assert k_red == 0
    u[layout.idx_Yg(k_red, 0)] = 0.05  # new active mass fraction

    updated = apply_u_to_state(state, u, layout)
    closure_idx = layout.gas_closure_index
    assert closure_idx is not None

    expected_closure = 1.0 - (updated.Yg[0, 0] + updated.Yg[1, 0])
    assert np.isclose(updated.Yg[closure_idx, 0], expected_closure)
    assert np.isclose(np.sum(updated.Yg[:, 0]), 1.0)
    assert np.isclose(updated.Yg[0, 0], 0.21)
    assert np.isclose(updated.Yg[1, 0], 0.05)
