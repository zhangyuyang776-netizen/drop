import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.equilibrium import (  # noqa: E402
    EquilibriumModel,
    compute_interface_equilibrium_full,
    mass_to_mole,
)


def test_raoult_multicomp_partial_pressures_and_mass_fractions():
    gas_names = ["A_g", "B_g", "N2"]
    liq_names = ["A_l", "B_l"]
    idx_cond_l = np.array([0, 1], dtype=int)
    idx_cond_g = np.array([0, 1], dtype=int)
    M_g = np.array([30.0, 44.0, 28.0], dtype=np.float64)
    M_l = np.array([80.0, 100.0], dtype=np.float64)

    Yg_far = np.array([0.1, 0.1, 0.8], dtype=np.float64)
    Xg_far = mass_to_mole(Yg_far, M_g)

    model = EquilibriumModel(
        method="raoult_psat",
        psat_model="clausius",
        background_fill="farfield",
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=idx_cond_l,
        idx_cond_g=idx_cond_g,
        M_g=M_g,
        M_l=M_l,
        Yg_farfield=Yg_far,
        Xg_farfield=Xg_far,
        cp_backend="HEOS",
        cp_fluids=["A_l", "B_l"],
        T_ref=350.0,
        psat_ref={"A_l": 2.0e4, "B_l": 1.0e4},
    )

    Ts = model.T_ref  # ensures psat == psat_ref under Clausius fallback
    Pg = 101325.0
    Yl_face = np.array([0.2, 0.3], dtype=np.float64)  # intentionally not normalized (will be renormalized)
    Yg_face = np.array([0.05, 0.15, 0.8], dtype=np.float64)

    res = compute_interface_equilibrium_full(
        model=model,
        Ts=Ts,
        Pg=Pg,
        Yl_face=Yl_face,
        Yg_face=Yg_face,
    )

    expected_x_cond = res.X_liq[model.idx_cond_l]
    expected_p_partial = expected_x_cond * res.psat[model.idx_cond_l]

    assert np.allclose(res.x_cond, expected_x_cond)
    assert np.allclose(res.p_partial, expected_p_partial)
    assert np.allclose(res.y_cond, res.p_partial / max(Pg, 1.0))

    assert np.isclose(np.sum(res.Yg_eq), 1.0, atol=1e-12)
    assert np.all(res.Yg_eq >= 0.0)
