import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.interface_bc import _build_mpp_row  # noqa: E402
from core.types import State  # noqa: E402


class DummyLayoutMass:
    def __init__(self):
        self.Ng = 1
        self.Ns_g_eff = 2
        self.gas_species_full = ["A_g", "B_g", "N2"]
        self.gas_full_to_reduced = {"A_g": 0, "B_g": 1, "N2": None}
        self.gas_reduced_to_full_idx = [0, 1]
        self.gas_closure_index = 2
        self.liq_species_full = ["A_l", "B_l"]
        self.blocks = {"Yg": slice(0, self.Ns_g_eff)}

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def idx_Yg(self, k_red: int, ig: int) -> int:
        return self.blocks["Yg"].start + ig * self.Ns_g_eff + k_red


def _make_common_inputs():
    grid = SimpleNamespace(
        r_f=np.array([0.0, 0.5, 1.0]),
        r_c=np.array([0.25, 0.75]),
        A_f=np.ones(3),
    )
    cfg = SimpleNamespace(
        physics=SimpleNamespace(default_D_g=None, include_mpp=True),
        species=SimpleNamespace(liq_balance_species="A_l", liq2gas_map={"A_l": "A_g", "B_l": "B_g"}),
    )
    layout = DummyLayoutMass()
    il_global = 0
    il_local = 0
    ig_global = 1
    ig_local = 1 - 1  # first gas cell in local numbering
    iface_f = 1
    idx_mpp = 10
    return grid, cfg, layout, il_global, il_local, ig_global, ig_local, iface_f, idx_mpp


def test_mpp_flux_sum_matches_mpp():
    grid, cfg, layout, ilg, ill, igg, igl, iface_f, idx_mpp = _make_common_inputs()
    rho = 2.0
    D_vec = np.array([[1.0], [2.0], [3.0]])
    props = SimpleNamespace(rho_g=np.array([rho]), D_g=D_vec)
    Yg_cell = np.array([[0.1], [0.2], [0.7]])
    Yg_eq = np.array([0.2, 0.3, 0.5])
    Yl = np.array([[0.6], [0.4]])
    state = State(
        Tg=np.array([300.0]),
        Yg=Yg_cell,
        Tl=np.array([300.0]),
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=1.0e-4,
    )
    eq_result = {"Yg_eq": Yg_eq}

    dr_g = float(grid.r_c[igg] - grid.r_f[iface_f])
    alpha = rho * D_vec[:, 0] / dr_g
    j_raw = -alpha * (Yg_cell[:, 0] - Yg_eq)
    j_sum = float(np.sum(j_raw))
    j_corr = j_raw - Yg_eq * j_sum
    deltaY = float(Yl[0, 0] - Yg_eq[0])
    mpp_expected = float(j_corr[0] / deltaY)
    J_full = mpp_expected * Yg_eq + j_corr

    row, diag = _build_mpp_row(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
        il_global=ilg,
        il_local=ill,
        ig_global=igg,
        ig_local=igl,
        iface_f=iface_f,
        idx_mpp=idx_mpp,
    )

    evap = diag["evaporation"]
    assert np.isclose(evap["mpp_eval"], mpp_expected)
    assert np.isclose(float(J_full.sum() - mpp_expected), 0.0, atol=1e-12)
    assert np.isclose(evap["sumJ_minus_mpp"], 0.0, atol=1e-12)
    assert row.row == idx_mpp


def test_mpp_zero_when_Y_equals_equilibrium():
    grid, cfg, layout, ilg, ill, igg, igl, iface_f, idx_mpp = _make_common_inputs()
    rho = 1.5
    D_vec = np.array([[1.0], [1.5], [2.0]])
    props = SimpleNamespace(rho_g=np.array([rho]), D_g=D_vec)
    Yg_eq = np.array([0.3, 0.2, 0.5])
    Yg_cell = Yg_eq.reshape(-1, 1)
    Yl = np.array([[0.7], [0.3]])
    state = State(
        Tg=np.array([310.0]),
        Yg=Yg_cell,
        Tl=np.array([305.0]),
        Yl=Yl,
        Ts=308.0,
        mpp=0.123,
        Rd=1.0e-4,
    )
    eq_result = {"Yg_eq": Yg_eq}

    row, diag = _build_mpp_row(
        grid=grid,
        state=state,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
        il_global=ilg,
        il_local=ill,
        ig_global=igg,
        ig_local=igl,
        iface_f=iface_f,
        idx_mpp=idx_mpp,
    )

    evap = diag["evaporation"]
    assert np.isclose(evap["mpp_eval"], 0.0, atol=1e-14)
    assert np.isclose(evap["sumJ_minus_mpp"], 0.0, atol=1e-12)
    assert row.row == idx_mpp
