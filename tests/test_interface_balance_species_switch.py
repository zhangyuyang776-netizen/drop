import sys
from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.interface_bc import _build_mpp_row  # noqa: E402
from core.types import State  # noqa: E402


class DummyLayout:
    """
    Minimal layout stub:
    - gas: 3 species full, closure is N2 (index 2), reduced are [0,1]
    """

    def __init__(self):
        self.gas_species_full = ["A", "B", "N2"]
        self.gas_reduced_to_full_idx = [0, 1]
        self.gas_closure_index = 2
        self.gas_full_to_reduced = {"A": 0, "B": 1, "N2": None}
        self.Ns_g_eff = 2

        self.liq_species_full = ["A_l", "B_l"]
        self.liq_reduced_to_full_idx = [0]  # B_l is closure (full index 1)
        self.liq_closure_index = 1

    def idx_Yg(self, k_red: int, ig_local: int) -> int:
        # Only used to build row sparsity; exact value not important for this test.
        return 1000 + ig_local * 10 + k_red

    def has_block(self, name: str) -> bool:
        return name == "Yg"


def _call_build_mpp_row(**kwargs):
    """Call _build_mpp_row with signature-robust kwargs."""
    sig = inspect.signature(_build_mpp_row)
    use = {k: v for k, v in kwargs.items() if k in sig.parameters}
    row, diag = _build_mpp_row(**use)
    return row, diag


def test_interface_balance_species_switch_not_exploding():
    # Construct a consistent case where either balance species yields the same mpp
    layout = DummyLayout()

    Ng = 1
    Nl = 1

    grid = SimpleNamespace(
        r_f=np.array([0.0, 0.0]),
        r_c=np.array([1.0]),
        A_f=np.array([1.0, 1.0]),
        iface_f=1,
        Ng=Ng,
        Nl=Nl,
    )

    props = SimpleNamespace(
        rho_g=np.array([1.0]),
        D_g=np.ones((3, Ng), dtype=float),
    )

    Yg_eq = np.array([0.10, 0.05, 0.85], dtype=float)
    Yl = np.array([[0.70], [0.30]], dtype=float)

    mpp_true = 0.02
    j_corr_A = mpp_true * (Yl[0, 0] - Yg_eq[0])
    j_corr_B = mpp_true * (Yl[1, 0] - Yg_eq[1])
    j_corr_N2 = -(j_corr_A + j_corr_B)

    Yg_cell = np.array(
        [Yg_eq[0] - j_corr_A, Yg_eq[1] - j_corr_B, Yg_eq[2] - j_corr_N2], dtype=float
    )

    state = State(
        Tg=np.zeros((Ng,), float),
        Yg=Yg_cell.reshape(3, Ng),
        Tl=np.zeros((Nl,), float),
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=1e-4,
    )

    eq_result = {"Yg_eq": Yg_eq}

    species_base = SimpleNamespace(
        gas_species=["A", "B", "N2"],
        liq_species=["A_l", "B_l"],
        liq2gas_map={"A_l": "A", "B_l": "B"},
        gas_balance_species="N2",
        liq_balance_species="A_l",
    )
    cfgA = SimpleNamespace(species=species_base, physics=SimpleNamespace())
    cfgB = SimpleNamespace(species=SimpleNamespace(**{**species_base.__dict__, "liq_balance_species": "B_l"}),
                           physics=SimpleNamespace())

    _, diagA = _call_build_mpp_row(
        grid=grid, state=state, props=props, layout=layout, cfg=cfgA,
        eq_result=eq_result,
        il_global=0, il_local=0,
        ig_global=0, ig_local=0,
        iface_f=grid.iface_f,
        idx_mpp=999,
    )
    mppA = float(diagA["evaporation"]["mpp_eval"])
    errA = float(diagA["evaporation"]["sumJ_minus_mpp"])

    _, diagB = _call_build_mpp_row(
        grid=grid, state=state, props=props, layout=layout, cfg=cfgB,
        eq_result=eq_result,
        il_global=0, il_local=0,
        ig_global=0, ig_local=0,
        iface_f=grid.iface_f,
        idx_mpp=999,
    )
    mppB = float(diagB["evaporation"]["mpp_eval"])
    errB = float(diagB["evaporation"]["sumJ_minus_mpp"])

    assert np.isclose(errA, 0.0, atol=1e-12)
    assert np.isclose(errB, 0.0, atol=1e-12)

    assert np.isclose(mppA, mpp_true, rtol=1e-12, atol=1e-12)
    assert np.isclose(mppB, mpp_true, rtol=1e-12, atol=1e-12)

    assert mppA > 0.0 and mppB > 0.0
    assert abs(mppA / mppB - 1.0) < 1e-10
