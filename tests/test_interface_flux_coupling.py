import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_species_system_SciPy import build_gas_species_system_global  # noqa: E402
from assembly.build_system_SciPy import build_transport_system  # noqa: E402
from core.types import State  # noqa: E402


class DummyLayoutSpecies:
    def __init__(self, Ns_g_eff: int, Ns_g_full: int, Ng: int):
        self.Ns_g_eff = Ns_g_eff
        self.Ns_g_full = Ns_g_full
        self.Ng = Ng
        self.blocks = {"Yg": slice(0, Ns_g_eff * Ng)}
        self.gas_species_reduced = ["A", "B"]
        self.gas_species_full = ["A", "B", "N2"]
        self.gas_full_to_reduced = {"A": 0, "B": 1, "N2": None}
        self.gas_reduced_to_full_idx = [0, 1]
        self.gas_closure_index = 2
        self.size = Ns_g_eff * Ng

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def n_dof(self) -> int:
        return self.size

    def idx_Yg(self, k_red: int, ig: int) -> int:
        return self.blocks["Yg"].start + ig * self.Ns_g_eff + k_red


class DummyLayoutEnergy:
    def __init__(self, Ng: int):
        self.Ng = Ng
        self.blocks = {"Tg": slice(0, Ng), "mpp": slice(Ng, Ng + 1)}
        self.size = Ng + 1
        self.Ns_g_eff = 0
        self.gas_closure_index = 1
        self.gas_species_full = ["A", "N2"]
        self.gas_full_to_reduced = {"A": None, "N2": None}
        self.gas_reduced_to_full_idx: list[int] = []
        self.liq_species_full = ["A_l"]

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def n_dof(self) -> int:
        return self.size

    def idx_Tg(self, ig: int) -> int:
        return self.blocks["Tg"].start + ig

    def idx_mpp(self) -> int:
        return self.blocks["mpp"].start


def test_species_interface_flux_injects_into_rhs():
    # Grid with Ng=2 gas cells; interface face at f=1
    grid = SimpleNamespace(
        Nl=1,
        Ng=2,
        Nc=3,
        gas_slice=slice(1, 3),
        iface_f=1,
        r_c=np.array([0.2, 0.6, 1.0]),
        r_f=np.array([0.0, 0.4, 0.8, 1.2]),
        V_c=np.full(3, 2.0),
        A_f=np.array([1.0, 2.5, 1.0, 1.0]),  # interface area scaled to 2.5
    )
    cfg = SimpleNamespace(
        discretization=SimpleNamespace(theta=1.0),
        physics=SimpleNamespace(include_mpp=False, stefan_velocity=False, species_convection=False),
        initial=SimpleNamespace(Yg={"A": 0.5, "B": 0.5, "N2": 0.0}),
    )
    layout = DummyLayoutSpecies(Ns_g_eff=2, Ns_g_full=3, Ng=2)

    Ns_full = 3
    Ng = 2
    Yg_old = np.zeros((Ns_full, Ng))
    props = SimpleNamespace(rho_g=np.ones(Ng), D_g=np.ones((Ns_full, Ng)))
    state_old = State(
        Tg=np.ones(Ng),
        Yg=Yg_old,
        Tl=np.ones(1),
        Yl=np.ones((1, 1)),
        Ts=300.0,
        mpp=0.0,
        Rd=1.0,
    )
    dt = 0.5

    # Interface flux: only species A has non-zero flux
    J_full = np.array([0.5, 0.0, -0.5])
    iface_evap = {"J_full": J_full}

    A, b, diag = build_gas_species_system_global(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        eq_result=None,
        interface_evap=iface_evap,
        return_diag=True,
    )

    row_A_ig0 = layout.idx_Yg(0, 0)
    # Time term is zero (Yg_old=0), so RHS is purely from interface flux with area scaling (positive)
    assert np.isclose(b[row_A_ig0], grid.A_f[grid.iface_f] * J_full[0])
    assert any("inner_flux" in key or key == "inner_flux" for key in diag["bc"])


def test_energy_rhs_receives_interface_enthalpy_flux():
    grid = SimpleNamespace(
        Nl=1,
        Ng=2,
        Nc=3,
        gas_slice=slice(1, 3),
        iface_f=1,
        r_c=np.array([0.25, 0.75, 1.25]),
        r_f=np.array([0.0, 0.5, 1.0, 1.5]),
        V_c=np.full(3, 1.5),
        A_f=np.array([1.0, 3.0, 1.0, 1.0]),  # interface area scaled to 3
    )
    cfg = SimpleNamespace(
        discretization=SimpleNamespace(theta=1.0),
        physics=SimpleNamespace(
            include_mpp=True,
            include_Ts=False,
            include_Rd=False,
            solve_Yg=False,
            interface=SimpleNamespace(bc_mode="Ts_fixed", Ts_fixed=300.0),
        ),
        initial=SimpleNamespace(T_inf=300.0, Yg={}, Yl={}, P_inf=101325.0),
        species=SimpleNamespace(liq_balance_species="A_l", liq2gas_map={"A_l": "A"}),
        conventions=SimpleNamespace(
            radial_normal="+er",
            flux_sign="outward_positive",
            heat_flux_def="q=-k*dTdr",
            evap_sign="mpp_positive_evaporation",
        ),
    )
    layout = DummyLayoutEnergy(Ng=2)

    Ns_g = 2
    Ng = 2
    Yg = np.array([[0.6, 0.6], [0.4, 0.4]])
    Yl = np.array([[1.0], [0.0]])
    state_old = State(
        Tg=np.zeros(Ng),
        Yg=Yg,
        Tl=np.zeros(1),
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=1.0,
    )
    props = SimpleNamespace(
        rho_g=np.ones(Ng),
        cp_g=np.zeros(Ng),
        k_g=np.zeros(Ng),
        D_g=np.ones((Ns_g, Ng)),
        h_g=np.full(Ng, 10.0),
        h_gk=np.vstack([np.full(Ng, 1.0), np.full(Ng, 2.0)]),
        k_l=np.zeros(1),
    )
    eq_result = {"Yg_eq": np.array([0.5, 0.5])}
    dt = 0.5

    A, b, diag = build_transport_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        eq_result=eq_result,
        return_diag=True,
    )

    evap = diag.get("evaporation", {})
    mpp_state = float(state_old.mpp)
    assert "j_corr_full" in evap
    row_Tg0 = layout.idx_Tg(0)
    # With cp=k=0, the only source in b for Tg0 should come from enthalpy flux
    A_if = float(grid.A_f[grid.iface_f])
    h_mix_if = float(props.h_g[0])
    h_k_if = np.array([1.0, 2.0])
    j_corr = np.asarray(evap["j_corr_full"], dtype=float)
    q_iface = mpp_state * h_mix_if + float(np.dot(h_k_if, j_corr))
    expected_rhs = -A_if * q_iface
    assert np.isclose(b[row_Tg0], expected_rhs)
