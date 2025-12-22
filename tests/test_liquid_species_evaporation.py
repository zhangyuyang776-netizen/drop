import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.build_liquid_species_system_SciPy import build_liquid_species_system  # noqa: E402
from core.types import State  # noqa: E402


class DummyLayoutYl:
    def __init__(self, Ns_l_eff: int, Ns_l_full: int, Nl: int):
        self.Ns_l_eff = Ns_l_eff
        self.Ns_l_full = Ns_l_full
        self.Nl = Nl
        self.blocks = {"Yl": slice(0, Ns_l_eff * Nl)}
        self.liq_reduced_to_full_idx = [0]  # only first species solved, second is closure
        self.liq_species_full = ["A", "B"]
        self.liq_species_reduced = ["A"]
        self.liq_closure_index = 1
        self.size = Ns_l_eff * Nl

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def n_dof(self) -> int:
        return self.size

    def idx_Yl(self, k_red: int, il: int) -> int:
        return self.blocks["Yl"].start + il * self.Ns_l_eff + k_red


def test_liquid_species_residual_zero_when_no_flux():
    Nl = 2
    grid = SimpleNamespace(
        Nl=Nl,
        Ng=1,
        Nc=3,
        gas_slice=slice(2, 3),
        iface_f=2,
        r_c=np.array([0.25, 0.75, 1.25]),
        r_f=np.array([0.0, 0.5, 1.0, 1.5]),
        V_c=np.array([1.0, 1.0, 1.0]),
        A_f=np.ones(4),
    )
    cfg = SimpleNamespace(
        discretization=SimpleNamespace(theta=1.0),
        conventions=SimpleNamespace(radial_normal="+er", flux_sign="outward_positive", heat_flux_def="q=-k*dTdr"),
    )
    layout = DummyLayoutYl(Ns_l_eff=1, Ns_l_full=2, Nl=Nl)

    Yl_full = np.array([[0.3, 0.3], [0.7, 0.7]])
    state_old = State(
        Tg=np.zeros(1),
        Yg=np.zeros((1, 1)),
        Tl=np.zeros(Nl),
        Yl=Yl_full,
        Ts=300.0,
        mpp=0.0,
        Rd=1.0,
    )
    props = SimpleNamespace(
        rho_l=np.ones(Nl),
        D_l=np.ones((2, Nl)),
        k_l=np.ones(Nl),
    )

    dt = 1.0
    A, b, _ = build_liquid_species_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        interface_evap=None,
        return_diag=True,
    )

    y_vec = np.array([state_old.Yl[0, 0], state_old.Yl[0, 1]])
    residual = A @ y_vec - b
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_liquid_species_interface_evap_sign_and_scale():
    Nl = 2
    grid = SimpleNamespace(
        Nl=Nl,
        Ng=1,
        Nc=3,
        gas_slice=slice(2, 3),
        iface_f=2,
        r_c=np.array([0.2, 0.6, 1.0]),
        r_f=np.array([0.0, 0.4, 0.8, 1.2]),
        V_c=np.array([2.0, 2.0, 1.0]),
        A_f=np.array([1.0, 2.5, 1.0, 1.0]),
    )
    cfg = SimpleNamespace(
        discretization=SimpleNamespace(theta=1.0),
        conventions=SimpleNamespace(radial_normal="+er", flux_sign="outward_positive", heat_flux_def="q=-k*dTdr"),
    )
    layout = DummyLayoutYl(Ns_l_eff=1, Ns_l_full=2, Nl=Nl)

    Yl_full = np.array([[0.4, 0.6], [0.6, 0.4]])
    state_old = State(
        Tg=np.zeros(1),
        Yg=np.zeros((1, 1)),
        Tl=np.zeros(Nl),
        Yl=Yl_full,
        Ts=300.0,
        mpp=0.2,
        Rd=1.0,
    )
    props = SimpleNamespace(
        rho_l=np.ones(Nl),
        D_l=np.zeros((2, Nl)),  # no diffusion to isolate interface flux
        k_l=np.ones(Nl),
    )

    dt = 0.5

    A, b, _ = build_liquid_species_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        interface_evap=None,
        return_diag=True,
    )

    y_vec = np.array([state_old.Yl[0, 0], state_old.Yl[0, 1]])
    residual = A @ y_vec - b

    # Only interface cell (il=1) should see divergence A_if * mpp * Yl_face
    A_if = grid.A_f[grid.iface_f]
    expected_div = A_if * state_old.mpp * state_old.Yl[0, -1]
    assert np.isclose(residual[1], expected_div)
    assert np.isclose(residual[0], 0.0)
