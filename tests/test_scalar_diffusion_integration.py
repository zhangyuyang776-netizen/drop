import sys
from pathlib import Path

import numpy as np
import pytest
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cantera  # noqa: F401
except Exception:
    pytest.skip("Cantera not available", allow_module_level=True)

try:
    import CoolProp  # noqa: F401
except Exception:
    pytest.skip("CoolProp not available", allow_module_level=True)

from assembly.build_system import build_transport_system_petsc  # noqa: E402
from core.layout import UnknownLayout  # noqa: E402
from core.types import (  # noqa: E402
    CaseChecks,
    CaseConventions,
    CaseCoolProp,
    CaseDiscretization,
    CaseEquilibrium,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseInterface,
    CaseMesh,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    CaseConfig,
    Grid1D,
    State,
)
from properties.aggregator import build_props_from_state  # noqa: E402
from properties.gas import build_gas_model  # noqa: E402
from properties.liquid import build_liquid_model  # noqa: E402


@pytest.fixture(scope="module")
def simple_cfg():
    meta = CaseMeta(id="test", title="integration", version=1, notes=None)
    paths = CasePaths(output_root=Path("."), case_dir=Path("."), mechanism_dir=Path("."), gas_mech="gri30.yaml")
    conventions = CaseConventions(
        radial_normal="+er",
        flux_sign="outward_positive",
        heat_flux_def="q=-k*dTdr",
        evap_sign="mpp_positive_evaporation",
        gas_closure_species="N2",
        index_source="unknown_layout_only",
        assembly_pure=True,
        grid_state_props_split=True,
    )
    eq_cfg = CaseEquilibrium(
        method="raoult_psat",
        psat_model="coolprop",
        background_fill="farfield",
        condensables_gas=["Water"],
        coolprop=CaseCoolProp(backend="HEOS", fluids=["Water"]),
    )
    interface_cfg = CaseInterface(type="no_condensation", bc_mode="Ts_fixed", Ts_fixed=300.0, equilibrium=eq_cfg)
    physics = CasePhysics(interface=interface_cfg)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=tuple(),
        liq_species=["Water"],
        liq_balance_species="Water",
        liq2gas_map={"Water": "Water"},
        mw_kg_per_mol={"N2": 28.0, "Water": 18.01528},
        molar_volume_cm3_per_mol={"Water": 18.0},
    )
    mesh = CaseMesh(
        liq_method="tanh",
        liq_beta=2.0,
        liq_center_bias=0.4,
        gas_method="tanh",
        gas_beta=2.0,
        gas_center_bias=-0.4,
        enforce_interface_continuity=False,
        continuity_tol=1e-12,
    )
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=1, N_gas=3, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-4, t_end=1e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 1.0},
        Yl={"Water": 1.0},
        Y_seed=1.0e-12,
    )
    petsc = CasePETSc(options_prefix="test_", ksp_type="cg", pc_type="jacobi", rtol=1e-8, atol=1e-12, max_it=100, restart=10, monitor=False)
    io_fields = CaseIOFields(scalars=[], gas=[], liquid=[])
    io = CaseIO(write_every=1, formats=["npz"], save_grid=False, fields=io_fields)
    checks = CaseChecks(
        enforce_sumY=True,
        sumY_tol=1.0e-10,
        clamp_negative_Y=True,
        min_Y=0.0,
        enforce_T_bounds=True,
        T_min=200.0,
        T_max=4000.0,
        enforce_unique_index=True,
        enforce_grid_state_props_split=True,
        enforce_assembly_purity=True,
    )
    return CaseConfig(
        case=meta,
        paths=paths,
        conventions=conventions,
        physics=physics,
        species=species,
        geometry=geometry,
        time=time,
        discretization=disc,
        initial=initial,
        petsc=petsc,
        io=io,
        checks=checks,
    )


@pytest.fixture(scope="module")
def simple_grid():
    Nl = 1
    Ng = 3
    Nc = Nl + Ng
    r_f = np.array([0.0, 0.5e-4, 1.0e-4, 1.5e-4, 2.0e-4])
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    return Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=1)


@pytest.fixture(scope="module")
def gas_model(simple_cfg):
    return build_gas_model(simple_cfg)


@pytest.fixture(scope="module")
def liq_model(simple_cfg):
    return build_liquid_model(simple_cfg)


@pytest.fixture()
def simple_state(simple_cfg, simple_grid, gas_model):
    Ng = simple_grid.Ng
    Nl = simple_grid.Nl
    Ns_g = gas_model.gas.n_species
    Ns_l = 1

    T_inf = float(simple_cfg.initial.T_inf)
    Tg = np.linspace(700.0, T_inf, Ng)
    Tl = np.full(Nl, 300.0)

    Yg = np.zeros((Ns_g, Ng))
    Yg[0, :] = 1.0  # all N2

    Yl = np.zeros((Ns_l, Nl))
    Yl[0, :] = 1.0

    return State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=float(simple_cfg.geometry.a0),
    )


class DummyLayout:
    """Minimal layout for Tg-only Step 6 assembly."""

    def __init__(self, Ng: int):
        self._Ng = Ng

    def has_block(self, name: str) -> bool:
        return name == "Tg"

    def n_dof(self) -> int:
        return self._Ng

    def idx_Tg(self, ig: int) -> int:
        return ig


def test_aggregator_and_build_system_end_to_end(simple_cfg, simple_grid, simple_state, gas_model, liq_model):
    props, extras = build_props_from_state(simple_cfg, simple_grid, simple_state, gas_model, liq_model)

    layout = DummyLayout(simple_grid.Ng)
    dt = simple_cfg.time.dt
    A, b = build_transport_system_petsc(simple_cfg, simple_grid, layout, simple_state, props, dt)

    x = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("jacobi")
    ksp.setFromOptions()
    ksp.solve(b, x)

    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    assert reason > 0, f"KSP did not converge, reason={reason}, iters={its}"

    Ng = simple_grid.Ng
    Tg_new = np.array([x.getValue(layout.idx_Tg(i)) for i in range(Ng)])
    assert np.all(np.isfinite(Tg_new))

    T_inf = float(simple_cfg.initial.T_inf)
    assert abs(Tg_new[-1] - T_inf) < 1e-6

    Tg_old = simple_state.Tg
    assert np.all(Tg_new <= np.maximum(Tg_old, T_inf) + 1e-8)
    assert np.all(Tg_new >= np.minimum(Tg_old, T_inf) - 1e-8)
