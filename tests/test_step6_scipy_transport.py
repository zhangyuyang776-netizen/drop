import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import logging

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

from assembly.build_system_SciPy import build_transport_system as build_transport_system_scipy  # noqa: E402
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
from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def cfg_step6():
    meta = CaseMeta(id="step6_scipy_Tg_diffusion", title="step6_scipy", version=1, notes=None)
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
    petsc = CasePETSc(options_prefix="test_", ksp_type="cg", pc_type="jacobi", rtol=1e-10, atol=1e-12, max_it=100, restart=10, monitor=False)
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
def grid_step6():
    Nl = 1
    Ng = 3
    Nc = Nl + Ng
    r_f = np.array([0.0, 0.5e-4, 1.0e-4, 1.5e-4, 2.0e-4])
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    return Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=1)


@pytest.fixture(scope="module")
def gas_model(cfg_step6):
    return build_gas_model(cfg_step6)


@pytest.fixture(scope="module")
def liq_model(cfg_step6):
    return build_liquid_model(cfg_step6)


@pytest.fixture()
def state_step6(cfg_step6, grid_step6, gas_model):
    Ng = grid_step6.Ng
    Nl = grid_step6.Nl
    Ns_g = gas_model.gas.n_species
    Ns_l = 1
    T_inf = float(cfg_step6.initial.T_inf)
    Tg = np.linspace(700.0, T_inf, Ng)
    Tl = np.full(Nl, 300.0)
    Yg = np.zeros((Ns_g, Ng))
    Yg[0, :] = 1.0
    Yl = np.zeros((Ns_l, Nl))
    Yl[0, :] = 1.0
    return State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=300.0,
        mpp=0.0,
        Rd=float(cfg_step6.geometry.a0),
    )


class DummyLayout:
    """Minimal Tg-only layout for Step 6 tests."""

    def __init__(self, Ng: int):
        self._Ng = Ng

    def has_block(self, name: str) -> bool:
        return name == "Tg"

    def n_dof(self) -> int:
        return self._Ng

    def idx_Tg(self, ig: int) -> int:
        return ig


def test_step6_scipy_closed_loop_Tg_diffusion(cfg_step6, grid_step6, state_step6, gas_model, liq_model):
    props, extras = build_props_from_state(cfg_step6, grid_step6, state_step6, gas_model, liq_model=None)
    Ng = grid_step6.Ng
    assert props.rho_g.shape == (Ng,)
    assert np.all(props.rho_g > 0)

    layout = DummyLayout(Ng)
    dt = cfg_step6.time.dt
    A, b = build_transport_system_scipy(cfg_step6, grid_step6, layout, state_step6, props, dt)

    assert A.shape[0] == Ng and A.shape[1] == Ng
    assert b.shape == (Ng,)
    assert np.all(np.diag(A) > 0)

    result = solve_linear_system_scipy(A, b, cfg_step6, method="direct")
    assert isinstance(result.x, np.ndarray)
    assert result.x.shape == (Ng,)
    assert np.all(np.isfinite(result.x))
    assert result.converged

    Tg_new = np.zeros_like(state_step6.Tg)
    for ig in range(Ng):
        Tg_new[ig] = result.x[layout.idx_Tg(ig)]

    state_new = replace(state_step6, Tg=Tg_new)

    # Human-readable log of temperature changes
    gas_start = grid_step6.gas_slice.start
    r_c = grid_step6.r_c
    logger.info("===== Step 6 SciPy Tg diffusion closed-loop check =====")
    logger.info("Ng = %d, dt = %.3e s, T_inf = %.2f K", grid_step6.Ng, cfg_step6.time.dt, float(cfg_step6.initial.T_inf))
    header = " ig |      r_c [m] |  Tg_old [K] |  Tg_new [K] |    dT [K]"
    logger.info(header)
    logger.info("-" * len(header))
    for ig in range(grid_step6.Ng):
        cell_idx = gas_start + ig
        logger.info(
            "%3d | %12.6e | %11.4f | %11.4f | % .4e",
            ig,
            r_c[cell_idx],
            state_step6.Tg[ig],
            state_new.Tg[ig],
            state_new.Tg[ig] - state_step6.Tg[ig],
        )
    logger.info("Tg_old = %s", np.array2string(state_step6.Tg, precision=4, separator=", "))
    logger.info("Tg_new = %s", np.array2string(state_new.Tg, precision=4, separator=", "))
    logger.info("dT     = %s", np.array2string(state_new.Tg - state_step6.Tg, precision=4, separator=", "))

    assert state_new is not state_step6
    assert np.all(state_new.Tg == Tg_new)
    assert np.all(state_new.Tl == state_step6.Tl)
    assert np.all(state_new.Yg == state_step6.Yg)

    T_inf = float(cfg_step6.initial.T_inf)
    assert abs(state_new.Tg[-1] - T_inf) < 1e-8
    Tg_old = state_step6.Tg
    # ensure some change occurred
    assert np.any(np.abs(state_new.Tg - Tg_old) > 1e-10)
    T_min = min(Tg_old.min(), T_inf)
    T_max = max(Tg_old.max(), T_inf)
    atol_range = 1e-3
    assert np.all(state_new.Tg >= T_min - atol_range)
    assert np.all(state_new.Tg <= T_max + atol_range)
