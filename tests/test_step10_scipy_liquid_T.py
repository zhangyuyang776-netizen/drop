import logging
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import CaseConfig, CaseMeta, CasePaths, CaseDiscretization, CaseConventions, CaseInitial, CaseTime, CaseChecks, CaseMesh, CaseGeometry, CasePhysics, CaseSpecies, CaseIO, CaseIOFields, CasePETSc, Grid1D, State, Props  # noqa: E402
from assembly.build_liquid_T_system_SciPy import build_liquid_T_system  # noqa: E402
from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402
from properties.liquid import LiquidPropertiesModel, build_liquid_model, compute_liquid_props  # noqa: E402

logger = logging.getLogger(__name__)


class DummyLayoutTl:
    """
    Minimal layout for Step 10:
    - Only a Tl block, size Nl.
    - Global index = il.
    """

    def __init__(self, Nl: int):
        self.Nl = Nl

    def n_dof(self) -> int:
        return self.Nl

    def has_block(self, name: str) -> bool:
        return name == "Tl"

    def idx_Tl(self, il: int) -> int:
        if not (0 <= il < self.Nl):
            raise IndexError(f"il={il} out of range for Nl={self.Nl}")
        return il


@pytest.fixture
def cfg_step10() -> CaseConfig:
    meta = CaseMeta(id="step10_liq_T", title="step10_liq", version=1, notes=None)
    paths = CasePaths(output_root=Path("."), case_dir=Path("."), mechanism_dir=Path("."), gas_mech="dummy.yaml")
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
    physics = CasePhysics()
    physics.interface.equilibrium.coolprop.fluids = ("Water",)
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["N2"],
        liq_species=["WATER_L"],
        liq_balance_species="WATER_L",
        liq2gas_map={"WATER_L": "H2O"},
        mw_kg_per_mol={"N2": 28.0, "WATER_L": 18.0},
        molar_volume_cm3_per_mol={"WATER_L": 18.0},
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
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=3, N_gas=3, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-5, t_end=1e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 1.0},
        Yl={"WATER_L": 1.0},
        Y_seed=1e-12,
    )
    petsc = CasePETSc(
        options_prefix="liq_",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-8,
        atol=1e-12,
        max_it=50,
        restart=10,
        monitor=False,
    )
    io_fields = CaseIOFields(scalars=[], gas=[], liquid=[])
    io = CaseIO(write_every=1, formats=["npz"], save_grid=False, fields=io_fields)
    checks = CaseChecks(
        enforce_sumY=True,
        sumY_tol=1e-10,
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


@pytest.fixture
def grid_step10_liq(cfg_step10: CaseConfig) -> Grid1D:
    Nl = cfg_step10.geometry.N_liq
    Ng = cfg_step10.geometry.N_gas
    Nc = Nl + Ng
    # simple uniform radii for testing
    r_f = np.linspace(0.0, cfg_step10.geometry.R_inf, Nc + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    A_f = 4.0 * np.pi * r_f * r_f
    V_c = 4.0 / 3.0 * np.pi * (r_f[1:] ** 3 - r_f[:-1] ** 3)
    return Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nc,
        r_c=r_c,
        r_f=r_f,
        V_c=V_c,
        A_f=A_f,
        iface_f=Nl,
        liq_slice=slice(0, Nl),
        gas_slice=slice(Nl, Nc),
    )


@pytest.fixture
def state_step10(grid_step10_liq: Grid1D) -> State:
    Nl = grid_step10_liq.Nl
    Tl = np.linspace(340.0, 300.0, Nl, dtype=np.float64)  # center hot, interface cooler
    Ts = 300.0
    Tg = np.zeros(grid_step10_liq.Ng, dtype=np.float64)
    Yg = np.zeros((1, max(grid_step10_liq.Ng, 1)), dtype=np.float64)
    Yl = np.ones((1, Nl), dtype=np.float64)
    return State(
        Tg=Tg,
        Yg=Yg,
        Tl=Tl,
        Yl=Yl,
        Ts=Ts,
        mpp=0.0,
        Rd=float(grid_step10_liq.r_f[grid_step10_liq.iface_f]),
    )


@pytest.fixture
def liq_model(cfg_step10: CaseConfig) -> LiquidPropertiesModel:
    return build_liquid_model(cfg_step10)


def test_step10_scipy_liquid_T_diffusion(cfg_step10, grid_step10_liq, state_step10, liq_model, caplog):
    caplog.set_level(logging.INFO)
    log = logging.getLogger(__name__)

    Nl = grid_step10_liq.Nl

    # properties (liquid only) via liquid model
    liq_core, liq_extra = compute_liquid_props(liq_model, state_step10, grid_step10_liq)
    rho_l = liq_core["rho_l"]
    cp_l = liq_core["cp_l"]
    k_l = liq_core["k_l"]
    # build Props manually; gas part dummy
    props = Props(
        rho_g=np.zeros(grid_step10_liq.Ng, dtype=np.float64),
        cp_g=np.zeros(grid_step10_liq.Ng, dtype=np.float64),
        k_g=np.zeros(grid_step10_liq.Ng, dtype=np.float64),
        D_g=None,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
        D_l=None,
    )

    layout = DummyLayoutTl(Nl)
    dt = cfg_step10.time.dt
    A, b = build_liquid_T_system(cfg_step10, grid_step10_liq, layout, state_step10, props, dt)

    assert A.shape == (Nl, Nl)
    assert b.shape == (Nl,)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(b))

    result = solve_linear_system_scipy(A, b, cfg_step10, method="direct")
    assert result.converged
    Tl_new = np.asarray(result.x, dtype=float)
    assert Tl_new.shape == (Nl,)
    assert np.all(np.isfinite(Tl_new))

    state_new = replace(state_step10, Tl=Tl_new)

    Tl_old = state_step10.Tl
    r_c = grid_step10_liq.r_c
    Ts_fixed = float(state_step10.Ts)

    log.info("===== Step 10 SciPy liquid T diffusion closed-loop check =====")
    log.info("Nl = %d, dt = %.3e s, Ts_fixed = %.2f K", Nl, dt, Ts_fixed)
    header = " il |      r_c [m] |  Tl_old [K] |  Tl_new [K] |    dT [K]"
    log.info(header)
    log.info("-" * len(header))
    for il in range(Nl):
        log.info(
            "%3d | %12.6e | %11.4f | %11.4f | % .4e",
            il,
            float(r_c[il]),
            float(Tl_old[il]),
            float(Tl_new[il]),
            float(Tl_new[il] - Tl_old[il]),
        )
    log.info("Tl_old = %s", np.array2string(Tl_old, precision=4, separator=", "))
    log.info("Tl_new = %s", np.array2string(Tl_new, precision=4, separator=", "))
    log.info("dT     = %s", np.array2string(Tl_new - Tl_old, precision=4, separator=", "))

    # Interface Dirichlet
    assert np.isclose(Tl_new[-1], Ts_fixed, rtol=0.0, atol=1e-10)

    # At least one temperature changed
    assert np.any(np.abs(Tl_new - Tl_old) > 1e-12)

    # Trend: if Ts is cooler than neighbor, neighbor should not heat up beyond tolerance (and vice versa)
    if Nl > 1:
        Tl_nb_old = Tl_old[-2]
        Tl_nb_new = Tl_new[-2]
        if Ts_fixed > Tl_nb_old:
            assert Tl_nb_new >= Tl_nb_old - 1e-8
        elif Ts_fixed < Tl_nb_old:
            assert Tl_nb_new <= Tl_nb_old + 1e-8

    assert state_new is not state_step10
