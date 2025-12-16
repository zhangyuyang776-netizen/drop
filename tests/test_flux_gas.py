import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import (  # noqa: E402
    CaseChecks,
    CaseConventions,
    CaseDiscretization,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseMesh,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    CaseConfig,
    Grid1D,
    Props,
)
from physics.flux_gas import compute_gas_diffusive_flux_T  # noqa: E402


def make_minimal_cfg() -> CaseConfig:
    """Minimal CaseConfig sufficient for flux computations."""
    meta = CaseMeta(id="flux_test", title="flux", version=1, notes=None)
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
    geometry = CaseGeometry(a0=1e-4, R_inf=1e-3, N_liq=1, N_gas=3, mesh=mesh)
    time = CaseTime(t0=0.0, dt=1e-6, t_end=1e-3, max_steps=None)
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
        options_prefix="flux_",
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


def make_simple_grid() -> Grid1D:
    """Uniform tiny grid with 1 liquid + 3 gas cells."""
    Nl = 1
    Ng = 3
    Nc = Nl + Ng
    r_f = np.linspace(0.0, 1.0e-3, Nc + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = np.ones(Nc)
    A_f = np.ones(Nc + 1)
    return Grid1D(Nl=Nl, Ng=Ng, Nc=Nc, r_c=r_c, r_f=r_f, V_c=V_c, A_f=A_f, iface_f=Nl)


def test_flux_shapes_and_signs():
    cfg = make_minimal_cfg()
    grid = make_simple_grid()
    Ng = grid.Ng
    # Non-uniform Tg so internal and outer flux are positive (heat flows outward).
    Tg = np.array([700.0, 500.0, 400.0])
    k_g = np.ones(Ng)
    props = Props(rho_g=np.ones(Ng), cp_g=np.ones(Ng), k_g=k_g, rho_l=np.ones(grid.Nl), cp_l=np.ones(grid.Nl), k_l=np.ones(grid.Nl))

    q = compute_gas_diffusive_flux_T(cfg, grid, props, Tg)

    assert q.shape == (grid.Nc + 1,)
    # interface face zero-flux placeholder
    assert q[grid.iface_f] == pytest.approx(0.0)
    # internal faces should be outward positive (since temperature decreases outward)
    assert q[grid.iface_f + 1] > 0.0
    assert q[grid.iface_f + 2] > 0.0
    # outer boundary should drive heat outward when Tg_last > T_inf
    assert q[-1] > 0.0


def test_flux_zero_for_uniform_temperature():
    cfg = make_minimal_cfg()
    grid = make_simple_grid()
    Ng = grid.Ng
    Tg = np.full(Ng, 300.0)
    k_g = np.full(Ng, 2.0)
    props = Props(rho_g=np.ones(Ng), cp_g=np.ones(Ng), k_g=k_g, rho_l=np.ones(grid.Nl), cp_l=np.ones(grid.Nl), k_l=np.ones(grid.Nl))

    q = compute_gas_diffusive_flux_T(cfg, grid, props, Tg)
    assert np.allclose(q, 0.0)


def test_flux_bad_conventions_raises():
    cfg = make_minimal_cfg()
    cfg.conventions.heat_flux_def = "q=k*dTdr"
    grid = make_simple_grid()
    Ng = grid.Ng
    Tg = np.array([700.0, 500.0, 400.0])
    props = Props(rho_g=np.ones(Ng), cp_g=np.ones(Ng), k_g=np.ones(Ng), rho_l=np.ones(grid.Nl), cp_l=np.ones(grid.Nl), k_l=np.ones(grid.Nl))
    with pytest.raises(ValueError):
        compute_gas_diffusive_flux_T(cfg, grid, props, Tg)


def test_flux_shape_mismatch_raises():
    cfg = make_minimal_cfg()
    grid = make_simple_grid()
    Ng = grid.Ng
    Tg = np.array([300.0, 400.0])  # wrong length
    props = Props(rho_g=np.ones(Ng), cp_g=np.ones(Ng), k_g=np.ones(Ng), rho_l=np.ones(grid.Nl), cp_l=np.ones(grid.Nl), k_l=np.ones(grid.Nl))
    with pytest.raises(ValueError):
        compute_gas_diffusive_flux_T(cfg, grid, props, Tg)
