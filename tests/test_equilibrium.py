import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.equilibrium import (  # noqa: E402
    EquilibriumModel,
    build_equilibrium_model,
    compute_interface_equilibrium,
    mass_to_mole,
    mole_to_mass,
)
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
)


def make_minimal_cfg_single_fuel(background_fill: str = "farfield") -> CaseConfig:
    meta = CaseMeta(id="test", title="test", version=1, notes=None)
    paths = CasePaths(
        output_root=Path("."),
        case_dir=Path("."),
        mechanism_dir=Path("."),
        gas_mech="dummy.yaml",
    )
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["N2", "FUEL"],
        liq_species=["FUEL_L"],
        liq_balance_species="FUEL_L",
        liq2gas_map={"FUEL_L": "FUEL"},
        mw_kg_per_mol={"N2": 28.0, "FUEL": 170.0, "FUEL_L": 170.0},
        molar_volume_cm3_per_mol={"FUEL_L": 200.0},
    )
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
    from core.types import CaseEquilibrium, CaseInterface, CaseCoolProp

    eq_cfg = CaseEquilibrium(
        method="raoult_psat",
        psat_model="coolprop",
        background_fill=background_fill,
        condensables_gas=["FUEL"],
        coolprop=CaseCoolProp(backend="HEOS", fluids=["FUEL_L"]),
    )
    interface_cfg = CaseInterface(
        type="no_condensation",
        bc_mode="Ts_fixed",
        Ts_fixed=300.0,
        equilibrium=eq_cfg,
    )
    physics = CasePhysics(interface=interface_cfg)
    geometry = CaseGeometry(
        a0=1.0e-4,
        R_inf=1.0e-3,
        N_liq=1,
        N_gas=2,
        mesh=CaseMesh(
            liq_method="tanh",
            liq_beta=2.0,
            liq_center_bias=0.4,
            gas_method="tanh",
            gas_beta=2.0,
            gas_center_bias=-0.4,
            enforce_interface_continuity=False,
            continuity_tol=1e-12,
        ),
    )
    time = CaseTime(t0=0.0, dt=1.0e-6, t_end=1.0e-3, max_steps=None)
    disc = CaseDiscretization(time_scheme="BE", theta=1.0, mass_matrix="rhoVc_p")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={"N2": 0.8, "FUEL": 0.2},
        Yl={"FUEL_L": 1.0},
        Y_seed=1.0e-12,
    )
    petsc = CasePETSc(
        options_prefix="test_",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1.0e-8,
        atol=1.0e-12,
        max_it=50,
        restart=10,
        monitor=False,
    )
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


def test_mass_mole_roundtrip_simple():
    M = np.array([2.0, 32.0, 100.0])
    Y = np.array([0.2, 0.3, 0.5])
    X = mass_to_mole(Y, M)
    Y_back = mole_to_mass(X, M)
    assert np.isclose(np.sum(X), 1.0)
    assert np.allclose(Y_back, Y, rtol=1e-12, atol=1e-12)
    Y_zero = np.array([0.0, 0.0, 0.0])
    X_zero = mass_to_mole(Y_zero, M)
    assert np.all(X_zero == 0.0)


def test_build_equilibrium_model_basic_mapping():
    cfg = make_minimal_cfg_single_fuel()
    M_g = np.array([28.0, 170.0])
    M_l = np.array([170.0])
    model = build_equilibrium_model(cfg, Ns_g=2, Ns_l=1, M_g=M_g, M_l=M_l)
    assert np.array_equal(model.idx_cond_l, np.array([0]))
    assert np.array_equal(model.idx_cond_g, np.array([1]))
    assert np.allclose(model.Yg_farfield, np.array([0.8, 0.2]))
    assert np.isclose(np.sum(model.Xg_farfield), 1.0)


def test_compute_interface_equilibrium_single_condensable_farfield_fill(monkeypatch):
    cfg = make_minimal_cfg_single_fuel(background_fill="farfield")
    M_g = np.array([28.0, 170.0])
    M_l = np.array([170.0])
    model = build_equilibrium_model(cfg, Ns_g=2, Ns_l=1, M_g=M_g, M_l=M_l)

    def fake_psat_vec_all(model_local, T):
        return np.array([2.0e4])

    import properties.equilibrium as eq_mod  # noqa: E401,E402

    monkeypatch.setattr(eq_mod, "_psat_vec_all", fake_psat_vec_all)

    Ts = 500.0
    Pg = 1.0e5
    Yl_face = np.array([1.0])
    Yg_face = model.Yg_farfield.copy()

    Yg_eq, y_cond, psat = compute_interface_equilibrium(model, Ts, Pg, Yl_face, Yg_face)
    assert np.isclose(np.sum(y_cond), 0.2, rtol=1e-12, atol=1e-12)
    assert np.isclose(np.sum(Yg_eq), 1.0)
    y_all = np.array([0.8, 0.2])
    numer = y_all * M_g
    Y_expected = numer / numer.sum()
    assert np.allclose(Yg_eq, Y_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(psat, np.array([2.0e4]))


def test_compute_interface_equilibrium_pressure_cap(monkeypatch):
    cfg = make_minimal_cfg_single_fuel(background_fill="farfield")
    M_g = np.array([28.0, 170.0])
    M_l = np.array([170.0])
    model = build_equilibrium_model(cfg, Ns_g=2, Ns_l=1, M_g=M_g, M_l=M_l)

    def fake_psat_vec_all(model_local, T):
        return np.array([1.0e7])

    import properties.equilibrium as eq_mod  # noqa: E401,E402

    monkeypatch.setattr(eq_mod, "_psat_vec_all", fake_psat_vec_all)

    Ts = 400.0
    Pg = 1.0e5
    Yl_face = np.array([1.0])
    Yg_face = model.Yg_farfield.copy()

    Yg_eq, y_cond, psat = compute_interface_equilibrium(model, Ts, Pg, Yl_face, Yg_face)
    assert np.isclose(np.sum(y_cond), 0.995, rtol=1e-12, atol=1e-12)
    y_all = np.array([0.005, 0.995])
    numer = y_all * M_g
    Y_expected = numer / numer.sum()
    assert np.isclose(np.sum(Yg_eq), 1.0)
    assert np.allclose(Yg_eq, Y_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(psat, np.array([1.0e7]))


def test_compute_interface_equilibrium_interface_noncondensables(monkeypatch):
    cfg = make_minimal_cfg_single_fuel(background_fill="interface_noncondensables")
    M_g = np.array([28.0, 170.0])
    M_l = np.array([170.0])
    model = build_equilibrium_model(cfg, Ns_g=2, Ns_l=1, M_g=M_g, M_l=M_l)

    def fake_psat_vec_all(model_local, T):
        return np.array([2.0e4])

    import properties.equilibrium as eq_mod  # noqa: E401,E402

    monkeypatch.setattr(eq_mod, "_psat_vec_all", fake_psat_vec_all)

    Ts = 400.0
    Pg = 1.0e5
    Yl_face = np.array([1.0])
    # Interface gas richer in fuel than farfield to trigger interface_noncondensables behavior
    Yg_face = np.array([0.6, 0.4])

    Yg_eq, y_cond, psat = compute_interface_equilibrium(model, Ts, Pg, Yl_face, Yg_face)

    assert Yg_eq.shape == (2,)
    assert y_cond.shape == (1,)
    assert psat.shape == (1,)
    assert np.isclose(np.sum(Yg_eq), 1.0, atol=1e-12)
    assert np.all(Yg_eq >= 0.0)
    assert y_cond[0] > 0.0
