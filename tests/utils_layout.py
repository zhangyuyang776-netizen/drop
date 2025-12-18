from pathlib import Path

import numpy as np

from core.types import (
    CaseChecks,
    CaseConfig,
    CaseConventions,
    CaseDiscretization,
    CaseGeometry,
    CaseInitial,
    CaseIO,
    CaseIOFields,
    CaseMeta,
    CaseMesh,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    Grid1D,
)


def make_simple_grid(Nl: int = 1, Ng: int = 2) -> Grid1D:
    """Create a minimal spherical grid suitable for layout tests."""
    Nc = Nl + Ng
    r_f = np.linspace(0.0, 2.0e-4, Nc + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    V_c = (4.0 * np.pi / 3.0) * (r_f[1:] ** 3 - r_f[:-1] ** 3)
    A_f = 4.0 * np.pi * r_f**2
    return Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nc,
        r_c=r_c,
        r_f=r_f,
        V_c=V_c,
        A_f=A_f,
        iface_f=Nl,
    )


def make_case_config(
    grid: Grid1D,
    species: CaseSpecies,
    *,
    solve_Tg: bool = True,
    solve_Yg: bool = True,
    solve_Tl: bool = False,
    solve_Yl: bool = False,
    case_id: str = "layout_test",
) -> CaseConfig:
    """Build a lightweight CaseConfig for layout/unit tests."""
    physics = CasePhysics(
        solve_Tg=solve_Tg,
        solve_Yg=solve_Yg,
        solve_Tl=solve_Tl,
        solve_Yl=solve_Yl,
        include_Ts=False,
        include_mpp=False,
        include_Rd=False,
    )

    mesh = CaseMesh(
        liq_method="uniform",
        liq_beta=1.0,
        liq_center_bias=0.0,
        gas_method="uniform",
        gas_beta=1.0,
        gas_center_bias=0.0,
        enforce_interface_continuity=True,
        continuity_tol=1e-12,
    )
    geometry = CaseGeometry(
        a0=float(grid.r_f[grid.iface_f]),
        R_inf=float(grid.r_f[-1]),
        N_liq=grid.Nl,
        N_gas=grid.Ng,
        mesh=mesh,
    )

    time = CaseTime(t0=0.0, dt=1.0e-6, t_end=1.0e-5)
    discretization = CaseDiscretization(time_scheme="backward_euler", theta=1.0, mass_matrix="lumped")
    initial = CaseInitial(
        T_inf=300.0,
        P_inf=101325.0,
        T_d0=300.0,
        Yg={species.gas_balance_species: 1.0},
        Yl={species.liq_balance_species: 1.0},
        Y_seed=0.0,
    )
    petsc = CasePETSc(
        options_prefix="",
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1e-6,
        atol=1e-12,
        max_it=20,
        restart=10,
        monitor=False,
    )
    io = CaseIO(write_every=0, formats=[], save_grid=False, fields=CaseIOFields(scalars=[], gas=[], liquid=[]))
    checks = CaseChecks(
        enforce_sumY=False,
        sumY_tol=1.0e-8,
        clamp_negative_Y=False,
        min_Y=0.0,
        enforce_T_bounds=False,
        T_min=0.0,
        T_max=5000.0,
        enforce_unique_index=True,
        enforce_grid_state_props_split=True,
        enforce_assembly_purity=True,
    )
    conventions = CaseConventions(
        radial_normal="+r",
        flux_sign="outward_positive",
        heat_flux_def="q_positive_outward",
        evap_sign="mpp_positive_outward",
        gas_closure_species=species.gas_balance_species,
        index_source="layout",
        assembly_pure=True,
        grid_state_props_split=True,
    )
    paths = CasePaths(
        output_root=Path("."),
        case_dir=Path("."),
        mechanism_dir=Path("."),
        gas_mech="mech.yaml",
    )
    case = CaseMeta(id=case_id, title="layout", version=1)

    return CaseConfig(
        case=case,
        paths=paths,
        conventions=conventions,
        physics=physics,
        species=species,
        geometry=geometry,
        time=time,
        discretization=discretization,
        initial=initial,
        petsc=petsc,
        io=io,
        checks=checks,
    )
