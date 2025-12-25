from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Bridge phase is serial-only (COMM_WORLD size must be 1).")
    return PETSc


def _import_chemistry_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("Cantera not available")
    try:
        import CoolProp  # noqa: F401
    except Exception:
        pytest.skip("CoolProp not available")


def _norm_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


class _PetscCfg:
    def __init__(self) -> None:
        self.options_prefix = ""
        self.ksp_type = "preonly"
        self.pc_type = "lu"
        self.rtol = 1e-10
        self.atol = 1e-12
        self.max_it = 1
        self.restart = 30
        self.monitor = False


class _Case:
    id = "test_linear_spd"


class _Cfg:
    def __init__(self) -> None:
        self.petsc = _PetscCfg()
        self.case = _Case()


@pytest.mark.parametrize("n", [20, 50])
def test_linear_backend_equivalence_random_spd(n: int):
    """
    Fast sanity check:
    - Build a deterministic SPD-ish matrix (well-conditioned).
    - SciPy direct vs PETSc preonly+lu should match tightly.
    """
    PETSc = _import_petsc_or_skip()

    rng = np.random.default_rng(1234)
    M = rng.standard_normal((n, n))
    A = M.T @ M + 1e-2 * np.eye(n)
    b = rng.standard_normal(n)

    from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402
    from assembly.build_system_petsc import numpy_dense_to_petsc_aij  # noqa: E402
    from solvers.petsc_linear import solve_linear_system_petsc  # noqa: E402

    cfg = _Cfg()

    t0 = time.perf_counter()
    xs = solve_linear_system_scipy(A, b, cfg, method="direct")
    t_scipy = time.perf_counter() - t0
    assert xs.converged

    A_p, b_p = numpy_dense_to_petsc_aij(A, b, comm=PETSc.COMM_WORLD)
    t0 = time.perf_counter()
    xp = solve_linear_system_petsc(A_p, b_p, cfg, method="direct")
    t_petsc = time.perf_counter() - t0
    assert xp.converged

    rel = _norm_rel(xp.x, xs.x)
    assert rel < 1e-8

    print(
        f"[random_spd n={n}] rel={rel:.3e} "
        f"scipy={t_scipy*1e3:.2f}ms petsc={t_petsc*1e3:.2f}ms"
    )


@pytest.mark.parametrize("Ng", [3, 5])
def test_linear_backend_equivalence_from_assembly(Ng: int, tmp_path: Path):
    """
    Assembly-based check (slower but meaningful):
    - Build a tiny case (Nl=1, Ng=3/5).
    - Assemble numpy A,b once.
    - Solve with SciPy direct.
    - Convert A,b -> PETSc and solve with PETSc preonly+lu.
    """
    PETSc = _import_petsc_or_skip()
    _import_chemistry_or_skip()

    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = Ng
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.physics.include_mpp = False
    cfg.physics.include_Rd = False
    cfg.physics.include_Ts = False
    cfg.physics.solve_Yg = False
    cfg.nonlinear.enabled = False

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)

    cfg.io.write_every = 10**9
    cfg.io.scalars_write_every = 10**9
    cfg.io.formats = []
    cfg.io.fields.scalars = []
    cfg.io.fields.gas = []
    cfg.io.fields.liquid = []
    cfg.io.fields.interface = []

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402
    from assembly.build_system_SciPy import (  # noqa: E402
        build_transport_system as build_transport_system_numpy,
    )
    from assembly.build_system_petsc import numpy_dense_to_petsc_aij  # noqa: E402
    from solvers.scipy_linear import solve_linear_system_scipy  # noqa: E402
    from solvers.petsc_linear import solve_linear_system_petsc  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    try:
        props0, _extras = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _extras = compute_props(cfg, grid, state0)

    A, b = build_transport_system_numpy(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props=props0,
        dt=float(cfg.time.dt),
        state_guess=state0,
        eq_result=None,
        return_diag=False,
    )

    xs = solve_linear_system_scipy(A, b, cfg, method="direct")
    assert xs.converged

    cfg2 = copy.deepcopy(cfg)
    cfg2.petsc.ksp_type = "preonly"
    cfg2.petsc.pc_type = "lu"
    cfg2.petsc.max_it = 1
    cfg2.petsc.monitor = False

    A_p, b_p = numpy_dense_to_petsc_aij(A, b, comm=PETSc.COMM_WORLD)
    xp = solve_linear_system_petsc(A_p, b_p, cfg2, method="direct")
    assert xp.converged

    rel = _norm_rel(xp.x, xs.x)
    assert rel < 1e-8
