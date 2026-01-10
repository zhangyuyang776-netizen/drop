from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_petsc_or_skip():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
    bootstrap_mpi_before_petsc()
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


def _build_ctx_and_x0(cfg, tmp_path: Path, Ng: int):
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = Ng
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True

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
    from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402

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
        props0, _ = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg, grid, state0)

    ctx, u0 = build_nonlinear_context_for_step(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg.time.t0),
        dt=float(cfg.time.dt),
    )
    return ctx, u0


def _build_dense_fd_jacobian(ctx, x0, eps=1.0e-8):
    x0 = np.asarray(x0, dtype=np.float64)
    N = x0.size

    scale_u = np.asarray(ctx.scale_u, dtype=np.float64)
    scale_u_safe = np.where(scale_u > 0.0, scale_u, 1.0)
    meta = getattr(ctx, "meta", None)
    if meta is None or not isinstance(meta, dict):
        meta = {}
    scale_F = np.asarray(meta.get("residual_scale_F", np.ones_like(x0)), dtype=np.float64)
    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)
    x_space = str(meta.get("petsc_x_space", "physical")).lower()
    f_space = str(meta.get("petsc_f_space", "physical")).lower()

    from assembly.residual_global import residual_only

    def _x_to_u_phys(x_arr):
        if x_space in ("physical", "phys", "unscaled"):
            return x_arr
        if x_space in ("scaled", "eval"):
            return x_arr * scale_u_safe
        raise ValueError(f"Unknown petsc_x_space={x_space!r}")

    def _res_phys_to_eval(res_phys):
        if f_space in ("physical", "phys", "unscaled"):
            return res_phys
        if f_space in ("scaled", "eval"):
            return res_phys / scale_F_safe
        raise ValueError(f"Unknown petsc_f_space={f_space!r}")

    u0_phys = _x_to_u_phys(x0)
    r0_phys = residual_only(u0_phys, ctx)
    if not np.all(np.isfinite(r0_phys)):
        r0_phys = np.where(np.isfinite(r0_phys), r0_phys, 1.0e20)
    r0_eval = _res_phys_to_eval(r0_phys)

    J = np.zeros((N, N), dtype=np.float64)
    x_work = x0.copy()
    for j in range(N):
        dx = eps * (1.0 + abs(x0[j]))
        if dx == 0.0:
            dx = eps
        x_work[j] = x0[j] + dx

        uj_phys = _x_to_u_phys(x_work)
        rj_phys = residual_only(uj_phys, ctx)
        if not np.all(np.isfinite(rj_phys)):
            rj_phys = np.where(np.isfinite(rj_phys), rj_phys, 1.0e20)
        rj_eval = _res_phys_to_eval(rj_phys)

        J[:, j] = (rj_eval - r0_eval) / dx
        x_work[j] = x0[j]

    return J


@pytest.mark.parametrize("Ng", [3, 5])
def test_sparse_fd_jacobian_matches_dense_small(tmp_path: Path, Ng: int):
    _import_petsc_or_skip()
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
    ctx, x0 = _build_ctx_and_x0(cfg, tmp_path, Ng)

    J_dense = _build_dense_fd_jacobian(ctx, x0, eps=1.0e-8)

    from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian  # noqa: E402
    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402

    P, stats = build_sparse_fd_jacobian(ctx, x0, eps=1.0e-8, drop_tol=0.0)

    ia, ja, a_vals = P.getValuesCSR()
    N = J_dense.shape[0]
    J_sp = np.zeros_like(J_dense)
    for i in range(N):
        start = ia[i]
        end = ia[i + 1]
        for k in range(start, end):
            j = int(ja[k])
            J_sp[i, j] = a_vals[k]

    pattern = build_jacobian_pattern(ctx.cfg, ctx.grid_ref, ctx.layout)
    indptr = np.asarray(pattern.indptr, dtype=np.int32)
    indices = np.asarray(pattern.indices, dtype=np.int32)

    diff_list = []
    ref_list = []
    for i in range(N):
        for j in indices[indptr[i] : indptr[i + 1]]:
            val_ref = J_dense[i, j]
            val_sp = J_sp[i, j]
            diff_list.append(abs(val_sp - val_ref))
            ref_list.append(abs(val_ref))

    diff_arr = np.asarray(diff_list, dtype=np.float64)
    ref_arr = np.asarray(ref_list, dtype=np.float64)
    rel_err = diff_arr / np.maximum(1.0, ref_arr)
    assert np.max(rel_err) < 1.0e-4

    assert stats["ncolors"] > 0
    assert stats["n_fd_calls"] > 0
