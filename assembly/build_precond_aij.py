"""
Build PETSc AIJ preconditioner matrices for serial runs.

Supports dense-bridge assembly and native PETSc assembly (assembly_mode=native_aij).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from solvers.nonlinear_context import NonlinearContext


def _get_linear_assembly_mode(ctx: NonlinearContext) -> str:
    solver_cfg = getattr(ctx.cfg, "solver", None)
    linear_cfg = getattr(solver_cfg, "linear", None)
    return str(getattr(linear_cfg, "assembly_mode", "bridge_dense")).lower()


def _build_state_props_eq(
    ctx: NonlinearContext,
    u_phys: np.ndarray,
) -> tuple[Any, Any, Any, Any, Dict[str, Any]]:
    cfg = ctx.cfg
    grid = ctx.grid_ref
    layout = ctx.layout
    diag: Dict[str, Any] = {}

    try:
        state_guess = ctx.make_state(u_phys)
        diag["state_source"] = "u_phys"
    except Exception:
        state_guess = ctx.state_old
        diag["state_source"] = "state_old"

    t_min_cfg = float(getattr(getattr(cfg, "checks", None), "T_min", 1.0))
    if not np.isfinite(t_min_cfg) or t_min_cfg <= 0.0:
        t_min_cfg = 1.0

    state_props = state_guess.copy()
    Tg_clamped = int(np.count_nonzero(state_props.Tg < t_min_cfg)) if state_props.Tg.size else 0
    Tl_clamped = int(np.count_nonzero(state_props.Tl < t_min_cfg)) if state_props.Tl.size else 0
    Ts_clamped = int(float(state_props.Ts) < t_min_cfg)
    if Tg_clamped:
        state_props.Tg = np.maximum(state_props.Tg, t_min_cfg)
    if Tl_clamped:
        state_props.Tl = np.maximum(state_props.Tl, t_min_cfg)
    if Ts_clamped:
        state_props.Ts = t_min_cfg

    try:
        from properties.compute_props import compute_props
        props, _props_extras = compute_props(cfg, grid, state_props)
        if Tg_clamped or Tl_clamped or Ts_clamped:
            diag["props_source"] = "state_guess_clamped"
        else:
            diag["props_source"] = "state_guess"
    except Exception:
        props = ctx.props_old
        diag["props_source"] = "props_old"

    eq_result = None
    eq_model = None
    eq_source = "disabled"
    needs_eq = bool(getattr(cfg.physics, "include_mpp", False) and layout.has_block("mpp"))
    if needs_eq:
        from assembly.residual_global import _get_or_build_eq_model
        from properties.equilibrium import compute_interface_equilibrium

        cache = ctx.meta.get("eq_result_cache")
        eq_model = _get_or_build_eq_model(ctx, state_guess)
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Ts_if = float(state_props.Ts)
            Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yl_face = np.asarray(state_props.Yl[:, il_if], dtype=np.float64)
            Yg_face = np.asarray(state_props.Yg[:, ig_if], dtype=np.float64)
            Yg_eq, y_cond, psat = compute_interface_equilibrium(
                eq_model,
                Ts=Ts_if,
                Pg=Pg_if,
                Yl_face=Yl_face,
                Yg_face=Yg_face,
            )
            eq_result = {"Yg_eq": np.asarray(Yg_eq), "y_cond": np.asarray(y_cond), "psat": np.asarray(psat)}
            eq_source = "computed"
            ctx.meta["eq_result_cache"] = dict(eq_result)
        except Exception:
            if cache is not None:
                eq_result = cache
                eq_source = "cached"
            else:
                raise

    diag["eq_source"] = eq_source
    return state_guess, props, eq_model, eq_result, diag


def _get_petsc():
    try:
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc AIJ builder.") from exc
    return PETSc


def dense_to_aij(
    A: np.ndarray,
    *,
    drop_tol: float = 0.0,
    comm=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convert dense numpy matrix A into PETSc AIJ matrix.

    drop_tol:
      entries with abs(val) <= drop_tol will be dropped (helps remove noise zeros).
    """
    PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    A = np.asarray(A, dtype=PETSc.ScalarType)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square 2D array, got {A.shape}")

    n = int(A.shape[0])
    if n == 0:
        M = PETSc.Mat().createAIJ([0, 0], comm=comm)
        M.setUp()
        diag = {
            "drop_tol": float(drop_tol),
            "nnz_sum": 0,
            "nnz_max": 0,
            "nnz_avg": 0.0,
            "nnz_total": 0,
            "nnz_max_row": 0,
        }
        return M, diag

    absA = np.abs(A)
    finite_mask = np.isfinite(A)
    mask = finite_mask & (absA > float(drop_tol))

    nnz = mask.sum(axis=1).astype(np.int32, copy=False)
    nnz_max = int(nnz.max()) if n > 0 else 0
    nnz_sum = int(nnz.sum())

    M = PETSc.Mat().createAIJ([n, n], nnz=nnz.tolist(), comm=comm)
    M.setUp()

    itype = PETSc.IntType
    stype = PETSc.ScalarType

    for i in range(n):
        cols = np.nonzero(mask[i])[0].astype(itype, copy=False)
        if cols.size == 0:
            continue
        vals = A[i, cols].astype(stype, copy=False)
        M.setValues(int(i), cols, vals, addv=False)

    M.assemblyBegin()
    M.assemblyEnd()

    diag = {
        "drop_tol": float(drop_tol),
        "nnz_sum": nnz_sum,
        "nnz_max": nnz_max,
        "nnz_avg": float(nnz.mean()) if n > 0 else 0.0,
        "nnz_total": nnz_sum,
        "nnz_max_row": nnz_max,
    }
    return M, diag


def _dense_mask_stats(A: np.ndarray, drop_tol: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    absA = np.abs(A)
    finite_mask = np.isfinite(A)
    mask = finite_mask & (absA > float(drop_tol))

    nnz = mask.sum(axis=1).astype(np.int32, copy=False)
    nnz_max = int(nnz.max()) if nnz.size else 0
    nnz_sum = int(nnz.sum())
    diag = {
        "drop_tol": float(drop_tol),
        "nnz_sum": nnz_sum,
        "nnz_max": nnz_max,
        "nnz_avg": float(nnz.mean()) if nnz.size else 0.0,
        "nnz_total": nnz_sum,
        "nnz_max_row": nnz_max,
    }
    return mask, nnz, diag


def _fill_aij_from_dense(M, A: np.ndarray, mask: np.ndarray) -> None:
    PETSc = _get_petsc()
    itype = PETSc.IntType
    stype = PETSc.ScalarType
    n = int(A.shape[0])

    M.zeroEntries()
    for i in range(n):
        cols = np.nonzero(mask[i])[0].astype(itype, copy=False)
        if cols.size == 0:
            continue
        vals = A[i, cols].astype(stype, copy=False)
        M.setValues(int(i), cols, vals, addv=False)

    M.assemblyBegin()
    M.assemblyEnd()


def build_precond_mat_aij_from_A(
    ctx: NonlinearContext,
    u_phys: np.ndarray,
    *,
    drop_tol: float = 0.0,
    comm=None,
    max_nnz_row: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a PETSc AIJ matrix P from dense A(u) and return diagnostics.
    """
    PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD
    if comm.getSize() != 1:
        raise NotImplementedError("Stage 2 precond builder is serial-only for now.")

    mode = _get_linear_assembly_mode(ctx)
    if mode == "native_aij":
        from assembly.build_system_petsc import build_transport_system_petsc_native
        state_guess, props, eq_model, eq_result, diag_ctx = _build_state_props_eq(ctx, u_phys)
        A_p, _b, diag_sys = build_transport_system_petsc_native(
            cfg=ctx.cfg,
            grid=ctx.grid_ref,
            layout=ctx.layout,
            state_old=ctx.state_old,
            props=props,
            dt=float(ctx.dt),
            state_guess=state_guess,
            eq_model=eq_model,
            eq_result=eq_result,
            return_diag=True,
            comm=comm,
        )
        diag = {"assembly_mode": "native_aij"}
        diag.update(diag_ctx)
        if isinstance(diag_sys, dict) and diag_sys:
            diag["diag_sys"] = diag_sys
        diag["shape"] = tuple(A_p.getSize())
        return A_p, diag

    from assembly.residual_global import build_transport_system_from_ctx

    A_dense, _b = build_transport_system_from_ctx(ctx, u_phys)
    A_dense = np.asarray(A_dense, dtype=PETSc.ScalarType)
    if A_dense.ndim != 2 or A_dense.shape[0] != A_dense.shape[1]:
        raise ValueError(f"A must be square 2D array, got {A_dense.shape}")

    n = int(A_dense.shape[0])
    mask, nnz, diag = _dense_mask_stats(A_dense, drop_tol)
    if max_nnz_row is not None:
        max_nnz_row = int(max_nnz_row)
        nnz_max = int(nnz.max()) if nnz.size else 0
        if max_nnz_row < nnz_max:
            max_nnz_row = nnz_max
        nnz_alloc = max_nnz_row
        diag["nnz_max_row_alloc"] = max_nnz_row
    else:
        nnz_alloc = nnz.tolist()

    P = PETSc.Mat().createAIJ([n, n], nnz=nnz_alloc, comm=comm)
    P.setUp()
    _fill_aij_from_dense(P, A_dense, mask)
    diag["shape"] = tuple(A_dense.shape)
    return P, diag


def fill_precond_mat_aij_from_A(
    P,
    ctx: NonlinearContext,
    u_phys: np.ndarray,
    *,
    drop_tol: float = 0.0,
) -> Dict[str, Any]:
    """
    Refill an existing PETSc AIJ matrix P with values from dense A(u).
    """
    PETSc = _get_petsc()
    mode = _get_linear_assembly_mode(ctx)
    if mode == "native_aij":
        from assembly.build_system_petsc import build_transport_system_petsc_native
        state_guess, props, eq_model, eq_result, diag_ctx = _build_state_props_eq(ctx, u_phys)
        A_new, _b, diag_sys = build_transport_system_petsc_native(
            cfg=ctx.cfg,
            grid=ctx.grid_ref,
            layout=ctx.layout,
            state_old=ctx.state_old,
            props=props,
            dt=float(ctx.dt),
            state_guess=state_guess,
            eq_model=eq_model,
            eq_result=eq_result,
            return_diag=True,
            comm=P.getComm(),
        )
        if P.getSize() != A_new.getSize():
            raise ValueError(f"P size {P.getSize()} does not match native A size {A_new.getSize()}")
        P.zeroEntries()
        try:
            P.axpy(1.0, A_new, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        except Exception:
            P.axpy(1.0, A_new)
        P.assemblyBegin()
        P.assemblyEnd()
        diag = {"assembly_mode": "native_aij"}
        diag.update(diag_ctx)
        if isinstance(diag_sys, dict) and diag_sys:
            diag["diag_sys"] = diag_sys
        diag["shape"] = tuple(A_new.getSize())
        return diag

    from assembly.residual_global import build_transport_system_from_ctx

    A_dense, _b = build_transport_system_from_ctx(ctx, u_phys)
    A_dense = np.asarray(A_dense, dtype=PETSc.ScalarType)
    if A_dense.ndim != 2 or A_dense.shape[0] != A_dense.shape[1]:
        raise ValueError(f"A must be square 2D array, got {A_dense.shape}")

    n = int(A_dense.shape[0])
    size = P.getSize()
    if size != (n, n):
        raise ValueError(f"P size {size} does not match A shape {(n, n)}")

    mask, _nnz, diag = _dense_mask_stats(A_dense, drop_tol)
    _fill_aij_from_dense(P, A_dense, mask)
    diag["shape"] = tuple(A_dense.shape)
    return diag


def build_precond_aij_from_ctx(
    ctx: NonlinearContext,
    u_phys: np.ndarray,
    *,
    drop_tol: float = 0.0,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build P ~ A(u) as PETSc AIJ matrix from current nonlinear context (serial Stage 2).
    """
    return build_precond_mat_aij_from_A(ctx, u_phys, drop_tol=drop_tol)
