"""
Build sparse FD Jacobian in PETSc AIJ using a conservative pattern and coloring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from solvers.nonlinear_context import NonlinearContext
from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
from parallel.mat_prealloc import (
    get_global_ownership_ranges_from_vec,
    build_owner_map_from_ownership_ranges,
    count_diag_off_nnz_for_local_rows,
)

logger = logging.getLogger(__name__)


def _get_petsc():
    try:
        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PETSc not available") from exc
    return PETSc


def _mat_setvalues_local(mat, rows, cols, values, *, addv: bool = False) -> None:
    """
    Thin wrapper around PETSc.Mat.setValues.

    Intended for test-time monkeypatching to validate local-row writes.
    """
    mat.setValues(rows, cols, values, addv=addv)


def _res_phys_to_eval_rows(
    res_phys_rows: np.ndarray,
    scale_F_safe: np.ndarray,
    row_ids_global: np.ndarray,
) -> np.ndarray:
    """
    Convert residual rows in physical space to evaluation space.
    """
    if scale_F_safe is None:
        return res_phys_rows
    if scale_F_safe.ndim != 1:
        scale_F_safe = scale_F_safe.ravel()
    scale_rows = scale_F_safe[row_ids_global]
    return res_phys_rows / scale_rows


def _build_sparse_fd_jacobian_serial(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
    *,
    PETSc=None,
    comm=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian in the scaled (eval) space using a coloring of the pattern.
    """
    if PETSc is None:
        PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    x0 = np.asarray(x0, dtype=np.float64)
    N = int(x0.size)

    scale_u = np.asarray(ctx.scale_u, dtype=np.float64)
    if scale_u.shape != x0.shape:
        raise ValueError(f"scale_u shape {scale_u.shape} does not match x0 {x0.shape}")
    scale_u_safe = np.where(scale_u > 0.0, scale_u, 1.0)

    scale_F = np.asarray(ctx.meta.get("residual_scale_F", np.ones_like(x0)), dtype=np.float64)
    if scale_F.shape != x0.shape:
        raise ValueError(f"residual_scale_F shape {scale_F.shape} does not match x0 {x0.shape}")
    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)

    from assembly.residual_global import residual_only

    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        return x_arr * scale_u_safe

    def _res_phys_to_eval(res_phys: np.ndarray) -> np.ndarray:
        return res_phys / scale_F_safe

    u0_phys = _x_to_u_phys(x0)
    r0_phys = residual_only(u0_phys, ctx)
    if not np.all(np.isfinite(r0_phys)):
        r0_phys = np.where(np.isfinite(r0_phys), r0_phys, 1.0e20)
    r0_eval = _res_phys_to_eval(r0_phys)
    if r0_eval.shape != x0.shape:
        raise ValueError(f"residual shape {r0_eval.shape} does not match x0 {x0.shape}")

    if pattern is None:
        pattern = build_jacobian_pattern(ctx.cfg, ctx.grid_ref, ctx.layout)
    indptr = np.asarray(pattern.indptr, dtype=np.int32)
    indices = np.asarray(pattern.indices, dtype=np.int32)
    if indptr.size != N + 1:
        raise ValueError(f"pattern indptr size {indptr.size} does not match N+1 {N+1}")

    col_adj: List[set[int]] = [set() for _ in range(N)]
    col_rows: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        row_cols = indices[indptr[i] : indptr[i + 1]]
        for c in row_cols:
            col_rows[int(c)].append(i)
        for a in range(row_cols.size):
            ca = int(row_cols[a])
            for b in range(a + 1, row_cols.size):
                cb = int(row_cols[b])
                if ca == cb:
                    continue
                col_adj[ca].add(cb)
                col_adj[cb].add(ca)

    order = sorted(range(N), key=lambda j: len(col_adj[j]), reverse=True)
    colors = [-1] * N
    ncolors = 0
    for j in order:
        used = {colors[nbr] for nbr in col_adj[j] if colors[nbr] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[j] = c
        if c + 1 > ncolors:
            ncolors = c + 1

    groups: List[List[int]] = [[] for _ in range(ncolors)]
    for j, c in enumerate(colors):
        groups[c].append(j)

    if mat is None:
        row_nnz = indptr[1:] - indptr[:-1]
        max_nnz_row = int(row_nnz.max()) if N > 0 else 0
        P = PETSc.Mat().createAIJ(size=(N, N), nnz=max_nnz_row, comm=comm)
        P.setUp()
    else:
        P = mat
        P.zeroEntries()

    n_fd_calls = 0
    x_work = x0.copy()

    for color_idx, cols_in_color in enumerate(groups):
        if not cols_in_color:
            continue

        dx_by_col: Dict[int, float] = {}
        for j in cols_in_color:
            dx = eps * (1.0 + abs(x0[j]))
            if dx == 0.0:
                dx = eps
            x_work[j] = x0[j] + dx
            dx_by_col[j] = dx

        u_phys = _x_to_u_phys(x_work)
        r_phys = residual_only(u_phys, ctx)
        if not np.all(np.isfinite(r_phys)):
            r_phys = np.where(np.isfinite(r_phys), r_phys, 1.0e20)
        r_eval = _res_phys_to_eval(r_phys)
        diff = r_eval - r0_eval
        n_fd_calls += 1

        for j in cols_in_color:
            rows = col_rows[j]
            if not rows:
                continue
            dx = dx_by_col[j]
            vals = diff[rows] / dx
            for row, val in zip(rows, vals):
                if drop_tol > 0.0 and abs(val) < drop_tol:
                    continue
                P.setValue(int(row), int(j), float(val), addv=False)

        for j in cols_in_color:
            x_work[j] = x0[j]

    P.assemblyBegin()
    P.assemblyEnd()

    ia, _ja, a_vals = P.getValuesCSR()
    nnz_total = int(a_vals.size)
    nnz_max = int((ia[1:] - ia[:-1]).max()) if N > 0 else 0
    nnz_avg = float(nnz_total) / float(N) if N > 0 else 0.0

    stats: Dict[str, Any] = {
        "ncolors": int(ncolors),
        "n_fd_calls": int(n_fd_calls),
        "nnz_total": nnz_total,
        "nnz_max_row": nnz_max,
        "nnz_avg": nnz_avg,
        "shape": (N, N),
        "eps": float(eps),
        "drop_tol": float(drop_tol),
        "pattern_nnz": int(indices.size),
    }
    return P, stats


def _build_sparse_fd_jacobian_mpi(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
    *,
    PETSc=None,
    comm=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian in MPI mode (local rows only, no coloring/prealloc).
    """
    if PETSc is None:
        PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    x0 = np.asarray(x0, dtype=np.float64)
    N = int(x0.size)
    if N == 0:
        raise ValueError("x0 must be non-empty for MPI Jacobian build.")

    from assembly.jacobian_pattern_dist import build_jacobian_pattern_local
    from assembly.residual_global import residual_only_owned_rows

    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        """
        并行 PETSc 后端：x0 就是物理 unknown（layout 向量），不做缩放。
        """
        return x_arr

    def _res_phys_to_eval(res_phys: np.ndarray) -> np.ndarray:
        """
        并行 PETSc 后端：在物理残差空间工作，不做 residual_scale_F 缩放。
        这里只做 NaN/Inf 防护。
        """
        if not np.all(np.isfinite(res_phys)):
            res_phys = np.where(np.isfinite(res_phys), res_phys, 1.0e20)
        return res_phys


    def _get_ownership_info(mat_local=None) -> Tuple[Tuple[int, int], np.ndarray]:
        rstart = None
        rend = None
        ranges = None

        if mat_local is not None:
            try:
                rstart, rend = mat_local.getOwnershipRange()
            except Exception:
                rstart, rend = None, None
            try:
                ranges = mat_local.getOwnershipRanges()
            except Exception:
                ranges = None

            if ranges is None:
                vec = None
                try:
                    if hasattr(mat_local, "getVecs"):
                        vec, _ = mat_local.getVecs()
                    elif hasattr(mat_local, "createVecs"):
                        vec, _ = mat_local.createVecs()
                except Exception:
                    vec = None
                if vec is not None:
                    try:
                        ranges = get_global_ownership_ranges_from_vec(vec)
                        if rstart is None or rend is None:
                            rstart, rend = vec.getOwnershipRange()
                    finally:
                        try:
                            vec.destroy()
                        except Exception:
                            pass

        if ranges is None or rstart is None or rend is None:
            dm_ctx = None
            try:
                dm_ctx = ctx.meta.get("dm", None)
            except Exception:
                dm_ctx = None
            if dm_ctx is None:
                dm_ctx = getattr(ctx, "dm", None)
            if dm_ctx is not None:
                vec = None
                try:
                    vec = dm_ctx.createGlobalVec()
                    rstart, rend = vec.getOwnershipRange()
                    ranges = get_global_ownership_ranges_from_vec(vec)
                except Exception:
                    pass
                finally:
                    if vec is not None:
                        try:
                            vec.destroy()
                        except Exception:
                            pass

        if ranges is None or rstart is None or rend is None:
            tmp = PETSc.Vec().create(comm=comm)
            tmp.setSizes(N)
            tmp.setFromOptions()
            tmp.setUp()
            rstart, rend = tmp.getOwnershipRange()
            ranges = get_global_ownership_ranges_from_vec(tmp)
            tmp.destroy()

        return (int(rstart), int(rend)), ranges

    ownership_range, ownership_ranges = _get_ownership_info(mat)
    rstart, rend = ownership_range
    n_owned = int(rend - rstart)

    local_pattern = build_jacobian_pattern_local(
        ctx.cfg,
        ctx.grid_ref,
        ctx.layout,
        ownership_range=ownership_range,
    )
    rows_global = np.asarray(local_pattern.rows_global, dtype=PETSc.IntType)
    indices = np.asarray(local_pattern.indices, dtype=PETSc.IntType)
    indptr = np.asarray(local_pattern.indptr, dtype=np.int64)
    nloc = int(rows_global.size)
    if rows_global.size:
        rmin = int(rows_global.min())
        rmax = int(rows_global.max())
        if rmin < rstart or rmax >= rend:
            raise RuntimeError(
                "[FD Jacobian] pattern rows not local: "
                f"min={rmin}, max={rmax}, ownership=[{rstart}, {rend})"
            )

    if nloc == 0 or indices.size == 0:
        if mat is None:
            P = PETSc.Mat().createAIJ(size=((n_owned, N), (n_owned, N)), comm=comm)
            try:
                P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            except Exception:
                pass
            P.setUp()
        else:
            P = mat
            try:
                P.zeroEntries()
            except Exception:
                pass
            try:
                P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            except Exception:
                pass
        stats: Dict[str, Any] = {
            "ncolors": 0,
            "n_fd_calls": 0,
            "nnz_total_local": 0,
            "n_local_rows": int(nloc),
            "shape": (N, N),
            "eps": float(eps),
            "drop_tol": float(drop_tol),
            "pattern_nnz_local": int(indices.size),
            "prealloc_nnz_local": 0,
            "cols_to_perturb": [],
            "ownership_range": ownership_range,
            "mpi_size": int(comm.getSize()),
            "max_diag_abs_local": 0.0,
            "pattern_local": local_pattern,
        }
        return P, stats

    owner_map = build_owner_map_from_ownership_ranges(ownership_ranges)
    myrank = int(comm.getRank())
    debug_fd = os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1"
    debug_fd_once = os.environ.get("DROPLET_PETSC_DEBUG_ONCE", "1") == "1"
    debug_fd_logged = False

    local_row_cols: List[List[int]] = []
    col_to_rows: Dict[int, List[int]] = {}
    for k in range(nloc):
        start = int(indptr[k])
        end = int(indptr[k + 1])
        cols = indices[start:end]
        gi = int(rows_global[k])
        cols_list = [int(c) for c in cols]
        if gi not in cols_list:
            cols_list.append(gi)
        cols_list = sorted(set(cols_list))
        local_row_cols.append(cols_list)
        for c in cols_list:
            col_to_rows.setdefault(c, []).append(k)

    d_nz, o_nz = count_diag_off_nnz_for_local_rows(
        local_rows_global=rows_global,
        local_row_cols=local_row_cols,
        owner_map=owner_map,
        myrank=myrank,
        ownership_range=ownership_range,
    )
    d_nz = np.asarray(d_nz, dtype=PETSc.IntType)
    o_nz = np.asarray(o_nz, dtype=PETSc.IntType)
    prealloc_nnz_local = int(d_nz.sum() + o_nz.sum())

    if mat is None:
        P = PETSc.Mat().createAIJ(size=((n_owned, N), (n_owned, N)), nnz=(d_nz, o_nz), comm=comm)
        P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        P.setUp()
    else:
        P = mat
        try:
            P.zeroEntries()
        except Exception:
            pass
        try:
            P.setPreallocationNNZ((d_nz, o_nz))
        except Exception:
            pass
        try:
            P.setUp()
        except Exception:
            pass
        try:
            P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        except Exception:
            pass

    cols_to_perturb = np.asarray(sorted(col_to_rows.keys()), dtype=PETSc.IntType)
    rows_global_list = rows_global

    local_row_ids = rows_global_list.astype(np.int64) - int(rstart)
    u0_phys = _x_to_u_phys(x0)
    r0_phys = residual_only_owned_rows(u0_phys, ctx, ownership_range)
    if not np.all(np.isfinite(r0_phys)):
        r0_phys = np.where(np.isfinite(r0_phys), r0_phys, 1.0e20)
    r0_eval = _res_phys_to_eval(r0_phys[local_row_ids])
    if r0_eval.shape != (nloc,):
        raise ValueError(f"local residual shape {r0_eval.shape} does not match nloc {nloc}")

    x_work = x0.copy()
    def _assert_local_rows(rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        rmin = int(rows.min())
        rmax = int(rows.max())
        if rmin < rstart or rmax >= rend:
            raise RuntimeError(
                f"[FD Jacobian] rank {myrank} writing non-local rows "
                f"[{rmin}, {rmax}] with ownership [{rstart}, {rend})"
            )
    n_fd_calls = 0
    for j in cols_to_perturb:
        j = int(j)
        iloc_list = col_to_rows.get(j)
        if not iloc_list:
            continue
        dx = eps * (1.0 + abs(x0[j]))
        if dx == 0.0:
            dx = eps

        x_work[j] = x0[j] + dx
        if debug_fd and (not debug_fd_once or not debug_fd_logged):
            try:
                dx_inf = float(np.max(np.abs(x_work - x0)))
                logger.warning(
                    "[DBG][rank=%d] MFPC_FD eval: ||xpert-x||_inf=%.3e col=%d",
                    myrank,
                    dx_inf,
                    j,
                )
                try:
                    state_dbg = ctx.make_state_from_u(x_work)
                    Yg = np.asarray(state_dbg.Yg, dtype=np.float64)
                    cell_idx = 7
                    if Yg.ndim == 2 and Yg.shape[1] > cell_idx:
                        Y = Yg[:, cell_idx]
                        s = float(np.sum(Y)) if Y.size else 0.0
                        mn = float(np.min(Y)) if Y.size else 0.0
                        mx = float(np.max(Y)) if Y.size else 0.0
                        idx = np.argsort(-Y)[:3] if Y.size else np.array([], dtype=np.int64)
                        top = [(int(i), float(Y[i])) for i in idx]
                        logger.warning(
                            "[DBG][rank=%d] MFPC_FD_EVAL Y@cell%d sum=%.6g min=%.3e max=%.3e top=%s",
                            myrank,
                            cell_idx,
                            s,
                            mn,
                            mx,
                            top,
                        )
                except Exception as exc:
                    logger.warning("[DBG][rank=%d] MFPC_FD_EVAL Y debug failed: %r", myrank, exc)
            except Exception as exc:
                logger.warning("[DBG][rank=%d] MFPC_FD eval debug failed: %r", myrank, exc)
            debug_fd_logged = True
        u1_phys = _x_to_u_phys(x_work)
        r1_phys = residual_only_owned_rows(u1_phys, ctx, ownership_range)
        if not np.all(np.isfinite(r1_phys)):
            r1_phys = np.where(np.isfinite(r1_phys), r1_phys, 1.0e20)
        r1_eval = _res_phys_to_eval(r1_phys[local_row_ids])

        dF_local = (r1_eval - r0_eval) / dx
        rows_sel = rows_global_list[iloc_list]
        vals_sel = dF_local[iloc_list]

        if drop_tol > 0.0:
            mask = np.abs(vals_sel) >= drop_tol
            if np.any(mask):
                rows_use = rows_sel[mask]
                _assert_local_rows(rows_use)
                _mat_setvalues_local(
                    P,
                    rows_use,
                    np.asarray([j], dtype=PETSc.IntType),
                    vals_sel[mask].reshape(-1, 1),
                    addv=False,
                )
        else:
            _assert_local_rows(rows_sel)
            _mat_setvalues_local(
                P,
                rows_sel,
                np.asarray([j], dtype=PETSc.IntType),
                vals_sel.reshape(-1, 1),
                addv=False,
            )

        x_work[j] = x0[j]
        n_fd_calls += 1

    P.assemblyBegin()
    P.assemblyEnd()

    nnz_local = 0
    max_diag_abs_local = 0.0
    for gi in rows_global_list:
        cols, vals = P.getRow(int(gi))
        nnz_local += int(len(cols))
        mask = cols == gi
        if np.any(mask):
            diag_val = float(vals[mask][0])
            max_diag_abs_local = max(max_diag_abs_local, abs(diag_val))
        if hasattr(P, "restoreRow"):
            P.restoreRow(int(gi), cols, vals)

    stats = {
        "ncolors": 0,
        "n_fd_calls": int(n_fd_calls),
        "nnz_total_local": int(nnz_local),
        "n_local_rows": int(nloc),
        "shape": (N, N),
        "eps": float(eps),
        "drop_tol": float(drop_tol),
        "pattern_nnz_local": int(indices.size),
        "prealloc_nnz_local": int(prealloc_nnz_local),
        "cols_to_perturb": [int(c) for c in cols_to_perturb],
        "ownership_range": ownership_range,
        "mpi_size": int(comm.getSize()),
        "max_diag_abs_local": float(max_diag_abs_local),
        "pattern_local": local_pattern,
    }
    return P, stats


def build_sparse_fd_jacobian(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian in the scaled (eval) space.

    If mat is provided, its ownership range defines local rows.
    """
    PETSc = _get_petsc()
    comm = PETSc.COMM_WORLD
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")

    if comm.getSize() == 1:
        return _build_sparse_fd_jacobian_serial(
            ctx,
            x0,
            eps=eps,
            drop_tol=drop_tol,
            pattern=pattern,
            mat=mat,
            PETSc=PETSc,
            comm=comm,
        )

    return _build_sparse_fd_jacobian_mpi(
        ctx,
        x0,
        eps=eps,
        drop_tol=drop_tol,
        pattern=pattern,
        mat=mat,
        PETSc=PETSc,
        comm=comm,
    )
