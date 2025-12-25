"""
Build sparse FD Jacobian in PETSc AIJ using a conservative pattern and coloring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from solvers.nonlinear_context import NonlinearContext
from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern


def _get_petsc():
    try:
        from petsc4py import PETSc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PETSc not available") from exc
    return PETSc


def build_sparse_fd_jacobian(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian in the scaled (eval) space using a coloring of the pattern.
    """
    PETSc = _get_petsc()
    comm = PETSc.COMM_WORLD
    if comm.getSize() != 1:
        raise NotImplementedError("Sparse FD Jacobian builder is serial-only for Stage 2.")

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
