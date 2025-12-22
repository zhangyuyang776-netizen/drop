"""
SciPy-based nonlinear solver wrapper (global Newton/Krylov).

This module only coordinates solver calls; residual assembly is handled elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import optimize

from solvers.nonlinear_context import NonlinearContext
from assembly.residual_global import residual_only

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NewtonDiagnostics:
    converged: bool
    method: str
    n_iter: int
    res_norm_2: float
    res_norm_inf: float
    history_res_inf: List[float] = field(default_factory=list)
    message: Optional[str] = None


@dataclass(slots=True)
class NewtonSolveResult:
    u: np.ndarray
    diag: NewtonDiagnostics


def solve_nonlinear_scipy(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NewtonSolveResult:
    """
    Solve F(u)=0 using SciPy nonlinear solvers with optional scaling.
    """
    cfg = ctx.cfg
    nl = getattr(cfg, "nonlinear", None)
    if nl is None or not getattr(nl, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    solver = str(getattr(nl, "solver", "newton_krylov"))
    krylov_method = str(getattr(nl, "krylov_method", "lgmres"))
    max_outer_iter = int(getattr(nl, "max_outer_iter", 20))
    inner_maxiter = int(getattr(nl, "inner_maxiter", 20))
    f_rtol = float(getattr(nl, "f_rtol", 1.0e-6))
    f_atol = float(getattr(nl, "f_atol", 1.0e-10))
    use_scaled_u = bool(getattr(nl, "use_scaled_unknowns", True))
    use_scaled_res = bool(getattr(nl, "use_scaled_residual", True))
    verbose = bool(getattr(nl, "verbose", False))

    if use_scaled_u:
        u0_s = ctx.to_scaled_u(u0)
    else:
        u0_s = np.asarray(u0, dtype=np.float64)

    history: List[float] = []
    scale = np.asarray(ctx.scale_u, dtype=np.float64)
    scale_safe = np.where(scale > 0.0, scale, 1.0)

    def _F_scaled(u_s: np.ndarray) -> np.ndarray:
        if use_scaled_u:
            u_phys = ctx.from_scaled_u(u_s)
        else:
            u_phys = u_s

        res = residual_only(u_phys, ctx)
        if use_scaled_res:
            res = res / scale_safe

        res_norm_inf = float(np.linalg.norm(res, ord=np.inf))
        history.append(res_norm_inf)
        return res

    converged = False
    msg = None

    if solver == "newton_krylov":
        try:
            sol_s = optimize.newton_krylov(
                _F_scaled,
                u0_s,
                method=krylov_method,
                inner_maxiter=inner_maxiter,
                maxiter=max_outer_iter,
                f_rtol=f_rtol,
                f_tol=f_atol,
                verbose=verbose,
            )
            converged = True
        except optimize.NoConvergence as exc:
            sol_raw = getattr(exc, "x", None)
            if sol_raw is None:
                sol_raw = exc.args[0] if exc.args else u0_s
            sol_s = np.asarray(sol_raw, dtype=np.float64)
            converged = False
            msg = str(exc)
            logger.warning("newton_krylov did not converge: %s", msg)
    elif solver in ("root_hybr", "hybr"):
        sol = optimize.root(
            _F_scaled,
            u0_s,
            method="hybr",
            tol=f_rtol,
            options={"maxfev": max_outer_iter},
        )
        sol_s = np.asarray(sol.x, dtype=np.float64)
        converged = bool(sol.success)
        msg = None if converged else str(sol.message)
        if not converged:
            logger.warning("root(hybr) did not converge: %s", msg)
    else:
        raise ValueError(f"Unknown cfg.nonlinear.solver={solver!r}")

    if use_scaled_u:
        u_final = ctx.from_scaled_u(sol_s)
    else:
        u_final = np.asarray(sol_s, dtype=np.float64)

    res_final = residual_only(u_final, ctx)
    res_norm_2 = float(np.linalg.norm(res_final))
    res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))
    n_iter = len(history)

    diag = NewtonDiagnostics(
        converged=converged,
        method=solver,
        n_iter=n_iter,
        res_norm_2=res_norm_2,
        res_norm_inf=res_norm_inf,
        history_res_inf=history,
        message=msg,
    )
    return NewtonSolveResult(u=u_final, diag=diag)
