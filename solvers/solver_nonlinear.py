"""
Nonlinear solve dispatcher for SciPy/PETSc backends.
"""

from __future__ import annotations

import numpy as np

from solvers.nonlinear_types import NonlinearSolveResult


def solve_nonlinear(ctx, u0: np.ndarray) -> NonlinearSolveResult:
    """Dispatch to SciPy or PETSc nonlinear solver based on cfg.nonlinear.backend."""
    cfg = ctx.cfg
    nl = getattr(cfg, "nonlinear", None)
    if nl is None or not getattr(nl, "enabled", False):
        raise ValueError("solve_nonlinear() called but cfg.nonlinear.enabled is False/missing")
    backend = str(getattr(nl, "backend", "scipy")).strip().lower()
    backend_alias = {
        "petsc": "petsc",
        "snes": "petsc",
        "petsc_snes": "petsc",
        "scipy": "scipy",
    }
    backend = backend_alias.get(backend, backend)

    if backend == "scipy":
        from solvers.newton_scipy import solve_nonlinear_scipy
        return solve_nonlinear_scipy(ctx, u0)
    if backend == "petsc":
        from solvers.petsc_snes import solve_nonlinear_petsc
        return solve_nonlinear_petsc(ctx, u0)

    raise ValueError(f"Unknown nonlinear backend: {backend!r}")
