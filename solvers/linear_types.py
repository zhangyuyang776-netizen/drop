"""
Shared linear solver result types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LinearSolveResult:
    x: np.ndarray
    converged: bool
    n_iter: int
    residual_norm: float
    rel_residual: float
    method: str
    message: Optional[str] = None
    diag: Optional[Dict[str, Any]] = None


class JacobianMode:
    """
    Canonical names and helpers for PETSc Jacobian / preconditioner modes.

    These are strings to stay compatible with cfg.petsc.jacobian_mode.
    """

    FD: str = "fd"
    MF: str = "mf"
    MFPC_SPARSE_FD: str = "mfpc_sparse_fd"
    MFPC_AIJA: str = "mfpc_aija"

    _ALIASES = {
        "mfpc_sparse": MFPC_SPARSE_FD,
        "mfpc_sparse_fd": MFPC_SPARSE_FD,
        "mfpc_aij": MFPC_AIJA,
        "mfpc_aija": MFPC_AIJA,
    }

    @classmethod
    def normalize(cls, value: Optional[str]) -> str:
        """
        Normalize user/cfg input into one of the canonical strings.

        Returns the lowercased string if it's not a known alias.
        """
        if value is None:
            return cls.FD
        v = str(value).strip().lower()
        if not v:
            return cls.FD
        if v in cls._ALIASES:
            return cls._ALIASES[v]
        return v

    @classmethod
    def canonical_set(cls) -> set[str]:
        return {cls.FD, cls.MF, cls.MFPC_SPARSE_FD, cls.MFPC_AIJA}
