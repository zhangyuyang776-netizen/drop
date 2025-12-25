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
