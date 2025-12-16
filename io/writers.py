"""
Minimal output helpers (Step 12.4):
- write_step_scalars: append key scalars per timestep to a CSV.
- write_step_spatial: placeholder for future spatial field output.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.types import CaseConfig, Grid1D, State

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from solvers.timestepper import StepDiagnostics


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_step_scalars(cfg: CaseConfig, t: float, state: State, diag: "StepDiagnostics") -> None:
    """
    Append scalar diagnostics for a timestep to CSV.

    Columns:
    t, Rd, Ts, mpp, Tg_mean, Tl_mean, Tg_if, Tg_far, Tl_center, Tl_if, energy_balance_if, mass_balance_rd
    """
    out_dir = (Path(cfg.paths.case_dir) / "scalars") if hasattr(cfg, "paths") else Path("scalars")
    out_path = out_dir / "scalars.csv"
    _ensure_parent(out_path)

    Tg_mean = float(np.mean(state.Tg)) if state.Tg.size else np.nan
    Tl_mean = float(np.mean(state.Tl)) if state.Tl.size else np.nan
    Tg_if = float(state.Tg[0]) if state.Tg.size else np.nan
    Tg_far = float(state.Tg[-1]) if state.Tg.size else np.nan
    Tl_center = float(state.Tl[0]) if state.Tl.size else np.nan
    Tl_if = float(state.Tl[-1]) if state.Tl.size else np.nan

    row = {
        "t": t,
        "Rd": float(state.Rd),
        "Ts": float(state.Ts),
        "mpp": float(state.mpp),
        "Tg_mean": Tg_mean,
        "Tl_mean": Tl_mean,
        "Tg_if": Tg_if,
        "Tg_far": Tg_far,
        "Tl_center": Tl_center,
        "Tl_if": Tl_if,
        "energy_balance_if": diag.energy_balance_if,
        "mass_balance_rd": diag.mass_balance_rd,
    }

    header = list(row.keys())
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_step_spatial(cfg: CaseConfig, grid: Grid1D, state: State) -> None:
    """
    Placeholder for spatial field output.
    MVP: no-op; can be extended to write npz/csv at selected steps.
    """
    _ = (cfg, grid, state)
    # Implement spatial dumps in later steps as needed.
    return
