"""
Minimal output helpers (Step 12.4):
- write_step_scalars: append key scalars per timestep to a CSV.
- write_step_spatial: placeholder for future spatial field output.
"""

from __future__ import annotations

import csv
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


def write_step_spatial(cfg: CaseConfig, grid: Grid1D, state: State, step_id: int | None = None, t: float | None = None) -> None:
    """
    Write spatial snapshot to ``spatial/snapshot_XXXXXX.npz`` under the case directory.

    - Uses cfg.io.fields.{gas, liquid, interface} as allow-lists.
    - Includes r_c/r_f (if present) and a running r_index for convenience.
    - Adds step_id/t when provided by the caller.
    """
    out_dir = (Path(cfg.paths.case_dir) / "spatial") if hasattr(cfg, "paths") else Path("spatial")
    out_dir.mkdir(parents=True, exist_ok=True)

    counter_path = out_dir / "_spatial_index.txt"
    try:
        idx = int(counter_path.read_text().strip())
    except FileNotFoundError:
        idx = 0
    except ValueError:
        idx = 0
    counter_path.write_text(str(idx + 1))

    out_path = out_dir / f"snapshot_{idx:06d}.npz"

    data = {}

    # Grid metadata (best effort)
    try:
        rc = getattr(grid, "r_c", None)
        rf = getattr(grid, "r_f", None)
        if rc is not None:
            rc_arr = np.asarray(rc)
            data["r_c"] = rc_arr
            data["r_index"] = np.arange(rc_arr.size)
        if rf is not None:
            data["r_f"] = np.asarray(rf)
    except Exception:
        pass

    if step_id is not None:
        data["step_id"] = np.asarray(step_id)
    if t is not None:
        data["t"] = np.asarray(t)

    fields = getattr(cfg, "io", None)
    field_cfg = getattr(fields, "fields", None) if fields is not None else None

    gas_fields = getattr(field_cfg, "gas", []) or []
    for name in gas_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    liq_fields = getattr(field_cfg, "liquid", []) or []
    for name in liq_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    iface_fields = getattr(field_cfg, "interface", []) or []
    for name in iface_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    np.savez(out_path, **data)
