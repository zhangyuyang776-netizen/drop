from __future__ import annotations

import numpy as np

from core.types import CaseConfig, Grid1D, State
from core.layout import UnknownLayout


def _interp_cell_centered(r_old: np.ndarray, v_old: np.ndarray, r_new: np.ndarray) -> np.ndarray:
    """Simple 1D linear interpolation for cell-centered quantities."""
    return np.interp(r_new, r_old, v_old)


def _reconstruct_gas_closure(Yg_full: np.ndarray, layout: UnknownLayout) -> np.ndarray:
    """Recompute gas closure species so each column sums to one (includes unsolved background species)."""
    k_cl = getattr(layout, "gas_closure_index", None)
    if k_cl is None:
        return Yg_full
    if k_cl >= Yg_full.shape[0]:
        raise ValueError(f"Gas closure index {k_cl} out of bounds for Yg_full shape {Yg_full.shape}")

    # Sum over all species except closure itself
    sum_other = np.sum(Yg_full, axis=0) - Yg_full[k_cl, :]
    closure = 1.0 - sum_other

    # Soft clamp small numerical violations
    tol = 1e-10
    closure = np.where((closure < 0.0) & (closure >= -tol), 0.0, closure)
    closure = np.where((closure > 1.0) & (closure <= 1.0 + tol), 1.0, closure)
    closure = np.clip(closure, 0.0, 1.0)

    Yg_full[k_cl, :] = closure

    # Optional final renormalization to mitigate interpolation noise
    sums = np.sum(Yg_full, axis=0)
    mask = sums > 0.0
    Yg_full[:, mask] /= sums[mask]

    return Yg_full


def remap_state_to_new_grid(
    state_old: State,
    grid_old: Grid1D,
    grid_new: Grid1D,
    cfg: CaseConfig,
    layout: UnknownLayout,
) -> State:
    """
    Remap state arrays from grid_old to grid_new (cell-centered linear interpolation).

    Assumes Nl/Ng are unchanged; only geometry (r) moves with Rd.
    """
    state = state_old.copy()

    Nl, Ng = grid_old.Nl, grid_old.Ng
    r_c_old = grid_old.r_c
    r_c_new = grid_new.r_c

    r_c_liq_old = r_c_old[:Nl]
    r_c_gas_old = r_c_old[Nl:]
    r_c_liq_new = r_c_new[:Nl]
    r_c_gas_new = r_c_new[Nl:]

    # Temperatures
    if state.Tl.size:
        state.Tl = _interp_cell_centered(r_c_liq_old, state.Tl, r_c_liq_new)
    if state.Tg.size:
        state.Tg = _interp_cell_centered(r_c_gas_old, state.Tg, r_c_gas_new)

    # Liquid species
    if state.Yl.size:
        Ns_l, _ = state.Yl.shape
        Yl_new = np.empty((Ns_l, Nl), dtype=np.float64)
        for k in range(Ns_l):
            Yl_new[k, :] = _interp_cell_centered(r_c_liq_old, state.Yl[k, :], r_c_liq_new)
        state.Yl = Yl_new

    # Gas species
    if state.Yg.size:
        Ns_g, _ = state.Yg.shape
        Yg_new = np.empty((Ns_g, Ng), dtype=np.float64)
        for k in range(Ns_g):
            Yg_new[k, :] = _interp_cell_centered(r_c_gas_old, state.Yg[k, :], r_c_gas_new)
        Yg_new = _reconstruct_gas_closure(Yg_new, layout)
        Yg_new = np.clip(Yg_new, 0.0, 1.0)
        state.Yg = Yg_new

    # Scalars (Ts/mpp/Rd) stay as provided in state_old
    return state
