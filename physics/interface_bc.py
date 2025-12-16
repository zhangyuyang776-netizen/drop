"""
Interface boundary conditions (Step 11.2+ with Ts energy jump).

Responsibilities (current version):
- Resolve interface geometry and unknown indices (Ts / mpp / Rd).
- Provide matrix-row definitions:
  * Ts: conduction (gas/liquid) + latent heat jump.
  * mpp: simplified single-condensable Stefan condition (diffusion-only).
  * Rd: index recorded only; equation elsewhere.
- Record diagnostic info (geometry, indices, equilibrium preview).

Direction and sign conventions:
- Radial coordinate r increases outward (from droplet center to far field).
- Interface normal n = +e_r, pointing from liquid to gas.
- All fluxes (mass, heat) and mpp are defined positive along +e_r
  ("out of the droplet"); mpp > 0 means evaporation (liquid -> gas).

This module DOES:
- assemble Ts and mpp interface equations using provided props/eq_result.

This module MUST NOT:
- compute thermophysical properties,
- normalize mass fractions,
- mutate State or Props.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, State, Props
from core.layout import UnknownLayout

# Placeholder type for equilibrium results; echoed into diag only.
EqResultLike = Optional[Mapping[str, np.ndarray]]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InterfaceRow:
    """Single matrix row definition for an interface unknown.

    Attributes
    ----------
    row : int
        Global row index in the linear system (matches UnknownLayout).
    cols : List[int]
        Global column indices; skeleton stage: can be left empty.
    vals : List[float]
        Coefficients corresponding to cols; skeleton stage: left empty.
    rhs : float
        Right-hand-side contribution.
    """

    row: int
    cols: List[int] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)
    rhs: float = 0.0


@dataclass(slots=True)
class InterfaceCoeffs:
    """Container for all interface-related matrix rows + diagnostics."""

    rows: List[InterfaceRow] = field(default_factory=list)
    diag: Dict[str, Any] = field(default_factory=dict)


def _require_block(layout: UnknownLayout, name: str) -> None:
    """Raise if a required block is missing to catch config/layout mismatch early."""
    if name not in layout.blocks:
        logger.error("Layout missing required block '%s' while cfg requests it.", name)
        raise ValueError(f"Layout missing required block '{name}' while cfg requests it.")


def build_interface_coeffs(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    eq_result: EqResultLike = None,
) -> InterfaceCoeffs:
    """
    Build skeleton matrix-row definitions for interface unknowns (Ts, mpp, Rd).

    This Step 11.2 version:
    - does NOT implement physical formulas,
    - does NOT modify state/props,
    - only resolves indices/geometry and prepares empty rows.

    Parameters
    ----------
    grid : Grid1D
        Spherical mesh with liquid and gas segments; iface_f defines the interface face.
    state : State
        Current state (Tg, Yg, Tl, Yl, Ts, mpp, Rd); Ts/mpp/Rd values are not changed.
    props : Props
        Cell-centered properties (rho, cp, k, D); not used in the skeleton version.
    layout : UnknownLayout
        Global unknown layout; used to query indices for Ts/mpp/Rd and nearby cells.
    cfg : CaseConfig
        Full case configuration; `cfg.physics.include_*` toggles decide which rows exist.
    eq_result : EqResultLike, optional
        Optional interface-equilibrium result produced by `properties.equilibrium`.
        Only passed through into diagnostics in this skeleton version.

    Returns
    -------
    InterfaceCoeffs
        Container with:
        - rows: list of InterfaceRow (one per interface scalar unknown),
        - diag: diagnostic dictionary (geometry, indices, optional equilibrium preview).
    """
    rows: List[InterfaceRow] = []
    diag: Dict[str, Any] = {}

    # Basic availability checks for mpp-dependent equations
    if cfg.physics.include_mpp and layout.has_block("mpp") and eq_result is None:
        logger.error("Interface mpp equation requires eq_result (with Yg_eq).")
        raise ValueError("eq_result must be provided when cfg.physics.include_mpp is True.")

    # Geometry and indexing near the interface
    iface_f = grid.iface_f
    il_global = grid.Nl - 1  # last liquid cell (global)
    ig_global = grid.Nl      # first gas cell (global)

    il_local = grid.Nl - 1   # Tl local index
    ig_local = 0             # Tg local index

    diag["iface_f"] = iface_f
    diag["cell_liq_global"] = il_global
    diag["cell_gas_global"] = ig_global
    diag["cell_liq_local"] = il_local
    diag["cell_gas_local"] = ig_local
    diag["r_if"] = float(grid.r_f[iface_f])
    diag["A_if"] = float(grid.A_f[iface_f])

    # Which interface unknowns exist
    phys = cfg.physics
    has_Ts = phys.include_Ts and ("Ts" in layout.blocks)
    has_mpp = phys.include_mpp and ("mpp" in layout.blocks)
    has_Rd = phys.include_Rd and ("Rd" in layout.blocks)

    diag["include_Ts"] = has_Ts
    diag["include_mpp"] = has_mpp
    diag["include_Rd"] = has_Rd

    # Defensive: cfg requests a variable but layout misses it
    if phys.include_Ts and not has_Ts:
        _require_block(layout, "Ts")
    if phys.include_mpp and not has_mpp:
        _require_block(layout, "mpp")
    if phys.include_Rd and not has_Rd:
        _require_block(layout, "Rd")

    # Global unknown indices (no manual offsets)
    if has_Ts:
        idx_Ts = layout.idx_Ts()
        diag["idx_Ts"] = idx_Ts

        if not has_mpp:
            logger.error("Ts equation (interface energy jump) requires mpp unknown.")
            raise ValueError("Ts equation requires mpp unknown (cfg.physics.include_mpp).")

        idx_mpp = layout.idx_mpp()
        diag["idx_mpp"] = idx_mpp

        ts_row, ts_diag = _build_Ts_row(
            grid=grid,
            state=state,
            props=props,
            layout=layout,
            cfg=cfg,
            il_global=il_global,
            ig_global=ig_global,
            il_local=il_local,
            ig_local=ig_local,
            iface_f=iface_f,
            idx_Ts=idx_Ts,
            idx_mpp=idx_mpp,
        )
        rows.append(ts_row)
        diag.update(ts_diag)

    if has_mpp:
        if "idx_mpp" not in diag:
            idx_mpp = layout.idx_mpp()
            diag["idx_mpp"] = idx_mpp
        else:
            idx_mpp = diag["idx_mpp"]
        mpp_row, mpp_diag = _build_mpp_row(
            grid=grid,
            state=state,
            props=props,
            layout=layout,
            cfg=cfg,
            eq_result=eq_result,
            ig_global=ig_global,
            ig_local=ig_local,
            iface_f=iface_f,
            idx_mpp=idx_mpp,
        )
        rows.append(mpp_row)
        diag.update(mpp_diag)

    if has_Rd:
        idx_Rd = layout.idx_Rd()
        diag["idx_Rd"] = idx_Rd
        # Rd equation row usually built elsewhere (e.g., radius_eq.py); index recorded for diagnostics only.

    # Optional snapshots of nearby state (read-only)
    if state.Tl.size:
        diag["Tl_if"] = float(state.Tl[il_local])
    if state.Tg.size:
        diag["Tg_if"] = float(state.Tg[ig_local])
    if state.Yg.size:
        diag["Yg_if"] = np.array(state.Yg[:, ig_local], copy=True)

    # Equilibrium preview passthrough
    if eq_result is not None:
        diag["equilibrium"] = eq_result

    # Direction/sign conventions for diagnostics
    diag["direction_convention"] = {
        "n": "+e_r (liquid -> gas)",
        "mpp_positive": "evaporation (liquid -> gas)",
        "flux_positive": "outward along +r",
    }

    coeffs = InterfaceCoeffs(rows=rows, diag=diag)
    logger.debug(
        "interface_bc built: %d rows (Ts=%s, mpp=%s, Rd=%s)",
        len(rows),
        has_Ts,
        has_mpp,
        has_Rd,
    )
    return coeffs


def _build_Ts_row(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    il_global: int,
    ig_global: int,
    il_local: int,
    ig_local: int,
    iface_f: int,
    idx_Ts: int,
    idx_mpp: int,
) -> Tuple[InterfaceRow, Dict[str, Any]]:
    """
    Build Ts interface energy-jump row (single-species, conduction + latent).

    Discrete form (per unit area):

        q_g + q_l - q_lat = 0

    with
        q_g   = -k_g * (Tg1 - Ts) / dr_g * A_if
        q_l   = -k_l * (Ts - TlN) / dr_l * A_if
        q_lat = mpp * L_v * A_if

    Matrix row (global unknowns: Tg1, TlN, Ts, mpp):

        A_if * [(-k_g/dr_g) * Tg1
                + (k_g/dr_g - k_l/dr_l) * Ts
                + (k_l/dr_l) * TlN
                - L_v * mpp] = 0
    """
    A_if = float(grid.A_f[iface_f])
    r_if = float(grid.r_f[iface_f])
    dr_g = float(grid.r_c[ig_global] - r_if)
    dr_l = float(r_if - grid.r_c[il_global])
    if dr_g <= 0.0 or dr_l <= 0.0:
        logger.error("Non-positive interface spacings: dr_g=%g, dr_l=%g", dr_g, dr_l)
        raise ValueError(f"Interface spacings must be positive: dr_g={dr_g}, dr_l={dr_l}")

    # Conductivities: follow Props naming used across gas/liquid modules
    try:
        k_g = float(props.k_g[ig_local])
    except Exception as exc:
        logger.error("Failed to access gas thermal conductivity k_g at cell %d: %s", ig_local, exc)
        raise
    try:
        k_l = float(props.k_l[il_local])
    except Exception as exc:
        logger.error("Failed to access liquid thermal conductivity k_l at cell %d: %s", il_local, exc)
        raise

    L_v = _get_latent_heat(props, cfg)

    # Unknown indices near the interface (layout is gas/liquid-local)
    idx_Tg = layout.idx_Tg(ig_local)
    idx_Tl = layout.idx_Tl(il_local)

    coeff_Tg = -A_if * k_g / dr_g
    coeff_Tl = A_if * k_l / dr_l
    coeff_Ts = A_if * (k_g / dr_g - k_l / dr_l)
    coeff_mpp = -A_if * L_v

    cols = [idx_Tg, idx_Ts, idx_Tl, idx_mpp]
    vals = [coeff_Tg, coeff_Ts, coeff_Tl, coeff_mpp]
    rhs = 0.0

    # Diagnostic preview using current state (not mutating state)
    Ts_cur = float(state.Ts)
    Tg_cur = float(state.Tg[ig_local]) if state.Tg.size else np.nan
    Tl_cur = float(state.Tl[il_local]) if state.Tl.size else np.nan
    mpp_cur = float(state.mpp)

    q_g = -k_g * (Tg_cur - Ts_cur) / dr_g * A_if
    q_l = -k_l * (Ts_cur - Tl_cur) / dr_l * A_if
    q_lat = mpp_cur * L_v * A_if

    diag_update: Dict[str, Any] = {
        "Ts_energy": {
            "A_if": A_if,
            "dr_g": dr_g,
            "dr_l": dr_l,
            "k_g": k_g,
            "k_l": k_l,
            "L_v": L_v,
            "q_g": q_g,
            "q_l": q_l,
            "q_lat": q_lat,
            "balance": q_g + q_l - q_lat,
            "coeffs": {
                "Tg": coeff_Tg,
                "Ts": coeff_Ts,
                "Tl": coeff_Tl,
                "mpp": coeff_mpp,
            },
            "state": {
                "Tg_if": Tg_cur,
                "Ts": Ts_cur,
                "Tl_if": Tl_cur,
                "mpp": mpp_cur,
            },
        }
    }

    return InterfaceRow(row=idx_Ts, cols=cols, vals=vals, rhs=rhs), diag_update


def _build_mpp_row(
    grid: Grid1D,
    state: State,
    props: Props,
    layout: UnknownLayout,
    cfg: CaseConfig,
    eq_result: EqResultLike,
    ig_global: int,
    ig_local: int,
    iface_f: int,
    idx_mpp: int,
) -> Tuple[InterfaceRow, Dict[str, Any]]:
    """
    Build mpp interface mass-balance row (single-condensable Stefan-like condition).

    Simplified form (single condensable, ignore convective Stefan term):

        J_cond · n - mpp = 0

    with
        J_cond ≈ -rho_g * D_cond * (Yg_cond,cell - Yg_cond,eq) / dr_g
    """
    if eq_result is None or "Yg_eq" not in eq_result:
        logger.error("mpp equation requires eq_result with 'Yg_eq'.")
        raise ValueError("eq_result with 'Yg_eq' is required for mpp equation.")
    Yg_eq = np.asarray(eq_result["Yg_eq"], dtype=np.float64)

    A_if = float(grid.A_f[iface_f])  # diagnostic only
    r_if = float(grid.r_f[iface_f])
    dr_g = float(grid.r_c[ig_global] - r_if)
    if dr_g <= 0.0:
        logger.error("Non-positive gas-side spacing at interface: dr_g=%g", dr_g)
        raise ValueError(f"Gas-side spacing must be positive: dr_g={dr_g}")

    rho_g = float(props.rho_g[ig_local])

    k_full, k_red, g_name = _get_condensable_indices(cfg, layout)
    if Yg_eq.shape[0] <= k_full:
        raise ValueError(f"eq_result['Yg_eq'] too short for species index {k_full}")
    D_cond = _get_gas_diffusivity(props, cfg, k_full, ig_local)

    Yg_cell_cond = float(state.Yg[k_full, ig_local])
    Yg_eq_cond = float(Yg_eq[k_full])

    coeff_Yg = -rho_g * D_cond / dr_g
    coeff_mpp = -1.0
    rhs = -rho_g * D_cond * Yg_eq_cond / dr_g

    idx_Yg = layout.idx_Yg(k_red, ig_local)

    cols = [idx_Yg, idx_mpp]
    vals = [coeff_Yg, coeff_mpp]

    mpp_cur = float(state.mpp)
    J_cond_cur = -rho_g * D_cond * (Yg_cell_cond - Yg_eq_cond) / dr_g
    R_cur = J_cond_cur - mpp_cur

    diag_update: Dict[str, Any] = {
        "mpp_mass": {
            "rho_g": rho_g,
            "D_cond": D_cond,
            "dr_g": dr_g,
            "k_full": k_full,
            "k_red": k_red,
            "species": g_name,
            "Yg_cell": Yg_cell_cond,
            "Yg_eq": Yg_eq_cond,
            "J_cond": J_cond_cur,
            "mpp": mpp_cur,
            "residual": R_cur,
            "coeffs": {"Yg": coeff_Yg, "mpp": coeff_mpp},
            "rhs": rhs,
            "A_if": A_if,
        }
    }

    return InterfaceRow(row=idx_mpp, cols=cols, vals=vals, rhs=rhs), diag_update


def _get_latent_heat(props: Props, cfg: CaseConfig) -> float:
    """Resolve latent heat L_v; prefer props if available, else cfg fallback."""
    candidates = (
        getattr(props, "h_vap_if", None),
        getattr(props, "lv", None),
        getattr(props, "latent_heat", None),
    )
    for cand in candidates:
        if cand is None:
            continue
        try:
            return float(cand)
        except Exception:
            continue

    L_v_cfg = getattr(cfg.physics, "latent_heat_default", None)
    if L_v_cfg is not None:
        return float(L_v_cfg)

    raise ValueError("Latent heat L_v not provided in props or cfg.physics.latent_heat_default.")


def _get_condensable_indices(
    cfg: CaseConfig,
    layout: UnknownLayout,
) -> Tuple[int, int, str]:
    """
    Return (k_cond_full, k_cond_red, gas_name) for the single condensable species.

    k_cond_full : index in full gas species list (State.Yg / Props.D_g first axis)
    k_cond_red  : reduced-species index used in UnknownLayout Yg block
    gas_name    : gas species name (for diagnostics)
    """
    l_name = cfg.species.liq_balance_species
    g_map = cfg.species.liq2gas_map
    if l_name not in g_map:
        raise ValueError(f"liq_balance_species '{l_name}' not found in liq2gas_map {g_map}")
    g_name = g_map[l_name]

    gas_full = layout.gas_species_full
    if g_name not in gas_full:
        raise ValueError(f"Condensable gas species '{g_name}' not found in gas_species_full {gas_full}")
    k_cond_full = gas_full.index(g_name)

    k_red = layout.gas_full_to_reduced.get(g_name)
    if k_red is None:
        raise ValueError(f"Condensable species '{g_name}' is configured as gas closure; cannot be condensable.")

    return k_cond_full, k_red, g_name


def _get_gas_diffusivity(
    props: Props,
    cfg: CaseConfig,
    k_full: int,
    ig_local: int,
) -> float:
    """
    Return D_g for given full-species index and gas cell.

    Prefer props.D_g if present; optionally fall back to cfg.physics.default_D_g.
    """
    if props.D_g is not None:
        if props.D_g.shape[0] <= k_full or props.D_g.shape[1] <= ig_local:
            raise ValueError(
                f"D_g shape {props.D_g.shape} too small for species {k_full} or cell {ig_local}"
            )
        D_val = float(props.D_g[k_full, ig_local])
        if D_val <= 0.0:
            raise ValueError(f"Non-positive D_g[{k_full},{ig_local}]={D_val}")
        return D_val

    D_default = getattr(cfg.physics, "default_D_g", None)
    if D_default is not None:
        return float(D_default)

    raise ValueError("No D_g available in props and cfg.physics.default_D_g missing.")
