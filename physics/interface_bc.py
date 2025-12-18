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
from physics.energy_flux import split_energy_flux_cond_diff_single

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
            il_global=il_global,
            il_local=il_local,
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

    # Unknown indices near the interface
    idx_Tg = layout.idx_Tg(ig_local)

    coeff_Tg = -A_if * k_g / dr_g
    coeff_Tl = A_if * k_l / dr_l
    coeff_Ts = A_if * (k_g / dr_g - k_l / dr_l)
    coeff_mpp = -A_if * L_v

    # Check if Tl is in layout (coupled) or fixed (explicit Gauss-Seidel)
    if layout.has_block("Tl"):
        # Fully coupled: Tl is an unknown in this system
        idx_Tl = layout.idx_Tl(il_local)
        cols = [idx_Tg, idx_Ts, idx_Tl, idx_mpp]
        vals = [coeff_Tg, coeff_Ts, coeff_Tl, coeff_mpp]
        rhs = 0.0
    else:
        # Gauss-Seidel split: Tl fixed at old value
        Tl_fixed = float(state.Tl[il_local]) if state.Tl.size > il_local else 0.0
        cols = [idx_Tg, idx_Ts, idx_mpp]
        vals = [coeff_Tg, coeff_Ts, coeff_mpp]
        rhs = -coeff_Tl * Tl_fixed  # Move Tl term to RHS

    # Diagnostic preview using current state (not mutating state)
    Ts_cur = float(state.Ts)
    Tg_cur = float(state.Tg[ig_local]) if state.Tg.size else np.nan
    Tl_cur = float(state.Tl[il_local]) if state.Tl.size else np.nan
    mpp_cur = float(state.mpp)

    q_g = -k_g * (Tg_cur - Ts_cur) / dr_g * A_if
    q_l = -k_l * (Ts_cur - Tl_cur) / dr_l * A_if
    q_lat = mpp_cur * L_v * A_if

    # Enthalpy-based split diagnostics (no matrix impact)
    if not hasattr(props, "h_g") or props.h_g is None:
        raise ValueError("Props.h_g is required for Ts enthalpy diagnostics.")
    if not hasattr(props, "h_l") or props.h_l is None:
        raise ValueError("Props.h_l is required for Ts enthalpy diagnostics.")
    h_g_if = float(props.h_g[ig_local])
    h_l_if = float(props.h_l[il_local])

    dTdr_g_if = (Tg_cur - Ts_cur) / dr_g
    dTdr_l_if = (Ts_cur - Tl_cur) / dr_l
    J_if = mpp_cur  # outward (+r) positive

    q_tot_g_area, q_cond_g_area, q_diff_g_area = split_energy_flux_cond_diff_single(
        cfg, k_g, dTdr_g_if, h_g_if, J_if
    )
    q_tot_l_area, q_cond_l_area, q_diff_l_area = split_energy_flux_cond_diff_single(
        cfg, k_l, dTdr_l_if, h_l_if, J_if
    )

    q_cond_g_pow = q_cond_g_area * A_if
    q_cond_l_pow = q_cond_l_area * A_if
    q_diff_g_pow = q_diff_g_area * A_if
    q_diff_l_pow = q_diff_l_area * A_if
    q_tot_g_pow = q_tot_g_area * A_if
    q_tot_l_pow = q_tot_l_area * A_if

    Lv_eff = h_g_if - h_l_if
    q_lat_eff_pow = q_diff_g_pow - q_diff_l_pow
    latent_mismatch_pow = q_lat - q_lat_eff_pow

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
            "enthalpy_split": {
                "h_g_if": h_g_if,
                "h_l_if": h_l_if,
                "Lv_eff": Lv_eff,
                "q_cond_g_area": q_cond_g_area,
                "q_cond_l_area": q_cond_l_area,
                "q_diff_g_area": q_diff_g_area,
                "q_diff_l_area": q_diff_l_area,
                "q_total_g_area": q_tot_g_area,
                "q_total_l_area": q_tot_l_area,
                "q_cond_g_pow": q_cond_g_pow,
                "q_cond_l_pow": q_cond_l_pow,
                "q_diff_g_pow": q_diff_g_pow,
                "q_diff_l_pow": q_diff_l_pow,
                "q_total_g_pow": q_tot_g_pow,
                "q_total_l_pow": q_tot_l_pow,
                "q_lat_old_pow": q_lat,
                "q_lat_eff_pow": q_lat_eff_pow,
                "latent_mismatch_pow": latent_mismatch_pow,
                "balance_old_pow": q_g + q_l - q_lat,
                "balance_eff_pow": q_g + q_l - q_lat_eff_pow,
                "units": {"area": "W/m^2", "pow": "W"},
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
    il_global: int,
    il_local: int,
    ig_global: int,
    ig_local: int,
    iface_f: int,
    idx_mpp: int,
) -> Tuple[InterfaceRow, Dict[str, Any]]:
    """
    Build mpp interface mass-balance row using multicomponent Raoult equilibrium.

    Steps (Step16.2):
    1) j_raw = -rho_g * D_i/dr_g * (Yg_cell_i - Yg_eq_i)
    2) j_corr = j_raw - Yg_eq * sum(j_raw)  (enforce sum j = 0)
    3) mpp = j_corr_b / (Yl_b - Yg_eq_b) using balance species b
    4) J_i = mpp * Yg_eq_i + j_corr_i (diagnostics; sum J = mpp)
    5) Linear row couples all reduced Yg (closure eliminated) + mpp.
    """
    if eq_result is None or "Yg_eq" not in eq_result:
        logger.error("mpp equation requires eq_result with 'Yg_eq'.")
        raise ValueError("eq_result with 'Yg_eq' is required for mpp equation.")
    Yg_eq_full = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
    Ns_g = Yg_eq_full.shape[0]
    if Ns_g == 0:
        raise ValueError("eq_result['Yg_eq'] is empty; cannot build mpp row.")

    A_if = float(grid.A_f[iface_f])  # diagnostic only
    r_if = float(grid.r_f[iface_f])
    dr_g = float(grid.r_c[ig_global] - r_if)
    if dr_g <= 0.0:
        logger.error("Non-positive gas-side spacing at interface: dr_g=%g", dr_g)
        raise ValueError(f"Gas-side spacing must be positive: dr_g={dr_g}")

    rho_g = float(props.rho_g[ig_local])
    bal = _get_balance_species_indices(cfg, layout)
    k_b_full = bal["k_g_full"]
    k_b_red = bal["k_g_red"]
    g_name = bal["g_name"]
    l_name = bal["l_name"]

    if Yg_eq_full.shape[0] <= k_b_full:
        raise ValueError(f"eq_result['Yg_eq'] too short for species index {k_b_full}")
    if state.Yg.shape[0] < Ns_g:
        raise ValueError(f"state.Yg has {state.Yg.shape[0]} species, expected at least {Ns_g}")

    D_full = _get_gas_diffusivity_vec(props, cfg, ig_local=ig_local, Ns_g=Ns_g)
    alpha = rho_g * D_full / dr_g

    Yg_cell_full = np.asarray(state.Yg[:, ig_local], dtype=np.float64).reshape(Ns_g)
    Yl_b = float(state.Yl[bal["k_l_full"], il_local])
    Yg_eq_b = float(Yg_eq_full[k_b_full])

    # 1) raw and corrected diffusive fluxes (per-area, +r outward)
    j_raw = -alpha * (Yg_cell_full - Yg_eq_full)
    j_sum = float(np.sum(j_raw))
    j_corr = j_raw - Yg_eq_full * j_sum

    # 2) mpp from balance species
    delta_Y = Yl_b - Yg_eq_b
    eps_delta = 1e-14
    if abs(delta_Y) < eps_delta:
        mpp_cur = 0.0
    else:
        mpp_cur = float(j_corr[k_b_full] / delta_Y)

    # 2b) Apply no_condensation constraint if configured
    mpp_unconstrained = mpp_cur
    interface_type = getattr(cfg.physics.interface, "type", "no_condensation")
    if interface_type == "no_condensation" and mpp_cur < 0.0:
        # Force evaporation-only: clamp negative mpp (condensation) to zero
        mpp_cur = 0.0

    # 3) total species fluxes for diagnostics
    J_full = mpp_cur * Yg_eq_full + j_corr

    # Linearization coefficients for j_corr_b vs reduced Yg (closure eliminated)
    k_cl = layout.gas_closure_index
    if k_cl is None:
        raise ValueError("gas_closure_index missing in layout; required to eliminate closure species.")

    a_full = Yg_eq_b * alpha.copy()
    a_full[k_b_full] = alpha[k_b_full] * (Yg_eq_b - 1.0)
    S_eq = float(np.sum(alpha * Yg_eq_full))
    const_j = alpha[k_b_full] * Yg_eq_b - Yg_eq_b * S_eq
    const2 = const_j + a_full[k_cl]

    cols: List[int] = []
    vals: List[float] = []

    if abs(delta_Y) < eps_delta:
        # Degenerate: enforce mpp = 0 directly
        rhs = 0.0
        cols.append(idx_mpp)
        vals.append(1.0)
    else:
        coeff_mpp = delta_Y
        rhs = const2
        if layout.has_block("Yg"):
            for k_red in range(layout.Ns_g_eff):
                k_full = layout.gas_reduced_to_full_idx[k_red]
                coeff_Yg = -(a_full[k_full] - a_full[k_cl])
                cols.append(layout.idx_Yg(k_red, ig_local))
                vals.append(coeff_Yg)
        else:
            # Explicit Yg: move contribution to RHS using current state
            for k_red in range(layout.Ns_g_eff):
                k_full = layout.gas_reduced_to_full_idx[k_red]
                rhs -= -(a_full[k_full] - a_full[k_cl]) * float(Yg_cell_full[k_full])

        cols.append(idx_mpp)
        vals.append(coeff_mpp)

    diag_update: Dict[str, Any] = {
        "evaporation": {
            "balance_liq": l_name,
            "balance_gas": g_name,
            "k_b_full": k_b_full,
            "k_b_red": k_b_red,
            "k_closure_full": k_cl,
            "dr_g": dr_g,
            "rho_g": rho_g,
            "DeltaY": delta_Y,
            "j_sum": j_sum,
            "j_raw_sum": float(j_raw.sum()),
            "j_corr_sum": float(j_corr.sum()),
            "mpp_eval": mpp_cur,
            "mpp_unconstrained": mpp_unconstrained,
            "no_condensation_applied": (interface_type == "no_condensation" and mpp_unconstrained < 0.0),
            "interface_type": interface_type,
            "sumJ_minus_mpp": float(J_full.sum() - mpp_cur),
            "A_if": A_if,
            "Yg_eq_full": Yg_eq_full,
            "j_corr_full": j_corr,
            "J_full": J_full,
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


def _get_balance_species_indices(
    cfg: CaseConfig,
    layout: UnknownLayout,
) -> Dict[str, Any]:
    """
    Return indices/names for the liquid balance species and its gas counterpart.

    Returns dict with:
        l_name, g_name, k_l_full, k_g_full, k_g_red (may be None if closure)
    """
    l_name = cfg.species.liq_balance_species
    g_map = cfg.species.liq2gas_map
    if l_name not in g_map:
        raise ValueError(f"liq_balance_species '{l_name}' not found in liq2gas_map {g_map}")
    g_name = g_map[l_name]

    if l_name not in layout.liq_species_full:
        raise ValueError(f"Liquid balance species '{l_name}' not found in liq_species_full {layout.liq_species_full}")
    k_l_full = layout.liq_species_full.index(l_name)

    if g_name not in layout.gas_species_full:
        raise ValueError(f"Gas balance species '{g_name}' not found in gas_species_full {layout.gas_species_full}")
    k_g_full = layout.gas_species_full.index(g_name)

    k_g_red = layout.gas_full_to_reduced.get(g_name)

    return {
        "l_name": l_name,
        "g_name": g_name,
        "k_l_full": k_l_full,
        "k_g_full": k_g_full,
        "k_g_red": k_g_red,
    }


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


def _get_gas_diffusivity_vec(
    props: Props,
    cfg: CaseConfig,
    ig_local: int,
    Ns_g: int,
) -> np.ndarray:
    """
    Return D_g vector (Ns_g,) at gas cell ig_local; prefer props.D_g.
    """
    if props.D_g is not None:
        if props.D_g.shape[0] < Ns_g or props.D_g.shape[1] <= ig_local:
            raise ValueError(
                f"D_g shape {props.D_g.shape} too small for Ns_g={Ns_g} or cell {ig_local}"
            )
        D_vec = np.asarray(props.D_g[:Ns_g, ig_local], dtype=np.float64)
        if np.any(D_vec <= 0.0):
            raise ValueError(f"D_g contains non-positive entries at cell {ig_local}: {D_vec}")
        return D_vec

    D_default = getattr(cfg.physics, "default_D_g", None)
    if D_default is not None:
        val = float(D_default)
        if val <= 0.0:
            raise ValueError(f"default_D_g must be positive, got {val}")
        return np.full((Ns_g,), val, dtype=np.float64)

    raise ValueError("No D_g available in props and cfg.physics.default_D_g missing.")
