"""
Global nonlinear residual F(u) for Tg/Tl/Yg/Yl/Ts/mpp/Rd (SciPy backend, Step 19.4.4).

Current status:
- Uses build_transport_system with state_guess = ctx.make_state(u).
- Material properties are recomputed from state_guess each residual evaluation.
- Interface equilibrium (Yg_eq) is recomputed from state_guess when include_mpp is enabled.
- All transport terms depend on props(state_guess); fallback to props_old on failure.
- Defines F(u) = A(u) @ u - b(u).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from assembly.build_system_SciPy import build_transport_system
from properties.compute_props import compute_props, get_or_build_models
from properties.equilibrium import build_equilibrium_model, compute_interface_equilibrium
from solvers.nonlinear_context import NonlinearContext

logger = logging.getLogger(__name__)


def _build_molar_mass_from_cfg(names: list[str], Ns: int, mw_map: dict[str, float]) -> np.ndarray:
    """Build molar mass array (kg/mol) aligned with provided names."""
    M = np.ones(Ns, dtype=np.float64)
    for i, name in enumerate(names):
        if i >= Ns:
            break
        if name in mw_map:
            try:
                M[i] = float(mw_map[name])
            except Exception:
                M[i] = 1.0
    return M


def _get_or_build_eq_model(
    ctx: NonlinearContext,
    state_guess,
) -> Any:
    """Return cached equilibrium model or build a new one when species sizes change."""
    meta = ctx.meta
    Ns_g = int(state_guess.Yg.shape[0])
    Ns_l = int(state_guess.Yl.shape[0])
    key = (Ns_g, Ns_l)

    if meta.get("eq_model_key") == key and meta.get("eq_model") is not None:
        return meta["eq_model"]

    cfg = ctx.cfg
    mw_map = dict(getattr(cfg.species, "mw_kg_per_mol", {}) or {})
    gas_names = list(getattr(cfg.species, "gas_species_full", []) or [])
    liq_names = list(getattr(cfg.species, "liq_species", []) or [])
    if len(gas_names) != Ns_g:
        raise ValueError(f"gas_species_full length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq_species length {len(liq_names)} != Ns_l {Ns_l}")
    missing_l = [name for name in liq_names if name not in mw_map]
    if missing_l:
        raise ValueError(f"mw_kg_per_mol missing liquid entries: {missing_l}")

    gas_model, _ = get_or_build_models(cfg)
    mech_names = list(getattr(gas_model.gas, "species_names", []))
    if mech_names and mech_names != gas_names:
        raise ValueError("gas_species_full does not match Cantera mechanism species order.")
    M_g = np.asarray(gas_model.gas.molecular_weights, dtype=np.float64) / 1000.0
    if M_g.size != Ns_g:
        raise ValueError(f"Cantera molecular_weights size {M_g.size} != Ns_g {Ns_g}")

    M_l = _build_molar_mass_from_cfg(liq_names, Ns_l, mw_map)

    eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
    meta["eq_model"] = eq_model
    meta["eq_model_key"] = key
    return eq_model


def _closure_clamp_diag(
    Y_full: np.ndarray,
    closure_idx: int,
    *,
    tol: float,
    phase: str,
) -> Dict[str, Any]:
    """Compute closure diagnostics to detect out-of-bounds closure before clamping."""
    sum_other = np.sum(Y_full, axis=0) - Y_full[closure_idx, :]
    closure_raw = 1.0 - sum_other
    soft = ((closure_raw < 0.0) & (closure_raw >= -tol)) | ((closure_raw > 1.0) & (closure_raw <= 1.0 + tol))
    hard = (closure_raw < -tol) | (closure_raw > 1.0 + tol)
    any_out = (closure_raw < 0.0) | (closure_raw > 1.0)
    return {
        f"{phase}_closure_raw_min": float(np.min(closure_raw)) if closure_raw.size else np.nan,
        f"{phase}_closure_raw_max": float(np.max(closure_raw)) if closure_raw.size else np.nan,
        f"{phase}_closure_oob_any": int(np.count_nonzero(any_out)),
        f"{phase}_closure_oob_soft": int(np.count_nonzero(soft)),
        f"{phase}_closure_oob_hard": int(np.count_nonzero(hard)),
    }


def build_global_residual(
    u: np.ndarray,
    ctx: NonlinearContext,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the global residual F(u) for one timestep.

    Parameters
    ----------
    u : ndarray
        Global unknown vector aligned with ctx.layout.
    ctx : NonlinearContext
        Timestep context containing cfg/grid/layout/state_old/props_old/dt.
    Notes
    -----
    This always uses state_guess = ctx.make_state(u) and recomputes props from state_guess.
    """
    u = np.asarray(u, dtype=np.float64)
    N = ctx.layout.n_dof()
    if u.shape != (N,):
        raise ValueError(f"Global unknown vector shape {u.shape} != ({N},)")

    cfg = ctx.cfg
    grid = ctx.grid_ref
    layout = ctx.layout
    state_old = ctx.state_old
    props_old = ctx.props_old
    dt = float(ctx.dt)

    try:
        state_guess = ctx.make_state(u)
    except Exception as exc:
        logger.warning("make_state(u) failed in residual_global: %s; fallback to state_old.", exc)
        state_guess = state_old

    props_source = "props_old"
    props_extras = None
    try:
        props, props_extras = compute_props(cfg, grid, state_guess)
        props_source = "state_guess"
    except Exception:
        logger.exception("compute_props failed in residual_global; falling back to props_old.")
        props = props_old
        props_extras = None

    eq_result = None
    eq_model = None
    eq_source = "none"
    phys = cfg.physics
    needs_eq = bool(getattr(phys, "include_mpp", False) and layout.has_block("mpp"))
    if needs_eq:
        cache = ctx.meta.get("eq_result_cache")
        eq_model = _get_or_build_eq_model(ctx, state_guess)
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Ts_if = float(state_guess.Ts)
            Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yl_face = np.asarray(state_guess.Yl[:, il_if], dtype=np.float64)
            Yg_face = np.asarray(state_guess.Yg[:, ig_if], dtype=np.float64)
            Yg_eq, y_cond, psat = compute_interface_equilibrium(
                eq_model,
                Ts=Ts_if,
                Pg=Pg_if,
                Yl_face=Yl_face,
                Yg_face=Yg_face,
            )
            eq_result = {"Yg_eq": np.asarray(Yg_eq), "y_cond": np.asarray(y_cond), "psat": np.asarray(psat)}
            eq_source = "computed"
            ctx.meta["eq_result_cache"] = dict(eq_result)
        except Exception as exc:
            if cache is not None:
                logger.warning("compute_interface_equilibrium failed; using cached eq_result: %s", exc)
                eq_result = cache
                eq_source = "cached"
            else:
                raise
    else:
        eq_source = "disabled"

    result = build_transport_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        state_guess=state_guess,
        eq_model=eq_model,
        eq_result=eq_result,
        return_diag=True,
    )

    if isinstance(result, tuple) and len(result) == 3:
        A, b, diag_sys = result
    else:
        A, b = result  # type: ignore[misc]
        diag_sys = {}

    if A.shape != (N, N):
        raise ValueError(f"Assembly produced A shape {A.shape}, expected {(N, N)}")
    if b.shape != (N,):
        raise ValueError(f"Assembly produced b shape {b.shape}, expected {(N,)}")
    if diag_sys is None:
        diag_sys = {}

    res = A @ u - b

    res_norm_2 = float(np.linalg.norm(res))
    res_norm_inf = float(np.linalg.norm(res, ord=np.inf))

    diag: Dict[str, Any] = {
        "assembly": diag_sys,
        "residual_norm_2": res_norm_2,
        "residual_norm_inf": res_norm_inf,
        "u_min": float(np.min(u)) if u.size else np.nan,
        "u_max": float(np.max(u)) if u.size else np.nan,
    }

    diag["props"] = {"source": props_source}
    diag["eq_result"] = {"source": eq_source, "enabled": needs_eq}
    try:
        diag["props"]["rho_g_min"] = float(np.min(props.rho_g)) if props.rho_g.size else np.nan
        diag["props"]["rho_g_max"] = float(np.max(props.rho_g)) if props.rho_g.size else np.nan
        diag["props"]["state_Tg_min"] = float(np.min(state_guess.Tg)) if state_guess.Tg.size else np.nan
        diag["props"]["state_Tg_max"] = float(np.max(state_guess.Tg)) if state_guess.Tg.size else np.nan
    except Exception:
        pass
    if isinstance(props_extras, dict):
        try:
            diag["props"]["extras_keys"] = list(props_extras.keys())
        except Exception:
            pass

    clamp_diag: Dict[str, Any] = {}
    tol_sumY = float(getattr(cfg.checks, "sumY_tol", 1e-12))
    if layout.has_block("Yg") and layout.gas_closure_index is not None:
        try:
            clamp_diag.update(
                _closure_clamp_diag(
                    state_guess.Yg,
                    layout.gas_closure_index,
                    tol=tol_sumY,
                    phase="gas",
                )
            )
        except Exception:
            logger.debug("Failed to compute gas closure clamp diagnostics.", exc_info=True)
    if layout.has_block("Yl") and layout.liq_closure_index is not None:
        try:
            clamp_diag.update(
                _closure_clamp_diag(
                    state_guess.Yl,
                    layout.liq_closure_index,
                    tol=tol_sumY,
                    phase="liq",
                )
            )
        except Exception:
            logger.debug("Failed to compute liquid closure clamp diagnostics.", exc_info=True)

    evap_diag = diag_sys.get("evaporation", {}) if isinstance(diag_sys, dict) else {}
    try:
        no_cond = bool(evap_diag.get("no_condensation_applied", False))
        deltaY_raw = float(evap_diag.get("DeltaY_raw", np.nan))
        deltaY_eff = float(evap_diag.get("DeltaY_eff", np.nan))
        clamp_diag["no_condensation_applied"] = no_cond
        if np.isfinite(deltaY_raw) and np.isfinite(deltaY_eff):
            clamp_diag["deltaY_min_applied"] = abs(deltaY_eff) > abs(deltaY_raw)
    except Exception:
        pass

    if clamp_diag:
        diag["clamp"] = clamp_diag

    if res.size:
        idx_max = int(np.argmax(np.abs(res)))
        entry = layout.entries[idx_max] if idx_max < len(layout.entries) else None
        diag["residual_argmax"] = {
            "idx": idx_max,
            "abs": float(abs(res[idx_max])),
            "value": float(res[idx_max]),
            "kind": getattr(entry, "kind", ""),
            "name": getattr(entry, "name", ""),
            "phase": getattr(entry, "phase", ""),
            "cell": getattr(entry, "cell", None),
            "spec": getattr(entry, "spec", None),
        }

    try:
        if layout.has_block("Ts"):
            diag["Ts_guess"] = float(u[layout.idx_Ts()])
        if layout.has_block("Rd"):
            diag["Rd_guess"] = float(u[layout.idx_Rd()])
    except Exception:
        logger.debug("Failed to extract Ts/Rd from u for diagnostics.", exc_info=True)

    return res, diag


def residual_only(u: np.ndarray, ctx: NonlinearContext) -> np.ndarray:
    """
    Wrapper for solvers that only accept residuals.
    """
    res, _ = build_global_residual(u, ctx)
    return res
