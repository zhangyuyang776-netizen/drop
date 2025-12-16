"""
Interface phase-equilibrium utilities (Raoult + psat) with configurable background fill.

Responsibilities:
- Build an EquilibriumModel from CaseConfig and species data (indices, molar masses, farfield Y).
- Compute interface-equilibrium gas composition Yg_eq given Ts, Pg, Yl_face, Yg_face.
- Provide psat helpers (CoolProp when available, Clausius fallback).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import coolprop.CoolProp as CP
except Exception:  # pragma: no cover - environment dependent
    CP = None

from core.types import CaseConfig

FloatArray = np.ndarray
EPS = 1e-30


def mass_to_mole(Y: FloatArray, M: FloatArray) -> FloatArray:
    """Convert mass fractions to mole fractions."""
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    denom = np.sum(Y / np.maximum(M, EPS))
    if denom <= 0.0:
        return np.zeros_like(Y)
    X = (Y / np.maximum(M, EPS)) / denom
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def mole_to_mass(X: FloatArray, M: FloatArray) -> FloatArray:
    """Convert mole fractions to mass fractions."""
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    numer = X * np.maximum(M, EPS)
    denom = np.sum(numer)
    if denom <= 0.0:
        return np.zeros_like(X)
    Y = numer / denom
    return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class EquilibriumModel:
    method: str  # "raoult_psat"
    psat_model: str  # "auto" | "coolprop" | "clausius"
    background_fill: str  # "farfield" | "interface_noncondensables"

    gas_names: List[str]
    liq_names: List[str]
    idx_cond_l: np.ndarray
    idx_cond_g: np.ndarray

    M_g: np.ndarray
    M_l: np.ndarray

    Yg_farfield: np.ndarray
    Xg_farfield: np.ndarray

    cp_backend: str
    cp_fluids: List[str]

    T_ref: float = 298.15
    psat_ref: Dict[str, float] = field(default_factory=dict)


def _psat_coolprop_single(fluid_label: str, T: float) -> Optional[float]:
    """Return psat [Pa] using CoolProp if available."""
    if CP is None:
        return None
    try:
        val = float(CP.PropsSI("P", "T", T, "Q", 0, fluid_label))
        if not np.isfinite(val) or val < 0.0:
            return None
        return val
    except Exception:
        return None


def _psat_clausius_single(fluid_label: str, T: float, T_ref: float, p_ref: Optional[float]) -> float:
    """Simple Clausius-Clapeyron fallback: p = p_ref * exp(-B(1/T - 1/T_ref)); crude placeholder."""
    if p_ref is None or p_ref <= 0.0:
        return 0.0
    # Placeholder constant slope; can be replaced with real parameters if available
    B = 2000.0
    val = p_ref * np.exp(-B * (1.0 / max(T, EPS) - 1.0 / max(T_ref, EPS)))
    return float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))


def _compute_psat_single(
    fluid_label: str, T: float, psat_model: str, T_ref: float, p_ref: Optional[float]
) -> float:
    """Compute psat with CoolProp first (if requested/available), Clausius fallback."""
    if psat_model in ("coolprop", "auto"):
        val = _psat_coolprop_single(fluid_label, T)
        if val is not None:
            return val
        if psat_model == "coolprop":
            warnings.warn(f"CoolProp psat failed for {fluid_label}; falling back to Clausius.")
    # Clausius fallback
    return _psat_clausius_single(fluid_label, T, T_ref, p_ref)


def _psat_vec_all(model: EquilibriumModel, T: float) -> np.ndarray:
    """Compute psat vector for all liquid species using model settings."""
    psat = np.zeros(len(model.liq_names), dtype=np.float64)
    for i, name in enumerate(model.liq_names):
        p_ref = model.psat_ref.get(name)
        fluid_label = model.cp_fluids[i] if i < len(model.cp_fluids) else name
        psat[i] = _compute_psat_single(fluid_label, T, model.psat_model, model.T_ref, p_ref)
    psat = np.nan_to_num(psat, nan=0.0, posinf=0.0, neginf=0.0)
    return psat


def build_equilibrium_model(
    cfg: CaseConfig,
    Ns_g: int,
    Ns_l: int,
    M_g: np.ndarray,
    M_l: np.ndarray,
) -> EquilibriumModel:
    """Construct EquilibriumModel from config and provided molar masses."""
    eq_cfg = cfg.physics.interface.equilibrium
    sp_cfg = cfg.species

    gas_names = list(cfg.species.gas_species) if hasattr(cfg.species, "gas_species") else list(eq_cfg.condensables_gas)
    liq_names = list(sp_cfg.liq_species)

    if len(gas_names) != Ns_g:
        raise ValueError(f"gas species length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq species length {len(liq_names)} != Ns_l {Ns_l}")

    # map condensables: provided as gas-phase names; map to liquid via liq2gas_map values
    idx_cond_l: List[int] = []
    idx_cond_g: List[int] = []
    liq2gas_map = dict(sp_cfg.liq2gas_map)
    gas_index = {name: i for i, name in enumerate(gas_names)}
    liq_index = {name: i for i, name in enumerate(liq_names)}
    for g_name in eq_cfg.condensables_gas:
        # find corresponding liquid species
        l_name = None
        for k, v in liq2gas_map.items():
            if v == g_name:
                l_name = k
                break
        if l_name is None:
            raise ValueError(f"Condensable gas {g_name} not mapped in liq2gas_map.")
        if l_name not in liq_index or g_name not in gas_index:
            raise ValueError(f"Condensable mapping {l_name}->{g_name} not found in species lists.")
        idx_cond_l.append(liq_index[l_name])
        idx_cond_g.append(gas_index[g_name])

    idx_cond_l_arr = np.array(idx_cond_l, dtype=int)
    idx_cond_g_arr = np.array(idx_cond_g, dtype=int)

    # farfield gas from initial.Yg ordered by gas species
    Yg_far = np.zeros(Ns_g, dtype=np.float64)
    init_Yg = cfg.initial.Yg
    for name, frac in init_Yg.items():
        if name in gas_index:
            Yg_far[gas_index[name]] = float(frac)
    Yg_far = np.nan_to_num(Yg_far, nan=0.0, posinf=0.0, neginf=0.0)
    # renormalize if sum !=1
    s_far = np.sum(Yg_far)
    if s_far > EPS:
        Yg_far /= s_far
    Xg_far = mass_to_mole(Yg_far, M_g)

    return EquilibriumModel(
        method=eq_cfg.method,
        psat_model=eq_cfg.psat_model,
        background_fill=eq_cfg.background_fill,
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=idx_cond_l_arr,
        idx_cond_g=idx_cond_g_arr,
        M_g=np.asarray(M_g, dtype=np.float64),
        M_l=np.asarray(M_l, dtype=np.float64),
        Yg_farfield=Yg_far,
        Xg_farfield=Xg_far,
        cp_backend=eq_cfg.coolprop.backend,
        cp_fluids=list(eq_cfg.coolprop.fluids),
        T_ref=298.15,
        psat_ref={},
    )


def compute_interface_equilibrium(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute interface-equilibrium gas composition.

    Returns:
        Yg_eq : (Ns_g,) equilibrium gas mass fractions
        y_cond : (Nv,) condensable gas mole fractions
        psat : (Ns_l,) saturation pressures for all liquid species
    """
    Ns_g = len(model.M_g)
    Ns_l = len(model.M_l)
    Yl_face = np.asarray(Yl_face, dtype=np.float64).reshape(Ns_l)
    Yg_face = np.asarray(Yg_face, dtype=np.float64).reshape(Ns_g)

    # 1) liquid mole fractions
    X_liq = mass_to_mole(Yl_face, model.M_l)

    # 2) psat vector
    psat = _psat_vec_all(model, Ts)

    # 3) condensable partial pressures (Raoult)
    idxL = model.idx_cond_l
    idxG = model.idx_cond_g
    x_cond = X_liq[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = x_cond * psat[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = np.nan_to_num(p_partial, nan=0.0, posinf=0.0, neginf=0.0)
    sum_partials = float(np.sum(p_partial))

    # cap to avoid exceeding Pg
    Pg_safe = max(float(Pg), 1.0)
    cap = 0.995 * Pg_safe
    if sum_partials > cap and sum_partials > 0.0:
        p_partial *= cap / sum_partials
        sum_partials = cap

    y_cond = p_partial / Pg_safe if idxG.size else np.zeros(0, dtype=np.float64)

    # 4) background gas mole fractions
    mask_bg = np.ones(Ns_g, dtype=bool)
    mask_bg[idxG] = False

    X_g_all_face = mass_to_mole(Yg_face, model.M_g)
    if model.background_fill == "interface_noncondensables":
        X_source = X_g_all_face
    elif model.background_fill == "farfield":
        X_source = model.Xg_farfield
    else:
        raise ValueError(f"Unknown background_fill mode: {model.background_fill}")

    X_bg = np.where(mask_bg, X_source, 0.0)
    s_bg = float(np.sum(X_bg))
    if s_bg > EPS:
        X_bg_norm = X_bg / s_bg
    else:
        # If no background left, distribute uniformly among non-condensables
        X_bg_norm = np.zeros_like(X_bg)
        if np.any(mask_bg):
            X_bg_norm[mask_bg] = 1.0 / np.sum(mask_bg)

    y_bg_total = 1.0 - float(np.sum(y_cond))
    y_bg_total = max(y_bg_total, 0.0)
    y_bg = y_bg_total * X_bg_norm

    # 5) assemble full gas mole fractions
    y_all = np.zeros(Ns_g, dtype=np.float64)
    if idxG.size:
        y_all[idxG] = y_cond
    y_all[mask_bg] = y_bg[mask_bg]

    # 6) mole -> mass
    Yg_eq = mole_to_mass(y_all, model.M_g)

    return Yg_eq, y_cond, psat
