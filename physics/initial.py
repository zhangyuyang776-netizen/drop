from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy import special

from core.types import CaseConfig, Grid1D, State
from properties.equilibrium import build_equilibrium_model
from properties.gas import GasPropertiesModel
from properties.liquid import LiquidPropertiesModel


def _build_mass_fractions(
    names: list[str],
    values: Mapping[str, float],
    closure_name: str,
    seed: float,  # kept for signature compatibility; no longer used
    n_cells: int,
) -> np.ndarray:
    """Build full mass-fraction array with closure species filled as complement."""
    Ns = len(names)
    # Start from zeros; do not pre-fill with seed
    Y = np.zeros((Ns, n_cells), dtype=np.float64)
    for i, name in enumerate(names):
        if name in values:
            Y[i, :] = float(values[name])

    if closure_name in names:
        idx = names.index(closure_name)
        others = np.sum(Y, axis=0) - Y[idx, :]
        Y[idx, :] = np.maximum(1.0 - others, 0.0)

    sums = np.sum(Y, axis=0)
    for j in range(n_cells):
        s = float(sums[j])
        if s > 0.0:
            Y[:, j] /= s
        elif Ns > 0:
            idx_cl = names.index(closure_name) if closure_name in names else 0
            Y[idx_cl, j] = 1.0
    return Y


def _get_molar_masses(
    cfg: CaseConfig, gas_model: GasPropertiesModel, Ns_g: int, Ns_l: int
) -> tuple[np.ndarray, np.ndarray]:
    M_g = np.ones(Ns_g, dtype=np.float64)
    M_l = np.ones(Ns_l, dtype=np.float64)
    try:
        M_g = np.asarray(gas_model.gas.molecular_weights, dtype=np.float64) / 1000.0
    except Exception:
        pass
    for i, name in enumerate(cfg.species.liq_species):
        if name in cfg.species.mw_kg_per_mol:
            try:
                M_l[i] = float(cfg.species.mw_kg_per_mol[name])
            except Exception:
                continue
    return M_g, M_l


def build_initial_state_erfc(
    cfg: CaseConfig,
    grid: Grid1D,
    gas_model: GasPropertiesModel,
    liq_model: Optional[LiquidPropertiesModel],
) -> State:
    """
    Build initial State using erfc profiles for temperature and gas species.
    """
    Nl = grid.Nl
    Ng = grid.Ng
    Rd0 = float(cfg.geometry.a0)

    T_inf = float(cfg.initial.T_inf)
    T_d0 = float(cfg.initial.T_d0)
    P_inf = float(cfg.initial.P_inf)
    # Unified initialization time scale for both temperature and species
    t_init = max(float(getattr(cfg.initial, "t_init_T", 1.0e-6)), 0.0)

    # Ts initial
    if getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
        Ts0 = float(getattr(cfg.physics.interface, "Ts_fixed", T_d0))
    else:
        Ts0 = T_d0

    gas_names = list(cfg.species.gas_species_full)
    liq_names = list(cfg.species.liq_species)

    Yg_inf_full = _build_mass_fractions(
        gas_names,
        cfg.initial.Yg,
        closure_name=cfg.species.gas_balance_species,
        seed=float(cfg.initial.Y_seed),
        n_cells=Ng,
    )
    Yl_full = _build_mass_fractions(
        liq_names,
        cfg.initial.Yl,
        closure_name=cfg.species.liq_balance_species,
        seed=float(cfg.initial.Y_seed),
        n_cells=Nl,
    )

    # Gas thermal diffusivity at far field
    alpha_g = 1.0e-5
    try:
        gas = gas_model.gas
        gas.TPY = T_inf, P_inf, Yg_inf_full[:, 0]
        rho_g = float(gas.density)
        cp_g = float(gas.cp_mass)
        k_g = float(gas.thermal_conductivity)
        if rho_g > 0 and cp_g > 0:
            alpha_g = k_g / (rho_g * cp_g)
    except Exception:
        alpha_g = 1.0e-5

    rc = grid.r_c
    rc_gas = rc[Nl:]

    # Temperature profiles
    Tl0 = np.full(Nl, T_d0, dtype=np.float64)
    xi_T = (rc_gas - Rd0) / (2.0 * np.sqrt(max(alpha_g, 1e-30) * max(t_init, 1e-30)))
    xi_T = np.maximum(xi_T, 0.0)
    Tg0 = T_inf - (T_inf - T_d0) * special.erfc(xi_T)

    # Species profiles (gas): tiny seeded vapor + background in far-field ratio
    Ns_g = len(gas_names)
    Ns_l = len(liq_names)
    M_g, M_l = _get_molar_masses(cfg, gas_model, Ns_g, Ns_l)

    # 1) condensable indices from liq2gas_map (all fuel species)
    cond_idx_from_map: list[int] = []
    for liq_name in liq_names:
        g_name = cfg.species.liq2gas_map.get(liq_name, None)
        if g_name is None:
            continue
        if g_name in gas_names:
            cond_idx_from_map.append(gas_names.index(g_name))

    # 2) optionally augment with equilibrium model indices
    cond_idx_from_eq: list[int] = []
    try:
        eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
        cond_idx_from_eq = list(np.asarray(eq_model.idx_cond_g, dtype=int))
    except Exception:
        eq_model = None
        cond_idx_from_eq = []

    # 3) union of both sources
    cond_idx_all = sorted(set(cond_idx_from_map) | set(cond_idx_from_eq))
    idx_cond_g = np.asarray(cond_idx_all, dtype=int)

    Yg_far = np.asarray(Yg_inf_full[:, 0], dtype=np.float64)

    mask_bg = np.ones(Ns_g, dtype=bool)
    if idx_cond_g.size > 0:
        mask_bg[idx_cond_g] = False
    bg_indices = np.where(mask_bg & (Yg_far > 0.0))[0]
    sum_bg_far = float(np.sum(Yg_far[bg_indices]))
    bg_frac = np.zeros(Ns_g, dtype=np.float64)
    if sum_bg_far > 0.0:
        bg_frac[bg_indices] = Yg_far[bg_indices] / sum_bg_far

    # Use far-field thermal diffusivity as effective D for species erfc
    D_eff = max(alpha_g, 1.0e-30)
    xi_Y = (rc_gas - Rd0) / (2.0 * np.sqrt(D_eff * max(t_init, 1.0e-30)))
    xi_Y = np.maximum(xi_Y, 0.0)

    Y_vap_if0 = float(cfg.initial.Y_vap_if0)

    Yg0 = np.zeros((Ns_g, Ng), dtype=np.float64)

    if idx_cond_g.size > 0 and Y_vap_if0 > 0.0:
        share = Y_vap_if0 / float(idx_cond_g.size)
        for k in idx_cond_g:
            Y_inf_k = float(Yg_far[k])
            Y0_k = share
            Yg0[k, :] = Y_inf_k - (Y_inf_k - Y0_k) * special.erfc(xi_Y)

    Y_vap_total = np.sum(Yg0, axis=0)
    Y_remain = np.clip(1.0 - Y_vap_total, 0.0, 1.0)
    for k in bg_indices:
        Yg0[k, :] += bg_frac[k] * Y_remain

    # Enforce closure per cell
    k_cl = gas_names.index(cfg.species.gas_balance_species) if cfg.species.gas_balance_species in gas_names else None
    for j in range(Ng):
        if k_cl is None:
            s = float(np.sum(Yg0[:, j]))
            if s > 0:
                Yg0[:, j] /= s
            continue
        sum_others = float(np.sum(Yg0[:, j]) - Yg0[k_cl, j])
        Yg0[k_cl, j] = max(0.0, 1.0 - sum_others)
        s = float(np.sum(Yg0[:, j]))
        if s > 0:
            Yg0[:, j] /= s

    state0 = State(
        Tg=Tg0,
        Yg=Yg0,
        Tl=Tl0,
        Yl=Yl_full,
        Ts=Ts0,
        mpp=0.0,
        Rd=Rd0,
    )
    return state0
