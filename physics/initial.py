from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy import special

from core.types import CaseConfig, Grid1D, State
from properties.equilibrium import build_equilibrium_model, compute_interface_equilibrium
from properties.gas import GasPropertiesModel
from properties.liquid import LiquidPropertiesModel


def _build_mass_fractions(
    names: list[str],
    values: Mapping[str, float],
    closure_name: str,
    seed: float,
    n_cells: int,
) -> np.ndarray:
    """Build full mass-fraction array with closure species filled as complement."""
    Ns = len(names)
    Y = np.full((Ns, n_cells), float(seed), dtype=np.float64)
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
            Y[0, j] = 1.0
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
    t_init_T = max(float(getattr(cfg.initial, "t_init_T", 1.0e-6)), 0.0)
    t_init_Y = max(float(getattr(cfg.initial, "t_init_Y", 1.0e-6)), 0.0)
    D_init_Y = float(getattr(cfg.initial, "D_init_Y", 1.0e-5))

    # Ts initial
    if getattr(cfg.physics.interface, "bc_mode", "") == "Ts_fixed":
        Ts0 = float(getattr(cfg.physics.interface, "Ts_fixed", T_d0))
    else:
        Ts0 = T_d0

    gas_names = list(cfg.species.gas_species)
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
    rc_liq = rc[:Nl]
    rc_gas = rc[Nl:]

    # Temperature profiles
    Tl0 = np.full(Nl, T_d0, dtype=np.float64)
    xi_T = (rc_gas - Rd0) / (2.0 * np.sqrt(max(alpha_g, 1e-30) * max(t_init_T, 1e-30)))
    xi_T = np.maximum(xi_T, 0.0)
    Tg0 = T_inf + (Rd0 / rc_gas) * (T_d0 - T_inf) * special.erfc(xi_T)

    # Interface equilibrium for saturated vapor
    Ns_g = len(gas_names)
    Ns_l = len(liq_names)
    M_g, M_l = _get_molar_masses(cfg, gas_model, Ns_g, Ns_l)
    Yg_eq = Yg_inf_full[:, 0]
    try:
        eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
        Yl_face = Yl_full[:, -1]
        Yg_face = Yg_inf_full[:, 0]
        Yg_eq, _, _ = compute_interface_equilibrium(eq_model, Ts=Ts0, Pg=P_inf, Yl_face=Yl_face, Yg_face=Yg_face)
        Yg_eq = np.asarray(Yg_eq, dtype=np.float64)
    except Exception:
        Yg_eq = Yg_inf_full[:, 0]

    # Species profiles (gas), using erfc from interface value toward far field
    xi_Y = (rc_gas - Rd0) / (2.0 * np.sqrt(max(D_init_Y, 1e-30) * max(t_init_Y, 1e-30)))
    xi_Y = np.maximum(xi_Y, 0.0)

    Yg0 = np.empty((Ns_g, Ng), dtype=np.float64)
    for k in range(Ns_g):
        Y_inf_k = float(Yg_inf_full[k, 0])
        Y0_k = float(Yg_eq[k]) if k < Yg_eq.shape[0] else Y_inf_k
        Yg0[k, :] = Y_inf_k + (Rd0 / rc_gas) * (Y0_k - Y_inf_k) * special.erfc(xi_Y)

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

