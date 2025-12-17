"""
Properties aggregator: build a unified Props from state + gas/liquid models.

Scope (Step 6.1):
- Call gas/liquid property evaluators.
- Package into Props with shape validation.
- Return extras dict for downstream interface/equilibrium diagnostics.

No mapping, no renormalization: state.Yg/Yl must already be full-length and normalized.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, Props, State
from properties.gas import GasPropertiesModel, compute_gas_props
from properties.liquid import LiquidPropertiesModel, compute_liquid_props

FloatArray = np.ndarray


def build_props_from_state(
    cfg: CaseConfig,
    grid: Grid1D,
    state: State,
    gas_model: GasPropertiesModel,
    liq_model: LiquidPropertiesModel | None = None,
) -> Tuple[Props, Dict[str, Dict[str, FloatArray]]]:
    """
    Aggregate properties from gas/liquid models into Props and extras.

    Contract:
    - state.Yg shape == (gas_model.gas.n_species, grid.Ng); already normalized.
    - state.Yl shape == (len(liq_model.liq_names), grid.Nl) if liq_model is provided; normalized.
    - No renormalization or species mapping performed here.
    """
    Ng = grid.Ng
    Nl = grid.Nl

    Ns_g = gas_model.gas.n_species
    if state.Yg.shape != (Ns_g, Ng):
        raise ValueError(f"Gas mass fractions shape {state.Yg.shape} != ({Ns_g},{Ng})")

    gas_core, gas_extra = compute_gas_props(gas_model, state, grid)
    rho_g = gas_core["rho_g"]
    cp_g = gas_core["cp_g"]
    k_g = gas_core["k_g"]
    D_g = gas_core.get("D_g", None)
    h_g = cp_g * state.Tg  # (Ng,) J/kg

    if liq_model is not None:
        Ns_l = len(liq_model.liq_names)
        if state.Yl.shape != (Ns_l, Nl):
            raise ValueError(f"Liquid mass fractions shape {state.Yl.shape} != ({Ns_l},{Nl})")
        liq_core, liq_extra = compute_liquid_props(liq_model, state, grid)
        rho_l = liq_core["rho_l"]
        cp_l = liq_core["cp_l"]
        k_l = liq_core["k_l"]
        D_l = None
        h_l = cp_l * state.Tl  # (Nl,) J/kg
    else:
        Ns_l = state.Yl.shape[0]
        rho_l = np.zeros(Nl, dtype=np.float64)
        cp_l = np.zeros(Nl, dtype=np.float64)
        k_l = np.zeros(Nl, dtype=np.float64)
        D_l = None
        liq_extra = {}
        h_l = np.zeros(Nl, dtype=np.float64)

    props = Props(
        rho_g=rho_g,
        cp_g=cp_g,
        k_g=k_g,
        D_g=D_g,
        h_g=h_g,
        h_l=h_l,
        rho_l=rho_l,
        cp_l=cp_l,
        k_l=k_l,
        D_l=D_l,
    )

    props.validate_shapes(grid, Ns_g=Ns_g, Ns_l=Ns_l)

    extras: Dict[str, Dict[str, FloatArray]] = {"gas": gas_extra, "liquid": liq_extra}
    return props, extras
