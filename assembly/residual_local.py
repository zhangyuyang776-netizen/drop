from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ResidualLocalCtx:
    layout: object
    ld: object


def _local_index(global_i: int, gxs: int) -> int:
    return int(global_i - gxs)


def _get_component(arr, i_arr: int, comp: int) -> float:
    try:
        return float(arr[i_arr, comp])
    except Exception:
        if comp != 0:
            raise
        return float(arr[i_arr])


def _set_component(arr, i_arr: int, comp: int, value: float) -> None:
    try:
        arr[i_arr, comp] = value
    except Exception:
        if comp != 0:
            raise
        arr[i_arr] = value


def pack_local_to_layout(
    ctx: ResidualLocalCtx,
    Xl_liq,
    Xl_gas,
    Xl_if,
    *,
    rank: int,
) -> np.ndarray:
    """
    Pack local (ghosted) DMDA arrays into a global layout-ordered vector.
    Only owned cells are populated; others remain zero.
    Interface values are only populated on rank 0 to avoid duplication.
    """
    layout = ctx.layout
    ld = ctx.ld
    u = np.zeros(int(layout.n_dof()), dtype=np.float64)

    if "Tl" in ld.comp_liq:
        comp = int(ld.comp_liq["Tl"][0])
        for i_local in range(ld.liq_xm):
            il = ld.liq_xs + i_local
            i_arr = _local_index(il, ld.liq_gxs)
            u[layout.idx_Tl(il)] = _get_component(Xl_liq, i_arr, comp)

    if "Yl" in ld.comp_liq:
        comps = ld.comp_liq["Yl"]
        for i_local in range(ld.liq_xm):
            il = ld.liq_xs + i_local
            i_arr = _local_index(il, ld.liq_gxs)
            for k_red, comp in enumerate(comps):
                u[layout.idx_Yl(k_red, il)] = _get_component(Xl_liq, i_arr, int(comp))

    if "Tg" in ld.comp_gas:
        comp = int(ld.comp_gas["Tg"][0])
        for i_local in range(ld.gas_xm):
            ig = ld.gas_xs + i_local
            i_arr = _local_index(ig, ld.gas_gxs)
            u[layout.idx_Tg(ig)] = _get_component(Xl_gas, i_arr, comp)

    if "Yg" in ld.comp_gas:
        comps = ld.comp_gas["Yg"]
        for i_local in range(ld.gas_xm):
            ig = ld.gas_xs + i_local
            i_arr = _local_index(ig, ld.gas_gxs)
            for k_red, comp in enumerate(comps):
                u[layout.idx_Yg(k_red, ig)] = _get_component(Xl_gas, i_arr, int(comp))

    if rank == 0 and ld.n_if > 0:
        if "Ts" in ld.comp_if:
            j = int(ld.comp_if["Ts"][0])
            u[layout.idx_Ts()] = float(Xl_if[j])
        if "mpp" in ld.comp_if:
            j = int(ld.comp_if["mpp"][0])
            u[layout.idx_mpp()] = float(Xl_if[j])
        if "Rd" in ld.comp_if:
            j = int(ld.comp_if["Rd"][0])
            u[layout.idx_Rd()] = float(Xl_if[j])

    return u


def scatter_layout_to_local(
    ctx: ResidualLocalCtx,
    vec_layout: np.ndarray,
    Fl_liq,
    Fl_gas,
    Fl_if,
    *,
    rank: int,
) -> None:
    """
    Scatter a layout-ordered vector into local residual arrays.
    Only owned cells are written; ghost regions remain untouched.
    Interface values are written only on rank 0.
    """
    layout = ctx.layout
    ld = ctx.ld

    if "Tl" in ld.comp_liq:
        comp = int(ld.comp_liq["Tl"][0])
        for i_local in range(ld.liq_xm):
            il = ld.liq_xs + i_local
            i_arr = _local_index(il, ld.liq_gxs)
            _set_component(Fl_liq, i_arr, comp, float(vec_layout[layout.idx_Tl(il)]))

    if "Yl" in ld.comp_liq:
        comps = ld.comp_liq["Yl"]
        for i_local in range(ld.liq_xm):
            il = ld.liq_xs + i_local
            i_arr = _local_index(il, ld.liq_gxs)
            for k_red, comp in enumerate(comps):
                _set_component(
                    Fl_liq,
                    i_arr,
                    int(comp),
                    float(vec_layout[layout.idx_Yl(k_red, il)]),
                )

    if "Tg" in ld.comp_gas:
        comp = int(ld.comp_gas["Tg"][0])
        for i_local in range(ld.gas_xm):
            ig = ld.gas_xs + i_local
            i_arr = _local_index(ig, ld.gas_gxs)
            _set_component(Fl_gas, i_arr, comp, float(vec_layout[layout.idx_Tg(ig)]))

    if "Yg" in ld.comp_gas:
        comps = ld.comp_gas["Yg"]
        for i_local in range(ld.gas_xm):
            ig = ld.gas_xs + i_local
            i_arr = _local_index(ig, ld.gas_gxs)
            for k_red, comp in enumerate(comps):
                _set_component(
                    Fl_gas,
                    i_arr,
                    int(comp),
                    float(vec_layout[layout.idx_Yg(k_red, ig)]),
                )

    if ld.n_if > 0:
        if rank == 0:
            if "Ts" in ld.comp_if:
                j = int(ld.comp_if["Ts"][0])
                Fl_if[j] = float(vec_layout[layout.idx_Ts()])
            if "mpp" in ld.comp_if:
                j = int(ld.comp_if["mpp"][0])
                Fl_if[j] = float(vec_layout[layout.idx_mpp()])
            if "Rd" in ld.comp_if:
                j = int(ld.comp_if["Rd"][0])
                Fl_if[j] = float(vec_layout[layout.idx_Rd()])
        else:
            Fl_if[:] = 0.0
