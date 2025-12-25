from __future__ import annotations

import numpy as np
import pytest

from core.grid import build_grid
from core.layout import build_layout
from tests._helpers_step15 import make_cfg_base


def _import_petsc_or_skip():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc

    return PETSc


def _build_layout_case():
    cfg = make_cfg_base(
        Nl=2,
        Ng=3,
        gas_species=("FUEL", "N2"),
        gas_balance="N2",
        liq_species=("FUEL_L", "O2_L"),
        liq_balance="FUEL_L",
        solve_Yg=True,
        include_Ts=True,
        include_mpp=True,
        include_Rd=True,
    )
    cfg.physics.solve_Yl = True
    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    return cfg, grid, layout


def _indices_from_is(iset):
    idx = iset.getIndices()
    arr = np.array(idx, dtype=np.int64, copy=True)
    if hasattr(iset, "restoreIndices"):
        try:
            iset.restoreIndices()
        except TypeError:
            iset.restoreIndices(idx)
    return arr


def test_layout_build_is_petsc_serial_cover_and_disjoint():
    PETSc = _import_petsc_or_skip()
    _cfg, grid, layout = _build_layout_case()

    is_dict = layout.build_is_petsc(comm=PETSc.COMM_SELF, ownership_range=None)

    exp = {}
    if layout.has_block("Tg"):
        exp["Tg"] = grid.Ng
    if layout.has_block("Yg"):
        exp["Yg"] = layout.Ns_g_eff * grid.Ng
    if layout.has_block("Tl"):
        exp["Tl"] = grid.Nl
    if layout.has_block("Yl"):
        exp["Yl"] = layout.Ns_l_eff * grid.Nl
    iface_n = 0
    for b in ("Ts", "mpp", "Rd"):
        if layout.has_block(b) and layout.block_size(b) > 0:
            iface_n += 1
    if iface_n > 0:
        exp["iface"] = iface_n

    for name, size in exp.items():
        assert name in is_dict, f"missing IS '{name}', got {list(is_dict.keys())}"
        assert int(is_dict[name].getSize()) == int(size)

    idx_list = []
    for iset in is_dict.values():
        idx = _indices_from_is(iset)
        if idx.size:
            idx_list.append(idx)

    if idx_list:
        concat = np.concatenate(idx_list)
        uniq = np.unique(concat)
        assert uniq.size == concat.size, "IS overlap detected"

    N = int(layout.n_dof())
    if idx_list:
        uniq = np.unique(np.concatenate(idx_list))
        assert uniq.size == N
        assert int(uniq.min()) == 0
        assert int(uniq.max()) == N - 1


def test_layout_build_is_petsc_with_ownership_range_filters():
    PETSc = _import_petsc_or_skip()
    _cfg, _grid, layout = _build_layout_case()

    N = int(layout.n_dof())
    r0, r1 = 0, max(1, N // 2)

    is_dict = layout.build_is_petsc(comm=PETSc.COMM_SELF, ownership_range=(r0, r1))

    idx_list = []
    for iset in is_dict.values():
        idx = _indices_from_is(iset)
        if idx.size:
            idx_list.append(idx)

    if idx_list:
        uniq = np.unique(np.concatenate(idx_list))
        assert np.all(uniq >= r0)
        assert np.all(uniq < r1)
