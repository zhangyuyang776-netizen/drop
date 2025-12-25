from __future__ import annotations

import pytest

from core.grid import build_grid
from core.layout import UnknownLayout, VarEntry, build_layout
from tests._helpers_step15 import make_cfg_base


def _make_cfg(
    *,
    solve_Yg: bool = True,
    solve_Yl: bool = False,
    include_Ts: bool = False,
    include_mpp: bool = False,
    include_Rd: bool = False,
    gas_species: tuple[str, ...] = ("FUEL", "N2"),
    gas_balance: str = "N2",
    liq_species: tuple[str, ...] = ("FUEL_L",),
    liq_balance: str = "FUEL_L",
    Nl: int = 2,
    Ng: int = 3,
):
    cfg = make_cfg_base(
        Nl=Nl,
        Ng=Ng,
        gas_species=gas_species,
        gas_balance=gas_balance,
        liq_species=liq_species,
        liq_balance=liq_balance,
        solve_Yg=solve_Yg,
        include_Ts=include_Ts,
        include_mpp=include_mpp,
        include_Rd=include_Rd,
    )
    cfg.physics.solve_Yl = bool(solve_Yl)
    return cfg


def _build_layout_from_cfg(cfg):
    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    return grid, layout


def _expect_sizes(layout, grid):
    exp = {}
    if layout.has_block("Tg"):
        exp["Tg"] = grid.Ng
    if layout.has_block("Yg"):
        exp["Yg"] = layout.Ns_g_eff * grid.Ng
    if layout.has_block("Tl"):
        exp["Tl"] = grid.Nl
    if layout.has_block("Yl"):
        exp["Yl"] = layout.Ns_l_eff * grid.Nl
    iface = 0
    for b in ("Ts", "mpp", "Rd"):
        if layout.has_block(b):
            iface += 1
    if iface > 0:
        exp["iface"] = iface
    return exp


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"include_Ts": True, "include_mpp": True, "include_Rd": True},
        {"solve_Yg": False},
        {"solve_Yl": True, "liq_species": ("FUEL_L", "O2_L"), "liq_balance": "FUEL_L"},
    ],
    ids=["with_iface", "no_Yg_no_iface", "with_Yl"],
)
def test_fieldsplit_by_layout_sizes(cfg_kwargs):
    cfg = _make_cfg(**cfg_kwargs)
    grid, layout = _build_layout_from_cfg(cfg)
    plan = layout.default_fieldsplit_plan(cfg=cfg, scheme="by_layout")
    info = layout.describe_fieldsplits(plan)

    names = [s["name"] for s in info["splits"]]
    exp = _expect_sizes(layout, grid)

    for name, size in exp.items():
        assert name in names, f"missing split '{name}', got {names}"
        got = next(s["size"] for s in info["splits"] if s["name"] == name)
        assert got == size, f"split '{name}' size={got}, expected {size}"


def test_fieldsplit_iface_blocks_match_existing():
    cfg = _make_cfg(include_Ts=True, include_mpp=False, include_Rd=True)
    _grid, layout = _build_layout_from_cfg(cfg)
    plan = layout.default_fieldsplit_plan(cfg=cfg, scheme="by_layout")

    iface = [p for p in plan if p.name == "iface"]
    expected_blocks = tuple(b for b in ("Ts", "mpp", "Rd") if layout.has_block(b))

    if expected_blocks:
        assert len(iface) == 1
        assert iface[0].blocks == expected_blocks
    else:
        assert len(iface) == 0


def _make_dummy_layout(entries):
    return UnknownLayout(
        size=len(entries),
        entries=list(entries),
        blocks={},
        Ng=1,
        Nl=1,
        Ns_g_full=0,
        Ns_g_eff=0,
        Ns_l_full=0,
        Ns_l_eff=0,
        gas_species_full=[],
        gas_species_reduced=[],
        gas_closure_species=None,
        gas_full_to_reduced={},
        gas_reduced_to_full_idx=[],
        gas_closure_index=None,
        liq_species_full=[],
        liq_species_reduced=[],
        liq_closure_species=None,
        liq_full_to_reduced={},
        liq_reduced_to_full_idx=[],
        liq_closure_index=None,
    )


def test_block_slice_requires_contiguous():
    entries = [
        VarEntry(i=0, kind="Tg", phase="gas", cell=0, spec=None, name="Tg[0]"),
        VarEntry(i=1, kind="Yg", phase="gas", cell=0, spec=0, name="Yg[0,0]"),
        VarEntry(i=2, kind="Tg", phase="gas", cell=1, spec=None, name="Tg[1]"),
    ]
    with pytest.raises(RuntimeError, match="not contiguous"):
        _make_dummy_layout(entries)
