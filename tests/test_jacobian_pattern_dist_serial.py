from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402
from assembly.jacobian_pattern_dist import (  # noqa: E402
    LocalJacPattern,
    build_jacobian_pattern_local,
)


class DummyPhysics:
    def __init__(self, solve_Yg: bool = True, solve_Yl: bool = True) -> None:
        self.solve_Yg = solve_Yg
        self.solve_Yl = solve_Yl


class DummyCfg:
    def __init__(self, solve_Yg: bool = True, solve_Yl: bool = True) -> None:
        self.physics = DummyPhysics(solve_Yg=solve_Yg, solve_Yl=solve_Yl)


class DummyLayout:
    def __init__(self) -> None:
        self.Ng = 2
        self.Nl = 2
        self.Ns_g_eff = 2
        self.Ns_l_eff = 1

        self.blocks = {}
        offset = 0

        self.blocks["Tg"] = slice(offset, offset + self.Ng)
        offset += self.Ng

        self.blocks["Tl"] = slice(offset, offset + self.Nl)
        offset += self.Nl

        self.blocks["Yg"] = slice(offset, offset + self.Ng * self.Ns_g_eff)
        offset += self.Ng * self.Ns_g_eff

        self.blocks["Yl"] = slice(offset, offset + self.Nl * self.Ns_l_eff)
        offset += self.Nl * self.Ns_l_eff

        self.blocks["Ts"] = slice(offset, offset + 1)
        offset += 1

        self.blocks["mpp"] = slice(offset, offset + 1)
        offset += 1

        self.blocks["Rd"] = slice(offset, offset + 1)
        offset += 1

        self.N_total = offset


def _make_dummy_cfg_layout():
    cfg = DummyCfg(solve_Yg=True, solve_Yl=True)
    layout = DummyLayout()
    grid = None
    return cfg, grid, layout


def test_local_pattern_full_range_matches_global():
    cfg, grid, layout = _make_dummy_cfg_layout()
    global_pat = build_jacobian_pattern(cfg, grid, layout)

    n = global_pat.shape[0]
    local_pat = build_jacobian_pattern_local(
        cfg,
        grid,
        layout,
        ownership_range=(0, n),
    )

    assert isinstance(local_pat, LocalJacPattern)
    assert local_pat.shape == global_pat.shape
    assert np.array_equal(local_pat.rows_global, np.arange(n, dtype=np.int32))
    assert local_pat.indptr.shape == global_pat.indptr.shape
    assert local_pat.indices.shape == global_pat.indices.shape

    for i in range(n):
        g_slice = slice(global_pat.indptr[i], global_pat.indptr[i + 1])
        l_slice = slice(local_pat.indptr[i], local_pat.indptr[i + 1])

        g_row = np.sort(global_pat.indices[g_slice])
        l_row = np.sort(local_pat.indices[l_slice])

        assert np.array_equal(g_row, l_row)


def test_local_pattern_subrange_matches_global():
    cfg, grid, layout = _make_dummy_cfg_layout()
    global_pat = build_jacobian_pattern(cfg, grid, layout)
    n = global_pat.shape[0]

    rstart, rend = 3, min(n, 8)
    local_pat = build_jacobian_pattern_local(
        cfg,
        grid,
        layout,
        ownership_range=(rstart, rend),
    )

    assert np.array_equal(
        local_pat.rows_global,
        np.arange(rstart, rend, dtype=np.int32),
    )

    nloc = rend - rstart
    assert local_pat.indptr.shape == (nloc + 1,)

    for k, gi in enumerate(local_pat.rows_global):
        g_slice = slice(global_pat.indptr[gi], global_pat.indptr[gi + 1])
        l_slice = slice(local_pat.indptr[k], local_pat.indptr[k + 1])

        g_row = np.sort(global_pat.indices[g_slice])
        l_row = np.sort(local_pat.indices[l_slice])

        assert np.array_equal(g_row, l_row)


def test_local_pattern_empty_range():
    cfg, grid, layout = _make_dummy_cfg_layout()
    local_pat = build_jacobian_pattern_local(
        cfg,
        grid,
        layout,
        ownership_range=(0, 0),
    )

    assert local_pat.rows_global.size == 0
    assert local_pat.indptr.shape == (1,)
    assert local_pat.indices.size == 0
    assert local_pat.meta["nnz_total"] == 0.0
