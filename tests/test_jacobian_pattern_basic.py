from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_chemistry_or_skip():
    try:
        import cantera  # noqa: F401
    except Exception:
        pytest.skip("Cantera not available")
    try:
        import CoolProp  # noqa: F401
    except Exception:
        pytest.skip("CoolProp not available")


@pytest.mark.parametrize("Ng", [3, 5])
def test_jacobian_pattern_basic(Ng: int, tmp_path: Path):
    _import_chemistry_or_skip()

    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = Ng
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / "case"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from properties.compute_props import get_or_build_models  # noqa: E402
    from assembly.jacobian_pattern import build_jacobian_pattern  # noqa: E402

    gas_model, _liq_model = get_or_build_models(cfg)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)

    pattern = build_jacobian_pattern(cfg, grid, layout)

    indptr = pattern.indptr
    indices = pattern.indices
    N = pattern.shape[0]

    assert indptr.shape == (N + 1,)
    assert indices.ndim == 1
    assert pattern.shape == (N, N)

    nnz_per_row = np.diff(indptr)
    assert np.all(nnz_per_row >= 1)
    assert np.max(nnz_per_row) < max(64, layout.Ns_g_eff + layout.Ns_l_eff + 10)

    nnz = int(indices.size)
    assert nnz <= 64 * N

    blocks = layout.blocks
    sl_Tg = blocks.get("Tg")
    if sl_Tg is not None and Ng >= 3:
        ig = 1
        row_idx = sl_Tg.start + ig
        row_cols = indices[indptr[row_idx] : indptr[row_idx + 1]]
        left = sl_Tg.start + (ig - 1)
        right = sl_Tg.start + (ig + 1)
        assert row_idx in row_cols
        assert left in row_cols
        assert right in row_cols

    sl_Ts = blocks.get("Ts")
    if sl_Tg is not None and sl_Ts is not None:
        idx_Ts = sl_Ts.start
        idx_Tg0 = sl_Tg.start
        row_ts = indices[indptr[idx_Ts] : indptr[idx_Ts + 1]]
        row_tg0 = indices[indptr[idx_Tg0] : indptr[idx_Tg0 + 1]]
        assert idx_Tg0 in row_ts
        assert idx_Ts in row_tg0

    print(
        f"[jac_pattern] Ng={Ng} N={N} nnz_total={nnz} "
        f"nnz_avg={float(nnz_per_row.mean()):.1f} nnz_max_row={int(nnz_per_row.max())}"
    )
