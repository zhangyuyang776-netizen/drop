import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import yaml

# Skip if dependencies for property models are unavailable
try:
    import cantera  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Cantera not available: {exc}", allow_module_level=True)

try:
    import CoolProp  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"CoolProp not available: {exc}", allow_module_level=True)

from driver.run_scipy_case import run_case


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "cases"
MECH_DIR = ROOT / "mechanism"


def _load_case_yaml(name: str, tmp_path: Path) -> Tuple[Path, str]:
    """Load template YAML, patch paths to tmp_path, and write to temp file."""
    base_path = CASE_DIR / name
    raw = yaml.safe_load(base_path.read_text())

    case_id = raw["case"]["id"]
    raw["paths"]["output_root"] = str(tmp_path)
    raw["paths"]["case_dir"] = str(tmp_path / case_id)
    raw["paths"]["mechanism_dir"] = str(MECH_DIR)

    tmp_yaml = tmp_path / f"{case_id}.yaml"
    tmp_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_yaml, case_id


def _find_latest_run_dir(output_root: Path, case_id: str) -> Path:
    case_root = output_root / case_id
    runs = sorted(case_root.iterdir())
    if not runs:
        pytest.fail(f"No run directories under {case_root}")
    return runs[-1]


def _find_scalars_file(run_dir: Path) -> Path:
    cands = list(run_dir.rglob("*.csv")) + list(run_dir.rglob("*.txt"))
    if not cands:
        pytest.fail(f"No scalar files found under {run_dir}")
    scalars = [p for p in cands if "scalars" in p.name.lower()]
    return scalars[0] if scalars else cands[0]


def _load_scalars(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def _tail(path: Path, n: int = 5) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[-n:])


def test_case_A_equilibrium(tmp_path: Path):
    tmp_yaml, case_id = _load_case_yaml("step13_A_equilibrium_const_props.yaml", tmp_path)

    rc = run_case(str(tmp_yaml), max_steps=25, log_level=logging.WARNING)
    assert rc == 0, f"run_case returned {rc}"

    run_dir = _find_latest_run_dir(tmp_path, case_id)
    scalars_path = _find_scalars_file(run_dir)
    data = _load_scalars(scalars_path)

    Rd = data["Rd"]
    Ts = data["Ts"]
    mpp = data["mpp"]

    tail = _tail(scalars_path)
    assert np.all(Rd > 0.0), f"Non-positive Rd\n{tail}"
    assert np.all(np.isfinite(Ts)), f"Non-finite Ts\n{tail}"
    assert np.all(np.isfinite(mpp)), f"Non-finite mpp\n{tail}"

    rel_change = abs(Rd[-1] - Rd[0]) / max(Rd[0], 1.0)
    assert rel_change < 1e-6, f"Rd drift too large ({rel_change:.3e})\n{tail}"
    assert np.max(np.abs(mpp)) < 1e-6, f"mpp not near zero\n{tail}"


def test_case_B_shrink(tmp_path: Path):
    tmp_yaml, case_id = _load_case_yaml("step13_B_shrink_const_props.yaml", tmp_path)

    rc = run_case(str(tmp_yaml), max_steps=30, log_level=logging.WARNING)
    assert rc == 0, f"run_case returned {rc}"

    run_dir = _find_latest_run_dir(tmp_path, case_id)
    scalars_path = _find_scalars_file(run_dir)
    data = _load_scalars(scalars_path)

    Rd = data["Rd"]
    Ts = data["Ts"]
    Tg_if = data["Tg_if"]
    Tl_if = data["Tl_if"]

    tail = _tail(scalars_path)
    assert np.all(Rd > 0.0), f"Non-positive Rd\n{tail}"
    assert np.all(np.isfinite(Ts)), f"Non-finite Ts\n{tail}"
    assert np.all(np.isfinite(Tg_if)), f"Non-finite Tg_if\n{tail}"
    assert np.all(np.isfinite(Tl_if)), f"Non-finite Tl_if\n{tail}"

    assert Rd[-1] < Rd[0], f"Rd did not decrease\n{tail}"
