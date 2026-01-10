from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_base_yaml() -> dict:
    yml = ROOT / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        pytest.skip(f"Case yaml not found: {yml}")
    return yaml.safe_load(yml.read_text())


def _write_yaml(tmp_path: Path, raw: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_yaml_rejects_invalid_pc_type(tmp_path: Path):
    from driver.run_scipy_case import _load_case_config  # noqa: E402

    raw = _load_base_yaml()
    linear = raw.setdefault("solver", {}).setdefault("linear", {})
    linear["pc_type"] = "jacobi"

    bad = _write_yaml(tmp_path, raw, "bad_pc_type.yaml")
    with pytest.raises(ValueError, match="linear\\.pc_type: invalid value"):
        _load_case_config(str(bad))


def test_yaml_rejects_invalid_fieldsplit_type(tmp_path: Path):
    from driver.run_scipy_case import _load_case_config  # noqa: E402

    raw = _load_base_yaml()
    linear = raw.setdefault("solver", {}).setdefault("linear", {})
    linear["pc_type"] = "fieldsplit"
    linear["fieldsplit"] = {"type": "add", "scheme": "bulk_iface"}

    bad = _write_yaml(tmp_path, raw, "bad_fieldsplit_type.yaml")
    with pytest.raises(ValueError, match="fieldsplit\\.type: invalid value"):
        _load_case_config(str(bad))
