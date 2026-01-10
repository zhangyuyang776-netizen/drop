from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(relpath: Path) -> str:
    path = ROOT / relpath
    return path.read_text(encoding="utf-8")


def test_p3_4_1_3_callsites_pass_real_ops_parallel():
    text = _read_text(Path("solvers") / "petsc_snes_parallel.py")
    bad = re.search(
        r"apply_structured_pc\([\s\S]*?A\s*=\s*P[\s\S]*?P\s*=\s*P",
        text,
    )
    assert bad is None, "apply_structured_pc must not be called with A=P, P=P in petsc_snes_parallel.py"


def test_p3_4_1_3_callsites_pass_real_ops_serial():
    text = _read_text(Path("solvers") / "petsc_snes.py")
    assert "A_for_range" not in text, "A_for_range workaround must be removed from petsc_snes.py"
    bad = re.search(
        r"apply_structured_pc\([\s\S]*?A\s*=\s*A_for_range",
        text,
    )
    assert bad is None, "apply_structured_pc must not use A_for_range in petsc_snes.py"
