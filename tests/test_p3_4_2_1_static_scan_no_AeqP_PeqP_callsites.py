from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PARTS = {
    "tests",
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    "out",
}

BAD_CALL_RE = re.compile(
    r"apply_structured_pc\([\s\S]*?\bA\s*=\s*P\b[\s\S]*?\bP\s*=\s*P\b",
    re.MULTILINE,
)


def _iter_source_py_files():
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        yield path


def test_p3_4_2_1_no_apply_structured_pc_called_with_AeqP_and_PeqP():
    hits = []
    for path in _iter_source_py_files():
        text = path.read_text(encoding="utf-8")
        match = BAD_CALL_RE.search(text)
        if match is None:
            continue
        lineno = text[: match.start()].count("\n") + 1
        hits.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert not hits, (
        "Found forbidden callsite(s): apply_structured_pc(..., A=P, P=P, ...)\n"
        + "\n".join(hits)
        + "\nFix: pass real operators (A=Aop, P=Pop) synchronized with ksp.setOperators(Aop,Pop)."
    )
