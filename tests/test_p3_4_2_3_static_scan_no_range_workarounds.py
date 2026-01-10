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

BAD_POP_CALL_RE = re.compile(
    r"apply_structured_pc\([\s\S]*?\bA\s*=\s*Pop\b[\s\S]*?\bP\s*=\s*Pop\b",
    re.MULTILINE,
)


def _iter_source_py_files():
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        yield path


def test_p3_4_2_3_no_range_workarounds_in_production_code():
    hits = []
    for path in _iter_source_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "A_for_range" in text:
            hits.append(f"{path.relative_to(ROOT)}:A_for_range")
        match = BAD_POP_CALL_RE.search(text)
        if match is not None:
            lineno = text[: match.start()].count("\n") + 1
            hits.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert not hits, (
        "Found range workaround(s) in production code:\n"
        + "\n".join(hits)
        + "\nFix: remove A_for_range and pass real operators (A=Aop, P=Pop)."
    )
