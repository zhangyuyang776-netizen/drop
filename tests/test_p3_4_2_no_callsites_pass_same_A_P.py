from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_FILENAMES = {"petsc_snes_parallel.py", "petsc_snes.py"}

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


def _iter_target_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if path.name not in TARGET_FILENAMES:
            continue
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def _expr_text(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node).strip()
        except Exception:
            pass
    return ast.dump(node, include_attributes=False)


def _is_apply_structured_pc(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "apply_structured_pc"
    if isinstance(func, ast.Attribute):
        return func.attr == "apply_structured_pc"
    return False


def test_p3_4_2_no_callsites_pass_same_A_P():
    hits: list[str] = []
    targets = _iter_target_files()
    assert targets, "No target files found to scan."
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_apply_structured_pc(node):
                continue
            kw_a = next((kw for kw in node.keywords if kw.arg == "A"), None)
            kw_p = next((kw for kw in node.keywords if kw.arg == "P"), None)
            expr_a = expr_p = None
            if kw_a is not None and kw_p is not None:
                expr_a = _expr_text(kw_a.value)
                expr_p = _expr_text(kw_p.value)
            elif len(node.args) >= 5:
                expr_a = _expr_text(node.args[3])
                expr_p = _expr_text(node.args[4])
            if expr_a is not None and expr_p is not None and expr_a == expr_p:
                lineno = getattr(node, "lineno", 0)
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: A={expr_a} P={expr_p}")

    assert not hits, (
        "Disallowed apply_structured_pc callsite where A and P are the same expression.\n"
        + "\n".join(hits)
        + "\nFix: pass real operators (A=Aop, P=Pop)."
    )
