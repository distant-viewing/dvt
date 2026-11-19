"""The package is handed to people who have no corpus, so it imports nothing that knows of one."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "dvt"
FORBIDDEN = {"tbm", "tbmtools", "batch"}


def imported(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_corpus_imports():
    bad = {
        str(p.relative_to(PACKAGE)): sorted(imported(p) & FORBIDDEN)
        for p in sorted(PACKAGE.rglob("*.py"))
        if imported(p) & FORBIDDEN
    }
    assert not bad, bad
