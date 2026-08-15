"""Architecture guard: gdstk may only be imported by GeomContext."""

import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "masklayout"
# The gdstk boundary is a closed, explicit allowlist — see the design document,
# section "The gdstk boundary". Adding an entry is a design change.
ALLOWED = {"geometry/context.py", "io/_gdstk_bridge.py"}


def _imports_gdstk(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "gdstk" for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "gdstk"
        ):
            return True
    return False


def test_only_geomcontext_imports_gdstk() -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path.relative_to(SRC).as_posix() not in ALLOWED
        and _imports_gdstk(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, (
        f"gdstk imported outside GeomContext: {offenders}. "
        "Route the call through masklayout.geometry.context instead."
    )


def test_the_guard_can_actually_detect_an_import() -> None:
    # Guards that never fire are worthless; prove this one fires.
    assert _imports_gdstk(ast.parse("import gdstk"))
    assert _imports_gdstk(ast.parse("from gdstk import boolean"))
    assert not _imports_gdstk(ast.parse("import numpy"))
