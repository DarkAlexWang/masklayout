"""Smoke test: the package is importable and declares a version."""

import masklayout


def test_package_exposes_version() -> None:
    assert isinstance(masklayout.__version__, str)
    assert masklayout.__version__.count(".") == 2
