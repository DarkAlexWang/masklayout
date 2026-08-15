"""Hierarchy inspection."""

import numpy as np
import pytest

from masklayout.model.cell import Cell, Reference
from masklayout.model.geometry import Polygon
from masklayout.model.layout import Layout, UnknownCellError


def _square(size: int = 100) -> Polygon:
    pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.int64)
    return Polygon(points=pts, layer=10)


def _nested() -> Layout:
    layout = Layout(name="LIB")
    layout.add(Cell(name="LEAF", polygons=[_square()]))
    layout.add(Cell(name="MID", references=[Reference("LEAF", (0, 0))]))
    layout.add(Cell(name="TOP", references=[Reference("MID", (0, 0))]))
    return layout


def test_top_cells_excludes_referenced_cells() -> None:
    assert _nested().top_cells() == ["TOP"]


def test_dependencies_are_transitive() -> None:
    assert _nested().dependencies("TOP") == {"MID", "LEAF"}
    assert _nested().dependencies("LEAF") == set()


def test_depth_counts_the_longest_chain() -> None:
    assert _nested().depth() == 2
    flat = Layout(name="LIB")
    flat.add(Cell(name="ONLY", polygons=[_square()]))
    assert flat.depth() == 0


def test_polygon_count_sums_all_cells() -> None:
    assert _nested().polygon_count() == 1


def test_duplicate_cell_name_is_rejected() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="A"))
    with pytest.raises(ValueError, match="already exists"):
        layout.add(Cell(name="A"))


def test_dependencies_on_unknown_cell_lists_known_cells() -> None:
    with pytest.raises(UnknownCellError, match="LEAF"):
        _nested().dependencies("NOPE")


def test_dangling_reference_is_reported() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP", references=[Reference("MISSING", (0, 0))]))
    with pytest.raises(UnknownCellError, match="MISSING"):
        layout.dependencies("TOP")


def test_reference_cycle_is_detected() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="A", references=[Reference("B", (0, 0))]))
    layout.add(Cell(name="B", references=[Reference("A", (0, 0))]))
    with pytest.raises(ValueError, match="cycle"):
        layout.depth()
