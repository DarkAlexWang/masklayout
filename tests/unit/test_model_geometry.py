"""Typed geometry model."""

import numpy as np
import pytest

from masklayout.model.cell import (
    Cell,
    ExplicitRepetition,
    RectangularRepetition,
    Reference,
)
from masklayout.model.geometry import Label, Polygon


def test_polygon_holds_integer_coordinates() -> None:
    poly = Polygon(points=np.array([[0, 0], [100, 0], [100, 50]], dtype=np.int64), layer=10)
    assert poly.points.dtype == np.int64
    assert poly.vertex_count == 3
    assert poly.datatype == 0


def test_polygon_rejects_non_integer_coordinates() -> None:
    with pytest.raises(TypeError, match="int64"):
        Polygon(points=np.array([[0.0, 0.5]], dtype=np.float64), layer=10)


def test_polygon_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        Polygon(points=np.array([0, 1, 2], dtype=np.int64), layer=10)


def test_polygon_bounds() -> None:
    poly = Polygon(points=np.array([[0, 0], [100, 0], [100, 50]], dtype=np.int64), layer=10)
    assert poly.bounds_dbu == (0, 0, 100, 50)


def test_label_holds_integer_origin() -> None:
    label = Label(text="A1", origin_dbu=(10, 20), layer=12)
    assert label.origin_dbu == (10, 20)
    assert label.text == "A1"


def test_rectangular_repetition_expands_to_offsets() -> None:
    rep = RectangularRepetition(columns=3, rows=2, spacing_dbu=(1000, 500))
    offsets = rep.offsets_dbu()
    assert offsets.dtype == np.int64
    assert offsets.shape == (6, 2)
    as_tuples = {tuple(row) for row in offsets.tolist()}
    assert (0, 0) in as_tuples
    assert (2000, 500) in as_tuples


def test_rectangular_repetition_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="columns"):
        RectangularRepetition(columns=0, rows=2, spacing_dbu=(10, 10))


def test_explicit_repetition_returns_its_offsets() -> None:
    given = np.array([[0, 0], [7, 9]], dtype=np.int64)
    assert np.array_equal(ExplicitRepetition(offsets_dbu_array=given).offsets_dbu(), given)


def test_reference_defaults_are_identity() -> None:
    ref = Reference(cell_name="LEAF", origin_dbu=(0, 0))
    assert ref.rotation_rad == 0.0
    assert ref.magnification == 1.0
    assert ref.x_reflection is False
    assert ref.repetition is None


def test_cell_starts_empty() -> None:
    cell = Cell(name="TOP")
    assert cell.polygons == []
    assert cell.labels == []
    assert cell.references == []
