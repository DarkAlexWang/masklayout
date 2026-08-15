"""Spatial index over model polygons."""

import numpy as np
import pytest

from masklayout.geometry.index import SpatialIndex
from masklayout.model.geometry import Polygon


def _rect_dbu(x0: int, y0: int, x1: int, y1: int, layer: int = 10) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=layer)


def test_index_reports_its_size() -> None:
    assert SpatialIndex([_rect_dbu(0, 0, 100, 100)], precision_um=0.001).polygon_count == 1


def test_bbox_query_finds_only_overlapping_polygons() -> None:
    index = SpatialIndex(
        [_rect_dbu(0, 0, 100, 100), _rect_dbu(1000, 0, 1100, 100)], precision_um=0.001
    )
    assert index.query_bbox(-0.01, -0.01, 0.15, 0.15) == [0]
    assert index.query_bbox(0.95, -0.01, 1.15, 0.15) == [1]
    assert sorted(index.query_bbox(-1.0, -1.0, 5.0, 5.0)) == [0, 1]


def test_ray_query_returns_candidates_along_the_ray() -> None:
    index = SpatialIndex(
        [_rect_dbu(0, 0, 100, 100), _rect_dbu(200, 0, 300, 100)], precision_um=0.001
    )
    assert sorted(index.query_ray((0.05, 0.05), (1.0, 0.0), length_um=1.0)) == [0, 1]


def test_nearest_distance_measures_the_gap() -> None:
    # Two 100 nm bars separated by a 60 nm gap.
    index = SpatialIndex(
        [_rect_dbu(0, 0, 2000, 100), _rect_dbu(0, 160, 2000, 260)], precision_um=0.001
    )
    distance = index.nearest_distance_um((1.0, 0.1), (0.0, 1.0), max_length_um=1.0, exclude=0)
    assert distance == pytest.approx(0.060)


def test_nearest_distance_returns_none_when_nothing_is_hit() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 100, 100)], precision_um=0.001)
    assert (
        index.nearest_distance_um((0.05, 0.2), (0.0, 1.0), max_length_um=1.0, exclude=None) is None
    )


def test_exclude_skips_the_owning_polygon() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 2000, 100)], precision_um=0.001)
    # Casting outward from the top edge with the owner excluded finds nothing.
    assert index.nearest_distance_um((1.0, 0.1), (0.0, 1.0), max_length_um=1.0, exclude=0) is None


def test_inward_cast_without_exclusion_measures_the_polygon_width() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 2000, 100)], precision_um=0.001)
    width = index.nearest_distance_um((1.0, 0.0), (0.0, 1.0), max_length_um=1.0, exclude=None)
    assert width == pytest.approx(0.100)


def test_zero_direction_is_rejected() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 100, 100)], precision_um=0.001)
    with pytest.raises(ValueError, match="non-zero"):
        index.query_ray((0.0, 0.0), (0.0, 0.0), length_um=1.0)


def test_empty_index_answers_queries_without_error() -> None:
    index = SpatialIndex([], precision_um=0.001)
    assert index.polygon_count == 0
    assert index.query_bbox(-1.0, -1.0, 1.0, 1.0) == []
    assert index.nearest_distance_um((0.0, 0.0), (1.0, 0.0), 1.0, exclude=None) is None
