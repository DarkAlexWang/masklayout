"""Wire PCells."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.pcells.wires import (
    BezierWireParams,
    TaperedWireParams,
    build_bezier_wire,
    build_tapered_wire,
    offset_centerline_um,
)


def test_offset_of_a_straight_line_is_a_rectangle() -> None:
    ring = offset_centerline_um(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([2.0, 2.0]))
    assert ring.shape == (4, 2)
    assert ring[:, 1].min() == pytest.approx(-1.0)
    assert ring[:, 1].max() == pytest.approx(1.0)


def test_offset_width_varies_along_a_taper() -> None:
    centerline = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    ring = offset_centerline_um(centerline, np.array([2.0, 1.0, 0.5]))
    at_start = ring[np.isclose(ring[:, 0], 0.0)][:, 1]
    at_end = ring[np.isclose(ring[:, 0], 10.0)][:, 1]
    assert at_start.max() - at_start.min() == pytest.approx(2.0)
    assert at_end.max() - at_end.min() == pytest.approx(0.5)


def test_offset_rejects_mismatched_width_count() -> None:
    with pytest.raises(ValueError, match="one width per"):
        offset_centerline_um(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([1.0]))


def test_bezier_wire_is_a_valid_grid_aligned_polygon() -> None:
    polys = build_bezier_wire(
        BezierWireParams(
            control_points_um=((0.0, 0.0), (2.0, 3.0), (6.0, -3.0), (8.0, 0.0)),
            width_um=0.4,
        ),
        TechConfig(),
        10,
        0,
    )
    assert len(polys) == 1
    assert polys[0].points.dtype == np.int64
    assert signed_area(polys[0].points) > 0
    assert polys[0].vertex_count > 8


def test_bezier_wire_width_is_respected_at_the_start() -> None:
    polys = build_bezier_wire(
        BezierWireParams(
            control_points_um=((0.0, 0.0), (3.0, 0.0), (7.0, 0.0), (10.0, 0.0)),
            width_um=0.5,
        ),
        TechConfig(),
        10,
        0,
    )
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert high_y - low_y == 500  # 0.5 um at a 1 nm grid


def test_tapered_wire_narrows_from_start_to_end() -> None:
    polys = build_tapered_wire(
        TaperedWireParams(
            control_points_um=((0.0, 0.0), (3.0, 0.0), (7.0, 0.0), (10.0, 0.0)),
            start_width_um=1.0,
            end_width_um=0.2,
        ),
        TechConfig(),
        10,
        0,
    )
    pts = polys[0].points
    near_start = pts[pts[:, 0] < 100]
    near_end = pts[pts[:, 0] > 9900]
    start_span = near_start[:, 1].max() - near_start[:, 1].min()
    end_span = near_end[:, 1].max() - near_end[:, 1].min()
    assert start_span == pytest.approx(1000, abs=2)
    assert end_span == pytest.approx(200, abs=2)
    assert start_span > end_span


def test_wire_rejects_a_width_that_self_intersects_on_a_tight_curve() -> None:
    # A wire far wider than its curve radius folds through itself.
    with pytest.raises(ValueError, match="self-intersect"):
        build_bezier_wire(
            BezierWireParams(
                control_points_um=((0.0, 0.0), (0.5, 4.0), (-0.5, 4.0), (0.0, 0.0)),
                width_um=6.0,
            ),
            TechConfig(),
            10,
            0,
        )
