"""Shape PCells."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.pcells.shapes import (
    LineEndParams,
    RoundedRectParams,
    build_line_end,
    build_rounded_rect,
)


def test_rounded_rect_spans_the_requested_extent() -> None:
    polys = build_rounded_rect(
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(4.0, 2.0), radius_um=0.5),
        TechConfig(),
        10,
        0,
    )
    assert len(polys) == 1
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert (low_x, low_y) == (0, 0)
    assert (high_x, high_y) == (4000, 2000)


def test_rounded_rect_is_counterclockwise_and_grid_aligned() -> None:
    polys = build_rounded_rect(
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(4.0, 2.0), radius_um=0.5),
        TechConfig(),
        10,
        0,
    )
    assert polys[0].points.dtype == np.int64
    assert signed_area(polys[0].points) > 0


def test_line_end_at_zero_angle_has_the_expected_extent() -> None:
    polys = build_line_end(
        LineEndParams(centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=0.0),
        TechConfig(),
        11,
        0,
    )
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert (low_x, high_x) == (0, 40)  # extends forward only
    assert (low_y, high_y) == (-50, 50)  # centred on width


def test_line_end_rotates_rigidly() -> None:
    tech = TechConfig()
    flat = build_line_end(
        LineEndParams(centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=0.0),
        tech,
        11,
        0,
    )[0]
    turned = build_line_end(
        LineEndParams(centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=math.pi / 2),
        tech,
        11,
        0,
    )[0]
    # A rigid rotation preserves vertex count and swaps bounding-box dimensions.
    assert turned.vertex_count == flat.vertex_count
    fx0, fy0, fx1, fy1 = flat.bounds_dbu
    tx0, ty0, tx1, ty1 = turned.bounds_dbu
    assert (tx1 - tx0, ty1 - ty0) == (fy1 - fy0, fx1 - fx0)


def test_line_end_rejects_a_corner_radius_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="radius"):
        build_line_end(
            LineEndParams(
                centre_um=(0.0, 0.0),
                width_um=0.1,
                extension_um=0.04,
                angle_rad=0.0,
                corner_radius_um=0.09,
            ),
            TechConfig(),
            11,
            0,
        )


def test_rounded_rect_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError):
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(0.0, 2.0), radius_um=0.1)
