"""Polyline normalization."""

import numpy as np
import pytest

from masklayout.geometry.normalize import (
    drop_collinear_points,
    drop_duplicate_points,
    is_simple,
    normalize_polyline,
    orient_counterclockwise,
    signed_area,
)


def test_drop_duplicate_points_removes_consecutive_repeats() -> None:
    pts = np.array([[0, 0], [0, 0], [10, 0], [10, 10], [10, 10]], dtype=np.float64)
    assert drop_duplicate_points(pts, tolerance=1e-9).tolist() == [[0, 0], [10, 0], [10, 10]]


def test_drop_duplicate_points_closes_the_wrap_around() -> None:
    # Last point coincident with first: a closed ring should not repeat it.
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 0]], dtype=np.float64)
    assert len(drop_duplicate_points(pts, tolerance=1e-9)) == 3


def test_drop_collinear_points_removes_a_midpoint() -> None:
    pts = np.array([[0, 0], [5, 0], [10, 0], [10, 10]], dtype=np.float64)
    result = drop_collinear_points(pts, tolerance=1e-9)
    assert result.tolist() == [[0, 0], [10, 0], [10, 10]]


def test_drop_collinear_points_keeps_a_genuine_corner() -> None:
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert len(drop_collinear_points(pts, tolerance=1e-9)) == 4


def test_signed_area_sign_follows_winding() -> None:
    ccw = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert signed_area(ccw) > 0
    assert signed_area(ccw[::-1]) < 0
    assert abs(signed_area(ccw)) == pytest.approx(100.0)


def test_orient_counterclockwise_is_idempotent_and_corrects_clockwise() -> None:
    ccw = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert np.array_equal(orient_counterclockwise(ccw), ccw)
    assert signed_area(orient_counterclockwise(ccw[::-1])) > 0


def test_is_simple_detects_a_bowtie() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=np.float64)
    assert is_simple(square)
    assert not is_simple(bowtie)


def test_normalize_polyline_applies_every_stage() -> None:
    messy = np.array([[0, 0], [0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)[::-1]
    result = normalize_polyline(messy, duplicate_tolerance=1e-9, collinear_tolerance=1e-9)
    assert sorted(map(tuple, result.tolist())) == [(0, 0), (0, 10), (10, 0), (10, 10)]
    assert signed_area(result) > 0


def test_normalize_polyline_rejects_a_degenerate_ring() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        normalize_polyline(
            np.array([[0, 0], [1, 1]], dtype=np.float64),
            duplicate_tolerance=1e-9,
            collinear_tolerance=1e-9,
        )
