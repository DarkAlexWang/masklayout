"""Polyline cleanup.

gdstk provides no simplification, collinear removal, or validity check, so
these are implemented here. Functions take and return (N, 2) arrays and do
not care whether the dtype is float micrometres or integer DBU, except
``is_simple`` which is float-only by way of shapely.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon

MIN_RING_VERTICES = 3

Points = NDArray[np.floating] | NDArray[np.integer]


def drop_duplicate_points(points: Points, tolerance: float) -> Points:
    """Remove consecutive coincident points, treating the ring as closed."""
    if len(points) < 2:
        return points
    following = np.roll(points, -1, axis=0)
    distance = np.linalg.norm((following - points).astype(np.float64), axis=1)
    return points[distance > tolerance]


def drop_collinear_points(points: Points, tolerance: float) -> Points:
    """Remove points whose deviation from the neighbouring chord is negligible.

    The test is the perpendicular distance from each point to the line through
    its neighbours, so ``tolerance`` is a real distance in the input's units.
    """
    if len(points) < MIN_RING_VERTICES:
        return points
    previous = np.roll(points, 1, axis=0).astype(np.float64)
    current = points.astype(np.float64)
    following = np.roll(points, -1, axis=0).astype(np.float64)

    chord = following - previous
    chord_length = np.linalg.norm(chord, axis=1)
    cross = np.abs(
        chord[:, 0] * (current[:, 1] - previous[:, 1])
        - chord[:, 1] * (current[:, 0] - previous[:, 0])
    )
    # Where the neighbours coincide the chord is degenerate; fall back to the
    # raw offset from the previous point.
    safe_length = np.where(chord_length > 0, chord_length, 1.0)
    deviation = np.where(
        chord_length > 0,
        cross / safe_length,
        np.linalg.norm(current - previous, axis=1),
    )
    return points[deviation > tolerance]


def signed_area(points: Points) -> float:
    """Shoelace area. Positive means counterclockwise winding."""
    values = points.astype(np.float64)
    following = np.roll(values, -1, axis=0)
    return float(np.sum(values[:, 0] * following[:, 1] - following[:, 0] * values[:, 1]) / 2.0)


def orient_counterclockwise(points: Points) -> Points:
    """Return the ring wound counterclockwise, leaving CCW input untouched."""
    return points if signed_area(points) >= 0 else points[::-1]


def is_simple(points: Points) -> bool:
    """True when the ring does not self-intersect."""
    if len(points) < MIN_RING_VERTICES:
        return False
    return bool(ShapelyPolygon(points.astype(np.float64)).is_valid)


def normalize_polyline(
    points: Points,
    *,
    duplicate_tolerance: float,
    collinear_tolerance: float,
) -> Points:
    """Dedup, remove collinear points, and orient counterclockwise."""
    result = drop_duplicate_points(points, duplicate_tolerance)
    result = drop_collinear_points(result, collinear_tolerance)
    if len(result) < MIN_RING_VERTICES:
        raise ValueError(
            f"polyline collapsed to {len(result)} vertices; a ring needs "
            f"at least {MIN_RING_VERTICES}"
        )
    return orient_counterclockwise(result)
