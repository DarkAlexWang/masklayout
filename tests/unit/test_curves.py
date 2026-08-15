"""Curve tessellation within a chord-error budget."""

import math

import numpy as np
import pytest

from masklayout.geometry.curves import arc_um, bezier_um, circle_um, rounded_rect_um


def _max_radial_error(points: np.ndarray, centre: tuple[float, float], radius: float) -> float:
    """Largest sagitta between the polyline and the true circle."""
    shifted = points - np.array(centre, dtype=np.float64)
    midpoints = (shifted + np.roll(shifted, -1, axis=0)) / 2.0
    return float(np.max(radius - np.linalg.norm(midpoints, axis=1)))


@pytest.mark.parametrize("radius_um", [0.05, 1.0, 10.0])
@pytest.mark.parametrize("budget_um", [0.01, 0.001])
def test_circle_never_exceeds_the_chord_error_budget(radius_um: float, budget_um: float) -> None:
    pts = circle_um((0.0, 0.0), radius_um, max_chord_error_um=budget_um)
    assert _max_radial_error(pts, (0.0, 0.0), radius_um) <= budget_um


def test_tighter_budget_produces_more_vertices() -> None:
    coarse = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.01)
    fine = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.0001)
    assert len(fine) > len(coarse)


def test_circle_is_closed_without_repeating_the_first_point() -> None:
    pts = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.001)
    assert not np.allclose(pts[0], pts[-1])


def test_arc_spans_exactly_the_requested_angles() -> None:
    pts = arc_um((0.0, 0.0), 2.0, 0.0, math.pi / 2, max_chord_error_um=0.001)
    assert pts[0] == pytest.approx([2.0, 0.0], abs=1e-9)
    assert pts[-1] == pytest.approx([0.0, 2.0], abs=1e-9)


def test_arc_error_is_bounded() -> None:
    pts = arc_um((0.0, 0.0), 5.0, 0.0, math.pi, max_chord_error_um=0.002)
    mids = (pts[:-1] + pts[1:]) / 2.0
    assert float(np.max(5.0 - np.linalg.norm(mids, axis=1))) <= 0.002


def test_bezier_endpoints_are_exact() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -2.0], [4.0, 0.0]])
    pts = bezier_um(controls, max_chord_error_um=0.001)
    assert pts[0] == pytest.approx([0.0, 0.0])
    assert pts[-1] == pytest.approx([4.0, 0.0])


def test_bezier_refines_with_a_tighter_budget() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -2.0], [4.0, 0.0]])
    assert len(bezier_um(controls, 0.0001)) > len(bezier_um(controls, 0.01))


def test_rounded_rect_corner_radius_is_respected() -> None:
    pts = rounded_rect_um((0.0, 0.0), (10.0, 6.0), radius_um=1.0, max_chord_error_um=0.001)
    assert pts.min(axis=0) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert pts.max(axis=0) == pytest.approx([10.0, 6.0], abs=1e-6)
    # No vertex may sit in the square corner cut away by the radius.
    corner = np.logical_and(pts[:, 0] < 1e-9, pts[:, 1] < 1e-9)
    assert not corner.any()


def test_rounded_rect_rejects_a_radius_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="radius"):
        rounded_rect_um((0.0, 0.0), (2.0, 2.0), radius_um=5.0, max_chord_error_um=0.001)
