"""Analytic curve tessellation.

Every function returns float micrometre points. Quantization to the design
grid happens later, in ``compile.py`` — see the design document, section
"Units and coordinate model".

The vertex count comes from inverting the sagitta relation. For a circular
arc of radius r split into segments subtending angle t, the chord's maximum
deviation from the arc is r * (1 - cos(t / 2)). Solving for t at a budget e
gives t = 2 * arccos(1 - e / r), and the segment count is the arc span
divided by t, rounded up.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

#: Never emit a closed curve coarser than this, regardless of budget.
_MIN_CIRCLE_SEGMENTS = 8

#: Maximum refinement doublings before giving up on a Bezier budget.
_MAX_BEZIER_REFINEMENTS = 16


def _segments_for_arc(radius_um: float, span_rad: float, max_chord_error_um: float) -> int:
    """Segment count that keeps the sagitta within budget."""
    if radius_um <= 0.0:
        raise ValueError(f"radius must be positive, got {radius_um}")
    if max_chord_error_um <= 0.0:
        raise ValueError(f"max_chord_error_um must be positive, got {max_chord_error_um}")
    if max_chord_error_um >= radius_um:
        return _MIN_CIRCLE_SEGMENTS
    step = 2.0 * math.acos(1.0 - max_chord_error_um / radius_um)
    return max(math.ceil(abs(span_rad) / step), 1)


def arc_um(
    centre_um: tuple[float, float],
    radius_um: float,
    start_rad: float,
    end_rad: float,
    max_chord_error_um: float,
) -> NDArray[np.float64]:
    """Tessellate a circular arc. Both endpoints are exact."""
    span = end_rad - start_rad
    count = _segments_for_arc(radius_um, span, max_chord_error_um)
    angles = np.linspace(start_rad, end_rad, count + 1, dtype=np.float64)
    return np.column_stack(
        (
            centre_um[0] + radius_um * np.cos(angles),
            centre_um[1] + radius_um * np.sin(angles),
        )
    )


def circle_um(
    centre_um: tuple[float, float], radius_um: float, max_chord_error_um: float
) -> NDArray[np.float64]:
    """Tessellate a full circle as a closed ring with no repeated vertex."""
    count = max(
        _segments_for_arc(radius_um, 2.0 * math.pi, max_chord_error_um),
        _MIN_CIRCLE_SEGMENTS,
    )
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False, dtype=np.float64)
    return np.column_stack(
        (
            centre_um[0] + radius_um * np.cos(angles),
            centre_um[1] + radius_um * np.sin(angles),
        )
    )


def _bezier_at(controls: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    """De Casteljau evaluation for an arbitrary-degree Bezier."""
    points = controls.astype(np.float64)[:, None, :].repeat(len(t), axis=1)
    for _ in range(len(controls) - 1):
        points = points[:-1] * (1.0 - t)[None, :, None] + points[1:] * t[None, :, None]
    return np.asarray(points[0], dtype=np.float64)


def bezier_um(
    control_points_um: NDArray[np.float64], max_chord_error_um: float
) -> NDArray[np.float64]:
    """Tessellate a Bezier curve by refining until the budget is met.

    A chord's deviation from the curve is estimated by evaluating the curve at
    the chord's midpoint parameter and measuring the distance to the chord
    midpoint. Sampling doubles until the worst case is within budget.
    """
    if max_chord_error_um <= 0.0:
        raise ValueError(f"max_chord_error_um must be positive, got {max_chord_error_um}")
    controls = np.asarray(control_points_um, dtype=np.float64)
    if len(controls) < 2:
        raise ValueError(f"a Bezier needs at least 2 control points, got {len(controls)}")

    count = 8
    points = _bezier_at(controls, np.linspace(0.0, 1.0, count + 1, dtype=np.float64))
    for _ in range(_MAX_BEZIER_REFINEMENTS):
        t = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
        points = _bezier_at(controls, t)
        chord_mid = (points[:-1] + points[1:]) / 2.0
        curve_mid = _bezier_at(controls, (t[:-1] + t[1:]) / 2.0)
        if float(np.max(np.linalg.norm(curve_mid - chord_mid, axis=1))) <= max_chord_error_um:
            return points
        count *= 2
    return points


def rounded_rect_um(
    lower_um: tuple[float, float],
    upper_um: tuple[float, float],
    radius_um: float,
    max_chord_error_um: float,
) -> NDArray[np.float64]:
    """An axis-aligned rectangle with circular corner fillets."""
    width = upper_um[0] - lower_um[0]
    height = upper_um[1] - lower_um[1]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"degenerate rectangle {lower_um} to {upper_um}")
    if radius_um <= 0.0:
        raise ValueError(f"radius must be positive, got {radius_um}")
    if 2.0 * radius_um > min(width, height):
        raise ValueError(
            f"corner radius {radius_um} does not fit in a {width} x {height} rectangle"
        )

    left, bottom = lower_um
    right, top = upper_um
    corners = [
        ((right - radius_um, bottom + radius_um), -math.pi / 2, 0.0),
        ((right - radius_um, top - radius_um), 0.0, math.pi / 2),
        ((left + radius_um, top - radius_um), math.pi / 2, math.pi),
        ((left + radius_um, bottom + radius_um), math.pi, 3.0 * math.pi / 2),
    ]
    pieces = [
        arc_um(centre, radius_um, start, end, max_chord_error_um) for centre, start, end in corners
    ]
    return np.vstack(pieces)
