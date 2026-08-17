"""Fixture generators for the regression corpus.

Every fixture is a function, never a checked-in GDS: a committed binary
nobody can regenerate is a fixture nobody can fix.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import bezier_um
from masklayout.model.geometry import Polygon
from masklayout.pcells.wires import offset_centerline_um

TARGET_LAYER = 10


def _rect(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=TARGET_LAYER)


def _rotate(points: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.column_stack(
        (
            points[:, 0] * math.cos(a) - points[:, 1] * math.sin(a),
            points[:, 0] * math.sin(a) + points[:, 1] * math.cos(a),
        )
    )


def _rotated_bar(length_nm: int, width_nm: int, deg: float) -> Polygon:
    pts = np.array([[0, 0], [length_nm, 0], [length_nm, width_nm], [0, width_nm]], dtype=np.float64)
    return Polygon(points=np.round(_rotate(pts, deg)).astype(np.int64), layer=TARGET_LAYER)


def isolated_line() -> list[Polygon]:
    """A single bar with nothing near it: unbounded space at every edge."""
    return [_rect(0, 0, 4000, 100)]


def dense_lines() -> list[Polygon]:
    """Three bars on a 160 nm pitch: 60 nm spaces."""
    return [_rect(0, y, 4000, y + 100) for y in (0, 160, 320)]


def semi_isolated_lines() -> list[Polygon]:
    """Two bars far enough apart to be neither dense nor isolated."""
    return [_rect(0, 0, 4000, 100), _rect(0, 400, 4000, 500)]


def line_ends_manhattan() -> list[Polygon]:
    """Collinear bars with a 60 nm gap: line ends at 0 degrees, facing each other.

    Note the absolute coordinates. An earlier version wrote
    ``_rect(2060, 0, 2000, 100)`` as though the signature were
    (x0, length, width), which produced a 60 nm stub abutting the first bar
    rather than a separate bar across a gap — and the two merged under
    correction, which is what exposed it.
    """
    return [_rect(0, 0, 2000, 100), _rect(2060, 0, 4060, 100)]


def line_ends_45() -> list[Polygon]:
    """A bar at 45 degrees."""
    return [_rotated_bar(2000, 100, 45.0)]


def line_ends_arbitrary() -> list[Polygon]:
    """A bar at 37 degrees: the non-Manhattan case this toolkit exists for."""
    return [_rotated_bar(2000, 100, 37.0)]


def convex_and_concave_corners() -> list[Polygon]:
    """An L shape: five convex corners and one concave."""
    pts = np.array(
        [[0, 0], [3000, 0], [3000, 1000], [1000, 1000], [1000, 3000], [0, 3000]],
        dtype=np.int64,
    )
    return [Polygon(points=pts, layer=TARGET_LAYER)]


def acute_corner() -> list[Polygon]:
    """A sharp wedge: a corner well below 90 degrees."""
    pts = np.array([[0, 0], [3000, 100], [3000, 200], [0, 60]], dtype=np.int64)
    return [Polygon(points=pts, layer=TARGET_LAYER)]


def narrow_neck() -> list[Polygon]:
    """A bar pinched to 8 nm: below any sane minimum width."""
    pts = np.array(
        [[0, 0], [2000, 0], [2000, 100], [1020, 100], [1000, 8], [980, 100], [0, 100]],
        dtype=np.int64,
    )
    return [Polygon(points=pts, layer=TARGET_LAYER)]


def contact_array() -> list[Polygon]:
    """A 3 x 3 grid of contacts, flattened for corpus comparison."""
    return [_rect(x, y, x + 200, y + 200) for x in (0, 500, 1000) for y in (0, 500, 1000)]


def curvilinear_wire() -> list[Polygon]:
    """A tessellated Bezier wire: hundreds of short chords at every angle."""
    tech = TechConfig()
    centre = bezier_um(
        np.array([[0.0, 0.0], [2.0, 3.0], [6.0, -3.0], [8.0, 0.0]]),
        tech.max_chord_error_nm / 1000.0,
    )
    widths = np.full(len(centre), 0.4)
    polygon, _ = compile_polyline(offset_centerline_um(centre, widths), tech, TARGET_LAYER, 0)
    return [polygon]


def pathological_sliver() -> list[Polygon]:
    """A polygon right at the minimum-area boundary."""
    return [_rect(0, 0, 2, 2)]  # 4 nm^2, exactly the default minimum


#: Every pattern class the corpus covers. Keys are golden filenames.
ALL_PATTERNS: dict[str, Callable[[], list[Polygon]]] = {
    "isolated_line": isolated_line,
    "dense_lines": dense_lines,
    "semi_isolated_lines": semi_isolated_lines,
    "line_ends_manhattan": line_ends_manhattan,
    "line_ends_45": line_ends_45,
    "line_ends_arbitrary": line_ends_arbitrary,
    "convex_and_concave_corners": convex_and_concave_corners,
    "acute_corner": acute_corner,
    "narrow_neck": narrow_neck,
    "contact_array": contact_array,
    "curvilinear_wire": curvilinear_wire,
    "pathological_sliver": pathological_sliver,
}
